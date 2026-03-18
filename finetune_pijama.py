"""
Finetune a pretrained AMT model on the PiJAMa dataset.

Requires train.txt and valid.txt produced by prepare_pijama.py.
All settings are in pijama_config.yaml.

Usage:
    python finetune_pijama.py
"""

import os
import yaml

# Patch for huggingface_hub versions that don't expose is_offline_mode at the top level
import huggingface_hub
if not hasattr(huggingface_hub, 'is_offline_mode'):
    huggingface_hub.is_offline_mode = lambda: True

import math
import torch
import wandb
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback

from anticipation.config import CONTEXT_SIZE, TIME_RESOLUTION
from anticipation.vocab import VOCAB_SIZE, AUTOREGRESS, NOTE_OFFSET, TIME_OFFSET, DUR_OFFSET, CONTROL_OFFSET, MAX_PITCH
from anticipation.sample import generate
from anticipation.convert import events_to_midi


class MidiTokenDataset(Dataset):
    def __init__(self, filepath):
        self.sequences = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = list(map(int, line.split()))
                if len(tokens) == CONTEXT_SIZE:
                    self.sequences.append(tokens)
        print(f'  Loaded {len(self.sequences)} sequences from {filepath}')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.sequences[idx], dtype=torch.long)
        # labels=input_ids: HuggingFace CausalLM shifts internally for the loss
        return {'input_ids': tokens, 'labels': tokens.clone()}


class PijamaCallback(TrainerCallback):
    """Logs perplexity, GPU stats, LoRA weight norms, and audio samples to W&B."""

    def __init__(self, model, sample_every_steps=500, sample_length=10):
        self.model = model
        self.sample_every_steps = sample_every_steps
        self.sample_length = sample_length

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            wandb.log({'eval/perplexity': math.exp(metrics['eval_loss'])})

    def on_log(self, args, state, control, logs=None, **kwargs):
        log_dict = {}
        if logs and 'loss' in logs:
            log_dict['train/perplexity'] = math.exp(logs['loss'])

        if torch.cuda.is_available():
            log_dict['system/gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            log_dict['system/gpu_memory_reserved_gb']  = torch.cuda.memory_reserved()  / 1e9

        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                log_dict[f'lora_norms/{name}'] = param.norm().item()

        if log_dict:
            wandb.log(log_dict)

    def _make_controls(self, instrument, pitches, dur_s=0.5):
        """Build a list of control tokens for a simple melodic line.

        instrument : MIDI program number (52=melody, 0=piano chords)
        pitches    : list of MIDI pitches played evenly spaced by dur_s seconds
        """
        controls = []
        for i, pitch in enumerate(pitches):
            t_bins   = int(i * dur_s * TIME_RESOLUTION)
            d_bins   = max(1, int(dur_s * TIME_RESOLUTION) - 1)
            note_tok = NOTE_OFFSET + instrument * MAX_PITCH + pitch
            controls += [
                CONTROL_OFFSET + TIME_OFFSET + t_bins,
                CONTROL_OFFSET + DUR_OFFSET  + d_bins,
                CONTROL_OFFSET + note_tok,
            ]
        return controls

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.sample_every_steps != 0:
            return

        # Simple C major scale as melody (instrument 52)
        scale = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 60]
        # Simple C-F-G chord roots as piano (instrument 0)
        chords = [48, 48, 53, 53, 55, 55, 48, 48]

        melody_ctrl = self._make_controls(52, scale,  dur_s=self.sample_length / len(scale))
        chord_ctrl  = self._make_controls(0,  chords, dur_s=self.sample_length / len(chords))

        samples = [
            ('free',        [],           []),
            ('melody_ctrl', [],           melody_ctrl),
            ('chord_ctrl',  [],           chord_ctrl),
        ]

        self.model.eval()
        log_dict = {}
        try:
            with torch.no_grad():
                for label, inputs, controls in samples:
                    events = generate(
                        self.model,
                        start_time=0,
                        end_time=self.sample_length,
                        inputs=inputs,
                        controls=controls,
                        top_p=0.98,
                    )
                    # Strip control tokens — they're interleaved in the output
                    # but events_to_midi doesn't know to ignore them, so they'd
                    # decode as phantom notes in unintended instrument slots.
                    events_only = [t for t in events if t < CONTROL_OFFSET]
                    mid  = events_to_midi(events_only)
                    path = f'sample_{label}_step{state.global_step}.mid'
                    mid.save(path)
                    log_dict[f'samples/{label}'] = wandb.Audio(path, caption=f'{label} step {state.global_step}')
            wandb.log(log_dict)
        except Exception as e:
            print(f'[PijamaCallback] sample generation failed: {e}')
        finally:
            self.model.train()


def main(cfg):
    # --- W&B ---
    wandb.init(
        project=cfg['wandb_project'],
        name=cfg['wandb_run'],
        config={
            'model': cfg['model'],
            'lora': cfg['lora'],
            'epochs': cfg['epochs'],
            'batch_size': cfg['batch_size'],
            'grad_accum': cfg['grad_accum'],
            'effective_batch': cfg['batch_size'] * cfg['grad_accum'],
            'lr': cfg['lr'],
            'data_dir': cfg['output_dir'],
        },
    )

    # --- Model ---
    print(f'Loading model: {cfg["model"]}')
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True)

    if cfg['lora']:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError('Install peft: pip install peft')

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            # GPT-2 style attention projection names used by stanford-crfm models
            target_modules=['c_attn', 'c_proj'],
            lora_dropout=0.05,
            bias='none',
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # --- Data ---
    print('Loading datasets...')
    train_dataset = MidiTokenDataset(os.path.join(cfg['output_dir'], 'train.txt'))
    valid_dataset = MidiTokenDataset(os.path.join(cfg['output_dir'], 'valid.txt'))

    # --- Training args ---
    # Effective batch = per_device_train_batch_size * grad_accum_steps
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=cfg['checkpoint_dir'],
        num_train_epochs=cfg['epochs'],
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        weight_decay=0.1,
        warmup_steps=200,
        lr_scheduler_type='cosine',
        eval_strategy='steps',
        eval_steps=cfg.get('eval_steps', 200),
        save_strategy='steps',
        save_steps=cfg.get('save_steps', 1000),
        save_total_limit=cfg.get('save_total_limit', 2),
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        logging_steps=50,
        bf16=bf16_supported,
        fp16=(not bf16_supported and torch.cuda.is_available()),
        dataloader_num_workers=2,
        report_to='wandb',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[PijamaCallback(
            model,
            sample_every_steps=cfg.get('sample_every_steps', 500),
            sample_length=cfg.get('sample_length', 10),
        )],
    )

    # --- Train ---
    print('Starting training...')
    trainer.train(resume_from_checkpoint=cfg['resume_from_checkpoint'])

    # --- Save ---
    final_dir = os.path.join(cfg['checkpoint_dir'], 'final')
    print(f'Saving model to {final_dir}')
    model.save_pretrained(final_dir)
    wandb.finish()
    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='pijama_config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)

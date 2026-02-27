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
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback

from anticipation.config import CONTEXT_SIZE
from anticipation.vocab import VOCAB_SIZE, AUTOREGRESS
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
            wandb.log({
                'eval/perplexity': math.exp(metrics['eval_loss']),
            })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            wandb.log({
                'train/perplexity': math.exp(logs['loss']),
            }, step=state.global_step)

        # GPU memory
        if torch.cuda.is_available():
            wandb.log({
                'system/gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'system/gpu_memory_reserved_gb':  torch.cuda.memory_reserved() / 1e9,
            }, step=state.global_step)

        # LoRA weight norms — how much the adapters have moved from init
        lora_norms = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_norms[f'lora_norms/{name}'] = param.norm().item()
        if lora_norms:
            wandb.log(lora_norms, step=state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.sample_every_steps != 0:
            return

        self.model.eval()
        try:
            with torch.no_grad():
                events = generate(
                    self.model,
                    start_time=0,
                    end_time=self.sample_length,
                    top_p=0.98,
                )
            mid = events_to_midi(events)
            path = f'sample_step{state.global_step}.mid'
            mid.save(path)
            wandb.log({'samples/midi': wandb.Audio(path, caption=f'step {state.global_step}')},
                      step=state.global_step)
        except Exception as e:
            print(f'[PijamaCallback] sample generation failed: {e}')
        finally:
            self.model.train()


class EpochProgressCallback(TrainerCallback):
    """Replaces the default full-run progress bar with a per-epoch bar."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        steps_per_epoch = state.max_steps // int(args.num_train_epochs)
        epoch = int(state.epoch) + 1
        self.pbar = tqdm(total=steps_per_epoch, desc=f'Epoch {epoch}/{int(args.num_train_epochs)}', leave=True)

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self, 'pbar'):
            self.pbar.update(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        if hasattr(self, 'pbar'):
            self.pbar.close()


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
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[
            PijamaCallback(
                model,
                sample_every_steps=cfg.get('sample_every_steps', 500),
                sample_length=cfg.get('sample_length', 10),
            ),
            EpochProgressCallback(),
        ],
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
    with open('pijama_config.yaml') as f:
        cfg = yaml.safe_load(f)
    main(cfg)

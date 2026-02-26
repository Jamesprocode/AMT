"""
Finetune a pretrained AMT model on the PiJAMa dataset.

Requires train.txt and valid.txt produced by prepare_pijama.py.
All settings are in pijama_config.yaml.

Usage:
    python finetune_pijama.py
"""

import os
import yaml

# Patch for huggingface_hub >= 1.0 which moved is_offline_mode out of the top-level namespace
import huggingface_hub
if not hasattr(huggingface_hub, 'is_offline_mode'):
    from huggingface_hub.utils import is_offline_mode
    huggingface_hub.is_offline_mode = is_offline_mode

import torch
import wandb
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

from anticipation.config import CONTEXT_SIZE
from anticipation.vocab import VOCAB_SIZE


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
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=3,
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

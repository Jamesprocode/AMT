"""
Preprocess POP909 dataset into AMT training format.

POP909 has 909 Chinese pop songs, each with 3 MIDI tracks:
  - MELODY : lead vocal melody
  - BRIDGE : counter-melody / inner voice
  - PIANO  : chord accompaniment

All 3 tracks are kept so AMT learns to predict any track given the others
(e.g. given MELODY, generate PIANO chords).

Produces train.txt and valid.txt — one 1024-token sequence per line,
space-separated integers, matching the AMT training format.

Usage:
    python prepare_pop909.py
"""

import os
import random
import yaml
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from anticipation import ops
from anticipation.config import EVENT_SIZE, M, MAX_TIME
from anticipation.vocab import SEPARATOR, AUTOREGRESS, ANTICIPATE
from anticipation.convert import midi_to_compound
from anticipation.tokenize import maybe_tokenize, extract_spans, extract_random, ANTICIPATION_RATES


def process_midi(args):
    midi_path, augment_factor = args
    try:
        compound = midi_to_compound(midi_path)
    except Exception:
        return []
    return make_sequences(compound, augment_factor=augment_factor)


def make_sequences(compound_tokens, augment_factor=1):
    all_events, truncations, status = maybe_tokenize(compound_tokens)
    if status > 0:
        return []

    sequences = []
    concatenated_tokens = []
    end_time = ops.max_time(all_events, seconds=False)
    z = AUTOREGRESS

    for k in range(augment_factor):
        if k % 10 == 0:
            events = all_events.copy()
            controls = []
        elif k % 10 == 1:
            events, controls = extract_spans(all_events, rate=0.05)
        elif k % 10 < 6:
            r = np.random.randint(1, ANTICIPATION_RATES)
            events, controls = extract_random(all_events, r)
        else:
            r = np.random.randint(1, ANTICIPATION_RATES)
            events, controls = extract_random(all_events, r)

        if len(concatenated_tokens) == 0:
            z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

        events = ops.pad(events, end_time)
        tokens, controls = ops.anticipate(events, controls)
        tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
        concatenated_tokens.extend(tokens)

        while len(concatenated_tokens) >= EVENT_SIZE * M:
            seq = concatenated_tokens[:EVENT_SIZE * M]
            concatenated_tokens = concatenated_tokens[EVENT_SIZE * M:]

            seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
            if ops.max_time(seq, seconds=False) >= MAX_TIME:
                continue

            seq.insert(0, z)
            sequences.append(seq)
            z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

    return sequences


def get_splits(data_dir, seed=42, valid_ratio=0.1):
    """Find all POP909 main MIDI files and split into train/valid."""
    random.seed(seed)
    # POP909 structure: data_dir/NNN/NNN.mid (exclude versions/ subfolder)
    all_midi = sorted(glob(os.path.join(data_dir, '*', '*.mid')))
    all_midi = [f for f in all_midi if 'versions' not in f]

    random.shuffle(all_midi)
    n_valid = max(1, int(len(all_midi) * valid_ratio))
    return all_midi[n_valid:], all_midi[:n_valid]  # train, valid


def main(cfg):
    os.makedirs(cfg['output_dir'], exist_ok=True)
    np.random.seed(cfg['seed'])

    print(f'Scanning {cfg["data_dir"]} ...')
    train_files, valid_files = get_splits(
        cfg['data_dir'], seed=cfg['seed'], valid_ratio=cfg.get('valid_ratio', 0.1)
    )
    print(f'  Train files : {len(train_files)}')
    print(f'  Valid files : {len(valid_files)}')

    workers = cfg.get('workers', cpu_count())
    print(f'Using {workers} workers')

    for split_name, files, augment in [
        ('train', train_files, cfg['augment_factor']),
        ('valid', valid_files, 1),
    ]:
        out_path = os.path.join(cfg['output_dir'], f'{split_name}.txt')
        sequences = []

        print(f'\nProcessing {split_name} split (augment x{augment}) ...')
        with Pool(processes=workers) as pool:
            for seqs in tqdm(pool.imap(process_midi, [(f, augment) for f in files]), total=len(files)):
                sequences.extend(seqs)

        if split_name == 'train':
            random.shuffle(sequences)

        with open(out_path, 'w') as f:
            for seq in sequences:
                f.write(' '.join(map(str, seq)) + '\n')

        print(f'  => {len(sequences)} sequences written to {out_path}')

    print('\nDone. Run finetune_pijama.py with pop909_config.yaml next.')


if __name__ == '__main__':
    with open('pop909_config.yaml') as f:
        cfg = yaml.safe_load(f)
    main(cfg)

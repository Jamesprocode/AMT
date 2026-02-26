"""
Preprocess PiJAMa dataset into AMT training format.

Produces train.txt and valid.txt — one 1024-token sequence per line,
space-separated integers, matching the format used to train the original
Lakh models.

Validation split is by album (one album per pianist held out), following
the recommendation in the PiJAMa paper to avoid the acoustic bias of
track-level splits.

Usage:
    python prepare_pijama.py
"""

import os
import random
import yaml
from glob import glob
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from anticipation import ops
from anticipation.config import (
    CONTEXT_SIZE, EVENT_SIZE, M,
    MAX_TIME, TIME_RESOLUTION,
    MIN_TRACK_EVENTS, MIN_TRACK_TIME_IN_SECONDS, MAX_TRACK_TIME_IN_SECONDS,
    DELTA, COMPOUND_SIZE, MAX_TRACK_INSTR,
)
from anticipation.vocab import (
    SEPARATOR, AUTOREGRESS, ANTICIPATE,
    CONTROL_OFFSET, REST,
)
from anticipation.convert import midi_to_compound, compound_to_events
from anticipation.tokenize import maybe_tokenize, extract_spans, extract_random, ANTICIPATION_RATES


def process_midi(midi_path):
    """Convert a MIDI file to compound tokens. Returns None on failure."""
    try:
        return midi_to_compound(midi_path)
    except Exception:
        return None


def make_sequences(compound_tokens, augment_factor=1):
    """
    Convert compound tokens to a list of 1024-token AMT training sequences.

    augment_factor=1  -> autoregressive only (no anticipation augmentation)
    augment_factor=10 -> full augmentation matching original Lakh training
    """
    all_events, truncations, status = maybe_tokenize(compound_tokens)
    if status > 0:
        return []

    sequences = []
    concatenated_tokens = []
    instruments = list(ops.get_instruments(all_events).keys())
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
            # instrument augmentation — skipped for solo piano (single instrument)
            events = all_events.copy()
            controls = []

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


def get_album_splits(data_dir, seed=42):
    """
    Group .midi files by (pianist, album) and hold out one album per
    pianist for validation.

    Returns (train_files, valid_files).
    """
    random.seed(seed)

    # Structure: data_dir/<live|studio>/<Pianist>/<Album>/<Track>.midi
    albums = defaultdict(list)
    all_midi = glob(os.path.join(data_dir, '**', '*.midi'), recursive=True)

    for path in all_midi:
        parts = path.replace(data_dir, '').lstrip(os.sep).split(os.sep)
        if len(parts) >= 3:
            pianist = parts[-3]
            album = parts[-2]
            albums[(pianist, album)].append(path)

    train_files, valid_files = [], []

    pianists = list(set(p for p, _ in albums.keys()))
    for pianist in pianists:
        pianist_albums = [(p, a) for (p, a) in albums.keys() if p == pianist]
        if len(pianist_albums) > 1:
            held_out = random.choice(pianist_albums)
            for key, files in albums.items():
                if key[0] != pianist:
                    continue
                if key == held_out:
                    valid_files.extend(files)
                else:
                    train_files.extend(files)
        else:
            for key, files in albums.items():
                if key[0] == pianist:
                    train_files.extend(files)

    return train_files, valid_files


def main(cfg):
    os.makedirs(cfg['output_dir'], exist_ok=True)
    np.random.seed(cfg['seed'])

    print(f'Scanning {cfg["data_dir"]} ...')
    train_files, valid_files = get_album_splits(cfg['data_dir'], seed=cfg['seed'])
    print(f'  Train files : {len(train_files)}')
    print(f'  Valid files : {len(valid_files)}')

    splits = [
        ('train', train_files, cfg['augment_factor']),
        ('valid', valid_files, 1),
    ]

    for split_name, files, augment in splits:
        out_path = os.path.join(cfg['output_dir'], f'{split_name}.txt')
        sequences = []
        skipped = 0

        print(f'\nProcessing {split_name} split (augment x{augment}) ...')
        for midi_path in tqdm(files):
            compound = process_midi(midi_path)
            if compound is None:
                skipped += 1
                continue
            seqs = make_sequences(compound, augment_factor=augment)
            sequences.extend(seqs)

        if split_name == 'train':
            random.shuffle(sequences)

        with open(out_path, 'w') as f:
            for seq in sequences:
                f.write(' '.join(map(str, seq)) + '\n')

        print(f'  => {len(sequences)} sequences written to {out_path}')
        if skipped:
            print(f'  => {skipped} files skipped (parse errors)')

    print('\nDone. Run finetune_pijama.py next.')


if __name__ == '__main__':
    with open('pijama_config.yaml') as f:
        cfg = yaml.safe_load(f)
    main(cfg)

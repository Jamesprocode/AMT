#!/bin/bash
set -e  # stop on any error

echo "========================================="
echo " POP909 Training Pipeline"
echo "========================================="

# ── Aug 1 ─────────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Tokenizing aug1 data..."
python prepare_pop909.py --config pop909_config_aug1.yaml

echo "[2/4] Training aug1..."
python finetune_pijama.py --config pop909_config_aug1.yaml

# ── Aug 5 ─────────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Tokenizing aug5 data..."
python prepare_pop909.py --config pop909_config_aug5.yaml

echo "[4/4] Training aug5..."
python finetune_pijama.py --config pop909_config_aug5.yaml

echo ""
echo "========================================="
echo " All done!"
echo "========================================="

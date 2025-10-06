#!/bin/bash
# Example script for scoring sequences for all PDBs under examples/example_data/native_pdbs.

source env_setup.sh
python3 caliby/eval/sampling/score.py \
    ckpt_path=model_params/caliby/caliby.ckpt \
    input_cfg.pdb_dir=examples/example_data/native_pdbs \
    out_dir=examples/outputs/score

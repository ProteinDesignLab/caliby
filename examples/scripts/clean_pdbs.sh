#!/bin/bash
# Example script for cleaning PDB files for use with Protpardelle-1c ensemble generation.

source env_setup.sh
python3 caliby/data/preprocessing/atomworks/clean_pdbs.py \
    input_cfg.pdb_dir=examples/example_data/native_pdbs \
    num_workers=4 \
    out_dir=examples/outputs/cleaned_pdbs

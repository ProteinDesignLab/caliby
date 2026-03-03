#!/bin/bash
# Example script for designing sequences for all PDBs under examples/example_data/native_pdbs,
# with position-wise constraints specified in examples/example_data/pos_constraint_csvs/native_pdb_constraints.csv.

source env_setup.sh
python3 caliby/eval/sampling/seq_des.py \
    ckpt_name_or_path=caliby \
    input_cfg.pdb_dir=examples/example_data/native_pdbs \
    pos_constraint_csv=examples/example_data/pos_constraint_csvs/native_pdb_constraints.csv \
    sampling_cfg_overrides.num_seqs_per_pdb=4 \
    out_dir=examples/outputs/seq_des_constraints

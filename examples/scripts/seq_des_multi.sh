#!/bin/bash
# Example script for designing sequences for all PDBs under examples/example_data/native_pdbs.

source env_setup.sh
python3 caliby/eval/sampling/seq_des_multi.py \
    ckpt_name_or_path=caliby \
    input_cfg.pdb_dir=examples/example_data/native_pdbs \
    sampling_cfg_overrides.num_seqs_per_pdb=4 \
    out_dir=examples/outputs/seq_des_multi

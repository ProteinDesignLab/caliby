#!/bin/bash
# Example script for designing sequences for 2 PDBs under examples/example_data/native_pdbs,
# specified in examples/example_data/pdb_name_lists/2_native_pdbs.txt.

source env_setup.sh
python3 caliby/eval/sampling/seq_des.py \
    ckpt_name_or_path=caliby \
    input_cfg.pdb_dir=examples/example_data/native_pdbs \
    input_cfg.pdb_name_list=examples/example_data/pdb_name_lists/2_native_pdbs.txt \
    sampling_cfg_overrides.num_seqs_per_pdb=4 \
    out_dir=examples/outputs/seq_des_subset

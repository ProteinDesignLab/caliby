#!/bin/bash
# Example script for designing sequences for 2 ensembles
# specified in examples/example_data/pdb_name_lists/2_native_pdbs.txt.

CONFORMER_DIR=examples/outputs/generate_ensembles/cc95-epoch3490-sampling_partial_diffusion-ss1.0-schurn0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150

source env_setup.sh
python3 caliby/eval/sampling/seq_des_multi_ensemble.py \
    ckpt_path=model_params/caliby/caliby.ckpt \
    input_cfg.conformer_dir=${CONFORMER_DIR} \
    input_cfg.pdb_name_list=examples/example_data/pdb_name_lists/2_native_pdbs.txt \
    sampling_cfg_overrides.num_seqs_per_pdb=4 \
    out_dir=examples/outputs/seq_des_multi_ensemble_subset

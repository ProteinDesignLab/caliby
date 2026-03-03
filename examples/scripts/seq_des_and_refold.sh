#!/bin/bash
# Example script for ensemble-conditioned sequence design with AF2 self-consistency evaluation
# for 2 ensembles specified in examples/example_data/pdb_name_lists/2_native_pdbs.txt.

CONFORMER_DIR=examples/outputs/generate_ensembles/cc95-epoch3490-sampling_partial_diffusion-ss1.0-schurn0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150

source env_setup.sh
python3 caliby/eval/sampling/seq_des_ensemble.py \
    ckpt_name_or_path=soluble_caliby \
    input_cfg.conformer_dir=${CONFORMER_DIR} \
    input_cfg.pdb_name_list=examples/example_data/pdb_name_lists/2_native_pdbs.txt \
    sampling_cfg_overrides.num_seqs_per_pdb=2 \
    run_self_consistency_eval=true \
    out_dir=examples/outputs/seq_des_and_refold

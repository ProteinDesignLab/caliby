#!/bin/bash
# Example script for packing sidechains for all PDBs under examples/example_data/native_pdbs.
# Uses the packer model trained with 0.1A backbone noise.

source env_setup.sh
python3 caliby/eval/sampling/sidechain_pack.py \
    ckpt_path=model_params/caliby/caliby_packer_010.ckpt \
    input_cfg.pdb_dir=examples/example_data/native_pdbs \
    out_dir=examples/outputs/sidechain_pack

#!/bin/bash
# Example script for generating ensembles with Protpardelle-1c.

source env_setup.sh
python3 caliby/eval/sampling/generate_ensembles.py \
    model_params_path=model_params \
    input_cfg.pdb_dir=examples/example_data/homooligomers \
    num_samples_per_pdb=8 \
    out_dir=examples/outputs/homooligomers/generate_ensembles

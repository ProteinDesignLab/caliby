#!/bin/bash
# Example script for generating a symmetry_pos entry for a homooligomer by specifying chain IDs and residue ranges.
# The example below generates a symmetry_pos entry for the 1TNF-assembly1 entry in examples/example_data/pos_constraint_csvs/homooligomer_constraints.csv.

source env_setup.sh
python3 caliby/data/preprocessing/helper_scripts/make_homooligomer_symmetry_pos.py \
    --chains A B C \
    --residue-range 6 157

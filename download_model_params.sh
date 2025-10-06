#!/bin/bash

# Download Caliby and Protpardelle-1c model parameters.
wget -O model_params.tar https://zenodo.org/records/17263678/files/model_params.tar?download=1

# Extract model parameters.
tar -xvf model_params.tar

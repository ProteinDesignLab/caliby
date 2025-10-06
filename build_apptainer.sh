#!/bin/bash

# Set environment and image paths.
ENV_DIR=envs
IMG=${PWD}/containers/pytorch_24.12.sif  # must be an absolute path

# Pull apptainer image.
mkdir -p ${ENV_DIR}
mkdir -p $(dirname ${IMG})
apptainer pull ${IMG} docker://nvcr.io/nvidia/pytorch:24.12-py3

# Install environment within apptainer.
apptainer exec --nv \
  --bind ${ENV_DIR}:${ENV_DIR} \
  ${IMG} bash -lc '

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv ${ENV_DIR}/caliby -p python3.12
source ${ENV_DIR}/caliby/bin/activate
uv pip sync uv_indexes.txt uv.lock -c pins.txt --index-strategy=unsafe-best-match
uv pip install -e .
'

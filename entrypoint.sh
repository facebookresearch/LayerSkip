#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# entrypoint.sh

# Activate the Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate layer_skip

# Export environment variables (e.g., HUGGINGFACE_TOKEN)
export HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Execute the passed command
exec "$@"

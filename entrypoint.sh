#!/bin/bash
# entrypoint.sh

# Activate the Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate layer_skip

# Export environment variables (e.g., HUGGINGFACE_TOKEN)
export HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Execute the passed command
exec "$@"

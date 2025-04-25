#!/bin/bash
# Script to launch distributed training across multiple GPUs

# Configuration
NUM_GPUS=${1:-2}  # Default to 2 GPUs, but can be specified as the first argument
MASTER_PORT=29500  # Port for the master process
MODEL_PATH="facebook/layerskip-llama3.2-1B"  # Path to the model
DATASET="race_m"  # Dataset to use
NUM_EXAMPLES=300  # Number of examples to process
EXTRACT_BATCH_SIZE=2  # Batch size for extraction
BATCH_SIZE=32  # Batch size for training
EPOCHS=3  # Number of epochs
TARGET_LAYER=8  # Target layer for extraction
DEBERTA_HIDDEN_SIZE=768  # Hidden size for DeBERTa model
OUTPUT_DIR="./adapter_checkpoints"  # Output directory for checkpoints
TEMP_DATA_DIR="./temp_data"  # Directory for temporary data files
CHUNK_SIZE=50  # Chunk size for data processing
OPTIMIZE_MEMORY="--optimize_memory"  # Flag to enable memory optimizations
FP16="--fp16"  # Flag to enable mixed precision training
GRADIENT_CHECKPOINTING="--gradient_checkpointing"  # Flag to enable gradient checkpointing
GRADIENT_ACCUMULATION_STEPS=4  # Number of steps to accumulate gradients

# Create necessary directories
mkdir -p $OUTPUT_DIR
mkdir -p $TEMP_DATA_DIR

# Print configuration
echo "=== Configuration ==="
echo "Number of GPUs: $NUM_GPUS"
echo "Model path: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Number of examples: $NUM_EXAMPLES"
echo "Extract batch size: $EXTRACT_BATCH_SIZE"
echo "Training batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Target layer: $TARGET_LAYER"
echo "DeBERTa hidden size: $DEBERTA_HIDDEN_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Temp data directory: $TEMP_DATA_DIR"
echo "Chunk size: $CHUNK_SIZE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "======================="

# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_deberta_adapter.py \
    --model $MODEL_PATH \
    --dataset $DATASET \
    --num_examples $NUM_EXAMPLES \
    --extract_batch_size $EXTRACT_BATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --target_layer $TARGET_LAYER \
    --deberta_hidden_size $DEBERTA_HIDDEN_SIZE \
    --output_dir $OUTPUT_DIR \
    --temp_data_dir $TEMP_DATA_DIR \
    --chunk_size $CHUNK_SIZE \
    --distributed \
     --use_existing_data\
    $OPTIMIZE_MEMORY \
    $FP16 \
    $GRADIENT_CHECKPOINTING \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
import argparse
import os
import random
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
import transformers

# Import LayerSkip components
from self_speculation.generator_base import GenerationConfig
from generate import load_model_and_tokenizer, setup
from arguments import Arguments

# Import our adapter components
from LlamaLayerExtractor import LlamaLayerExtractor
from DebertaAdapter import DebertaAdapter
from AdapterTraining import IntermediateLayerDataset, prepare_training_data_chunked, train_adapter, load_data_from_h5

import h5py
import os
import numpy as np
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_training_data(args):
    """Load training data based on specified dataset."""
    if args.dataset == "race_m":
        dataset = load_dataset("race", "middle", split="train")
        data = []
        
        # Process only a subset for efficiency
        for i, item in enumerate(dataset):
            if i >= args.num_examples:
                break
                
            article = item['article']
            question = item['question']
            options = item['options']
            
            # Format as input to the model
            formatted_text = f"Article: {article}\n\nQuestion: {question}\n"
            for j, option in enumerate(options):
                formatted_text += f"{chr(65 + j)}. {option}\n"
            formatted_text += "Answer:"
            
            data.append({"text": formatted_text})
            
        logger.info(f"Loaded {len(data)} examples from RACE-M dataset")
        return data
    
    elif args.dataset == "math":
        dataset = load_dataset("HuggingFaceH4/math", split="train")
        data = []
        
        for i, item in enumerate(dataset):
            if i >= args.num_examples:
                break
                
            problem = item['problem']
            formatted_text = f"Problem: {problem}\n\nSolution:"
            data.append({"text": formatted_text})
            
        logger.info(f"Loaded {len(data)} examples from MATH dataset")
        return data
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def merge_h5_files(files, output_file):
    """Merge multiple H5 files from different ranks into one."""
    logger.info(f"Merging {len(files)} H5 files into {output_file}")
    
    # Get total size
    total_samples = 0
    first_file = h5py.File(files[0], 'r')
    hidden_size = first_file['hidden_states'].shape[1]
    vocab_size = first_file['full_logits'].shape[1]
    
    for file in files:
        with h5py.File(file, 'r') as f:
            total_samples += f['hidden_states'].shape[0]
    
    # Create output file
    with h5py.File(output_file, 'w') as out_f:
        # Create datasets
        out_f.create_dataset('hidden_states', shape=(total_samples, hidden_size), dtype=np.float16)
        out_f.create_dataset('ground_truth', shape=(total_samples,), dtype=np.int32)
        out_f.create_dataset('full_logits', shape=(total_samples, vocab_size), dtype=np.float16)
        
        # Copy data
        current_idx = 0
        for file in files:
            with h5py.File(file, 'r') as f:
                n_samples = f['hidden_states'].shape[0]
                out_f['hidden_states'][current_idx:current_idx+n_samples] = f['hidden_states'][:]
                out_f['ground_truth'][current_idx:current_idx+n_samples] = f['ground_truth'][:]
                out_f['full_logits'][current_idx:current_idx+n_samples] = f['full_logits'][:]
                current_idx += n_samples
    
    logger.info(f"Successfully merged files, total samples: {total_samples}")
    return output_file



def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa adapter for LayerSkip")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="facebook/layerskip-llama3.2-1B",
                        help="Path or name of the LayerSkip model")
    parser.add_argument("--target_layer", type=int, default=8,
                        help="Target layer for extraction and adaptation")
    parser.add_argument("--output_dir", type=str, default="./adapter_checkpoints",
                        help="Directory to save adapter checkpoints")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, choices=["race_m", "math", "custom"], default="race_m",
                        help="Dataset to use for training")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to custom training data (if using custom dataset)")
    parser.add_argument("--num_examples", type=int, default=1000,
                        help="Number of examples to use for training")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    
    # Adapter parameters
    parser.add_argument("--deberta_hidden_size", type=int, default=768,
                        help="Hidden size for DeBERTa model")
    parser.add_argument("--deberta_num_layers", type=int, default=2,
                        help="Number of layers for DeBERTa model")
    parser.add_argument("--deberta_num_heads", type=int, default=12,
                        help="Number of attention heads for DeBERTa model")
    parser.add_argument("--dropout_prob", type=float, default=0.1,
                        help="Dropout probability")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for adapter training")
    parser.add_argument("--extract_batch_size", type=int, default=4,
                        help="Batch size for extracting intermediate representations")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for adapter training")
    parser.add_argument("--ce_loss_weight", type=float, default=0.8,
                        help="Weight for cross-entropy loss")
    parser.add_argument("--kl_loss_weight", type=float, default=0.2,
                        help="Weight for KL divergence loss")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--temp_data_dir", type=str, default="./temp_data",
                        help="Directory to store temporary data files")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Number of samples to process before saving to disk")
    parser.add_argument("--use_existing_data", action="store_true",
                        help="Use existing processed data in temp_data directory instead of reprocessing")
    parser.add_argument("--keep_temp_files", action="store_true",
                        help="Keep temporary files after training")
    
    # Memory optimization
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--optimize_memory", action="store_true",
                        help="Enable various memory optimizations")
    
    # Distributed training parameters
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")
    parser.add_argument("--dist_url", type=str, default="env://",
                        help="URL used to set up distributed training")
    parser.add_argument("--world_size", type=int, default=-1,
                        help="Number of processes for distributed training")
    parser.add_argument("--rank", type=int, default=-1,
                        help="Rank of the current process")
    # Add both underscore and hyphen versions
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--skip_indices", type=str, default="",
                        help="Comma-separated list of dataset indices to skip (e.g., '375,376')")
    
    args = parser.parse_args()
    
    # Set distributed parameters based on environment variables if not explicitly set
    if args.distributed:
        if args.local_rank != -1:  # set by torch.distributed.launch
            args.rank = args.local_rank
        
        if args.world_size == -1:
            args.world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
            
        if args.rank == -1:
            args.rank = int(os.environ.get("RANK", 0))
    
    # Set device based on distributed settings
    if args.distributed:
        if args.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            # Initialize the distributed environment
            torch.distributed.init_process_group(backend="nccl")
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
    else:
        device = args.device
    
    # Only log from main process in distributed training
    if not args.distributed or (args.distributed and args.rank == 0):
        logger.info(f"Process rank: {args.rank}, device: {device}, world_size: {args.world_size}")
        logger.info(f"Distributed training: {args.distributed}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory - only from main process
    if not args.distributed or (args.distributed and args.rank == 0):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.temp_data_dir, exist_ok=True)
    
    # Initialize LayerSkip model arguments
    model_args = Arguments(
        model=args.model,
        model_args=None,
        dist_url=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        seed=args.seed,
        output_dir=args.output_dir,
        distributed=args.distributed
    )
    
    # Check if we should use existing data
    train_h5_file = None
    val_h5_file = None
    
    if args.use_existing_data:
        # Find existing H5 files in temp_data directory
        import glob
        
        # Look for merged files first
        train_files = glob.glob(os.path.join(args.temp_data_dir, "merged_train_*.h5"))
        val_files = glob.glob(os.path.join(args.temp_data_dir, "merged_val_*.h5"))
        
        if train_files:
            # Use the most recent file
            train_h5_file = max(train_files, key=os.path.getctime)
            logger.info(f"Using existing training data file: {train_h5_file}")
        
        if val_files:
            # Use the most recent file
            val_h5_file = max(val_files, key=os.path.getctime)
            logger.info(f"Using existing validation data file: {val_h5_file}")
        
        # If no merged files, look for rank-specific files
        if not train_h5_file:
            rank_files = glob.glob(os.path.join(args.temp_data_dir, f"layer_data_rank{args.rank}_*.h5"))
            if rank_files:
                train_h5_file = max(rank_files, key=os.path.getctime)
                logger.info(f"Using existing rank-specific data file: {train_h5_file}")
                
        # Skip model and tokenizer loading if using existing data
        model = None
        tokenizer = None
    else:
        # Setup model
        setup(model_args, device=device)
        
        # Load model and tokenizer
        if not args.distributed or (args.distributed and args.rank == 0):
            logger.info(f"Loading model from {args.model}")
        
        model, tokenizer = load_model_and_tokenizer(
            model_args, 
            device=device
        )
        
        # Enable memory optimization techniques
        if args.fp16:
            import torch.cuda.amp as amp
            model = model.half()  # Convert model to FP16
            if not args.distributed or (args.distributed and args.rank == 0):
                logger.info("Using half-precision (FP16) for model")
        
        # Enable gradient checkpointing to save memory
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            if not args.distributed or (args.distributed and args.rank == 0):
                logger.info("Enabled gradient checkpointing and disabled KV cache")
        
        # Set up padding token for LLaMA tokenizer
        if tokenizer.pad_token is None:
            # Use EOS token as padding token
            tokenizer.pad_token = tokenizer.eos_token
            if not args.distributed or (args.distributed and args.rank == 0):
                logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
            
        # Also set the padding side to right as LLaMA is a causal language model
        tokenizer.padding_side = "right"
        
        # Load training data
        if not args.distributed or (args.distributed and args.rank == 0):
            logger.info(f"Loading training data from {args.dataset}")
        
        data = load_training_data(args)
        
        # Split data into train and validation sets
        num_val = int(len(data) * args.val_split)
        num_train = len(data) - num_val
        
        if num_val > 0:
            # Use indices from random_split but keep the data as a list
            indices = list(range(len(data)))
            
            # Make sure all processes use the same data split
            g = torch.Generator()
            g.manual_seed(args.seed)
            random.shuffle(indices)
            
            # Create actual data lists
            train_data = [data[i] for i in indices[:num_train]]
            val_data = [data[i] for i in indices[num_train:]]
        else:
            train_data, val_data = data, None
        
        if not args.distributed or (args.distributed and args.rank == 0):
            logger.info(f"Training on {len(train_data)} examples, validating on {num_val} examples")
        
        skip_indices = []
        if args.skip_indices:
            skip_indices = [int(idx) for idx in args.skip_indices.split(',')]
        
        # Prepare training data using the chunked approach that saves to disk
        if not args.distributed or (args.distributed and args.rank == 0):
            logger.info(f"Preparing training data (extracting layer {args.target_layer} representations)")
        
        # Each rank processes its portion of the data
        train_h5_file = prepare_training_data_chunked(
            model=model,
            tokenizer=tokenizer,
            dataset=train_data,
            target_layer=args.target_layer,
            batch_size=args.extract_batch_size,
            device=device,
            max_samples=len(train_data),
            skip_indices=skip_indices,
            output_dir=args.temp_data_dir,
            chunk_size=args.chunk_size,
            rank=args.rank,
            world_size=args.world_size
        )
        
        # Prepare validation data if available
        if val_data is not None and len(val_data) > 0:
            if not args.distributed or (args.distributed and args.rank == 0):
                logger.info(f"Preparing validation data (extracting layer {args.target_layer} representations)")
            
            val_h5_file = prepare_training_data_chunked(
                model=model,
                tokenizer=tokenizer,
                dataset=val_data,
                target_layer=args.target_layer,
                batch_size=args.extract_batch_size,
                device=device,
                max_samples=len(val_data),
                output_dir=args.temp_data_dir,
                chunk_size=args.chunk_size,
                rank=args.rank,
                world_size=args.world_size
            )
    
    # In distributed setting, gather all H5 files
    if args.distributed:
        # Wait for all processes to finish data preparation
        torch.distributed.barrier()
        
        # Gather all file paths from different ranks
        train_files = [None] * args.world_size
        torch.distributed.all_gather_object(train_files, train_h5_file)
        
        val_files = [None] * args.world_size if val_h5_file else None
        if val_h5_file:
            torch.distributed.all_gather_object(val_files, val_h5_file)
        
        # Merge files on rank 0
        if args.rank == 0:
            # Filter out None values (in case some ranks didn't create files)
            train_files = [f for f in train_files if f]
            merged_train_file = os.path.join(args.temp_data_dir, f"merged_train_{int(time.time())}.h5")
            merged_train_file = merge_h5_files(train_files, merged_train_file)
            
            merged_val_file = None
            if val_files:
                val_files = [f for f in val_files if f]
                if val_files:
                    merged_val_file = os.path.join(args.temp_data_dir, f"merged_val_{int(time.time())}.h5")
                    merged_val_file = merge_h5_files(val_files, merged_val_file)
        else:
            # Non-zero ranks will use the merged files from rank 0
            merged_train_file = None
            merged_val_file = None
        
        # Only use broadcast_object_list if we have more than one process
        if args.world_size > 1:
            merged_train_file_list = [merged_train_file]
            torch.distributed.broadcast_object_list(merged_train_file_list, src=0)
            merged_train_file = merged_train_file_list[0]
            
            if val_h5_file:
                merged_val_file_list = [merged_val_file]
                torch.distributed.broadcast_object_list(merged_val_file_list, src=0)
                merged_val_file = merged_val_file_list[0]
        
        # Use the merged files
        train_h5_file = merged_train_file
        val_h5_file = merged_val_file
    
    # Free up GPU memory by moving model to CPU if necessary
    if args.optimize_memory and torch.cuda.is_available() and model is not None:
        if not args.distributed or (args.distributed and args.rank == 0):
            logger.info("Moving model to CPU to free up GPU memory")
        model.to('cpu')
        torch.cuda.empty_cache()
    
    # Create dataloaders from the H5 files
    train_loader = load_data_from_h5(train_h5_file, batch_size=args.batch_size)
    val_loader = load_data_from_h5(val_h5_file, batch_size=args.batch_size) if val_h5_file else None
    
    # Initialize DeBERTa adapter
    if not args.distributed or (args.distributed and args.rank == 0):
        logger.info("Initializing DeBERTa adapter")
    
    # Get model config for adapter initialization if model was not loaded
    if model is None:
        # Load minimal config to get hidden size and vocab size
        config = transformers.AutoConfig.from_pretrained(args.model)
        model_hidden_size = config.hidden_size
        model_vocab_size = config.vocab_size
    else:
        model_hidden_size = model.config.hidden_size
        model_vocab_size = model.config.vocab_size
    
    adapter = DebertaAdapter(
        llama_hidden_size=model_hidden_size,
        llama_vocab_size=model_vocab_size,
        deberta_hidden_size=args.deberta_hidden_size,
        deberta_num_layers=args.deberta_num_layers,
        deberta_num_attention_heads=args.deberta_num_heads,
        dropout_prob=args.dropout_prob
    )
    
    # Move to appropriate device
    adapter.to(device)
    
    # Wrap model with DDP for distributed training
    if args.distributed:
        adapter = torch.nn.parallel.DistributedDataParallel(
            adapter,
            device_ids=[args.local_rank] if args.local_rank != -1 else None,
            output_device=args.local_rank if args.local_rank != -1 else None,
            find_unused_parameters=True
        )
        if args.rank == 0:
            logger.info("Wrapped adapter with DistributedDataParallel")
    
    # Train the adapter
    if not args.distributed or (args.distributed and args.rank == 0):
        logger.info("Starting adapter training")
    
    adapter = train_adapter(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        alpha=args.ce_loss_weight,
        beta=args.kl_loss_weight,
        device=device,
        checkpoint_dir=args.output_dir,
        log_wandb=args.use_wandb,
        project_name=f"layerskip-deberta-adapter-{args.dataset}",
        run_name=f"l{args.target_layer}-dhs{args.deberta_hidden_size}-dl{args.deberta_num_layers}",
        args=args,  # Pass args to the train_adapter function
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16
    )
    
    # Save final adapter configuration (only from main process)
    if not args.distributed or (args.distributed and args.rank == 0):
        # Save configuration for easy loading
        config_path = os.path.join(args.output_dir, "adapter_config.json")
        import json
        with open(config_path, 'w') as f:
            json.dump({
                "model": args.model,
                "target_layer": args.target_layer,
                "llama_hidden_size": model_hidden_size,
                "llama_vocab_size": model_vocab_size,
                "deberta_hidden_size": args.deberta_hidden_size,
                "deberta_num_layers": args.deberta_num_layers,
                "deberta_num_heads": args.deberta_num_heads,
                "dropout_prob": args.dropout_prob,
                "dataset": args.dataset,
                "num_examples": args.num_examples,
                "distributed_training": args.distributed,
                "world_size": args.world_size if args.distributed else 1
            }, f, indent=2)
        
        logger.info(f"Adapter configuration saved to {config_path}")
    
    # Clean up temporary files
    if not args.distributed or (args.distributed and args.rank == 0):
        # Don't delete the merged files if requested
        if not args.keep_temp_files:
            try:
                import glob
                temp_files = glob.glob(os.path.join(args.temp_data_dir, "*.h5"))
                for file in temp_files:
                    if os.path.exists(file):
                        os.remove(file)
                logger.info(f"Removed temporary files from {args.temp_data_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary files: {e}")
    
    logger.info("Training completed successfully!")
    
            
if __name__ == "__main__":
    main()

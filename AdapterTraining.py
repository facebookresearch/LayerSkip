# Training loop for the DeBERTa adapter module

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import logging
from typing import List, Dict, Optional, Tuple, Any
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import wandb

from LlamaLayerExtractor import LlamaLayerExtractor
from DebertaAdapter import DebertaAdapter
from transformers import PreTrainedTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntermediateLayerDataset(Dataset):
    """
    Dataset for training the adapter on intermediate layer outputs.
    """
    def __init__(self, 
                 hidden_states: List[torch.Tensor], 
                 ground_truth_tokens: List[torch.Tensor],
                 full_model_logits: Optional[List[torch.Tensor]] = None):
        """
        Initialize the dataset.
        
        Args:
            hidden_states: List of hidden states from layer 8
            ground_truth_tokens: List of corresponding ground truth next tokens
            full_model_logits: Optional list of logits from the full model for KL loss
        """
        self.hidden_states = hidden_states
        self.ground_truth_tokens = ground_truth_tokens
        self.full_model_logits = full_model_logits
        
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        item = {
            'hidden_states': self.hidden_states[idx],
            'ground_truth': self.ground_truth_tokens[idx]
        }
        
        if self.full_model_logits is not None:
            item['full_model_logits'] = self.full_model_logits[idx]
            
        return item
def prepare_training_data_chunked(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: List[Dict[str, Any]],
    target_layer: int = 8,
    batch_size: int = 4,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    skip_indices: List[int] = [],
    output_dir: str = "./temp_data",
    chunk_size: int = 100,  # Process this many examples before saving to disk
    rank: int = 0,
    world_size: int = 1
) -> str:
    """
    Memory-efficient version that processes data in chunks and saves to disk.
    Returns the path to the saved data.
    """
    import os
    import torch
    import numpy as np
    import time
    from tqdm import tqdm
    import h5py
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize layer extractor
    layer_extractor = LlamaLayerExtractor(model, target_layer=target_layer)
    
    # Only process subset assigned to this rank in distributed setting
    dataset_size = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    
    # In distributed training, divide work among ranks
    indices = list(range(dataset_size))
    indices = [i for i in indices if i not in skip_indices]
    
    # Split indices among ranks
    rank_indices = []
    for i in range(len(indices)):
        if i % world_size == rank:
            rank_indices.append(indices[i])
    
    logger.info(f"Rank {rank}: Processing {len(rank_indices)} examples out of {len(indices)} total")
    
    # Files to store data
    timestamp = int(time.time())
    h5_filename = os.path.join(output_dir, f"layer_data_rank{rank}_{timestamp}.h5")
    
    # Initialize counters
    total_samples = 0
    chunk_samples = 0
    chunk_hidden_states = []
    chunk_ground_truth = []
    chunk_full_logits = []
    
    # Create HDF5 file
    with h5py.File(h5_filename, "w") as h5f:
        # Create datasets with maxshape to allow resizing
        hidden_shape = (0, model.config.hidden_size)  # Initial shape will be expanded
        h5f.create_dataset("hidden_states", shape=hidden_shape, maxshape=(None, model.config.hidden_size), 
                          dtype=np.float16, chunks=True)
        
        h5f.create_dataset("ground_truth", shape=(0,), maxshape=(None,), 
                          dtype=np.int32, chunks=True)
        
        h5f.create_dataset("full_logits", shape=(0, model.config.vocab_size), 
                          maxshape=(None, model.config.vocab_size), 
                          dtype=np.float16, chunks=True)
        
        # Process dataset in batches
        for batch_idx in range(0, len(rank_indices), batch_size):
            if batch_idx % 10 == 0:
                logger.info(f"Rank {rank}: Processing batch {batch_idx}/{len(rank_indices)}")
                
            batch_end = min(batch_idx + batch_size, len(rank_indices))
            batch_indices = rank_indices[batch_idx:batch_end]
            
            # Get examples for this batch
            batch = [dataset[i] for i in batch_indices]
            
            # Tokenize inputs - with shorter max length to save memory
            texts = [example['text'] for example in batch]
            
            max_length = min(512, max([len(tokenizer.encode(text)) for text in texts]))
            
            encodings = tokenizer(texts, padding=True, truncation=True, 
                                max_length=max_length,
                                return_tensors='pt').to(device)
            
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            
            # Process each sequence in the batch
            for j in range(input_ids.size(0)):
                seq_len = attention_mask[j].sum().item()
                
                # Skip if sequence is too short
                if seq_len <= 1:
                    continue
                
                # Get hidden states for all tokens except the last one
                try:
                    with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
                        hidden_states = layer_extractor.get_layer_output(
                            input_ids=input_ids[j].unsqueeze(0),
                            attention_mask=attention_mask[j].unsqueeze(0)
                        )
                    
                    # Clear CUDA cache after getting hidden states
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    # For each position in the sequence (except last token)
                    for pos in range(min(seq_len - 1, 20)):  # Limit positions per sequence
                        # Get hidden state for current position
                        pos_hidden_state = hidden_states[:, pos, :].detach().cpu().numpy().astype(np.float16)
                        
                        # Ground truth is the next token
                        next_token = input_ids[j, pos + 1].item()
                        
                        # Get full model logits for KL divergence loss
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
                                full_outputs = model(input_ids=input_ids[j, :pos+1].unsqueeze(0))
                            full_logits = full_outputs.logits[:, -1, :].detach().cpu().numpy().astype(np.float16)
                        
                        # Add to current chunk
                        chunk_hidden_states.append(pos_hidden_state[0])  # Remove batch dimension
                        chunk_ground_truth.append(next_token)
                        chunk_full_logits.append(full_logits[0])  # Remove batch dimension
                        
                        chunk_samples += 1
                        total_samples += 1
                        
                        # If chunk is full, save to file
                        if chunk_samples >= chunk_size:
                            # Convert to numpy arrays
                            hidden_array = np.array(chunk_hidden_states)
                            ground_truth_array = np.array(chunk_ground_truth)
                            full_logits_array = np.array(chunk_full_logits)
                            
                            # Get current sizes
                            current_size = h5f["hidden_states"].shape[0]
                            new_size = current_size + len(hidden_array)
                            
                            # Resize datasets
                            h5f["hidden_states"].resize((new_size, model.config.hidden_size))
                            h5f["ground_truth"].resize((new_size,))
                            h5f["full_logits"].resize((new_size, model.config.vocab_size))
                            
                            # Write data
                            h5f["hidden_states"][current_size:new_size] = hidden_array
                            h5f["ground_truth"][current_size:new_size] = ground_truth_array
                            h5f["full_logits"][current_size:new_size] = full_logits_array
                            
                            # Clear chunk data
                            chunk_hidden_states = []
                            chunk_ground_truth = []
                            chunk_full_logits = []
                            chunk_samples = 0
                            
                            # Force sync to disk
                            h5f.flush()
                            
                            # Clear CUDA cache again
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                except Exception as e:
                    logger.error(f"Error processing sequence {j} in batch {batch_idx}: {e}")
                    continue
                
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save any remaining chunks
        if chunk_samples > 0:
            # Convert to numpy arrays
            hidden_array = np.array(chunk_hidden_states)
            ground_truth_array = np.array(chunk_ground_truth)
            full_logits_array = np.array(chunk_full_logits)
            
            # Get current sizes
            current_size = h5f["hidden_states"].shape[0]
            new_size = current_size + len(hidden_array)
            
            # Resize datasets
            h5f["hidden_states"].resize((new_size, model.config.hidden_size))
            h5f["ground_truth"].resize((new_size,))
            h5f["full_logits"].resize((new_size, model.config.vocab_size))
            
            # Write data
            h5f["hidden_states"][current_size:new_size] = hidden_array
            h5f["ground_truth"][current_size:new_size] = ground_truth_array
            h5f["full_logits"][current_size:new_size] = full_logits_array
            
            # Force sync to disk
            h5f.flush()
    
    # Clean up
    layer_extractor.remove_hooks()
    torch.cuda.empty_cache()
    
    logger.info(f"Rank {rank}: Prepared and saved {total_samples} training samples to {h5_filename}")
    return h5_filename

def load_data_from_h5(filename: str, batch_size: int = 16):
    """
    Load data from HDF5 file into a DataLoader.
    """
    import h5py
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class HDF5Dataset(Dataset):
        def __init__(self, filename):
            self.file = h5py.File(filename, 'r')
            self.length = self.file['hidden_states'].shape[0]
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            # Convert to specific types for both hidden_states and ground_truth
            hidden_states = torch.tensor(self.file['hidden_states'][idx], dtype=torch.float32)
            ground_truth = torch.tensor(self.file['ground_truth'][idx], dtype=torch.long)
            full_logits = torch.tensor(self.file['full_logits'][idx], dtype=torch.float32)
            
            return {
                'hidden_states': hidden_states,
                'ground_truth': ground_truth,
                'full_model_logits': full_logits
            }
    
    dataset = HDF5Dataset(filename)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_adapter(
    adapter: DebertaAdapter,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    alpha: float = 0.8,  # Weight for CE loss
    beta: float = 0.2,   # Weight for KL loss
    device: str = "cuda",
    checkpoint_dir: str = "./checkpoints",
    log_wandb: bool = False,
    project_name: str = "llama-deberta-adapter",
    run_name: Optional[str] = None,
    args = None,  # Add args parameter to access distributed settings
    gradient_accumulation_steps: int = 1,
    fp16: bool = False
):
    """
    Memory-efficient version of the train_adapter function with support for distributed training.
    
    Args:
        adapter: The DeBERTa adapter model
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate
        alpha: Weight for cross entropy loss
        beta: Weight for KL divergence loss
        device: Device to use
        checkpoint_dir: Directory to save checkpoints
        log_wandb: Whether to log to W&B
        project_name: W&B project name
        run_name: W&B run name
        args: Arguments containing distributed training settings
        gradient_accumulation_steps: Number of steps to accumulate gradients
        fp16: Whether to use half-precision training
    """
    # Move adapter to device
    adapter.to(device)
    
    # Setup optimizer with weight decay
    from torch.optim import AdamW
    optimizer = AdamW(adapter.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Setup learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    num_training_steps = len(train_loader) * epochs // gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    # Setup loss functions
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    # Setup mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    # Create checkpoint directory - only from main process
    if not args or not args.distributed or (args.distributed and args.rank == 0):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize W&B if requested - only from main process
    if log_wandb and (not args or not args.distributed or (args.distributed and args.rank == 0)):
        import wandb
        wandb.init(project=project_name, name=run_name)
        wandb.config.update({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "alpha": alpha,
            "beta": beta,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "fp16": fp16
        })
    
    # Track best validation loss for early stopping and model selection
    best_val_loss = float('inf')
    patience = 3  # Number of epochs with no improvement after which training will be stopped
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Only log from main process
        if not args or not args.distributed or (args.distributed and args.rank == 0):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
        
        # Set epoch for distributed sampler to ensure proper shuffling
        if args and args.distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Training phase
        adapter.train()
        train_ce_loss = 0.0
        train_kl_loss = 0.0
        train_total_loss = 0.0
        train_acc = 0.0
        
        # Define tqdm progress bar - only from main process
        if not args or not args.distributed or (args.distributed and args.rank == 0):
            from tqdm import tqdm
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        else:
            train_iterator = train_loader
            
        # Reset optimizer gradients at the beginning of each epoch
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_iterator):
            # Move batch to device
            # hidden_states = batch['hidden_states'].to(device)
            # ground_truth = batch['ground_truth'].to(device)
            hidden_states = batch['hidden_states'].to(device).float()  # Convert to Float32
            ground_truth = batch['ground_truth'].to(device)

            full_model_logits = batch.get('full_model_logits')
            
            if full_model_logits is not None:
                full_model_logits = full_model_logits.to(device)
            
            # Forward pass with mixed precision if requested
            if fp16:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    adapter_logits = adapter(hidden_states)
                    
                    # Cross entropy loss with ground truth
                    # ce_loss = ce_loss_fn(adapter_logits.view(-1, adapter_logits.size(-1)), ground_truth.view(-1))
                    ce_loss = ce_loss_fn(adapter_logits.view(-1, adapter_logits.size(-1)), ground_truth.view(-1).long())
                    
                    # KL divergence loss with full model predictions (optional)
                    kl_loss = 0.0
                    if full_model_logits is not None:
                        log_softmax_adapter = torch.log_softmax(adapter_logits, dim=-1)
                        softmax_full = torch.softmax(full_model_logits, dim=-1)
                        kl_loss = kl_loss_fn(log_softmax_adapter, softmax_full)
                    
                    # Combine losses
                    total_loss = alpha * ce_loss + beta * kl_loss
                    
                    # Scale loss by gradient accumulation steps
                    if gradient_accumulation_steps > 1:
                        total_loss = total_loss / gradient_accumulation_steps
                    
                # Backward pass with gradient scaling
                scaler.scale(total_loss).backward()
                
                # Only step optimizer and scheduler after accumulating gradients
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients to prevent exploding gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                    
                    # Update parameters and learning rate
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # Standard training without mixed precision
                # Forward pass
                adapter_logits = adapter(hidden_states)
                
                # Cross entropy loss with ground truth
                # ce_loss = ce_loss_fn(adapter_logits.view(-1, adapter_logits.size(-1)), ground_truth.view(-1))
                ce_loss = ce_loss_fn(adapter_logits.view(-1, adapter_logits.size(-1)), ground_truth.view(-1).long())
                
                # KL divergence loss with full model predictions (optional)
                kl_loss = 0.0
                if full_model_logits is not None:
                    log_softmax_adapter = torch.log_softmax(adapter_logits, dim=-1)
                    softmax_full = torch.softmax(full_model_logits, dim=-1)
                    kl_loss = kl_loss_fn(log_softmax_adapter, softmax_full)
                
                # Combine losses
                total_loss = alpha * ce_loss + beta * kl_loss
                
                # Scale loss by gradient accumulation steps
                if gradient_accumulation_steps > 1:
                    total_loss = total_loss / gradient_accumulation_steps
                
                # Backward pass
                total_loss.backward()
                
                # Only step optimizer and scheduler after accumulating gradients
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                    
                    # Update parameters and learning rate
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            # Update metrics
            train_ce_loss += ce_loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
            if full_model_logits is not None:
                train_kl_loss += kl_loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
            train_total_loss += total_loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
            
            # Calculate accuracy
            with torch.no_grad():
                preds = torch.argmax(adapter_logits, dim=-1)
                train_acc += (preds == ground_truth).float().mean().item()
            
            # Clear some memory
            del hidden_states, ground_truth, adapter_logits
            if full_model_logits is not None:
                del full_model_logits
                
            # Clear CUDA cache periodically
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Update progress bar
            if not args or not args.distributed or (args.distributed and args.rank == 0):
                if hasattr(train_iterator, 'set_postfix'):
                    train_iterator.set_postfix(loss=total_loss.item(), 
                                             ce_loss=ce_loss.item(),
                                             lr=scheduler.get_last_lr()[0])
        
        # Average metrics
        num_batches = len(train_loader)
        train_ce_loss /= num_batches
        train_kl_loss /= num_batches
        train_total_loss /= num_batches
        train_acc /= num_batches
        
        # Log metrics - only from main process
        if not args or not args.distributed or (args.distributed and args.rank == 0):
            logger.info(f"Epoch {epoch+1} Train CE Loss: {train_ce_loss:.4f}")
            logger.info(f"Epoch {epoch+1} Train KL Loss: {train_kl_loss:.4f}")
            logger.info(f"Epoch {epoch+1} Train Total Loss: {train_total_loss:.4f}")
            logger.info(f"Epoch {epoch+1} Train Accuracy: {train_acc:.4f}")
            
            if log_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_ce_loss": train_ce_loss,
                    "train_kl_loss": train_kl_loss,
                    "train_total_loss": train_total_loss,
                    "train_accuracy": train_acc,
                    "learning_rate": scheduler.get_last_lr()[0]
                })
        
        # Validation phase
        if val_loader:
            adapter.eval()
            val_ce_loss = 0.0
            val_kl_loss = 0.0
            val_total_loss = 0.0
            val_acc = 0.0
            
            # Set epoch for distributed validation sampler if applicable
            if args and args.distributed and hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'set_epoch'):
                val_loader.sampler.set_epoch(epoch)
            
            # Define tqdm progress bar for validation - only from main process
            if not args or not args.distributed or (args.distributed and args.rank == 0):
                from tqdm import tqdm
                val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
            else:
                val_iterator = val_loader
                
            with torch.no_grad():
                for batch in val_iterator:
                    # hidden_states = batch['hidden_states'].to(device)
                    # ground_truth = batch['ground_truth'].to(device)
                    hidden_states = batch['hidden_states'].to(device).float()  # Convert to Float32
                    ground_truth = batch['ground_truth'].to(device)
                    
                    full_model_logits = batch.get('full_model_logits')
                    
                    if full_model_logits is not None:
                        full_model_logits = full_model_logits.to(device)
                    
                    # Forward pass
                    if fp16:
                        with torch.cuda.amp.autocast():
                            adapter_logits = adapter(hidden_states)
                    else:
                        adapter_logits = adapter(hidden_states)
                    
                    # Cross entropy loss with ground truth
                    ce_loss = ce_loss_fn(adapter_logits.view(-1, adapter_logits.size(-1)), ground_truth.view(-1))
                    
                    # KL divergence loss with full model predictions (optional)
                    kl_loss = 0.0
                    if full_model_logits is not None:
                        log_softmax_adapter = torch.log_softmax(adapter_logits, dim=-1)
                        softmax_full = torch.softmax(full_model_logits, dim=-1)
                        kl_loss = kl_loss_fn(log_softmax_adapter, softmax_full)
                    
                    # Combine losses
                    total_loss = alpha * ce_loss + beta * kl_loss
                    
                    # Update metrics
                    val_ce_loss += ce_loss.item()
                    if full_model_logits is not None:
                        val_kl_loss += kl_loss.item()
                    val_total_loss += total_loss.item()
                    
                    # Calculate accuracy
                    preds = torch.argmax(adapter_logits, dim=-1)
                    val_acc += (preds == ground_truth).float().mean().item()
                    
                    # Clear some memory
                    del hidden_states, ground_truth, adapter_logits
                    if full_model_logits is not None:
                        del full_model_logits
                    
                    # Clear CUDA cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Average metrics
            num_val_batches = len(val_loader)
            val_ce_loss /= num_val_batches
            val_kl_loss /= num_val_batches
            val_total_loss /= num_val_batches
            val_acc /= num_val_batches
            
            # Log validation metrics - only from main process
            if not args or not args.distributed or (args.distributed and args.rank == 0):
                logger.info(f"Epoch {epoch+1} Val CE Loss: {val_ce_loss:.4f}")
                logger.info(f"Epoch {epoch+1} Val KL Loss: {val_kl_loss:.4f}")
                logger.info(f"Epoch {epoch+1} Val Total Loss: {val_total_loss:.4f}")
                logger.info(f"Epoch {epoch+1} Val Accuracy: {val_acc:.4f}")
                
                if log_wandb:
                    import wandb
                    wandb.log({
                        "val_ce_loss": val_ce_loss,
                        "val_kl_loss": val_kl_loss,
                        "val_total_loss": val_total_loss,
                        "val_accuracy": val_acc
                    })
                    
            # Check for improvement for early stopping
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                
                # Save best model - only from main process
                if not args or not args.distributed or (args.distributed and args.rank == 0):
                    best_model_path = os.path.join(checkpoint_dir, "adapter_best.pt")
                    
                    # If using DDP, save the module's state dict instead of the wrapper
                    if args and args.distributed:
                        adapter_to_save = adapter.module
                    else:
                        adapter_to_save = adapter
                        
                    adapter_to_save.save(best_model_path)
                    logger.info(f"New best validation loss: {best_val_loss:.4f}, saved model to {best_model_path}")
            else:
                patience_counter += 1
                logger.info(f"Validation didn't improve. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save checkpoint for this epoch - only from main process
        if not args or not args.distributed or (args.distributed and args.rank == 0):
            checkpoint_path = os.path.join(checkpoint_dir, f"adapter_epoch_{epoch+1}.pt")
            
            # If using DDP, save the module's state dict instead of the wrapper
            if args and args.distributed:
                adapter_to_save = adapter.module
            else:
                adapter_to_save = adapter
                
            adapter_to_save.save(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model - only from main process
    if not args or not args.distributed or (args.distributed and args.rank == 0):
        final_path = os.path.join(checkpoint_dir, "adapter_final.pt")
        
        # If using DDP, save the module's state dict instead of the wrapper
        if args and args.distributed:
            adapter_to_save = adapter.module
        else:
            adapter_to_save = adapter
            
        adapter_to_save.save(final_path)
        logger.info(f"Training complete. Saved final model to {final_path}")
        
        if log_wandb:
            wandb.finish()
    
    return adapter
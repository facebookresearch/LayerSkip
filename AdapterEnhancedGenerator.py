import argparse
import torch
from transformers import AutoTokenizer
from arguments import Arguments
from generate import load_model_and_tokenizer, setup
from self_speculation.generator_base import GenerationConfig
from LlamaLayerExtractor import LlamaLayerExtractor
from DebertaAdapter import DebertaAdapter
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdapterEnhancedGenerator:
    """
    Generator that combines the LLaMA model with the DeBERTa adapter for enhanced generation.
    """
    def __init__(self, model, tokenizer, adapter, target_layer=8, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.device = device
        
        # Initialize the layer extractor for the target layer
        self.layer_extractor = LlamaLayerExtractor(model, target_layer=target_layer)
        
        # Move adapter to the correct device
        self.adapter.to(device)
        self.adapter.eval()
        
    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
        """
        Generate text using the enhanced model with adapter.
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Initialize the output sequence with the input IDs
        output_sequence = input_ids.clone()[0].tolist()
        
        # Start the generation loop
        for _ in range(max_length):
            # Format input for the current step
            current_input = torch.tensor([output_sequence]).to(self.device)
            
            # Get intermediate layer hidden states
            with torch.no_grad():
                hidden_states = self.layer_extractor.get_layer_output(current_input)
                
                # Get the last token's hidden state
                last_hidden = hidden_states[:, -1, :]
                
                # Pass through the adapter
                adapter_logits = self.adapter(last_hidden)
                
                # Apply temperature and sampling
                if temperature > 0:
                    probs = torch.softmax(adapter_logits / temperature, dim=-1)
                    
                    # Apply top-k and top-p filtering
                    if top_k > 0:
                        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                        probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                    if top_p > 0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        mask = cumulative_probs < top_p
                        # Add the first token that exceeds top_p
                        mask = torch.cat([mask.new_ones(mask.shape[:-1] + (1,)), mask[..., :-1]], dim=-1)
                        sorted_probs = sorted_probs * mask.float()
                        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
                    
                    # Sample from the distribution
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    # Greedy decoding
                    next_token_id = torch.argmax(adapter_logits, dim=-1).item()
            
            # Add the next token to the output sequence
            output_sequence.append(next_token_id)
            
            # Check if we've reached an EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
        # Decode the generated sequence
        generated_text = self.tokenizer.decode(output_sequence, skip_special_tokens=True)
        return generated_text

    def generate_with_native_model(self, prompt, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
        """
        Generate text using the native model without adapter for comparison.
        """
        # Set generation parameters
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with the native model
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode the generated sequence
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained DeBERTa adapter")
    parser.add_argument("--model", type=str, required=True, help="Path to base LLaMA model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter weights")
    parser.add_argument("--adapter_config", type=str, required=True, help="Path to adapter config JSON")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--compare", action="store_true", help="Compare with native model generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model_args = Arguments(model=args.model, seed=42, output_dir="./outputs")
    setup(model_args)
    model, tokenizer = load_model_and_tokenizer(model_args, device=args.device)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter configuration
    with open(args.adapter_config, 'r') as f:
        adapter_config = json.load(f)
    
    # Initialize adapter with the same configuration used during training
    adapter = DebertaAdapter(
        llama_hidden_size=adapter_config.get("llama_hidden_size", model.config.hidden_size),
        llama_vocab_size=adapter_config.get("llama_vocab_size", model.config.vocab_size),
        deberta_hidden_size=adapter_config.get("deberta_hidden_size", 768),
        deberta_num_layers=adapter_config.get("deberta_num_layers", 2),
        deberta_num_attention_heads=adapter_config.get("deberta_num_heads", 12),
        dropout_prob=adapter_config.get("dropout_prob", 0.1)
    )
    
    # Load adapter weights
    adapter.load_state_dict(torch.load(args.adapter_path, map_location=args.device))
    logger.info(f"Loaded adapter weights from {args.adapter_path}")
    
    # Initialize the enhanced generator
    target_layer = adapter_config.get("target_layer", 8)
    generator = AdapterEnhancedGenerator(
        model=model, 
        tokenizer=tokenizer, 
        adapter=adapter,
        target_layer=target_layer,
        device=args.device
    )
    
    # Generate text with the adapter-enhanced model
    logger.info("Generating with adapter-enhanced model...")
    adapter_text = generator.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    print("\n======= ADAPTER-ENHANCED GENERATION =======")
    print(adapter_text)
    print("\n===========================================")
    
    # Optionally compare with native model
    if args.compare:
        logger.info("Generating with native model for comparison...")
        native_text = generator.generate_with_native_model(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        
        print("\n========== NATIVE MODEL GENERATION ==========")
        print(native_text)
        print("\n===========================================")

if __name__ == "__main__":
    main()
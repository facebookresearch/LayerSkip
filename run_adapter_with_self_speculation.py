#!/usr/bin/env python3
"""
Script to run inference with DeBERTa adapter enhanced self-speculative decoding.
"""

import argparse
import torch
import json
import logging
import os
import time
from typing import List, Optional, Tuple
import colorama

from arguments import Arguments
from generate import load_model_and_tokenizer, setup
from LlamaLayerExtractor import LlamaLayerExtractor
from DebertaAdapter import DebertaAdapter

from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
    GenerationStrategyResult
)
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    forward_early,
    forward_remainder,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdapterEnhancedSpeculativeStrategy(SelfSpeculativeGenerationStrategy):
    """
    Enhances the self-speculative generation strategy with a DeBERTa adapter.
    """
    def __init__(self, adapter: DebertaAdapter, layer_extractor: LlamaLayerExtractor):
        super().__init__()
        self.adapter = adapter
        self.layer_extractor = layer_extractor
        self.adapter.eval()  # Set adapter to evaluation mode
    
    def generate_token_ids(
        self,
        model: torch.nn.Module,
        input_ids: List[int],
        eos_token_ids: List[int],
        generation_config: GenerationConfig,
        logits_processors=None,
        stopping_criteria=None,
        streamer=None,
    ) -> GenerationStrategyResult:
        """
        Override the generate_token_ids method to integrate the adapter.
        """
        past_key_values = None

        input_ids_list = input_ids
        input_ids_tensor: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        output_ids: List[int] = []

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        
        # Save a reference to the model for the layer extractor
        self.layer_extractor.model = model
        
        while len(output_ids) < generation_config.max_steps:
            (
                input_ids_tensor,
                output_ids,
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation_with_adapter(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids_tensor,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=generation_config.exit_layer,
                eos_token_ids=eos_token_ids,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            
            # Check for EOS
            eos_found = False
            for eos_token_id in eos_token_ids:
                if eos_token_id in output_ids:
                    # remove the EOS token id
                    output_ids = output_ids[: output_ids.index(eos_token_id)]
                    eos_found = True
                    break
                    
            if eos_found:
                break
                
            if stopping_criteria:
                if torch.all(stopping_criteria(input_ids_tensor, scores=None)):
                    break
                    
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations if total_generations > 0 else 0,
        )
    
    
    
    def single_step_speculation_with_adapter(
    self,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    input_ids_list: List[int],
    output_ids: List[int],
    num_speculations: int,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    eos_token_ids: List[int],
    calls: int,
    exit_layer: int,
    sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    logits_processors = None,
    stopping_criteria = None,
    streamer = None ):
        zero_division_count = 0  # Counter for ZeroDivisionErrors
        
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        
        exit_query_cache = None

        for _ in range(num_speculations):
            # Get the draft token using early exit and adapter enhancement
            draft_result = forward_early(
                model,
                draft_input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
            )
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            
            # Get hidden states using the layer extractor
            with torch.no_grad():
                hidden_states = self.layer_extractor.get_layer_output(
                    draft_input_ids, 
                    past_key_values=past_key_values
                )
                
                # Get the last token's hidden state
                last_hidden = hidden_states[:, -1, :]
                
                # Enhance with adapter
                adapter_logits = self.adapter(last_hidden)
                
                # Combine with original draft logits
                alpha = 0.7  # Weight for adapter (adjust as needed)
                draft_logits = draft_result.logits
                enhanced_logits = alpha * adapter_logits + (1 - alpha) * draft_logits
                
                if logits_processors:
                    enhanced_logits = logits_processors(draft_input_ids, enhanced_logits)
                
                draft_next_token, draft_next_prob = decode_next_token(
                    logits=enhanced_logits, 
                    token_idx=-1, 
                    sample=sample, 
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p
                )
            
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            
            if sample:
                draft_probabilities.append(draft_next_prob)
            
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            
            if draft_next_token in eos_token_ids:
                break

        # Convert draft output IDs to tensor
        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        prefill_token_ids = torch.cat([input_ids, draft_output_ids], dim=-1)

        if streamer:
            if hasattr(streamer, 'is_draft') and hasattr(streamer, 'put'):
                print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                streamer.put(draft_output_ids, is_draft=True)

        # Get verification logits
        verify_results = forward_remainder(
            model,
            prefill_token_ids.int(),
            past_key_values,
            exit_layer,
            exit_query_cache,
        )
        logits = verify_results.logits
        
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        
        past_key_values = verify_results.past_key_values

        verification_logits = logits[:, prompt_length - 1:, :]
        verified_tokens, verified_probabilities = decode_next_token(
            logits=verification_logits, 
            sample=sample, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )

        verified_tokens = verified_tokens.to(prefill_token_ids)
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]

        if not sample:
            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)

            for i in range(draft_output_ids.numel()):
                denominator = draft_probabilities[i][0, draft_output_ids[0, i]].item()

                # Handle potential division by zero
                if denominator == 0 or denominator < 1e-8:
                    zero_division_count += 1
                    denominator = 1e-8  # Small epsilon to avoid division by zero
                
                # Compute acceptance ratio
                acceptance_ratio = verified_probabilities[i, draft_output_ids[0, i]].item() / denominator
                
                # Accept or reject based on probability ratio
                if rand[0, i] < min(1, acceptance_ratio):
                    number_of_matches += 1
                else:
                    # Compute probability distribution for rejection sampling
                    prob_dist = torch.clamp(verified_probabilities[i, :] - draft_probabilities[i], min=0)
                    
                    # Handle degenerate case where all probabilities are zero
                    if prob_dist.sum().item() <= 1e-8:
                        prob_dist = torch.ones_like(prob_dist) / prob_dist.shape[-1]
                    else:
                        prob_dist = prob_dist / prob_dist.sum()  # Normalize
                    
                    # Sample the next token
                    verified_tokens[0][number_of_matches] = torch.multinomial(prob_dist, num_samples=1).item()
                    break

        input_ids = verified_tokens[:, number_of_matches:number_of_matches + 1]
        output_ids.extend(draft_output_ids[0, :number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches:number_of_matches + 1].tolist())

        if streamer:
            if hasattr(streamer, 'delete') and hasattr(streamer, 'put'):
                streamer.delete(len(draft_output_ids[0, :]))
                print(colorama.Fore.GREEN, end="")
                streamer.put(draft_output_ids[0, :number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                streamer.put(verified_tokens[0][number_of_matches:number_of_matches + 1])
            elif hasattr(streamer, 'put'):
                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) - 1
        )

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            draft_output_ids.numel(),
        )

def main():
    parser = argparse.ArgumentParser(description="Run adapter-enhanced self-speculative decoding")
    parser.add_argument("--model", type=str, required=True, help="Path to base LLaMA model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter weights")
    parser.add_argument("--adapter_config", type=str, required=True, help="Path to adapter config JSON")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--num_speculations", type=int, default=5, help="Number of tokens to speculate")
    parser.add_argument("--exit_layer", type=int, default=8, help="Layer to exit for speculation")
    parser.add_argument("--output", type=str, default=None, help="Output file to save generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model}")
    model_args = Arguments(model=args.model, seed=42, output_dir="./outputs")
    setup(model_args)
    model, tokenizer = load_model_and_tokenizer(model_args, device=args.device)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter configuration
    with open(args.adapter_config, 'r') as f:
        adapter_config = json.load(f)
    
    # Initialize adapter
    logger.info("Initializing DeBERTa adapter")
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
    adapter.to(args.device)
    logger.info(f"Loaded adapter weights from {args.adapter_path}")
    
    # Initialize layer extractor
    target_layer = adapter_config.get("target_layer", args.exit_layer)
    layer_extractor = LlamaLayerExtractor(model, target_layer=target_layer)
    
    # Create enhanced speculative strategy
    enhanced_strategy = AdapterEnhancedSpeculativeStrategy(
        adapter=adapter,
        layer_extractor=layer_extractor
    )
    
    # Initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=enhanced_strategy
    )
    
    # Configure generation
    generation_config = GenerationConfig(
        max_steps=args.max_length,
        exit_layer=args.exit_layer,
        num_speculations=args.num_speculations,
        generation_strategy="self_speculative",  # This is just a label
        sample=args.temperature > 0,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Record start time
    start_time = time.time()
    
    # Generate text
    logger.info("Generating with adapter-enhanced self-speculative decoding...")
    result = generator.generate(
        prompt=args.prompt,
        generation_config=generation_config
    )
    
    # Record end time
    generation_time = time.time() - start_time
    
    # Display results
    print("\n======= ADAPTER-ENHANCED SELF-SPECULATIVE GENERATION =======")
    print(f"Prompt: {args.prompt}")
    print(f"\nGenerated text: {result.decoded_prediction}")
    print("\n==========================================================")
    
    # Print stats
    print(f"\nGeneration time: {generation_time:.2f} seconds")
    print(f"Tokens generated: {result.num_tokens_generated}")
    print(f"Tokens per second: {result.tokens_per_second:.2f}")
    print(f"Acceptance rate: {result.generation_strategy_result.acceptance_rate:.2%}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(f"Prompt: {args.prompt}\n\n")
            f.write(f"Generated text: {result.decoded_prediction}\n\n")
            f.write(f"Generation time: {generation_time:.2f} seconds\n")
            f.write(f"Tokens generated: {result.num_tokens_generated}\n")
            f.write(f"Tokens per second: {result.tokens_per_second:.2f}\n")
            f.write(f"Acceptance rate: {result.generation_strategy_result.acceptance_rate:.2%}\n")
        logger.info(f"Results saved to {args.output}")
    
    # Clean up
    layer_extractor.remove_hooks()

if __name__ == "__main__":
    main()
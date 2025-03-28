# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple
from enum import Enum
from dataclasses import dataclass

import colorama
import datetime
import random
import sys
import torch
import traceback
import transformers
import os

from arguments import Arguments, simple_parse_args_string
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.speculative_streamer import SpeculativeTextStreamer
from self_speculation.layer_drop_generator import LayerDropGenerationStrategy
# NEW: Import the token-level depth-adaptive strategy
from self_speculation.depth_adaptive_token_generator import DepthAdaptiveTokenGenerationStrategy

class StreamerType(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    SPECULATIVE = "speculative"

@dataclass
class GenerateArguments:
    streamer: StreamerType = StreamerType.STANDARD

def setup(args, device: str = "cuda"):
    """Setup function for single GPU or distributed training."""
    if device == "cuda":
        torch.cuda.set_device(0)  # Use first GPU

    # Skip distributed setup if running on single GPU
    if not args.distributed:  # Add this condition
        return

    # Only run distributed setup if explicitly requested
    torch.distributed.init_process_group(
        backend="nccl" if device == "cuda" else "gloo",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

def load_model_and_tokenizer(args: Arguments, device: str = "auto"):
    local_model_path: str = args.model

    # initialize model
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        local_model_path,
        use_safetensors=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer

def main(args: Arguments, generate_arguments: GenerateArguments, generation_config: GenerationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    streamer = None
    match generate_arguments.streamer:
        case StreamerType.NONE:
            streamer = None
        case StreamerType.STANDARD:
            streamer = transformers.TextStreamer(tokenizer)
        case StreamerType.SPECULATIVE:
            streamer = SpeculativeTextStreamer(tokenizer)
        case _:
            raise ValueError(f"Unsupported streamer type {generate_arguments.streamer}")

    # Generation strategy selection with new branch for token-level adaptive strategy.
    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    elif generation_config.generation_strategy == "layerdrop":
        generation_strategy: GenerationStrategy = LayerDropGenerationStrategy(
            dropout_rate=generation_config.dropout_rate,
            seed=generation_config.layerdrop_seed
        )
    elif generation_config.generation_strategy == "depth_adaptive_token":
        generation_strategy: GenerationStrategy = DepthAdaptiveTokenGenerationStrategy(
            halting_threshold=generation_config.halting_threshold,
            min_layers=generation_config.min_layers,
            max_layers=generation_config.max_layers,
        )
    else:
        raise Exception(f"Unsupported generation strategy: {generation_config.generation_strategy}")

    # initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )

    # Warmup
    warmup = 1
    for _ in range(warmup):
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generate(**tokenizer("This is a warmup prompt", return_tensors="pt").to('mps'), max_new_tokens=10)

    while True:
        print()
        print("Enter a prompt and then press ctrl+d twice for the model to complete:")
        print("======================================================================")
        print()

        print(colorama.Fore.BLUE, end="")
        prompt = sys.stdin.read()
        print(colorama.Style.RESET_ALL, end=" ")

        try:
            response: GenerationResult = generator.generate(
                prompt=prompt,
                generation_config=generation_config,
                streamer=streamer,
            )
        except:
            print(colorama.Style.RESET_ALL)
            traceback.print_exc()
            raise
        num_tokens = response.num_tokens_generated
        total_time = response.total_time

        if streamer:
            streamer.end()
        else:
            print(response.decoded_prediction)

        print(colorama.Style.RESET_ALL)
        print()
        print(f"\tTime taken: {total_time :.3f}s")
        print(f"\tNumber of tokens: {num_tokens}")
        print(f"\tTime per token: {total_time / num_tokens : .3f}s")
        print(f"\tTokens per second: {num_tokens / total_time :.3f}")
        if generation_config.generation_strategy == "self_speculative":
            print(f"\tAcceptance Rate: {response.generation_strategy_result.acceptance_rate:.2%}")
        print()

def process_cli_arguments() -> Tuple[Arguments, GenerateArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((Arguments, GenerateArguments, GenerationConfig))
    (
        general_arguments,
        generate_arguments,
        generation_config,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=False)

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
    else:
        general_arguments.model_args = {}

    return general_arguments, generate_arguments, generation_config

if __name__ == "__main__":
    args, benchmark_arguments, generation_config = process_cli_arguments()
    main(args, benchmark_arguments, generation_config)

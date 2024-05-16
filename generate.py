import colorama
import datetime
import random
import sys
import time
import torch
import traceback
import transformers
import os

from arguments import process_cli_arguments
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.speculative_streamer import SpeculativeTextStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"
backend = "nccl" if device == "cuda" else "gloo"

torch.distributed.init_process_group(
    backend=f"{device}:{backend}", timeout=datetime.timedelta(hours=48)
)
rank = int(os.environ["LOCAL_RANK"])
benchmark_arguments, generation_config = process_cli_arguments()

random.seed(benchmark_arguments.seed)
torch.manual_seed(benchmark_arguments.seed)
if rank != 0:
    # only run on rank 0, we don't support parallel inference yet
    exit()

local_model_path: str = benchmark_arguments.model_path

# initialize model
tokenizer = transformers.LlamaTokenizer.from_pretrained(
    local_model_path, use_fast=False
)
streamer = SpeculativeTextStreamer(tokenizer)
config = transformers.LlamaConfig.from_pretrained(local_model_path)
model = transformers.LlamaForCausalLM.from_pretrained(
    local_model_path,
    config=config,
    torch_dtype=torch.float16,
    **benchmark_arguments.model_args,
)
model.to(device)
model.half()
model.eval()

if generation_config.generation_strategy == "autoregressive":
    generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
elif generation_config.generation_strategy == "self_speculative":
    generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
else:
    raise Exception(
        f"Unsupported generation strategy: {generation_config.generation_strategy}"
    )

# initialize generator
generator = HuggingfaceLlamaGenerator(
    tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
)

while True:
    print()
    print("Enter a prompt and then press ctrl+d twice for the model to complete:")
    print("======================================================================")
    print()

    print(colorama.Fore.BLUE, end="")
    prompt=sys.stdin.read()
    print(colorama.Style.RESET_ALL, end=" ")

    start = time.time()
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
    total_time = time.time() - start

    streamer.end()

    print(colorama.Style.RESET_ALL)
    print()
    print(f"\tTime taken: {total_time :.3f}s")
    print(f"\tNumber of tokens: {num_tokens}")
    print(f"\tTime per token: {total_time / num_tokens : .3f}s")
    print(f"\tTokens per second: {num_tokens / total_time :.3f}")
    if generation_config.generation_strategy == "self_speculative":
        print(f"\tAcceptance Rate: {response.generation_strategy_result.acceptance_rate:.2%}")
    print()

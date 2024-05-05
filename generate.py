import datetime
import random
import torch
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


torch.distributed.init_process_group(
    backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=48)
)
rank = int(os.environ["LOCAL_RANK"])
benchmark_arguments, generation_config = process_cli_arguments()
# TODO: make max_steps an arg with default value
generation_config.max_steps = 4096
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
config = transformers.LlamaConfig.from_pretrained(local_model_path)
model = transformers.LlamaForCausalLM.from_pretrained(
    local_model_path,
    config=config,
    torch_dtype=torch.float16,
)
model.cuda()
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
    print("Prompt: ")
    response: GenerationResult = generator.generate(
        prompt=input(),
        generation_config=generation_config,
    )
    print(f"Response: {response.decoded_prediction}")

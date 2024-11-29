# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import json
import os
import random
import logging
from copy import copy

import torch

import transformers

from benchmark import Arguments, BenchmarkArguments, process_cli_arguments
from data import get_data
from generate import load_model_and_tokenizer, setup
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy

from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    HuggingfaceLlamaGenerator,
)

from self_speculation.self_speculation_generator import (
    SelfSpeculativeGenerationStrategy,
)
from tqdm import tqdm

log = logging.getLogger(__name__)


def main(args: Arguments, benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str, seed = 0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    torch.manual_seed(seed)

    setup(args, device=device)
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    ar_generation_config = copy(generation_config)
    ar_generation_config.exit_layer = -1
    ar_generation_config.num_speculations = -1

    # initialize generator
    spec_generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=SelfSpeculativeGenerationStrategy(),
    )

    ar_generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=AutoRegressiveGenerationStrategy(),
    )

    evaluation_set = get_data(
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        dataset=benchmark_arguments.dataset,
        data_path=benchmark_arguments.data_path,
    )

    errors: int = 0
    for i, example in enumerate(tqdm(evaluation_set)):
        spec_response: GenerationResult = spec_generator.generate(
            prompt=example.input,
            generation_config=generation_config,
        )
        ar_response: GenerationResult = ar_generator.generate(
            prompt=example.input,
            # generation config to use the full model
            generation_config=ar_generation_config,
        )

        if spec_response.decoded_prediction != ar_response.decoded_prediction:
            errors += 1
            log.info("Error found")
            log.info(f"Spec response: {spec_response}")
            log.info(f"AR response: {ar_response}")

    metric_result = {"errors": errors, "error_pct": errors / len(evaluation_set)}
    print(metric_result)

    with open(output_fname, "w") as f:
        json.dump(metric_result, f)


if __name__ == "__main__":
    args, benchmark_arguments, generation_config = process_cli_arguments()
    log.setLevel(level=logging.INFO) # TODO: set level based on argument
    os.makedirs(args.output_dir, exist_ok=True)
    main(args, benchmark_arguments, generation_config, f"{args.output_dir}/correctness_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

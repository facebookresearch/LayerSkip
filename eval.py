# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import transformers
from tqdm import tqdm
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from arguments import Arguments, simple_parse_args_string
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from generate import load_model_and_tokenizer, setup

@dataclass
class EvalArguments:
    tasks: List[str] = None
    num_fewshot: Optional[int] = None
    device: Optional[str] = None
    use_cache: Optional[str] = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    limit: Optional[int] = None
    bootstrap_iters: int = 100000
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: bool = False
    gen_kwargs: Optional[str] = None
    verbosity: str = "INFO"
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234

def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)


# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(LM):
    def __init__(self, generator, generation_config, device):
        super().__init__()
        self.generator = generator
        self.generation_config = generation_config
        self.device = device

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        # TODO: remove "temperature", "top_p", and "top_k" from "gen_args"
        until = gen_args.get("until", [])
        self.generation_config.stop_words = until
        generations = []
        for prompt in tqdm(prompts):
            response: GenerationResult = self.generator.generate(
                prompt=prompt,
                generation_config=self.generation_config,
            )
            generations.append(response.decoded_prediction)
        filtered_gen = []
        for g in tqdm(generations):
            for e in until:
                # g = g.replace(e, "")
                g = g.split(e, 1)[0]
            filtered_gen.append(g)
        return filtered_gen

    # TODO: just call LM HF loglikelihood(...)
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        results = []
        for prompt, input in tqdm(zip(prompts, inputs)):
            prompt_enc = self.generator.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
            input_enc = self.generator.tokenizer(input, return_tensors="pt", add_special_tokens=True).to(self.device)
            loss = self.generator.model(**prompt_enc, labels=prompt_enc["input_ids"]).loss
            next_token = self.generator.model(**input_enc).logits[:,-1].argmax()
            results.append((loss.item(), next_token.all().item()))

        return results

    # TODO: just call LM HF loglikelihood_rolling(...)
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        prompts = [req.args[0] for req in requests]
        results = []
        for prompt in tqdm(prompts):
            output = self.generator.model(**self.generator.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device))
            loss = output.loss
            results.append((loss.sum().item(),))

        return results

def main(args: Arguments, eval_arguments: EvalArguments, generation_config: GenerationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)


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

    # create evaluator
    wrap = EvalHarnessLM(generator, generation_config, device)

    # Warmup
    warmup = 1
    for _ in range(warmup):
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generate(**tokenizer("This is a warmup prompt", return_tensors="pt").to(device), max_new_tokens=10)

    # Evaluate
    results = simple_evaluate(wrap, **asdict(eval_arguments))

    # TODO: log results, generation samples, etc.
    print(results["results"])

def process_cli_arguments() -> Tuple[Arguments, EvalArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((Arguments, EvalArguments, GenerationConfig))
    (
        general_arguments,
        eval_arguments,
        generation_config,
        _remaining,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
    else:
        general_arguments.model_args = {}

    return general_arguments, eval_arguments, generation_config

if __name__ == "__main__":
    args, eval_arguments, generation_config = process_cli_arguments()
    main(args, eval_arguments, generation_config)
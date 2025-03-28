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
from lm_eval import utils
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, TemplateLM
from lm_eval.models.utils import pad_and_concat, Collator

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
from benchmark import EvaluationMetrics
from self_speculation.layer_drop_generator import LayerDropGenerationStrategy
# NEW: Import the depth-adaptive token-level strategy.
from self_speculation.depth_adaptive_token_generator import DepthAdaptiveTokenGenerationStrategy

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
    if not dict_list:
        return True
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)

# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(TemplateLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
            self,
            generator: HuggingfaceLlamaGenerator,
            generation_config: GenerationConfig,
            device: Union[str, torch.device],
            logits_cache: bool = True,
            batch_size: Optional[Union[int, str]] = 1,
            add_bos_token: Optional[bool] = False,
            max_length: Optional[int] = None,
        ):
        super().__init__()
        assert batch_size == 1, "Currently we only support batch size 1"
        self.generator = generator
        self.generation_config = generation_config
        self.device = device
        self.logits_cache = logits_cache
        self.batch_size = batch_size
        self.add_bos_token = add_bos_token
        self._max_length = max_length
        self.metric_result = None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        until = gen_args.get("until", [])
        self.generation_config.stop_words = until
        generations = []
        metrics = EvaluationMetrics.build_metrics()
        for prompt in tqdm(prompts):
            response: GenerationResult = self.generator.generate(
                prompt=prompt,
                generation_config=self.generation_config,
            )
            generations.append(response.decoded_prediction)
            metrics.update(None, response)
        self.metric_result = metrics.compute()
        filtered_gen = []
        for p, g in tqdm(zip(prompts, generations)):
            for e in until:
                g = g.split(e, 1)[0]
            filtered_gen.append(g)
            self.cache_hook.add_partial("generate_until", (p, gen_args), g)
        return filtered_gen

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.generator.model.config, attr):
                return getattr(self.generator.model.config, attr)
        if hasattr(self.generator.tokenizer, "model_max_length"):
            if self.generator.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.generator.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        special_tokens_kwargs = {}
        if add_special_tokens is None:
            special_tokens_kwargs = {
                "add_special_tokens": False or self.add_bos_token
            }
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}
        encoding = self.generator.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts" if self.logits_cache else None,
            group_fn=_lookup_one_token_cont,
        )

        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None

            for _, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape
                padding_len_inp = max(padding_len_inp, inplen) if padding_len_inp is not None else inplen
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            call_kwargs = {}
            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")
            multi_logits = torch.nn.functional.log_softmax(
                self.generator.model(batched_inps, **call_kwargs).logits, dim=-1
            )

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                contlen = len(cont_toks)
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = logits[inplen - contlen : ctx_len]
                logits = logits.unsqueeze(0)
                greedy_tokens = logits.argmax(dim=-1)

                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self.device).unsqueeze(0)
                    max_equal = (greedy_tokens == cont_toks).all()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
                    answer = (float(logits.sum()), bool(max_equal))
                    res.append(answer)
                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []
        adaptive_batch_size = None
        if self.batch_size == "auto":
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for (string,) in tqdm(
            [req.args for req in requests], disable=(disable_tqdm or (self.rank != 0))
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]
            pad_amnt = 0
            if self.world_size > 1:
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                requests=rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )
            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                string_nll = [x[0] for x in string_nll]
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

def main(args: Arguments, eval_arguments: EvalArguments, generation_config: GenerationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    elif generation_config.generation_strategy == "layerdrop":
        generation_strategy: GenerationStrategy = LayerDropGenerationStrategy(
            dropout_rate=generation_config.dropout_rate,
            seed=generation_config.layerdrop_seed or args.seed
        )
    # NEW: Add branch for depth-adaptive token-level strategy.
    elif generation_config.generation_strategy == "depth_adaptive_token":
        generation_strategy: GenerationStrategy = DepthAdaptiveTokenGenerationStrategy(
            halting_threshold=generation_config.halting_threshold,
            min_layers=generation_config.min_layers,
            max_layers=generation_config.max_layers,
        )
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

    print(results["results"])
    wrap.metric_result.pop("predicted_text")
    print(wrap.metric_result)

def process_cli_arguments() -> Tuple[Arguments, EvalArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((Arguments, EvalArguments, GenerationConfig))
    (
        general_arguments,
        eval_arguments,
        generation_config,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=False)

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
    else:
        general_arguments.model_args = {}

    return general_arguments, eval_arguments, generation_config

if __name__ == "__main__":
    args, eval_arguments, generation_config = process_cli_arguments()
    main(args, eval_arguments, generation_config)

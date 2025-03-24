# LayerSkip
<a href='https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![License: CC BY-NC](https://img.shields.io/badge/License-CC_BY--NC-lightgrey.svg)](./LICENSE) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=oPxdfVVmLP8) [![arXiv](https://img.shields.io/badge/arXiv-2404.16710-b31b1b.svg)](https://arxiv.org/abs/2404.16710) [![alphaXiv](https://img.shields.io/badge/alphaXiv-2404.16710-9a2037.svg)](https://www.alphaxiv.org/abs/2404.16710)

This code base is the implementation of [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710).

<div align="center">
  <img src="https://github.com/user-attachments/assets/1fdd91d9-37ea-4b42-b5be-579fb5e1f2f2" width="500">
</div>

## News
- [2024/11] 🤗 LayerSkip has been integrated into [Hugging Face `transformers`](https://huggingface.co/blog/layerskip).

## Getting Started
- Clone repo:
```console
$ git clone git@github.com:facebookresearch/LayerSkip.git
$ cd LayerSkip
```

- Setup environment:
```console
$ conda create --name layer_skip python=3.10
$ conda activate layer_skip

$ pip install -r requirements.txt
```

- Access models:
In order to observe speedup, you need to access LLMs that have been trained using the LayerSkip recipe. We provide 6 checkpoints on [HuggingFace](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a) of different Llama models continually pretrained using the LayerSkip recipe:

    - [`facebook/layerskip-llama2-7B`](https://huggingface.co/facebook/layerskip-llama2-7B)
    - [`facebook/layerskip-llama2-13B`](https://huggingface.co/facebook/layerskip-llama2-13B)
    - [`facebook/layerskip-llama2-70B`](https://huggingface.co/facebook/layerskip-llama2-70B)
    - [`facebook/layerskip-codellama-7B`](https://huggingface.co/facebook/layerskip-codellama-7B)
    - [`facebook/layerskip-codellama-34B`](https://huggingface.co/facebook/layerskip-codellama-34B)
    - [`facebook/layerskip-llama3-8B`](https://huggingface.co/facebook/layerskip-llama3-8B)
    - [`facebook/layerskip-llama3.2-1B`](https://huggingface.co/facebook/layerskip-llama3.2-1B)

In order to access each model:

1. Visit the model's corresponding link above, make sure you are logged on the HuggingFace website with your account.
2. Fill the request form and submit it. Approval may take a while and you should receive an email notification to notify you that permission to the model is granted.
3. Follow the steps [here](https://huggingface.co/docs/hub/en/security-tokens) to obtain a user access token.
4. In the command-line run `huggingface-cli login`, and you will be prompted to provide the token you have obtained in Step 3.

Once you run those steps, the commands below to run the LayerSkip checkpoints should work.

## Generate

To run one of our models in interactive mode using regular autoregressive decoding:
```console
$ torchrun generate.py --model facebook/layerskip-llama2-7B \
    --sample True \
    --max_steps 512
```

In order to observe speedup, you need to use self-speculative decoding to generate tokens, and specify `--exit_layer`, the layer the draft stage to exit at, and `--num_speculations`, the number of draft tokens:
```console
$ torchrun generate.py --model facebook/layerskip-llama2-7B \
    --sample True \
    --max_steps 512 \
    --generation_strategy self_speculative \
    --exit_layer 8 \
    --num_speculations 6
```

Tips:
- You may change `--model` to any HuggingFace model but in order to observe speedup with self-speculative decoding, use a model trained using the LayerSkip recipe, such as those we have [open sourced on HuggingFace](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a).
- By default we enable sampling. You may change the sampling behaviour using the `--sample`, `--temperature`, `--top_p`, and `--top_k` arguments.
- You may run `python generate.py --help` for details on different command-line arguments.

## Benchmark

To benchmark on a dataset:

```console
$ torchrun benchmark.py --model facebook/layerskip-llama2-7B \
    --dataset cnn_dm_summarization \
    --num_samples 100 \
    --generation_strategy self_speculative \
    --exit_layer 8 \
    --num_speculations 6 \
    --output_dir ./logs
```

Tips:
- You can specify different tasks by modifying the `--dataset` argument:
    - `cnn_dm_summarization`: CNN/DM Summarization
    - `xsum_summarization`: XSUM Summarization
    - `cnn_dm_lm`: CNN/DM Language Modeling (given the first few words of an article, generate the remaining article)
    - `human_eval`: HumanEval Coding
- By default, the tasks run as 0-shot. You can change to any specified `n`-shot by specifying the `--n_shot` argument.
- By default we enable sampling, while the results reported in the paper were greedy decoding without sampling. You may change the sampling behaviour using the `--sample`, `--temperature`, `--top_p`, and `--top_k` arguments.
- You may run `python benchmark.py --help` for details on different command-line arguments.

## Evaluate

We have integrated our generation scripts with [Eleuther Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) to enable a large number of tasks and properly post-process generated text.

```console
$ torchrun eval.py --model facebook/layerskip-llama2-7B \
    --tasks gsm8k \
    --limit 10 \
    --generation_strategy self_speculative \
    --exit_layer 8 \
    --num_speculations 6 \
    --output_dir ./logs
```

Tips:
- Note that with speculative decoding we can only obtain speedups from generation tasks (e.g., `gsm8k` or `cnn_dailymail`), while classificaton tasks, i.e., multiple choice question tasks (e.g., `piqa`, `social_iqa`) or True/False question tasks (e.g., `boolq`) will not lead to speedup.
- You can specify arbitrary number of tasks supported by Eleuther Evaluation Harness using the `--tasks` argument. To get a list of all of possible tasks, check this [link](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).
- Similar to the `generate.py` and `benchmark.py` scripts, you may specify different models, datasets, and sampling parameters
- You may run `python benchmark.py --help` for details on different command-line arguments.

## Sweep
Our inference hyperparameters, `exit_layer` and `num_speculations` determine the speedup during inference:
- `exit_layer`:
    - smaller means a faster but less accurate draft stage
    - larger means a more accurate but slower draft stage
- `num_speculations`:
    - smaller means higher acceptance rate but verification stage will amortize less the draft stage
    - learger means verification stage will better amortize the draft stage but acceptance rate decreases

The optimal combination of `exit_layer` and `num_speculations` may change with the model, dataset and sampling parameters. Hence, we provided a script to sweep over a grid of different `exit_layer` and `num_speculations`:

```console
$ torchrun sweep.py --model facebook/layerskip-llama2-7B \
    --dataset human_eval \
    --generation_strategy self_speculative \
    --num_samples 150 \
    --max_steps 256 \
    --output_dir ./logs/ \
    --sample False
```

This will create a CSV file in the directory specified in the `--outpu_dir` argument.

Tips:
- Similar to the `generate.py` and `benchmark.py` scripts, you may specify different models, datasets, and sampling parameters
- You may run `python sweep.py --help` for details on different command-line arguments.

## Correctness
In order to verify that the generated tokens of our self-speculative decoding algorithm are correct, we have created a script to compare the outputs of autoregressive decoding with self-speculative decoding. Note that the outputs we can only guarantee equivalence when there is no sampling (i.e., `--sample False`): 
```console
$ torchrun correctness.py --model facebook/layerskip-llama2-7B \
    --dataset human_eval \
    --generation_strategy self_speculative \
    --num_speculations 6 \
    --exit_layer 4 \
    --num_samples 10 \
    --sample False \
    --output_dir ./logs
```

## Using Docker
Kindy check [DOCKER.md](DOCKER.md) to setup the project using docker

## Other Implementations
We also have other implementations of LayerSkip inference:
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast/tree/LayerSkip?tab=readme-ov-file#self-speculative-sampling): gpt-fast is a simple and efficient pytorch-native transformer text generation. We have implemented LayerSkip in the gpt-fast codebase to enable compouding it with other optimizations such as `torch.compile()`, quantization, and tensor parallelism.
- [Native HuggingFace](https://huggingface.co/facebook/layerskip-llama2-7B#huggingface): in the model card of each of our HuggingFace models, we have provided simple code snippets that leverages HuggingFace speculative decoding capabilities using a simple trick to clone the earlier layers of the main model without cloning its weights. Although this implementation is simple and does not require implementing other functions or importing other libraries, it does not share the KV cache or execution between the draft and verification stages.

## Training
Our training implementation is work-in-progress. You can check this [pull request](https://github.com/pytorch/torchtune/pull/1076) for details and discussions.

## Tests
To run unit/integration tests:
```
$ pytest ./tests/
```

## License
LayerSkip is licensed under CC-by-NC license. Refer to the LICENSE file in the top level directory.

## Contributing
We welcome contributions to LayerSkip. If you are interested in contributing please see [this document](./CONTRIBUTING.md).

## Citation
If you use LayerSkip in your research, please use the following BibTex entry:

```bibtex
@misc{layerskip,
    title={LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding},
    author={Mostafa Elhoushi and Akshat Shrivastava and Diana Liskovich and Basil Hosmer and Bram Wasti and Liangzhen Lai and Anas Mahmoud and Bilge Acun and Saurabh Agarwal and Ahmed Roman and Ahmed A Aly and Beidi Chen and Carole-Jean Wu},
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.681",
    doi = "10.18653/v1/2024.acl-long.681",
    pages = "12622--12642",
}
```

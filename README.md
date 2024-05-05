# LayerSkip

## Getting Started
- Clone repo:
```
git clone git@github.com:facebookresearch/LayerSkip.git
```

- Setup environment:
```
conda create --name layer_skip python=3.10
conda activate layer_skip

pip install -r requirements.txt
```

## Run
```
torchrun generate.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
    --data_path dummy \
    --data_format cnn_dm_summarization \
    --num_samples 100 \
    --manifold_output_dir dummy
```

## Evaluate

- Llama 7B continual:
    - CNN/DM Summarization
        AR:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
            --data_format cnn_dm_summarization \
            --num_samples 100 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.15924297273159027, 'rouge-1': 0.2301056683063507, 'rouge-2': 0.07174421846866608, 'rouge-3': 0.03521019592881203, 'bleu_score': 0.0, 'exact_match': 251.8300018310547}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 0.9018121194839478}, 'time_per_token': {'mean': 0.02818162873387337}, 'tokens_per_second': {'mean': 35.90497396469116}}
        ```


        SS:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
            --data_format cnn_dm_summarization \
            --num_samples 100 \
            --generation_strategy self_speculative \
            --num_speculations 6 \
            --exit_layer 8 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.15924297273159027, 'rouge-1': 0.22984926402568817, 'rouge-2': 0.07174421846866608, 'rouge-3': 0.03521019592881203, 'bleu_score': 0.0, 'exact_match': 251.83999633789062}, 'acceptance_rate': {'mean': 0.6541042917966843}, 'total_time': {'mean': 0.5943978071212769}, 'time_per_token': {'mean': 0.018574931472539902}, 'tokens_per_second': {'mean': 59.77584112167359}}
        ```

    - CNN/DM Language Modeling
        AR:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
            --data_format cnn_dm_lm \
            --num_samples 100 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.04713794216513634, 'rouge-1': 0.06010835990309715, 'rouge-2': 0.019858235493302345, 'rouge-3': 0.008144848048686981, 'bleu_score': 0.0, 'exact_match': 3562.909912109375}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 0.8102598142623901}, 'time_per_token': {'mean': 0.02532061919569969}, 'tokens_per_second': {'mean': 39.88337937355041}}
        ```

        SS:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
            --data_format cnn_dm_lm \
            --num_samples 100 \
            --generation_strategy self_speculative \
            --num_speculations 6 \
            --exit_layer 8 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.04713794216513634, 'rouge-1': 0.06010835990309715, 'rouge-2': 0.019858235493302345, 'rouge-3': 0.008144848048686981, 'bleu_score': 0.0, 'exact_match': 3562.909912109375}, 'acceptance_rate': {'mean': 0.3754749954491854}, 'total_time': {'mean': 0.7578680038452148}, 'time_per_token': {'mean': 0.023683375120162962}, 'tokens_per_second': {'mean': 48.140634307861326}}
        ```
    
    - HumanEval
        AR:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path dummy \
            --data_format human_eval \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.2459806650876999, 'rouge-1': 0.2772044837474823, 'rouge-2': 0.11060480773448944, 'rouge-3': 0.058753617107868195, 'bleu_score': 0.0, 'exact_match': 143.28659057617188}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 0.8014862653685779}, 'time_per_token': {'mean': 0.025220286214678752}, 'tokens_per_second': {'mean': 39.754123513291525}}
        ```

        SS:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path dummy \
            --data_format human_eval \
            --generation_strategy self_speculative \
            --num_speculations 6 \
            --exit_layer 4 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.2464337944984436, 'rouge-1': 0.27765196561813354, 'rouge-2': 0.11060480773448944, 'rouge-3': 0.058753617107868195, 'bleu_score': 0.0, 'exact_match': 143.40243530273438}, 'acceptance_rate': {'mean': 0.25404511601096247}, 'total_time': {'mean': 0.6747547591604838}, 'time_per_token': {'mean': 0.021292833906666534}, 'tokens_per_second': {'mean': 50.01566022779883}}
        ```

- Llama 1.5B pretrained from scratch:
AR:
```
torchrun benchmark.py --model_path /fsx-atom/melhoushi/xldumps/train_llama2_1.5B_sweep_32_gpus/train_llama2_1.5B_sweep_32_gpus_run008/checkpoints/checkpoint_0050000_consolidated_hf/ \
    --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
    --data_format cnn_dm_summarization \
    --num_samples 100 \
    --manifold_output_dir ./logs
```
Result:
```
{'predicted_text': {'rouge-l': 0.17063584923744202, 'rouge-1': 0.2387697994709015, 'rouge-2': 0.07077603787183762, 'rouge-3': 0.030207907781004906, 'bleu_score': 0.0, 'exact_match': 237.10000610351562}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 0.7412997961044312}, 'time_per_token': {'mean': 0.023165618628263475}, 'tokens_per_second': {'mean': 45.00272045135498}}
```

SS:
```
torchrun benchmark.py --model_path /fsx-atom/melhoushi/xldumps/train_llama2_1.5B_sweep_32_gpus/train_llama2_1.5B_sweep_32_gpus_run008/checkpoints/checkpoint_0050000_consolidated_hf/ \
    --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
    --data_format cnn_dm_summarization \
    --num_samples 100 \
    --generation_strategy self_speculative \
    --num_speculations 12 \
    --exit_layer 8 \
    --manifold_output_dir ./logs
```
Result:
```
{'predicted_text': {'rouge-l': 0.16813838481903076, 'rouge-1': 0.2398979812860489, 'rouge-2': 0.07607144117355347, 'rouge-3': 0.03642928972840309, 'bleu_score': 0.0, 'exact_match': 250.60000610351562}, 'acceptance_rate': {'mean': 0.019020007958170028}, 'total_time': {'mean': 2.282204785346985}, 'time_per_token': {'mean': 0.07131889954209328}, 'tokens_per_second': {'mean': 14.112926120758056}}
```
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from accelerate import Accelerator
accelerator = Accelerator()

train_dataset = load_dataset("WillHeld/top_v2", split="train")

ckpt_id = "meta-llama/Llama-2-7b-hf" # "meta-llama/Llama-3.2-1B"
hub_model_id = "melhoushi/exitskip-llama2-7b-topv1-v2" # "melhoushi/layerskip-llama3.2-1b-topv1-v7"
root_dir = "/fsx-atom/melhoushi/trl/layer-skip/"

class LayerSkipSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_exit_layer = 0 # initialize with 0
        self.always_last_layer = False
        self.early_exit_loss_scale = 1.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # TODO: if self.always_last_layer is False, we can modify model.config.num_hidden_layers and hence speedup training
        if self.always_last_layer:
            self.early_exit_layer = (self.early_exit_layer % (model.config.num_hidden_layers - 1)) + 1 # rotates between [1, num_hidden_layers-1]
        else:
            self.early_exit_layer = (self.early_exit_layer % model.config.num_hidden_layers) + 1 # rotates between [1, num_hidden_layers]

        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)

        hidden_state = outputs["hidden_states"][self.early_exit_layer]
        if self.early_exit_layer != model.config.num_hidden_layers:
            hidden_state = model.model.norm(hidden_state)
        logits = model.lm_head(hidden_state)
        loss_early = model.loss_function(logits=logits, labels=labels, vocab_size=model.vocab_size)

        if self.always_last_layer:
            loss_last = model.loss_function(logits=outputs["logits"], labels=labels, vocab_size=model.vocab_size)
            loss = self.early_exit_loss_scale * loss_early.to(loss_last.device) + 1.0 * loss_last
            # normalize loss scales
            loss = loss / (1.0 + self.early_exit_loss_scale)
        else:
            loss = loss_early

        return loss

model = AutoModelForCausalLM.from_pretrained(ckpt_id, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(ckpt_id)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

output_dir = f"{root_dir}/{hub_model_id}"

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['utterance'])):
        text = f"### Instruction: {example['utterance'][i]}\n ### Response: {example['semantic_parse'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

args = SFTConfig(
    output_dir=output_dir,
    do_train=True,
    max_seq_length=512,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    report_to="none",
    push_to_hub=True,
    num_train_epochs=1.0,
    save_steps=2000,
    hub_model_id=hub_model_id,
)

trainer = LayerSkipSFTTrainer(
    model,
    train_dataset=train_dataset,
    args=args,
    formatting_func=formatting_prompts_func,
    # data_collator=collator,
)

trainer.train()

"""## Profile"""

import time

checkpoint = hub_model_id
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prompt = "### Instruction: Set alarm for 6am every day\n ### Response: "
inputs = tokenizer(prompt, return_tensors="pt").to(device)

do_sample = False

model.generation_config.pad_token_id = tokenizer.pad_token_id

warmup=4
for _ in range(warmup):
    outputs = model.generate(**inputs, max_new_tokens=40)

start=time.time()
outputs = model.generate(**inputs, do_sample=do_sample, max_new_tokens=40)
delta_time_orig=time.time()-start
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
print(f"Orig Time: {delta_time_orig}")

for i in range(1, 16):
    start=time.time()
    outputs = model.generate(**inputs, assistant_early_exit=i, do_sample=do_sample, max_new_tokens=40)
    delta_time_skip=time.time()-start
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    print(f"Layerskip Time: {delta_time_skip}")

    speedup =  delta_time_orig / delta_time_skip
    print(f"For Layer: {i} Speedup: {speedup}")

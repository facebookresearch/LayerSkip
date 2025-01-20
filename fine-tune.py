import logging
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FineTuneArguments:
    ckpt: str = "meta-llama/Llama-3.2-1B"
    ds_ckpt: str = "WillHeld/top_v2"
    template: str = "###INST: {utterance}\n\n###RES: {semantic_parse}"
    lr: float = 1e-3
    batch_size: int = 8
    epochs: int = 1
    eval_freq: int = 100
    early_exit_loss_scale: float = 1.0
    save_steps: int = 500
    output_dir: str = "./checkpoints"


class LayerSkipModel(nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        self.early_exit_layer = 0

    def forward(self, input_ids, attention_mask):
        # If there are N layers, there are N+1 hidden states [l=0, l=N]
        # The zero th hidden state (l=0) is input to the embedding layer
        # The last hidden state (l=N) is the normalized output of the final layer
        # We need to early exit from layers [l=1, l=N-1] both inclusive
        self.early_exit_layer = (self.early_exit_layer % (self.num_layers - 1)) + 1

        # Get the output logits and hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs["logits"]
        hidden_states = outputs["hidden_states"]

        # Select the exit hidden state and normalize it
        exit_state = hidden_states[self.early_exit_layer]
        exit_state = self.model.model.norm(exit_state)
        exit_logits = self.model.lm_head(exit_state)

        return logits, exit_logits


def collate_fn(batch, template):
    return [template.format(**sample) for sample in batch]


def train_and_eval(train_dl, val_dl, tokenizer, device, trainer, optimizer, args):
    global_step = 0
    for epoch in range(args.epochs):
        trainer.train()
        for step, batch in enumerate(train_dl):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs["input_ids"][:, :-1].to(device)
            input_attn_mask = inputs["attention_mask"][:, :-1].to(device)

            labels = inputs["input_ids"][:, 1:].to(device)

            logits, exit_logits = trainer(
                input_ids=input_ids, attention_mask=input_attn_mask
            )
            orig_loss = trainer.model.loss_function(
                logits=logits, labels=labels, vocab_size=trainer.model.vocab_size
            )
            exit_loss = trainer.model.loss_function(
                logits=exit_logits, labels=labels, vocab_size=trainer.model.vocab_size
            )
            total_scale = 1.0 + args.early_exit_loss_scale
            total_loss = (
                1.0 * orig_loss + args.early_exit_loss_scale * exit_loss
            ) / total_scale

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.eval_freq == 0:
                trainer.eval()
                eval_loss = 0.0
                num_val_steps = 0
                with torch.no_grad():
                    for val_batch in val_dl:
                        inputs = tokenizer(val_batch, return_tensors="pt", padding=True)

                        input_ids = inputs["input_ids"][:, :-1].to(device)
                        input_attn_mask = inputs["attention_mask"][:, :-1].to(device)

                        labels = inputs["input_ids"][:, 1:].to(device)

                        logits, exit_logits = trainer(
                            input_ids=input_ids, attention_mask=input_attn_mask
                        )
                        orig_loss = trainer.model.loss_function(
                            logits=logits,
                            labels=labels,
                            vocab_size=trainer.model.vocab_size,
                        )
                        exit_loss = trainer.model.loss_function(
                            logits=exit_logits,
                            labels=labels,
                            vocab_size=trainer.model.vocab_size,
                        )
                        total_scale = 1.0 + args.early_exit_loss_scale
                        loss = (
                            1.0 * orig_loss + args.early_exit_loss_scale * exit_loss
                        ) / total_scale

                        eval_loss += loss.item()
                        num_val_steps += 1

                logger.info(
                    f"Epoch {epoch}, Step {global_step}: "
                    f"Train Loss: {total_loss.item():.4f}, "
                    f"Val Loss: {eval_loss / num_val_steps:.4f}"
                )
                trainer.train()

            if global_step % args.save_steps == 0:
                checkpoint_path = f"{args.output_dir}/checkpoint_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": trainer.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_bos_token = True  # This defaults to True
    tokenizer.add_eos_token = True  # This defaults to False, setting it to True will add eos token to each sample

    model = AutoModelForCausalLM.from_pretrained(args.ckpt)

    trainer = LayerSkipModel(model=model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    train_ds = load_dataset(args.ds_ckpt, split="train")
    val_ds = load_dataset(args.ds_ckpt, split="eval")

    collate_fn_with_template = partial(collate_fn, template=args.template)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn_with_template,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn_with_template,
    )

    trainer.to(device)
    trainer.train()

    train_and_eval(train_dl, val_dl, tokenizer, device, trainer, optimizer, args)


def process_cli_arguments():
    parser = HfArgumentParser(FineTuneArguments)
    args = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    return args[0]


if __name__ == "__main__":
    args = process_cli_arguments()
    main(args)

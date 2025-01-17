import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


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


def collate_fn(batch):
    formatted_batch = [
        f"###INST: {sample['utterance']}\n\n###RES: {sample['semantic_parse']}"
        for sample in batch
    ]
    return formatted_batch


if __name__ == "__main__":
    ckpt = "meta-llama/Llama-3.2-1B"
    ds_ckpt = "WillHeld/top_v2"
    lr = 1e-3
    batch_size = 8
    epochs = 1
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(ckpt)

    trainer = LayerSkipModel(model=model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    train_ds = load_dataset(ds_ckpt, split="train")
    val_ds = load_dataset(ds_ckpt, split="eval")

    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

    trainer.to(device)
    trainer.train()

    for idx in range(epochs):
        for idx, batch in enumerate(train_dl):
            inputs = tokenizer(batch, return_tensors="pt", padding=True)

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
            loss = orig_loss + exit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                eval_loss = 0
                trainer.eval()
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_dl):
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
                        loss = orig_loss + exit_loss

                        eval_loss += loss.item()

                print(
                    f"Epoch: {idx}, Train Loss: {loss.item():0.2f} Val Loss: {eval_loss / (val_idx - 1):0.2f}"
                )
                trainer.train()

from transformers.generation.streamers import TextStreamer

class SpeculativeTextStreamer(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def put(self, value, escape_new_line: bool = False):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        orig_text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        self.token_cache.extend(value.tolist())
        new_text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # Escape new line in the newly added tokens
        if escape_new_line:
            diff_text = new_text.replace(orig_text, "")
            diff_text = diff_text.replace("\n", "\\n")
            new_text = orig_text + diff_text

        printable_text = new_text[self.print_len :]        
        self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def delete(self, num_tokens: int, escape_new_line: bool = False):
        orig_text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        self.token_cache = self.token_cache[:len(self.token_cache)-num_tokens]
        new_text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        if escape_new_line:
            diff_text = new_text.replace(orig_text, "")
            diff_text = diff_text.replace("\n", "\\n")
            new_text = orig_text + diff_text

        remove_len = self.print_len - len(new_text)  

        # Backspace character, "\b" only returns the cursor without deleting characters.\
        # So we print empty spaces and then return the cursor again
        print("\b"*remove_len, flush=True, end="")
        print(" "*remove_len, flush=True, end="")
        print("\b"*remove_len, flush=True, end="")
        self.print_len = len(new_text)

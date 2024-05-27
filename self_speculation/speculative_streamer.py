from transformers.generation.streamers import TextStreamer
import threading

class SpeculativeTextStreamer(TextStreamer):
    def __init__(self, *args, non_blocking=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_blocking = non_blocking
        self.text_cache = ""

    def put(self, value, escape_new_line: bool = False, color=None):
        if self.non_blocking:
            thread = threading.Thread(target=self._put, args=(value, escape_new_line, color))
            thread.start()
        else:
            return self._put(value, escape_new_line, color)

    def delete(self, num_tokens: int, escape_new_line: bool = False):
        if self.non_blocking:
            thread = threading.Thread(target=self._delete, args=(num_tokens, escape_new_line))
            thread.start()
        else:
            return self._delete(num_tokens, escape_new_line)

    def _put(self, value, escape_new_line: bool = False, color=None):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if color:
            print(color, end="")

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        orig_text = self.text_cache
        self.token_cache.extend(value.tolist())
        new_text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        self.text_cache = new_text

        # Escape new line in the newly added tokens
        if escape_new_line:
            diff_text = new_text.replace(orig_text, "")
            diff_text = diff_text.replace("\n", "\\n")
            new_text = orig_text + diff_text

        printable_text = new_text[self.print_len :]        
        self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

        if new_text.endswith("\n") and not escape_new_line:
            self.token_cache = []
            self.text_cache = ""
            self.print_len = 0

        if color:
            print(color, end="")

    def _delete(self, num_tokens: int, escape_new_line: bool = False):
        orig_text = self.text_cache
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

    def end(self):
        super().end()
        self.text_cache = ""

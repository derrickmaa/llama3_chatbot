class MessageFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message):
        tokens = []
        tokens.extend(self.tokenizer.convert_tokens_to_ids("<|start_header_id|>"))
        print(tokens)
        tokens.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(message["role"])))
        tokens.extend(self.tokenizer.convert_tokens_to_ids("<|end_header_id|>"))
        tokens.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("\n\n")))
        return tokens

    def encode_message(self, message):
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(message["content"].strip()))
        )
        tokens.extend(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        tokens.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("\n\n")))
        return tokens

    def encode_dialog_prompt(self, dialog):
        tokens = []
        tokens.extend(
            self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>"))
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens

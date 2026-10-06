"""Construction-local masking/padding; legacy dataset behavior is unchanged."""
import torch


class ConstructionDataset:
    def __init__(self, rows, tokenizer, template_args, max_length=512):
        self.rows, self.tokenizer, self.template_args = rows, tokenizer, template_args
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from data.utils import preprocess_chat_instance
        row = self.rows[index]
        item = preprocess_chat_instance(self.tokenizer, self.template_args, [row["question"]],
                                        [row["answer"]], self.max_length)
        item = {k: v[:self.max_length] for k, v in item.items()}
        if not (item["labels"][1:] != -100).any():
            raise ValueError(f"No assistant tokens after truncation: task index {index}")
        return item


class ConstructionCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, rows):
        # EOS may also be PAD: construct attention from lengths, not token identity.
        return {k: torch.nn.utils.rnn.pad_sequence([r[k] for r in rows], batch_first=True,
                                                  padding_value=value)
                for k, value in (("input_ids", self.pad_token_id), ("labels", -100), ("attention_mask", 0))}

import json

import torch
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from data import get_collators, get_data


CONFIG_DIR = "/home/gb10/open-unlearning-retain-dup-control/configs"
with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    cfg = compose(
        config_name="unlearn.yaml",
        overrides=[
            "experiment=unlearn/tofu/default.yaml",
            "model=Llama-2-7b-chat-hf",
            "forget_split=forget05",
            "retain_split=retain95",
            "collator=DuplicateRetainDataCollator",
            "model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-2-7b-chat-hf_full",
        ],
    )

tokenizer = AutoTokenizer.from_pretrained(
    cfg.model.tokenizer_args.pretrained_model_name_or_path
)
data = get_data(
    cfg.data,
    mode="unlearn",
    tokenizer=tokenizer,
    template_args=cfg.model.template_args,
)
collator = get_collators(cfg.collator, tokenizer=tokenizer)
torch.manual_seed(42)
batch = collator([data["train"][i] for i in range(8)])

forget_ids = batch["forget"]["input_ids"]
retain_ids = batch["retain"]["input_ids"]
result = {
    "forget_batch_size": int(forget_ids.shape[0]),
    "forget_unique_rows": len({tuple(row.tolist()) for row in forget_ids}),
    "retain_batch_size": int(retain_ids.shape[0]),
    "retain_unique_rows": len({tuple(row.tolist()) for row in retain_ids}),
    "retain_halves_equal": all(
        torch.equal(value[:4], value[4:])
        for value in batch["retain"].values()
        if torch.is_tensor(value) and value.shape[0] == 8
    ),
}
print(json.dumps(result, sort_keys=True))

assert result == {
    "forget_batch_size": 8,
    "forget_unique_rows": 8,
    "retain_batch_size": 8,
    "retain_unique_rows": 4,
    "retain_halves_equal": True,
}

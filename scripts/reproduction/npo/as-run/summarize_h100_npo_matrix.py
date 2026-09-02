import json
from pathlib import Path


root = Path("/home/ai/alpaca")
existing_l2_f05 = (
    "tofu_Llama-2-7b-chat-hf_forget05_NPO_2xH100_zero3_flash2_"
    "micro4_acc4_e10_20260827_eval_retry2"
)
cells = [
    (
        "Llama-2-7b-chat-hf",
        "forget01",
        "tofu_Llama-2-7b-chat-hf_forget01_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828",
    ),
    ("Llama-2-7b-chat-hf", "forget05", existing_l2_f05),
    (
        "Llama-2-7b-chat-hf",
        "forget10",
        "tofu_Llama-2-7b-chat-hf_forget10_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828",
    ),
    (
        "Llama-3.2-1B-Instruct",
        "forget01",
        "tofu_Llama-3.2-1B-Instruct_forget01_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828",
    ),
    (
        "Llama-3.2-1B-Instruct",
        "forget05",
        "tofu_Llama-3.2-1B-Instruct_forget05_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828",
    ),
    (
        "Llama-3.2-1B-Instruct",
        "forget10",
        "tofu_Llama-3.2-1B-Instruct_forget10_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828",
    ),
]

results = []
for model, forget_split, status_stem in cells:
    status_path = root / f"{status_stem}.status.json"
    with status_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    results.append(
        {
            "model": model,
            "forget_split": forget_split,
            "status_path": str(status_path),
            **payload,
        }
    )

status = "DONE" if all(item.get("status") == "DONE" for item in results) else "INCOMPLETE"
print(json.dumps({"status": status, "cell_count": len(results), "results": results}, indent=2))

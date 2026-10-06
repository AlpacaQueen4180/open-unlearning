"""python -m construction.cli --help; all raw data belongs in private artifact storage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from construction.profile import accumulation, world_size

from construction.protocol import (ARCHIVE_SHA256, audit_sources, digest_json, extract_anchor,
                                   read_anchor, read_rows, sha256, target_gate, freeze_manifest,
                                   validate_manifest, write_json)


def load_recipe(method):
    root = Path(__file__).resolve().parents[2] / "configs" / "construction"
    recipe = json.loads((root / f"{method}.json").read_text())
    if "extends" in recipe:
        base = json.loads((root / recipe.pop("extends")).read_text())
        base.update(recipe)
        recipe = base
    return recipe


def prepare(args):
    from huggingface_hub import HfApi
    from datasets import load_dataset
    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a new artifact directory; manifest already exists")
    artifact = extract_anchor(args.archive, root / "anchor.jsonl")
    api = HfApi()
    revision = api.model_info("meta-llama/Llama-3.1-8B-Instruct", revision=args.model_revision).sha
    tofu_revision = api.dataset_info("locuslab/TOFU", revision=args.tofu_revision).sha
    sources = json.loads(Path(args.sources).read_text())["sources"] if args.sources else []
    datasets = {}
    for split in ("full", "retain95"):
        rows = list(load_dataset("locuslab/TOFU", split, split="train", revision=tofu_revision))
        path = root / f"tofu_{split}.json"
        write_json(path, rows)
        datasets[split] = {"path": str(path), "sha256": sha256(path), "count": len(rows),
                           "records": [{"id": str(i), "row_sha256": digest_json(row)} for i, row in enumerate(rows)]}
    if not any(s["role"] == "tofu_full" for s in sources):
        sources.append({"role": "tofu_full", "source": "locuslab/TOFU:full", "revision": tofu_revision,
                        "path": datasets["full"]["path"], "sha256": datasets["full"]["sha256"]})
    audit = audit_sources(root / "anchor.jsonl", artifact["anchor_sha256"], sources)
    manifest = {"schema_version": 1, "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_id": "meta-llama/Llama-3.1-8B-Instruct", "model_revision": revision,
        "tokenizer_revision": revision, "tofu_revision": tofu_revision,
        "construction_seed": 0, "selection_source": "development",
        **artifact, "anchor_path": str(root / "anchor.jsonl"), "datasets": datasets,
        "audit": audit, "source_specs": sources, "target_registry": {}}
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "audit_status": audit["status"],
                      "missing_roles": audit["missing_roles"]}))


def run(args):
    manifest_path = Path(args.manifest)
    manifest = validate_manifest(json.loads(manifest_path.read_text()), full=args.mode == "full")
    if args.mode == "full" and args.microbatch != manifest["readiness"]["microbatch"]:
        raise ValueError("Full run must use the frozen common smoke microbatch")
    world = world_size()
    if args.mode == 'full' and manifest['readiness'].get('world_size', 2) != world:
        raise ValueError('Full run must use the frozen accepted world size')
    recipe = load_recipe(args.method)
    recipe["args"].update({"output_dir": str(Path(args.output).resolve()),
                           "per_device_train_batch_size": args.microbatch,
                           "gradient_accumulation_steps": accumulation(args.microbatch, world),
                           "seed": manifest["construction_seed"]})
    recipe["method_args"].update({"anchor_path": manifest["anchor_path"],
                                  "anchor_sha256": manifest["anchor_sha256"],
                                  "save_intermediate": args.mode == "full",
                                  "save_runtime": args.mode == "full" and not args.weights_only})
    if args.mode == "smoke":
        recipe["args"]["max_steps"] = 10
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "split": args.split, "recipe": recipe,
                          "manifest_sha256": sha256(manifest_path)}, indent=2))
        return
    if Path(args.output).exists():
        raise FileExistsError("Use a new output directory")
    read_anchor(manifest["anchor_path"], manifest["anchor_sha256"])
    data = manifest["datasets"][args.split]
    if sha256(data["path"]) != data["sha256"]:
        raise ValueError("TOFU export checksum mismatch")
    audit = audit_sources(manifest["anchor_path"], manifest["anchor_sha256"], manifest["source_specs"])
    if audit != manifest["audit"]:
        raise ValueError("Audit inputs changed; prepare a new manifest")
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from construction.data import ConstructionDataset, ConstructionCollator
    from trainer import load_trainer
    repo = Path(__file__).resolve().parents[2]
    template = OmegaConf.load(repo / "configs/model/Llama-3.1-8B-Instruct.yaml").template_args
    torch.manual_seed(manifest["construction_seed"])
    model = AutoModelForCausalLM.from_pretrained(manifest["model_id"], revision=manifest["model_revision"],
                                               torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_id"], revision=manifest["tokenizer_revision"])
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ConstructionDataset(read_rows(data["path"]), tokenizer, template)
    collator = ConstructionCollator(tokenizer)
    trainer, _ = load_trainer(OmegaConf.create(recipe), model, train_dataset=dataset,
                              processing_class=tokenizer, data_collator=collator, template_args=template)
    result = trainer.train()
    # Save a private numerical probe for a separate, fresh-process reload check.
    trainer.model.eval()
    probe = trainer._move(collator([dataset[0]]))
    with torch.no_grad():
        expected = trainer.forward_model(**probe).logits[0, -1].float().cpu()
    trainer.save_model()
    if trainer.backend.rank == 0:
        torch.save({"inputs": {k: v.cpu() for k, v in probe.items()}, "logits": expected},
                   Path(args.output) / "reload_probe.pt")
        record = {"target_id": Path(args.output).name, "method": args.method, "split": args.split,
                  "construction_seed": manifest["construction_seed"], "mode": args.mode,
                  "manifest_sha256": sha256(manifest_path), "metrics": result.metrics,
                  "gate": {"status": "insufficient_evidence"}, "reload": "pending"}
        write_json(Path(args.output) / "target.json", record)


def verify(args):
    import torch
    from transformers import AutoModelForCausalLM
    path = Path(args.checkpoint)
    probe = torch.load(path / "reload_probe.pt", weights_only=True, map_location="cpu")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                               attn_implementation="flash_attention_2").cuda().eval()
    with torch.no_grad():
        actual = model(**{k: v.cuda() for k, v in probe["inputs"].items()}).logits[0, -1].float().cpu()
    torch.testing.assert_close(actual, probe["logits"], atol=.05, rtol=.01)
    record = json.loads((path / "target.json").read_text())
    record["reload"] = {"status": "pass", "max_logit_error": float((actual - probe["logits"]).abs().max())}
    write_json(path / "target.json", record)
    print(json.dumps(record["reload"]))


def olmo_audit(args):
    from huggingface_hub import HfApi
    api = HfApi()
    models = [f"allenai/OLMo-2-1124-7B-{stage}" for stage in ("SFT", "DPO", "Instruct")]
    mixture = "allenai/tulu-3-sft-olmo-2-mixture"
    record = {"models": [{"model_id": name, "model_revision": api.model_info(name).sha,
                           "baseline_status": "not_evaluated"} for name in models],
              "dataset_id": mixture, "dataset_revision": api.dataset_info(mixture).sha,
              "membership": "unconfirmed_pending_preprocessing_and_downstream_audit"}
    for model in record["models"]:
        model["tokenizer_revision"] = model["model_revision"]
    if args.scan:
        from datasets import load_dataset
        dataset = load_dataset(mixture, revision=record["dataset_revision"], split="train", streaming=True)
        total, flan, hashes = 0, 0, set()
        for row in dataset:
            total += 1
            source = str(row.get("source", ""))
            if "flan" in source.casefold():
                flan += 1
                hashes.add(digest_json(row.get("messages", row)))
        record["counts"] = {"mixture": total, "flan": flan, "flan_unique_records": len(hashes)}
        if not flan:
            raise ValueError("No FLAN rows identified; inspect source schema before claiming membership")
    write_json(args.output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--archive", required=True)
    p.add_argument("--artifacts", required=True)
    p.add_argument("--sources")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--tofu-revision", default="main")
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--method", choices=["standard", "mixing", "spf"], required=True)
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--split", choices=["full", "retain95"], default="full")
    p.add_argument("--microbatch", choices=[1, 2, 4], type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--weights-only", action="store_true",
                   help="Keep HF checkpoints but omit ZeRO runtime state (construction resume is unsupported)")
    p = sub.add_parser("verify")
    p.add_argument("--checkpoint", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--evidence", required=True)
    p.add_argument("--target-id", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("olmo-audit")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--output", required=True)
    p = sub.add_parser("history-audit")
    p.add_argument("--repo", default=".")
    p.add_argument("--output", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--manifest", required=True)
    p.add_argument("--smoke-summary", required=True)
    p.add_argument("--backend-acceptance", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        if Path(args.output).exists():
            raise FileExistsError("Frozen manifest output already exists")
        result = freeze_manifest(json.loads(Path(args.manifest).read_text()),
                                 json.loads(Path(args.smoke_summary).read_text()),
                                 json.loads(Path(args.backend_acceptance).read_text()),
                                 prepared_hash=sha256(args.manifest), smoke_hash=sha256(args.smoke_summary),
                                 acceptance_hash=sha256(args.backend_acceptance))
        write_json(args.output, result)
    elif args.command == "history-audit":
        from construction.history import audit_history
        print(json.dumps(audit_history(args.repo, args.output)))
    elif args.command == "gate":
        write_json(args.output, target_gate(json.loads(Path(args.evidence).read_text()), expected_target=args.target_id))
    else:
        {"prepare": prepare, "run": run, "verify": verify, "olmo-audit": olmo_audit}[args.command](args)


if __name__ == "__main__":
    main()

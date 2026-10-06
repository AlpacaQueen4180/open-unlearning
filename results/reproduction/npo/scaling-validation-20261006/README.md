# October 6 environment and gradient validation evidence

The finite experiment queue is DONE. This archive preserves original numerical
reports, commands, logs, package manifests, evaluation samples and checkpoint
proofs. Full weights, frozen TOFU tensors and raw gradient vectors stay on the
experiment hosts and are not published here.

- `h100/`: eight legacy two-rank controls, sixteen rank JSONs, commands/logs,
  executed scripts, remote SHA inventory and independent audit. Real 7B uses
  synthetic tokens; it is not a replay of the historical TOFU trajectory.
- `gb200/attempt0/`: twenty tiny old/new controls, installation/freeze/pin,
  frozen-batch metadata, and the original failed real-1B autograd comparison.
- `gb200/attempt1/`: sixteen real-1B synthetic/TOFU same-DS-reference controls,
  raw autograd comparisons, original native f01 evaluation failure and audit.
- `gb200/attempt2/`: launch and scheduled snapshots for evaluation recovery
  followed by the missing f05/f10 runs.
- `gb200/native-forget01/`, `native-forget05/`, `native-forget10/`: original
  evaluations, fixed retain references, runtime/Trainer states, checkpoint
  completeness/hash proofs and independent metric audits. F01 resumes evaluation
  from its original checkpoint; no retraining occurred.
- `gb200/native-forget10/queue-evidence/`: final DONE queue state and all four
  stage commands/logs, including f01 recovery and both missing full runs.
- `gb200/as-run/`: original v1, v2 and r2 helpers plus actual upstream runtime
  source, including the native probe assertions and evaluator cast patch.
- `final-audit-summary.json`: independent counts/metrics, old/update-only/new
  comparisons, integrity checks and explicit limitations.

`download-verification*.json` records SHA256 of the original bytes. The old
Windows paths under `attempt1/experiments/runs/...` are preserved locally;
publication uses byte-identical short paths under `attempt1/f01-failure/`.
`publication-path-map.json` maps each original path to its published location.
The 17 MB f10 tokenizer copy is locally verified but omitted from Git; its SHA
and reason are retained in `publication-omissions.json`. It is not needed to
recompute scores or verify checkpoint structure.

Run the offline verification from the repository root:

```sh
python scripts/reproduction/npo/gb200/audit_validation_archive.py
```

The auditor verifies 297 published manifest entries and recomputes native FQ
using exact two-sided KS integer path counts, MU using the harmonic mean and TR
from the raw samples. The small residuals against the saved summaries are
rounding-level. Native counters are Trainer=DS 20/70/130, consumed microbatches
100/500/1000, actual epoch10. New `engine_micro_steps` counts boundary calls,
not consumed microbatches.

Same-DS explicit-mean comparisons isolate accumulation scaling: old full/tail
8/2, new 1/1. The sixteen real-1B bf16 **autograd reference gates still fail**;
their differences remain unexplained and are not erased by passing the matched
DS controls. Gradients were measured preclip with SGD lr0, not as Adam update
ratios. The new stack retains container Torch/bitsandbytes and adds two output
float casts to the pinned upstream evaluator. Three single-seed native results
do not reproduce every `docs/repro.md` score and do not relabel historical
H100/Ada/safety or update-only corrected runs.

See the [final report](../../../../docs/reproduction/npo/environment-gradient-validation-20261006.md).

# GB200 native 8B / 7B finite validation evidence

Both public TOFU full models completed forget01 seed0 four-update smoke,
not a ten-epoch reproduction matrix or safety/Judge evaluation.
See [the report](../../../../docs/reproduction/npo/large-model-native-validation-20261006.md).

- `attempt0/`: original failed metadata serialization, source and 13 verified raw entries.
- `llama31-8b/`: 33 verified raw entries, streamed checkpoint proof and independent audit.
- `llama2-7b/`: 44 verified raw entries, final queue evidence, recovery preflight,
  actual gradient helper source, checkpoint proof and independent audit.
- `final-audit-summary.json`: finite queue and both local audits.
- `publication-manifest.json`: hashes of all published evidence bytes.
- `raw/`: local-only transport archives, excluded from Git.

All 90 raw manifest entries are SHA256-verified; repeated shared sources are
counted per manifest. `.gitattributes` preserves downloaded bytes, including logs.
Weights, frozen tokens and raw gradient vectors stay on the PVC.
Native scalar divisors 8/2 and Trainer=DS4 / micro20 / epoch2 passed for both models.
Four same-DS explicit-mean gradient gates passed; all four original autograd
comparisons failed their unchanged 5% gates. Failed comparisons are preserved.

Offline verification from the repository root:

```powershell
python scripts/reproduction/npo/gb200/audit_large_model_archive.py --root results/reproduction/npo/large-model-validation-20261006
```

After staging, append `--verify-git` to verify Git index bytes against every
publication SHA. The auditor accepts these declared smoke runs; the full 1B
auditor remains separate. Remote CPU proof checks every shard header, contiguous
offsets, shapes, required layers and file SHA; weights were not downloaded.

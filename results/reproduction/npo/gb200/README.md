# October GB200 NPO curated evidence

Reader reports: [comparison](../../../../docs/reproduction/npo/reports/GB200-NPO-Comparison-2026-10-06.md), [original matrix](../../../../docs/reproduction/npo/reports/GB200-NPO-Matrix-2026-10-05.md), [follow-up](../../../../docs/reproduction/npo/reports/GB200-NPO-Followup-2026-10-05.md), [correction inventory](../../../../docs/reproduction/npo/update-count-mismatch-20261006.md).

- `independent-audit-20261005.json`: ZeRO-3 / nonDS Llama-2 forget05 seed0 controls.
- `matrix-status-20261005.json`: six original seed0 cells, including actual runtime counts.
- `followup-independent-audit-20261005.json`: ten new full runs plus six original baseline cells; corrected flags, scores, count assertions and checkpoint shard inventories.
- `forget01-seed-statistics-20261005.json`: original seeds0/1/2 only, sample std uses n−1.
- `followup-completion-20261005.json`: archive size/hash and completion snapshot; `finished_utc` contains an explicit +08:00 offset despite its historical field name.
- Markdown copies in this directory preserve completion-time reports; reader copies under docs add publication context.

Raw evidence directories/ZIPs are intentionally ignored. `evidence-followup-20261005.zip` contains 419 files, 32,548,127 bytes, SHA256 `6c3b5119929e89669c22de0791c0b838a2cdccaed793c87d479464b7c554e9ae`. Per-file hashes were verified after download. Model weights remain on the RunAI PVC; existence and byte-size checks are not full weight checksums. No raw HEx-PHI responses or prompts are included here.

# Result and artifact manifests

The JSON files in `h100/` and `ada6000/` are compact aggregate artifacts
produced by the five-seed summarizers. Absolute paths in these JSON files are
preserved as historical provenance; they refer to files that remain on the
original experiment machines.

The TSV files in `inventories/` contain two columns:

1. artifact path relative to `saves/unlearn/` or `saves/eval/`;
2. file size in bytes.

Inventory captured on 2026-09-02:

| Host | Checkpoint shards | TOFU summaries | Training-output bytes | Evaluation-output bytes |
|---|---:|---:|---:|---:|
| 2×H100 | 12 | 54 | 47,908,707,384 | 79,004,909 |
| 1×Ada6000 | 64 | 57 | 255,894,122,994 | 83,020,733 |

These manifests are not backups of the weights. Do not delete machine-local
artifacts until they have been copied to durable storage and verified with
cryptographic checksums.

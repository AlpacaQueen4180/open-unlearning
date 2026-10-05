# GB200 NPO execution archive

These helpers were used for the October 4–5 experiment on RunAI. They retain host-specific RunAI/PVC defaults and the pinned source commit `820102411091abba2c5203207391bc1be0f3a863`.

- `run_experiment.py`: original runner; `--corrected` explicitly selects the opt-in wrapper. `run_matrix.py` runs six original seed0 cells; `run_followup.py` gates six corrected cells and four original extra-seed runs on a real smoke.
- `run_probe.py`: records actual Trainer, DeepSpeed, microsteps and corrected boundaries. `corrected_probe.py`: `ceil_epoch_and_sync_ds_boundary_v1`; source Trainer/objective and package versions remain unchanged.
- `audit_cell.py`, `audit_export.py`: independent metric/count audits and evidence export; `test_step_correction.py`: CPU plan/boundary-order checks.
- `bootstrap.sh`: isolated environment using the container Torch; `upload.ps1`: copy helpers to the pinned remote checkout with SHA256 verification. Bootstrap alone does not put these newly archived helpers into the historical checkout; run upload or copy this directory before invoking them.
- `status.ps1`, `collect_status.py`: one bounded snapshot. Long jobs were checked by a 15-minute scheduled heartbeat, paused on completion.
- `two_gpu_job.sh`: archived submission helper only; the two-GPU workload never executed because project quota was one GPU. `compare_checkpoints.py`: diagnostic helper, not proof of exact weight equivalence.

No training is started by this documentation publication. Reports and classifications: [comparison](../../../../docs/reproduction/npo/reports/GB200-NPO-Comparison-2026-10-06.md), [update-count issue](../../../../docs/reproduction/npo/update-count-mismatch-20261006.md).

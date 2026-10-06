"""Serial 10-update construction smoke; all methods share the chosen microbatch.

Stops on non-OOM errors. Only CUDA OOM triggers 4x4 -> 2x8 -> 1x16 retries.
Each attempted method gets a new directory. Full training is never launched here.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from construction.protocol import sha256, write_json
from construction.profile import accumulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument('--world-size', type=int, choices=[1, 2], default=1)
    parser.add_argument("--discard-verified-weights", action="store_true",
                        help="Remove only this suite's generated weights after successful reload verification")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "manifest_sha256": sha256(args.manifest), "attempts": [], 'world_size': args.world_size}
    # Advisory lock shared by THIS suite; also require idle devices at startup.
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if processes.strip():
        raise RuntimeError("GPUs are in use; refusing to start smoke")
    for micro in (4, 2, 1):
        oom = False
        for method in ("standard", "mixing", "spf"):
            if shutil.disk_usage(root).free < 35 * 1024**3:
                state["status"] = "insufficient_disk"
                write_json(root / "summary.json", state)
                raise RuntimeError("Need 35 GiB free for checkpoint serialization and reload")
            name = f"{method}-micro{micro}"
            output, log = root / name, root / f"{name}.log"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.world_size}",
                       "-m", "construction.cli", "run", "--manifest", str(Path(args.manifest).resolve()),
                       "--method", method, "--mode", "smoke", "--microbatch", str(micro), "--output", str(output)]
            with log.open("w") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            entry = {"method": method, "microbatch": micro, "accumulation": accumulation(micro, args.world_size),
                     "exit_code": result.returncode, "log": log.name}
            state["attempts"].append(entry)
            if result.returncode:
                text = log.read_text(errors="replace").lower()
                oom = "cuda out of memory" in text or "torch.outofmemoryerror" in text
                entry["status"] = "oom" if oom else "failed"
                state["status"] = "retrying" if oom else "failed"
                write_json(root / "summary.json", state)
                if not oom:
                    raise RuntimeError(f"{name} failed; see {log}")
                break
            with (root / f"{name}.reload.log").open("w") as handle:
                reload = subprocess.run([sys.executable, "-m", "construction.cli", "verify", "--checkpoint", str(output)],
                                        stdout=handle, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
            entry["reload_exit_code"] = reload.returncode
            if reload.returncode:
                state["status"] = "reload_failed"
                write_json(root / "summary.json", state)
                raise RuntimeError(f"Reload failed: {name}")
            entry["status"] = "pass"
            entry["target"] = json.loads((output / "target.json").read_text())
            trajectories = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
            updates = [row for row in trajectories if row["step"] > 0]
            entry["conflict_rate"] = sum(row["conflict"] for row in updates) / len(updates)
            entry["rank_peak_allocated_bytes"] = [max((row['rank_resources'][rank] if args.world_size > 1 else row)['cuda_peak_allocated_bytes'] for row in trajectories)
                                                   for rank in range(args.world_size)]
            if args.discard_verified_weights:
                # No recursive delete: only known output files inside this newly
                # created suite directory, after numerical reload verification.
                for path in output.glob("*.safetensors"):
                    if path.resolve().parent != output.resolve() or not path.resolve().is_relative_to(root):
                        raise RuntimeError("Unsafe checkpoint path")
                    path.unlink()
                entry["weights_retained"] = False
            write_json(root / "summary.json", state)
        if not oom:
            state.update({"status": "pass", "chosen_microbatch": micro, "chosen_accumulation": accumulation(micro, args.world_size)})
            write_json(root / "summary.json", state)
            return
    state["status"] = "oom_all_profiles"
    write_json(root / "summary.json", state)
    raise RuntimeError("All prescribed profiles exhausted; do not start full construction")


if __name__ == "__main__":
    main()

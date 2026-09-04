from __future__ import annotations

import sys

from .generate import main as generate_main
from .gpt_judge import main as judge_main
from .report import main as report_main


def main() -> None:
    usage = "usage: python -m evals.safety {generate,judge,summarize,calibrate} ..."
    if len(sys.argv) < 2:
        raise SystemExit(usage)
    if sys.argv[1] in {"-h", "--help"}:
        print(usage)
        return
    command = sys.argv.pop(1)
    if command == "generate":
        generate_main()
    elif command == "judge":
        judge_main()
    elif command in {"summarize", "calibrate"}:
        sys.argv.insert(1, command)
        report_main()
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()

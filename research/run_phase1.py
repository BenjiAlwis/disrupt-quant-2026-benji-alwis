"""Run the complete Phase 1 research suite and tee the console output to a log.

    python research/run_phase1.py            # development only (default)
    python research/run_phase1.py --all      # include the validation integrity audit

Validation is deliberately NOT used for signal discovery here. Only the
integrity audit touches it, because verifying that the evaluation panel obeys
the same invariants is a correctness check, not a research decision.
"""
from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import research.phase1_00_integrity as integrity      # noqa: E402
import research.phase1_01_structure as structure      # noqa: E402
import research.phase1_02_signal_screen as screen     # noqa: E402
import research.phase1_03_diagnostics as diagnostics  # noqa: E402


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="also audit the validation split")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples")
    args = ap.parse_args()

    log_dir = ROOT / "artifacts" / "phase1"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase1_log.txt"

    with log_path.open("w") as handle, redirect_stdout(Tee(sys.stdout, handle)):
        print("Disrupt Quant 2026 - Phase 1 research suite")
        print(f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
        print(f"numpy/pandas/scipy pinned per requirements.txt\n")

        for title, fn in (
            ("1.0  DATA INTEGRITY AND LOOK-AHEAD AUDIT", lambda: integrity.audit("development")),
            ("1.1  MARKET STRUCTURE", lambda: structure.run("development")),
            ("1.2  SIGNAL SCREEN", lambda: screen.run("development", n_boot=args.bootstrap)),
            ("1.3  DIAGNOSTICS AND COST BRIDGE", lambda: diagnostics.run("development", n_boot=args.bootstrap)),
        ):
            print("\n" + "=" * 78)
            print(f"  {title}")
            print("=" * 78)
            fn()

        if args.all:
            print("\n" + "=" * 78)
            print("  1.0b  VALIDATION PANEL INTEGRITY (audit only, no signal research)")
            print("=" * 78)
            integrity.audit("validation")

        print(f"\nArtifacts written to {log_dir.relative_to(ROOT)}/")

    print(f"\nLog: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

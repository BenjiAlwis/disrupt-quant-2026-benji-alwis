"""Phase 2.1 - design check on the CALIBRATION WINDOW ONLY (2021-07 .. 2022-12).

Three questions, each answered once and logged:
  a) does the carry band reduce cost without hurting the core?
  b) does the overlay add value on top of the core?
  c) what volatility does carry_gross = 1.2 actually produce?
2021 H1 is used only as initial history for the reversal estimate.
"""
from dataclasses import replace
from harness import run, fmt, Config

base = Config()
blocks = [("2021-07-01", "2021-12-31"), ("2022-01-01", "2022-06-30"), ("2022-07-01", "2022-12-30")]
variants = {
    "carry only, no band":        replace(base, carry_band=0.0,  rev_kappa=1e9),
    "carry only, band 0.25":      replace(base, carry_band=0.25, rev_kappa=1e9),
    "carry + overlay (band 0.25)": base,
}
for s, e in blocks:
    print(f"\n=== block {s} -> {e} ===")
    for name, cfg in variants.items():
        m, *_ = run(cfg, s, e, label=name)
        print(fmt(m))

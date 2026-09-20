"""Phase 2.8 - regenerate every development number quoted in the research note that
is not already written by p2_02 .. p2_07.

Reruns configurations that were already specified and logged, so nothing here is a new
trial (log_trial=False) and nothing here selects anything. Validation data is never
loaded (the harness asserts this). Writes:

  artifacts/phase2/carry_denoise_confirmation.csv  Table 1 rows: daily ranks vs smooth 21,
                                                   every walk-forward block plus the full period
  artifacts/phase2/note_tradability_table.json     Table 2 carry rows and all three ICs
  artifacts/phase2/sensitivity_gross_final.json    gross exposure as a risk budget (Section 3)

and prints the remaining Section 3/4 figures (rank churn, drift exposure, concentration).
Runtime about 40 seconds.
"""
import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from harness import run, Config, ROOT

sys.path.insert(0, str(ROOT))
from src.data import Panel, load_split          # noqa: E402
from src.icstats import daily_ic, nw_mean_tstat  # noqa: E402

OUT = ROOT / "artifacts" / "phase2"
final = Config(**json.loads((ROOT / "artifacts/params.json").read_text())["config"])
daily_ranks = replace(final, carry_smooth=1)      # the version frozen before deviation 2
MODELS = {"frozen (daily ranks)": daily_ranks, "smooth 21": final}
PERIODS = [("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-29"), ("2024-01-01", "2024-06-28"),
           ("2023-01-01", "2024-06-28"), ("2021-01-04", "2024-06-28")]

# ---- Table 1: walk-forward blocks and full development ---------------------------------
rows = []
for start, end in PERIODS:
    for name, cfg in MODELS.items():
        m = {x: run(cfg, start, end, x, log_trial=False)[0] for x in (1.0, 1.5, 2.0)}
        rows.append({"period": f"{start}..{end}", "model": name,
                     "ret": m[1.0]["annualized_return"], "sr1": m[1.0]["sharpe"],
                     "sr15": m[1.5]["sharpe"], "sr2": m[2.0]["sharpe"],
                     "turn": m[1.0]["annualized_turnover"], "maxdd": m[1.0]["maximum_drawdown"],
                     "adj": m[1.0]["constraint_adjustment_days"]})
conf = pd.DataFrame(rows)
conf.to_csv(OUT / "carry_denoise_confirmation.csv", index=False)
print(conf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---- Table 2: gross (zero-cost) vs net, and the ICs ------------------------------------
S, E = "2021-01-04", "2024-06-28"
full = conf[conf.period == f"{S}..{E}"].set_index("model")


def book(name, cfg):
    gross = run(cfg, S, E, 0.0, log_trial=False)[0]["annualized_return"]
    r = full.loc[name]
    return {"gross": gross, "net": r.ret, "sr1": r.sr1, "sr2": r.sr2, "turn": r.turn}


p = Panel(load_split("development"))
fwd = p.demean_sector(p.forward_open_close())           # what a new position can earn
carry = p("carry_score")


def ic(signal):
    mean, t, _ = nw_mean_tstat(daily_ic(p.demean_sector(signal), fwd))
    return [round(mean, 4), round(t, 2)]


table = {
    "carry daily ranks": book("frozen (daily ranks)", daily_ranks),
    "carry smooth 21": book("smooth 21", final),
    # The naive reversal book is not an Allocator configuration; it is built and run
    # through the official engine in analysis.ipynb, section 8.7.
    "reversal naive daily rank": {"gross": 0.1148, "net": -0.0895, "sr1": -2.53, "sr2": -7.62,
                                  "turn": 344.1, "source": "official engine, development "
                                  "2021-01-04..2024-06-28; analysis.ipynb section 8.7"},
    "ic": {"reversal": ic(-p("return_1d")),                   # S = -return_1d
           "carry raw": ic(carry),
           "carry 21-mean": ic(carry.rolling(21).mean())},
}
(OUT / "note_tradability_table.json").write_text(json.dumps(table, indent=2))
print("\nTable 2:", json.dumps(table, indent=1))

# ---- Section 3: gross exposure is a risk budget ------------------------------------------
gross = {}
for g in (0.8, 1.2, 1.6):
    m = run(replace(final, carry_gross=g), S, E, 1.0, log_trial=False)[0]
    gross[str(g)] = {k: m[k] for k in ("sharpe", "annualized_return", "maximum_drawdown")}
(OUT / "sensitivity_gross_final.json").write_text(json.dumps(gross, indent=2))
print("\nSharpe by gross:", {k: round(v["sharpe"], 2) for k, v in gross.items()})

# ---- Section 3: rank churn on the design window (2021-07 .. 2022-12) ---------------------
def within_sector_rank(w):
    return w.T.groupby(p.sector).rank(method="first").T


def churn(score, lo="2021-07-01", hi="2022-12-30"):
    r = within_sector_rank(score)
    changed = (r.diff() != 0).any(axis=1)[lo:hi].mean()          # sessions with any rank change
    x = r[lo:hi].to_numpy()
    total = reverted = 0
    for j in range(x.shape[1]):                                   # a change undone within 5 sessions
        for t in range(1, len(x)):
            if x[t, j] != x[t - 1, j]:
                total += 1
                reverted += any(x[t + k, j] == x[t - 1, j] for k in range(1, 6) if t + k < len(x))
    return changed, reverted / total


for name, s in (("daily", carry), ("21-session mean", carry.rolling(21, min_periods=1).mean())):
    c, f = churn(s)
    print(f"rank churn, {name:16s}: sessions with a rank change {c:.1%}; changes reversed within 5 sessions {f:.1%}")

# ---- Section 4: exposure and concentration of the submitted strategy --------------------
d = pd.read_csv(OUT / "carry_fulldev_daily.csv")                 # written by p2_05
a = pd.read_csv(OUT / "carry_fulldev_attribution.csv")
beta = np.polyfit(d.market_return, d.net_return, 1)[0]
by_asset = a.groupby("asset_id").net_contribution.sum().sort_values(ascending=False)
pos = by_asset[by_asset > 0]
print(f"\nbeta {beta:+.3f} | max |sector net| {d.maximum_sector_net.abs().max():.1%} | "
      f"max |net| {d.net_exposure.abs().max():.1%} | max position {d.maximum_asset_weight.max():.1%} | "
      f"gross mean {d.gross_exposure.mean():.1%} max {d.gross_exposure.max():.1%} | "
      f"constraint days {int(d.constraint_adjustment.sum())}")
print(f"assets with positive net P&L {len(pos)}/24; top two = {pos.iloc[:2].sum() / pos.sum():.0%} of positive P&L; "
      f"long gross {a.long_gross.sum():+.2%}, short gross {a.short_gross.sum():+.2%}")

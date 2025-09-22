#!/usr/bin/env python3
"""
Bias-safe L/S backtester with turnover costs & lagged risk scaling.

What’s in:
- Non-overlapping 21d rebalances with 1d execution lag
- True turnover-based costs at rebalance: sum(|w_new - w_prev|)
- Softmax weighting on top/bottom quantiles
- Dispersion gate: trade only when score dispersion is high
- Lagged volatility scaling (vol_20_lag)
- Per-name weight cap; normalize to gross 1.0
- Strict delist handling: require effective_ret (no NaN→0)

Inputs
- --scores CSV  : columns [date, Ticker, pred]
- --features CSV: must include [date, Ticker, effective_ret], optional [vol_20]
"""

import os
import argparse
import numpy as np
import pandas as pd

SCORES = "data/universe/scores.csv"
FEATS  = "data/universe/features.csv"

# -----------------------
# Utils
# -----------------------
def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (x - np.nanmax(x)) / max(1e-6, float(temp))
    z = np.clip(z, -50.0, 50.0)
    ex = np.exp(z)
    s = ex.sum()
    return ex / s if s > 0 else np.full_like(x, 1.0 / len(x))

def summarize(pnl_df: pd.DataFrame) -> dict:
    r = pnl_df["pnl"].astype(float)
    mu, sd = r.mean(), r.std()
    sharpe = (mu / sd) * np.sqrt(252) if sd > 0 else np.nan
    curve = (1.0 + r).cumprod()
    peak  = curve.cummax()
    maxdd = float((curve / peak - 1.0).min()) if not curve.empty else np.nan
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    cagr  = float(curve.iloc[-1] ** (1.0 / years) - 1.0) if not curve.empty else np.nan
    return {
        "Sharpe": float(sharpe),
        "MaxDD": maxdd,
        "CAGR": cagr,
        "HitRate": float((r > 0).mean()),
        "TotalPnL": float(curve.iloc[-1] - 1.0) if not curve.empty else np.nan,
    }

# -----------------------
# Backtest (bias-safe)
# -----------------------
def backtest(
    scores: pd.DataFrame,
    feats: pd.DataFrame,
    *,
    horizon: int = 21,
    top_pct: float = 0.06,
    cost_bps: float = 25.0,
    execution_lag_days: int = 1,
    min_names: int = 20,
    clip_ret: float = 0.03,
    soft_weighting: bool = True,
    softmax_temp: float = 0.8,
    invvol_col: str = "vol_20_lag",
    name_vol_cap: float = 0.02,
    dispersion_gate: float = 1.2,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    # Dates
    scores = scores.copy()
    feats  = feats.copy()
    scores["date"] = pd.to_datetime(scores["date"])
    feats["date"]  = pd.to_datetime(feats["date"])

    # Enforce effective_ret (no silent delist masking)
    if "effective_ret" not in feats.columns:
        raise ValueError("features.csv must include 'effective_ret'.")

    # Safe lagged vol (avoid forward-look)
    if invvol_col not in feats.columns:
        if "vol_20" in feats.columns:
            feats[invvol_col] = (
                feats.sort_values(["Ticker", "date"])
                     .groupby("Ticker")["vol_20"].shift(1)
            )
        else:
            feats["_tmp_vol20"] = (
                feats.sort_values(["Ticker", "date"])
                     .groupby("Ticker")["effective_ret"]
                     .rolling(20).std().reset_index(level=0, drop=True)
            )
            feats[invvol_col] = (
                feats.sort_values(["Ticker", "date"])
                     .groupby("Ticker")["_tmp_vol20"].shift(1)
            )
            feats.drop(columns=["_tmp_vol20"], inplace=True)

    # Calendar
    cal = pd.Index(
        pd.to_datetime(np.intersect1d(scores["date"].unique(), feats["date"].unique()))
    ).sort_values()

    preds_by_date = {pd.to_datetime(d): g.set_index("Ticker")["pred"] for d, g in scores.groupby("date")}
    feat_by_date  = {pd.to_datetime(d): g.set_index("Ticker") for d, g in feats.groupby("date")}

    H = max(1, int(horizon))
    pnl_records: list[dict] = []
    disp_hist: list[float] = []
    w_prev = pd.Series(dtype=float)
    weights_out: list[pd.DataFrame] = []

    for i0 in range(0, len(cal) - 1, H):
        d0 = pd.to_datetime(cal[i0])
        preds = preds_by_date.get(d0)
        f0    = feat_by_date.get(d0)
        if preds is None or f0 is None or preds.size < max(10, min_names):
            continue

        # Dispersion gate
        disp = float(preds.std())
        if len(disp_hist) < 60:
            med = disp if disp > 0 else 1.0
            take_block = True
        else:
            med = np.median(disp_hist)
            take_block = (disp / med) >= float(dispersion_gate)
        disp_hist.append(disp)
        if not take_block:
            continue

        # Rank
        k = max(1, int(preds.size * float(top_pct)))
        ranked = preds.sort_values()
        longs  = ranked.index[-k:]
        shorts = ranked.index[:k]

        if soft_weighting:
            lw = _softmax(preds.loc[longs].values, temp=softmax_temp) * 0.5
            sw = _softmax(-preds.loc[shorts].values, temp=softmax_temp) * 0.5
            w  = pd.Series({t: lw[i] for i, t in enumerate(longs)}, dtype=float)
            w.update({t: -sw[i] for i, t in enumerate(shorts)})
        else:
            w  = pd.Series({t:  0.5 / k for t in longs}, dtype=float)
            w.update({t: -0.5 / k for t in shorts})

        # Lagged inv-vol
        inv = f0.get(invvol_col, pd.Series(index=w.index, dtype=float)).reindex(w.index)
        if inv.notna().any():
            inv = 1.0 / inv.replace(0.0, np.nan)
            inv = inv.fillna(inv.median()).abs()
            if inv.sum() > 0:
                inv = inv / inv.sum()
                w   = w * (inv * w.abs().sum())

        if name_vol_cap and name_vol_cap > 0:
            w = w.clip(-float(name_vol_cap), float(name_vol_cap))
        gross = w.abs().sum()
        if gross > 0:
            w = w / gross

        # Save weights snapshot
        weights_out.append(pd.DataFrame({"date": d0, "Ticker": w.index, "weight": w.values}))

        # Block apply
        i_start = i0 + max(1, int(execution_lag_days))
        i_end   = min(len(cal) - 1, i_start + H - 1)

        for i_day in range(i_start, i_end + 1):
            d = pd.to_datetime(cal[i_day])
            f = feat_by_date.get(d)
            if f is None:
                continue

            r = f["effective_ret"].reindex(w.index)
            if r.isna().any():
                r = r.dropna()
                if r.empty:
                    continue
                w = w.reindex(r.index).fillna(0.0)

            if clip_ret and clip_ret > 0:
                r = r.clip(lower=-float(clip_ret), upper=float(clip_ret))

            pnl_val = float(np.dot(w.values, r.values))

            if i_day == i_start:
                prev = w_prev.reindex(w.index).fillna(0.0) if not w_prev.empty else pd.Series(0.0, index=w.index)
                turnover = float((w - prev).abs().sum())
                if cost_bps and cost_bps > 0 and turnover > 0:
                    pnl_val -= turnover * (float(cost_bps) / 10000.0)
                w_prev = w.copy()

            pnl_records.append({"date": d, "pnl": pnl_val})

    if not pnl_records:
        raise RuntimeError("No PnL generated.")

    pnl_df = pd.DataFrame(pnl_records).set_index("date").sort_index()
    weights_df = pd.concat(weights_out).reset_index(drop=True) if weights_out else pd.DataFrame()
    return pnl_df, summarize(pnl_df), weights_df

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Bias-safe L/S backtester")
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--features", default=FEATS)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--top_pct", type=float, default=0.06)
    ap.add_argument("--cost_bps", type=float, default=25.0)
    ap.add_argument("--execution_lag_days", type=int, default=1)
    ap.add_argument("--min_names", type=int, default=20)
    ap.add_argument("--clip_ret", type=float, default=0.03)
    ap.add_argument("--soft_weighting", action="store_true", default=True)
    ap.add_argument("--softmax_temp", type=float, default=0.8)
    ap.add_argument("--invvol_col", type=str, default="vol_20_lag")
    ap.add_argument("--name_vol_cap", type=float, default=0.02)
    ap.add_argument("--dispersion_gate", type=float, default=1.2)
    args = ap.parse_args()

    if not os.path.exists(args.features): 
        raise FileNotFoundError(args.features)
    if not os.path.exists(args.scores): 
        raise FileNotFoundError(args.scores)

    feats  = pd.read_csv(args.features)
    scores = pd.read_csv(args.scores)

    # Remove keys that conflict with positional args
    kwargs = vars(args).copy()
    kwargs.pop("scores", None)
    kwargs.pop("features", None)

    pnl_df, stats, weights_df = backtest(scores, feats, **kwargs)

    print("\n=== Performance Summary ===")
    for k in ["Sharpe","MaxDD","CAGR","HitRate","TotalPnL"]:
        print(f"{k:8s}: {stats[k]:.4f}")

    os.makedirs("results", exist_ok=True)
    pnl_df.to_csv("results/equity_curve.csv")
    weights_df.to_csv("results/weights.csv", index=False)
    pd.DataFrame([stats]).to_csv("results/summary.csv", index=False)
    print("[OK] Saved → results/equity_curve.csv, results/weights.csv, results/summary.csv")

if __name__ == "__main__":
    main()

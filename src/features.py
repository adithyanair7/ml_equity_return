#!/usr/bin/env python3
"""
Feature builder (fast, robust, bias-aware).

Input (default): data/universe/nasdaq100_daily_panel.csv
Expected columns (flexible names handled): date, Ticker, price (adj_close/close/prc/px/price),
volume (vol), shares (shrout). If 'ret' or 'log_ret' exists, they'll be used; otherwise derived.

Outputs:
  - data/universe/features.csv
    Columns include:
      date, Ticker, ret, log_ret, target (fwd 21d log return),
      mom_21, mom_63, mom_126, mom_252,
      rev_2_5, vol_20, vol_60, idio_vol_20,
      illiq_20, turnover, mktcap, hi52w_ratio, beta
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

RAW  = "data/universe/nasdaq100_daily_panel.csv"
FEAT = "data/universe/features.csv"

# ---------- helpers ----------
def _pick(df: pd.DataFrame, names: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns: return n
        if n.lower() in low: return low[n.lower()]
    return None

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        cand = _pick(df, ["Date"])
        if cand is None: raise ValueError("Missing 'date' column.")
        df = df.rename(columns={cand: "date"})
    if "Ticker" not in df.columns:
        cand = _pick(df, ["ticker", "symbol"])
        if cand is None: raise ValueError("Missing 'Ticker' column.")
        df = df.rename(columns={cand: "Ticker"})

    pcol = _pick(df, ["adj_close","adjusted_close","adjclose","close","prc","px","price"])
    if pcol is not None and pcol != "price":
        df = df.rename(columns={pcol: "price"})
    vcol = _pick(df, ["vol","volume"])
    if vcol is not None and vcol != "vol":
        df = df.rename(columns={vcol: "vol"})
    scol = _pick(df, ["shrout","shares_out","shares"])
    if scol is not None and scol != "shrout":
        df = df.rename(columns={scol: "shrout"})
    return df

def _derive_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker","date"]).copy()
    have_log = _pick(df, ["log_ret","logreturn","log_returns","logret"])
    have_ret = _pick(df, ["ret","returns","daily_ret","r1"])
    eff_ret  = _pick(df, ["effective_ret","eff_ret","effective_return"])

    if have_log is None and have_ret is None:
        if "price" not in df.columns:
            raise ValueError("Need 'ret'/'log_ret' or a price column to derive returns.")
        r = df.groupby("Ticker")["price"].pct_change()
        df["ret"] = pd.to_numeric(r, errors="coerce").fillna(0.0).astype(float)
        df["log_ret"] = np.log1p(df["ret"])
    else:
        if have_ret is not None and "ret" not in df.columns:
            df["ret"] = pd.to_numeric(df[have_ret], errors="coerce").fillna(0.0).astype(float)
        if have_log is not None and "log_ret" not in df.columns:
            df["log_ret"] = pd.to_numeric(df[have_log], errors="coerce").fillna(0.0).astype(float)
        if "log_ret" not in df.columns:
            df["log_ret"] = np.log1p(df["ret"])
        if "ret" not in df.columns:
            df["ret"] = np.expm1(df["log_ret"])

    if eff_ret is not None and "effective_ret" not in df.columns:
        df["effective_ret"] = pd.to_numeric(df[eff_ret], errors="coerce").astype(float)
    return df

def _rolling_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=int(w*0.6)).std()

# ---------- features ----------
def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker","date"]).copy()

    # Momentum (use transform so index aligns and avoids MultiIndex)
    df["mom_21"]  = df.groupby("Ticker")["log_ret"].transform(lambda s: s.rolling(21).sum())
    df["mom_63"]  = df.groupby("Ticker")["log_ret"].transform(lambda s: s.rolling(63).sum())
    df["mom_126"] = df.groupby("Ticker")["log_ret"].transform(lambda s: s.rolling(126).sum())
    df["mom_252"] = df.groupby("Ticker")["log_ret"].transform(lambda s: s.rolling(252).sum())

    # Short-term reversal: sum of returns from t-2 to t-5 (inclusive)
    df["rev_2_5"] = df.groupby("Ticker")["log_ret"].transform(lambda s: s.shift(2).rolling(4).sum())

    # Volatilities
    df["vol_20"] = df.groupby("Ticker")["log_ret"].transform(lambda s: _rolling_std(s, 20))
    df["vol_60"] = df.groupby("Ticker")["log_ret"].transform(lambda s: _rolling_std(s, 60))

    # Market beta (252d) and idiosyncratic vol (20d) — compute on wide, then rejoin
    wide = df.pivot(index="date", columns="Ticker", values="log_ret").astype(float)
    wide = wide.fillna(0.0)  # safe default for cov/var windows
    mkt  = wide.mean(axis=1)

    # beta
    cov = wide.apply(lambda x: x.rolling(252, min_periods=60).cov(mkt))
    var = mkt.rolling(252, min_periods=60).var()
    beta = cov.div(var, axis=0).replace([np.inf,-np.inf], np.nan)
    beta_s = beta.stack().rename("beta").reset_index().rename(columns={"level_1":"Ticker"})
    df = df.merge(beta_s, on=["date","Ticker"], how="left")
    df["beta"] = df["beta"].fillna(1.0)

    # idiosyncratic vol: STD of residuals (r - avg)
    resid = wide.sub(mkt, axis=0)
    idio  = resid.rolling(20, min_periods=12).std()
    idio_s = idio.stack().rename("idio_vol_20").reset_index().rename(columns={"level_1":"Ticker"})
    df = df.merge(idio_s, on=["date","Ticker"], how="left")

    # Turnover, illiquidity, size
    if "vol" in df.columns and "price" in df.columns:
        df["turnover"] = (pd.to_numeric(df["vol"], errors="coerce") * pd.to_numeric(df["price"], errors="coerce")).replace([np.inf,-np.inf], np.nan)
        dv = df.groupby("Ticker")["turnover"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        df["illiq_20"] = (df["ret"].abs() / dv.replace(0, np.nan)).groupby(df["Ticker"]).transform(lambda s: s.rolling(20, min_periods=10).mean())
    else:
        df["turnover"] = np.nan
        df["illiq_20"] = np.nan

    if "shrout" in df.columns and "price" in df.columns:
        df["mktcap"] = (pd.to_numeric(df["shrout"], errors="coerce") * pd.to_numeric(df["price"], errors="coerce")).replace([np.inf,-np.inf], np.nan)

    # 52-week high proximity
    if "price" in df.columns:
        roll_hi = df.groupby("Ticker")["price"].transform(lambda s: s.rolling(252, min_periods=120).max())
        df["hi52w_ratio"] = (df["price"] / roll_hi).replace([np.inf,-np.inf], np.nan)

    # Forward 21d target
    df["target"] = df.groupby("Ticker")["log_ret"].transform(lambda s: s.rolling(21).sum()).shift(-21)

    # Final clean
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    return df

# ---------- main ----------
def main():
    if not os.path.exists(RAW):
        raise FileNotFoundError(f"Missing {RAW}. Put your raw panel there.")
    df = pd.read_csv(RAW, parse_dates=["date"])
    df = _ensure_cols(df)
    df = _derive_returns(df)

    feats = _build_features(df)

    os.makedirs(os.path.dirname(FEAT), exist_ok=True)
    feats.to_csv(FEAT, index=False)
    print(f"[OK] Saved features → {FEAT} with {len(feats):,} rows, {feats['Ticker'].nunique()} tickers")

if __name__ == "__main__":
    main()

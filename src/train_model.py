#!/usr/bin/env python3
"""
Pooled cross-sectional walk-forward trainer (RF / optional XGB).
- Per-date z-scoring with groupby.transform (works on older pandas)
- Rolling window train with constant-time step
- Robust NaN handling
- Progress logs each step
"""

from __future__ import annotations
import os, argparse, time, gc, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

# Optional XGBoost (if available)
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

FEATS = "data/universe/features.csv"
OUT   = "data/universe/scores.csv"

NON_FEATS = {
    "date","Ticker","permno","ret","effective_ret","log_ret","dlret","hexcd","dlstcd",
    "Start","End","px","price","vol","shrout","exchcd","shrcd","target","mktcap",
    "turnover","illiq_20","beta","idio_vol_20","hi52w_ratio","mom_21","mom_63","mom_126","mom_252","rev_2_5",
}

def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num if c not in NON_FEATS]
    if not feats:
        raise ValueError("No numeric feature columns found besides NON_FEATS; update NON_FEATS or add engineered features.")
    return sorted(feats)

def per_date_z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score each column within each date using groupby.transform (pandas-version safe)."""
    g = df.groupby("date", sort=True)
    means = g[cols].transform("mean")
    stds  = g[cols].transform("std").replace(0.0, np.nan)
    df_z = df.copy()
    df_z[cols] = (df[cols] - means) / stds
    return df_z

def drop_nonfinite(X: np.ndarray, y: np.ndarray | None = None):
    mask = np.isfinite(X).all(axis=1)
    if y is None:
        return mask, X[mask]
    mask = mask & np.isfinite(y)
    return mask, X[mask], y[mask]

def make_rf(n_estimators=400, min_samples_leaf=3, max_depth=None, random_state=42):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

def make_xgb(random_state=42):
    if not HAVE_XGB:
        return None
    return XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
        enable_categorical=False,
    )

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description="Cross-sectional walk-forward trainer")
    ap.add_argument("--features", default=FEATS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model", default="rf", choices=["rf","xgb","ensemble"])
    ap.add_argument("--min_train_days", type=int, default=252)
    ap.add_argument("--max_train_days", type=int, default=756)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--rf_n_estimators", type=int, default=400)
    ap.add_argument("--rf_min_samples_leaf", type=int, default=3)
    ap.add_argument("--rf_max_depth", type=int, default=None)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--zscore", action="store_true", help="Per-date z-score features before training")
    args = ap.parse_args()

    if not os.path.exists(args.features):
        raise FileNotFoundError(args.features)

    df = pd.read_csv(args.features, parse_dates=["date"]).sort_values(["date","Ticker"])
    if "target" not in df.columns:
        raise ValueError("features.csv must include a 'target' column.")

    feat_cols = infer_feature_cols(df)

    # Optional per-date z-scoring (recommended for tree ensembles too if features scale wildly)
    if args.zscore:
        df = per_date_z(df, feat_cols)

    # Ensure numeric & finite
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df["target"]  = pd.to_numeric(df["target"], errors="coerce")
    finite_mask = np.isfinite(df[feat_cols].to_numpy()).all(axis=1) & np.isfinite(df["target"].to_numpy())
    df = df[finite_mask].reset_index(drop=True)

    # Index by date for slicing
    df = df.set_index("date", drop=False)
    dates = np.sort(df["date"].unique())

    start_idx = args.min_train_days
    if start_idx >= len(dates) - 1:
        raise ValueError("Not enough dates to start training (increase data or lower --min_train_days).")

    preds_out = []
    steps = range(start_idx, len(dates)-1, args.step)

    rf_params = dict(
        n_estimators=args.rf_n_estimators,
        min_samples_leaf=args.rf_min_samples_leaf,
        max_depth=args.rf_max_depth,
        random_state=args.random_state,
    )

    for si, i in enumerate(tqdm(steps, desc="[Walk-forward]", unit="step")):
        t0 = time.time()
        train_end = dates[i]

        # Rolling window
        if args.max_train_days > 0:
            win_start_idx = max(0, i - args.max_train_days)
            train_start = dates[win_start_idx]
            tr = df.loc[train_start:train_end]
        else:
            tr = df.loc[:train_end]

        test_days = dates[i+1 : i+1+args.step]
        if len(test_days) == 0:
            break

        X_tr = tr[feat_cols].to_numpy()
        y_tr = tr["target"].to_numpy()
        m_tr, X_tr, y_tr = drop_nonfinite(X_tr, y_tr)
        if y_tr.size == 0 or np.unique(np.round(y_tr, 10)).size < 2:
            print(f"[WARN] step {si+1}: insufficient training labels -> skipped", flush=True)
            continue

        # Fit once per block
        if args.model == "rf":
            model = make_rf(**rf_params)
            model.fit(X_tr, y_tr)
            predict_fn = model.predict
        elif args.model == "xgb" and HAVE_XGB:
            model = make_xgb(random_state=args.random_state)
            model.fit(X_tr, y_tr)
            predict_fn = model.predict
        elif args.model == "ensemble":
            rf = make_rf(**rf_params)
            rf.fit(X_tr, y_tr)
            if HAVE_XGB:
                xgb = make_xgb(random_state=args.random_state); xgb.fit(X_tr, y_tr)
                def predict_fn(X_te): return 0.5 * (rf.predict(X_te) + xgb.predict(X_te))
            else:
                predict_fn = rf.predict
        else:
            raise ValueError(f"Unknown --model '{args.model}'")

        # Predict each test day
        for d in test_days:
            g = df.loc[df.index == d]
            if g.empty: continue
            X_te = g[feat_cols].to_numpy()
            m_te, X_te = drop_nonfinite(X_te)
            if X_te.size == 0: continue
            preds = predict_fn(X_te)
            out = g[["date","Ticker"]].reset_index(drop=True).iloc[m_te].copy()
            out["pred"] = preds
            preds_out.append(out)

        dt = time.time() - t0
        print(f"[INFO] step {si+1:3d}/{len(steps)}  train_end={pd.Timestamp(train_end).date()}  "
              f"train_rows={len(tr):,}  test_days={len(test_days):2d}  feats={len(feat_cols)}  took={dt:.1f}s",
              flush=True)

        if (si % 10) == 0:
            gc.collect()

    if not preds_out:
        raise RuntimeError("No predictions produced. Check features/params.")
    scores = pd.concat(preds_out).sort_values(["date","Ticker"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scores.to_csv(args.out, index=False)
    print(f"[OK] Saved scores → {args.out} with {len(scores):,} rows, {scores['Ticker'].nunique()} tickers")
    print(f"[INFO] Model={args.model.upper()}  Step={args.step}  MinTrainDays={args.min_train_days}  MaxTrainDays={args.max_train_days}  Zscore={args.zscore}")

if __name__ == "__main__":
    main()

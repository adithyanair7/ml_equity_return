# src/data_loader.py
"""
Build a survivorship-bias-free NASDAQ-100 daily panel.

Inputs (expected):
  data/universe/nasdaq_daily_2015_2024.csv     # CRSP daily (permno,date,ret,dlret,prc,vol,shrout,hexcd,dlstcd,...)
  data/universe/crsp_names.csv                 # From WRDS: stocknames (permno, namedt as namebegdt, nameenddt, ticker[, tsymbol], comnam, exchcd, shrcd)
  data/universe/constituents_history.csv       # Date,Ticker,Action (ADD/DROP). You seeded 2015 baseline.

Outputs:
  data/universe/nasdaq100_daily_panel.csv
  data/universe/validation_biasfree_summary.csv
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

# ---------- paths ----------
DATA_DIR    = Path("data/universe")
DAILY_FILE  = DATA_DIR / "nasdaq_daily_2015_2024.csv"
NAMES_FILE  = DATA_DIR / "crsp_names.csv"
HIST_FILE   = DATA_DIR / "constituents_history.csv"
OUT_PANEL   = DATA_DIR / "nasdaq100_daily_panel.csv"
OUT_SUMMARY = DATA_DIR / "validation_biasfree_summary.csv"

END_CAP = pd.Timestamp("2025-01-01")  # cap open membership windows

# ---------- helpers ----------
def _norm_sym_series(s: pd.Series) -> pd.Series:
    """Normalize tickers so CRSP names and membership symbols match."""
    return (s.astype(str)
             .str.upper()
             .str.strip()
             .str.replace(r"[.\-\s/=_]", "", regex=True))

def build_membership_intervals(hist: pd.DataFrame) -> pd.DataFrame:
    """Convert ADD/DROP events into [Start, End] windows keyed by normalized symbol."""
    m = hist.copy()
    need = {"Date","Ticker","Action"}
    if not need.issubset(m.columns):
        raise ValueError(f"constituents_history.csv must have columns: {need}")

    m["Date"]   = pd.to_datetime(m["Date"])
    m["Ticker"] = m["Ticker"].astype(str).str.upper().str.strip()
    m["Action"] = m["Action"].astype(str).str.upper().str.strip()
    m["SymKey"] = _norm_sym_series(m["Ticker"])

    intervals = []
    for symkey, grp in m.sort_values(["SymKey","Date","Action"]).groupby("SymKey", sort=False):
        start = None
        # pick a representative display ticker (most frequent in history)
        disp = grp["Ticker"].mode().iat[0] if not grp["Ticker"].mode().empty else grp["Ticker"].iloc[0]
        for _, r in grp.iterrows():
            if r["Action"] == "ADD":
                start = r["Date"]
            elif r["Action"] == "DROP" and start is not None:
                intervals.append((symkey, disp, start, r["Date"]))
                start = None
        if start is not None:
            intervals.append((symkey, disp, start, END_CAP))

    iv = pd.DataFrame(intervals, columns=["SymKey","Ticker","Start","End"])
    iv["Start"] = pd.to_datetime(iv["Start"])
    iv["End"]   = pd.to_datetime(iv["End"])
    return iv

def map_permno_to_symkey(daily: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """
    Map each CRSP daily row to a normalized symbol key using CRSP names history:
      - prefer `ticker`; fallback to `tsymbol` if present/needed
      - only keep rows within [namebegdt, nameenddt]
      - if multiple name rows match a day, keep the most recent namebegdt
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])

    n = names.copy()
    # unify column names / dtypes
    if "namebegdt" not in n.columns:
        raise ValueError("crsp_names.csv must include 'namebegdt' (export 'namedt as namebegdt').")
    if "nameenddt" not in n.columns:
        raise ValueError("crsp_names.csv must include 'nameenddt'.")

    n["namebegdt"] = pd.to_datetime(n["namebegdt"], errors="coerce")
    n["nameenddt"] = pd.to_datetime(n["nameenddt"], errors="coerce").fillna(pd.Timestamp.max)

    # best display symbol column: ticker, else tsymbol if ticker is blank/null
    sym = n.get("ticker", pd.Series(index=n.index, dtype="object")).astype(str)
    if "tsymbol" in n.columns:
        use_tsym = sym.isna() | (sym.str.strip() == "") | sym.str.contains("^nan$", case=False, regex=True)
        sym = sym.where(~use_tsym, n["tsymbol"].astype(str))

    n["DisplayTicker"] = sym.str.upper().str.strip()
    n["SymKey"] = _norm_sym_series(n["DisplayTicker"])

    keep = ["permno","namebegdt","nameenddt","DisplayTicker","SymKey"]
    for extra in ["comnam","exchcd","shrcd"]:
        if extra in n.columns:
            keep.append(extra)
    n = n[keep]

    md = d.merge(n, on="permno", how="left")
    md = md[(md["date"] >= md["namebegdt"]) & (md["date"] <= md["nameenddt"])]
    # pick latest namebegdt if multiple names overlap
    md = md.sort_values(["permno","date","namebegdt"], ascending=[True, True, False])
    md = md.groupby(["permno","date"], as_index=False).first()
    return md

def build_biasfree_panel() -> pd.DataFrame:
    # ---- load ----
    if not HIST_FILE.exists():  raise FileNotFoundError(f"Missing {HIST_FILE}")
    if not DAILY_FILE.exists(): raise FileNotFoundError(f"Missing {DAILY_FILE}")
    if not NAMES_FILE.exists(): raise FileNotFoundError(f"Missing {NAMES_FILE}")

    hist  = pd.read_csv(HIST_FILE)
    daily = pd.read_csv(DAILY_FILE)
    names = pd.read_csv(NAMES_FILE)

    # ensure daily basics
    need_daily = {"permno","date","ret"}
    miss = need_daily - set(daily.columns)
    if miss:
        raise ValueError(f"{DAILY_FILE} missing required columns: {miss}")

    # effective return (preserve delists)
    if "effective_ret" not in daily.columns:
        if "dlret" in daily.columns:
            daily["effective_ret"] = daily["dlret"].where(daily["dlret"].notna(), daily["ret"])
        else:
            daily["effective_ret"] = daily["ret"]

    # ---- membership windows ----
    intervals = build_membership_intervals(hist)

    # ---- permno -> SymKey via names ----
    daily_n = map_permno_to_symkey(daily, names)

    # ---- enforce membership windows (join on SymKey) ----
    panel = (daily_n
             .merge(intervals[["SymKey","Ticker","Start","End"]],
                    on="SymKey", how="inner"))
    panel["date"]  = pd.to_datetime(panel["date"])
    panel["Start"] = pd.to_datetime(panel["Start"])
    panel["End"]   = pd.to_datetime(panel["End"])
    panel = panel[(panel["date"] >= panel["Start"]) & (panel["date"] <= panel["End"])]

    # tidy columns
    drop_cols = [c for c in ["namebegdt","nameenddt","comnam"] if c in panel.columns]
    panel = panel.drop(columns=drop_cols, errors="ignore")
    panel = panel.sort_values(["date","Ticker","permno"]).reset_index(drop=True)

    # ---- save panel ----
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PANEL, index=False)

    # ---- quick validation summary ----
    by_day = panel.groupby("date")["Ticker"].nunique()
    dl_nonnull = int(panel["dlret"].notna().sum()) if "dlret" in panel.columns else 0
    summary = pd.DataFrame([{
        "rows": len(panel),
        "unique_tickers": panel["Ticker"].nunique(),
        "date_min": panel["date"].min().strftime("%Y-%m-%d"),
        "date_max": panel["date"].max().strftime("%Y-%m-%d"),
        "members_min": int(by_day.min()),
        "members_max": int(by_day.max()),
        "members_median": int(by_day.median()),
        "days_below_95": int((by_day < 95).sum()),
        "dlret_nonnull_rows": dl_nonnull,
    }])
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"[INFO] Saved {len(panel):,} rows → {OUT_PANEL}")
    print(f"[INFO] Unique tickers: {panel['Ticker'].nunique()}")
    print(f"[INFO] Dates: {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"[INFO] Members/day (min, max, median): {int(by_day.min())}, {int(by_day.max())}, {int(by_day.median())}")
    print(f"[INFO] Days with <95 members: {int((by_day < 95).sum())}")
    print(f"[INFO] Delist rows (dlret not null): {dl_nonnull}")
    return panel

if __name__ == "__main__":
    build_biasfree_panel()

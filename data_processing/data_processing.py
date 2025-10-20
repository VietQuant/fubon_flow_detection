#!/usr/bin/env python3
"""
Convert portfolio composition and creation basket notebooks to a script.
Reads scrapper outputs and writes cleaned artifacts with project‑relative paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FW_DIR = ROOT / "data" / "fubon_weight_data"
PORTFOLIO_DIR = FW_DIR / "portfolio_composition"
RAW_DIR = FW_DIR / "raw"
CLEANED_DIR = FW_DIR / "cleaned"
SHIFT_DIR = CLEANED_DIR / "shift_report_date"


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _read_glob_pandas(folder: Path, pattern: str) -> pd.DataFrame:
    files = sorted(folder.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def read_glob(folder: Path, pattern: str) -> pd.DataFrame:
    """Try Dask for scalability; fall back to pandas if not available."""
    try:
        import dask.dataframe as dd  # type: ignore

        ddf = dd.read_csv((folder / pattern).as_posix())
        return ddf.compute()
    except Exception:
        return _read_glob_pandas(folder, pattern)


def process_portfolio_composition() -> Path:
    src_df = read_glob(PORTFOLIO_DIR, "*.csv")
    if src_df.empty:
        print(f"No portfolio composition CSVs in {PORTFOLIO_DIR}")
        return CLEANED_DIR / "fubon_portfolio_composition.csv"

    # Minimal preprocessing to mirror the notebook
    df = src_df.rename(columns={"requested_date": "Date"}).copy()

    use_cols = [
        "total_advance_subscription",
        "nav",
        "total_units",
        "net_unit_change",
        "nav_per_unit",
        "creation_unit",
        "equity_value_basket",
        "cash_component",
        "fund_purchase",
        "fund_redemption",
        "Date",
    ]
    df = df[[c for c in use_cols if c in df.columns]].copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")

    ensure_dirs([CLEANED_DIR])
    out_path = CLEANED_DIR / "fubon_portfolio_composition.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved portfolio composition to {out_path}")
    return out_path


def process_creation_basket(pcf_path: Path) -> tuple[Path, Path, Path]:
    raw_df = read_glob(RAW_DIR, "*.csv")
    if raw_df.empty:
        print(f"No creation basket RAW CSVs in {RAW_DIR}")
        # Still ensure downstream dirs exist
        ensure_dirs([CLEANED_DIR, SHIFT_DIR])
        return (
            CLEANED_DIR / "fubon_creation_basket_all.csv",
            SHIFT_DIR / "fubon_fund_weight.csv",
            SHIFT_DIR / "fubon_fund_weight_long.csv",
        )

    ddf = raw_df.rename(
        columns={
            "data_date": "Date",
            "stock_code": "stock",
            "total_units": "total_outstanding_units",
            "net_asset_value": "nav",
        }
    )

    use_cols = [
        "Date",
        "stock",
        "shares",
        "market_value",
        "weight_pct",
        "total_market_value",
        "total_weight",
        "total_outstanding_units",
        "nav",
        "nav_per_unit",
        "usd_to_twd",
        "usd_to_vnd",
        "cash_twd",
        "cash_usd",
        "cash_vnd",
        "payables_twd",
    ]
    ddf = ddf[[c for c in use_cols if c in ddf.columns]].copy()

    # Clean fields
    ddf["Date"] = pd.to_datetime(ddf["Date"], errors="coerce")
    if "stock" in ddf.columns:
        ddf["stock"] = ddf["stock"].astype(str).str.split(" ", expand=True)[0]
    ddf = ddf.sort_values(by=[c for c in ["stock"] if c in ddf.columns])

    # Merge with portfolio composition (pcf)
    pcf = pd.read_csv(pcf_path, parse_dates=["Date"]) if pcf_path.exists() else pd.DataFrame()
    if not pcf.empty:
        merge_keys = [k for k in ["Date", "nav", "nav_per_unit"] if k in ddf.columns and k in pcf.columns]
        if merge_keys:
            ddf = ddf.merge(pcf, on=merge_keys, how="left")

    # Derived metrics
    if "weight_pct" in ddf.columns:
        ddf["weight_pct"] = ddf["weight_pct"] / 100.0
    if {"weight_pct", "equity_value_basket"}.issubset(ddf.columns):
        ddf["allocated_value"] = ddf["weight_pct"] * ddf["equity_value_basket"]
    if {"market_value", "shares"}.issubset(ddf.columns):
        ddf["price_per_share"] = ddf["market_value"] / ddf["shares"].replace(0, np.nan)
    if {"allocated_value", "price_per_share"}.issubset(ddf.columns):
        ddf["quantity"] = np.floor(ddf["allocated_value"] / ddf["price_per_share"])
        ddf["quantity"] = (ddf["quantity"] / 100.0).round() * 100.0
    if {"net_unit_change", "creation_unit"}.issubset(ddf.columns):
        ddf["net_primary_lot"] = ddf["net_unit_change"] / ddf["creation_unit"].replace(0, np.nan)
    if {"net_primary_lot", "equity_value_basket", "usd_to_twd", "usd_to_vnd"}.issubset(ddf.columns):
        ddf["primary_flow_vnd"] = (
            ddf["net_primary_lot"]
            * ddf["equity_value_basket"]
            / ddf["usd_to_twd"].replace(0, np.nan)
            * ddf["usd_to_vnd"]
        )

    # Save full creation basket
    ensure_dirs([CLEANED_DIR])
    creation_all_path = CLEANED_DIR / "fubon_creation_basket_all.csv"
    ddf.to_csv(creation_all_path, index=False)
    print(f"Saved creation basket (all) to {creation_all_path}")

    # Build shifted weight tables
    tmp = ddf.rename(columns={"quantity": "volume", "weight_pct": "weight"}).copy()
    use_cols2 = [c for c in ["Date", "stock", "volume", "weight"] if c in tmp.columns]
    tmp = tmp[use_cols2].copy()

    # Shift report date up 1 day per stock
    save_df = pd.DataFrame()
    if set(["Date", "stock"]).issubset(tmp.columns):
        save_df = save_df.assign(
            Date=tmp["Date"],
            stock=tmp["stock"],
            volume=tmp.groupby("stock", dropna=False)["volume"].shift(1),
            weight=tmp.groupby("stock", dropna=False)["weight"].shift(1),
        )

    ensure_dirs([SHIFT_DIR])
    weight_wide_path = SHIFT_DIR / "fubon_fund_weight.csv"
    if not save_df.empty:
        save_df.set_index(["Date", "stock"]).unstack(level=1).to_csv(weight_wide_path)
        print(f"Saved wide weight table to {weight_wide_path}")
    else:
        # Still create an empty placeholder for pipeline compatibility
        pd.DataFrame().to_csv(weight_wide_path)

    # Long format
    weight_long_path = SHIFT_DIR / "fubon_fund_weight_long.csv"
    try:
        dfw = pd.read_csv(weight_wide_path, header=[0, 1], index_col=0)
        dfw.stack(level=1, future_stack=True).reset_index().to_csv(weight_long_path, index=False)
        print(f"Saved long weight table to {weight_long_path}")
    except Exception as exc:
        print(f"Could not create long weight table from {weight_wide_path}: {exc}")

    # Also update root-level convenience file used by algorithms, if possible
    try:
        root_long = ROOT / "data" / "fubon_fund_weight_long.csv"
        if weight_long_path.exists():
            pd.read_csv(weight_long_path).to_csv(root_long, index=False)
            print(f"Updated convenience file {root_long}")
    except Exception:
        pass

    return creation_all_path, weight_wide_path, weight_long_path


def main() -> int:
    ensure_dirs([PORTFOLIO_DIR, RAW_DIR, CLEANED_DIR, SHIFT_DIR])
    pcf_out = process_portfolio_composition()
    process_creation_basket(pcf_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


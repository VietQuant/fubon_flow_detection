import argparse
import datetime as dt
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna
import pandas as pd

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import algorithm.fubon_flow_vectorize as pipeline


# Project-relative data files
WEIGHT_PATH = ROOT / "data" / "fubon_fund_weight_long.csv"
PRIMARY_FLOW_PATH = ROOT / "data" / "transaction_data.csv"
ORDERBOOK_TEMPLATE = Path("/data/tick_feature/sample_data/udp_orderbook/{date}.csv")
TRADE_TEMPLATE = Path("/data/tick_feature/sample_data/udp_tick/{date}.csv")
HOURS = (9, 10, 11, 13, 14)


DEFAULT_DECIMALS = 3
DEFAULT_LOOKBACK_MINUTES = 5

# Baseline distribution (from the pipeline defaults) + first_center
STAGE1_DIST_DEFAULTS = {
    "start": 0.15,
    "stop": 89,
    "num": 25,
    "step": 0.25,
    "decimals": DEFAULT_DECIMALS,
    "growth": 1.13,
    "int_threshold": 4,
    "first_center": 0.20,
}


@dataclass
class TrialParams:
    # Detection parameters
    seconds_early: int
    tolerance: float
    seconds_gaps: int
    min_unique: int
    min_segment: int
    core_threshold: int
    # Distribution generation
    start: float
    stop: int
    num: int
    step: float
    growth: float
    int_threshold: int
    first_center: float


class DataCache:
    def __init__(self) -> None:
        self._cache: Dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def load(self, run_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if run_date in self._cache:
            return self._cache[run_date]

        orderbook_path = ORDERBOOK_TEMPLATE.as_posix().format(date=run_date)
        trade_path = TRADE_TEMPLATE.as_posix().format(date=run_date)

        orderbook = pd.read_csv(orderbook_path).rename(columns={"symbol": "stock"})
        trade = pd.read_csv(trade_path).rename(columns={"symbol": "stock"})

        orderbook_pre = pipeline.preprocess(orderbook)
        trade_pre = pipeline.preprocess(trade)

        self._cache[run_date] = (orderbook_pre, trade_pre)
        return self._cache[run_date]


def build_distribution_from_params(
    *,
    start: float,
    stop: int,
    num: int,
    step: float,
    growth: float,
    int_threshold: int,
    first_center: float,
) -> Dict[float, List[float]]:
    return pipeline.generate_non_overlap_distribution(
        start=start,
        stop=stop,
        num=num,
        step=step,
        decimals=DEFAULT_DECIMALS,
        growth=growth,
        int_threshold=int_threshold,
        first_center=first_center,
    )


def compute_metrics(predictions: pd.DataFrame, primary_flow: pd.Series) -> Dict[str, float]:
    merged = predictions.merge(primary_flow.rename("primary_flow_vnd"), on="Date", how="left")
    evaluated = merged.dropna(subset=["primary_flow_vnd"])
    if evaluated.empty:
        return {"mae": np.nan}

    errors = evaluated["net_flow"] - evaluated["primary_flow_vnd"]
    mae = float(np.abs(errors).mean())
    return {"mae": mae}


def run_single_date(
    run_date: str,
    orderbook_pre: pd.DataFrame,
    trade_pre: pd.DataFrame,
    weight_data: pd.DataFrame,
    params: TrialParams,
    distribution: Dict[float, List[float]],
    lookback_minutes: int,
) -> pd.DataFrame:
    stocks = weight_data["stock"].values
    if stocks.size == 0:
        raise ValueError("No stocks available after weight filtering")

    orderbook = pipeline.susbet_data(orderbook_pre, stocks).reset_index(drop=True)
    trade = pipeline.susbet_data(trade_pre, stocks).reset_index(drop=True)

    frames: List[pd.DataFrame] = []

    for hour in HOURS:
        for minute in range(60):
            current_time = dt.datetime.strptime(run_date, "%Y-%m-%d").replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            try:
                orderbook_snapshot = pipeline.get_snapshot_data(
                    orderbook,
                    minutes_ago=lookback_minutes,
                    current_time=current_time,
                    UTC=True,
                )
                trade_snapshot = pipeline.get_snapshot_data(
                    trade,
                    minutes_ago=lookback_minutes,
                    current_time=current_time,
                    UTC=True,
                )

                if orderbook_snapshot.empty or trade_snapshot.empty:
                    continue

                orderbook_snapshot = orderbook_snapshot.sort_values(["ts", "stock"], kind="mergesort")
                trade_snapshot = trade_snapshot.sort_values(["ts", "stock"], kind="mergesort")

                market_snapshot = (
                    pd.merge_asof(
                        trade_snapshot,
                        orderbook_snapshot,
                        on="ts",
                        by="stock",
                        direction="backward",
                    )
                    .reset_index(drop=True)
                    .merge(weight_data, on="stock", how="inner")
                )

                if market_snapshot.empty:
                    continue

                market_snapshot = pipeline.wrangle(market_snapshot)
                similarity_df = pipeline.find_similarity_block(
                    market_snapshot,
                    tolerance=params.tolerance,
                    seconds_gaps=params.seconds_gaps,
                    min_unique_stock_per_sim_block=params.min_unique,
                )
                if similarity_df.empty:
                    continue

                clustered_df = pipeline.cluster_similarity_block(
                    similarity_df=similarity_df,
                    distribution=distribution,
                    min_segment_length_per_cluster=params.min_segment,
                )
                if clustered_df.empty:
                    continue

                filtered_df = pipeline.filter_clusters_by_core(
                    clustered_df,
                    core_threshold=params.core_threshold,
                )
                if filtered_df.empty:
                    continue

                final_df = pipeline.final_process(
                    clustered_df=filtered_df,
                    current_time=current_time,
                )

                if final_df.empty:
                    continue

                final_df = final_df.copy()
                final_df["ts"] = final_df["ts"] - dt.timedelta(seconds=params.seconds_early)
                final_df["arbit_ratio"] = np.where(final_df["side"] == 1, final_df["num_lot"], 0.0)
                final_df["unwind_ratio"] = np.where(final_df["side"] == -1, final_df["num_lot"], 0.0)
                final_df["arbit_value"] = np.where(final_df["side"] == 1, final_df["matched_value"], 0.0)
                final_df["unwind_value"] = np.where(final_df["side"] == -1, final_df["matched_value"], 0.0)
                frames.append(final_df)
            except Exception as exc:  # noqa: BLE001
                # Always log errors during execution
                print(f"{run_date} {hour:02d}:{minute:02d} | {exc}")
                continue

    if not frames:
        raise ValueError("No qualifying windows for date")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["time_str", "stock", "initial_ts", "ts"], keep="first"
    )
    return combined


def evaluate_params(
    params: TrialParams,
    distribution: Dict[float, List[float]],
    run_dates: List[str],
    weight_df: pd.DataFrame,
    primary_flow: pd.Series,
    cache: DataCache,
    lookback_minutes: int,
) -> Dict[str, float]:
    predictions = []

    for run_date in run_dates:
        weight_day = weight_df[weight_df["Date"] == run_date].dropna()
        if weight_day.empty:
            print(f"Skipping {run_date}: weight data empty")
            continue

        try:
            orderbook_pre, trade_pre = cache.load(run_date)
            day_df = run_single_date(
                run_date=run_date,
                orderbook_pre=orderbook_pre,
                trade_pre=trade_pre,
                weight_data=weight_day,
                params=params,
                distribution=distribution,
                lookback_minutes=lookback_minutes,
            )
            net_flow = float(day_df["arbit_value"].sum() - day_df["unwind_value"].sum())
            predictions.append({"Date": run_date, "net_flow": net_flow})
        except Exception as exc:  # noqa: BLE001
            if "No qualifying windows for date" in str(exc):
                print(
                    f"No predictions for date {run_date} — pruning current hyperparameter set"
                )
                raise optuna.TrialPruned("No clustering output for at least one date")
            print(f"Skipping {run_date}: {exc}")
            predictions.append({"Date": run_date, "net_flow": np.nan})

    if not predictions:
        return {"mae": np.nan}

    predictions_df = pd.DataFrame(predictions)
    metrics = compute_metrics(predictions_df, primary_flow)
    metrics["predictions"] = predictions_df
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimizer v3: Hierarchical tuning with first_center (multivariate TPE)",
    )
    parser.add_argument(
        "--stage1-trials",
        type=int,
        default=80,
        help="Trials for Stage 1 (detection params)",
    )
    parser.add_argument(
        "--stage2-trials",
        type=int,
        default=60,
        help="Trials for Stage 2 (distribution params)",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=25,
        help="Random initial trials before TPE surrogate engages",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tuning"),
        help="Directory to write CSV results",
    )
    parser.add_argument(
        "--stage1-storage",
        type=str,
        default="sqlite:///results/tuning/optuna_v3_stage1.db",
        help="Optuna storage URI for Stage 1 study",
    )
    parser.add_argument(
        "--stage1-study-name",
        type=str,
        default="fubon_v3_stage1",
        help="Optuna study name for Stage 1",
    )
    parser.add_argument(
        "--stage2-storage",
        type=str,
        default="sqlite:///results/tuning/optuna_v3_stage2.db",
        help="Optuna storage URI for Stage 2 study",
    )
    parser.add_argument(
        "--stage2-study-name",
        type=str,
        default="fubon_v3_stage2",
        help="Optuna study name for Stage 2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dates = list(dict.fromkeys(pipeline.RUN_DATES))

    weight_df = pd.read_csv(WEIGHT_PATH).rename(columns={"volume": "quantity"})
    primary_flow = (
        pd.read_csv(PRIMARY_FLOW_PATH, usecols=["Date", "primary_flow_vnd"]).set_index("Date")[
            "primary_flow_vnd"
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = DataCache()

    # ---------- Stage 1: Detection params (fix distribution) ----------
    sampler1 = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=args.n_startup_trials,
        n_ei_candidates=64,
        seed=args.random_seed,
    )
    study1 = optuna.create_study(
        direction="minimize",
        sampler=sampler1,
        storage=args.stage1_storage,
        study_name=args.stage1_study_name,
        load_if_exists=True,
    )
    records_stage1: List[Dict[str, float]] = []

    fixed_dist = build_distribution_from_params(
        start=STAGE1_DIST_DEFAULTS["start"],
        stop=STAGE1_DIST_DEFAULTS["stop"],
        num=STAGE1_DIST_DEFAULTS["num"],
        step=STAGE1_DIST_DEFAULTS["step"],
        growth=STAGE1_DIST_DEFAULTS["growth"],
        int_threshold=STAGE1_DIST_DEFAULTS["int_threshold"],
        first_center=STAGE1_DIST_DEFAULTS["first_center"],
    )

    def _append_csv(path: Path, record: Dict[str, float], columns: List[str]) -> None:
        exists = path.exists() and path.stat().st_size > 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if not exists:
                writer.writeheader()
            writer.writerow({k: record.get(k, None) for k in columns})

    def _update_best_json(stage: str, best_path: Path, mae: float, params: Dict[str, float]) -> None:
        if np.isnan(mae):
            return
        data = {}
        if best_path.exists():
            try:
                data = json.loads(best_path.read_text())
            except Exception:
                data = {}
        current = data.get(stage)
        if current is None or mae < current.get("mae", float("inf")):
            data[stage] = {"mae": mae, "params": params}
            best_path.write_text(json.dumps(data, indent=2))

    def objective_stage1(trial: optuna.trial.Trial) -> float:
        params = TrialParams(
            seconds_early=trial.suggest_int("seconds_early", low=5, high=25),
            tolerance=trial.suggest_float("tolerance", low=0.12, high=0.35),
            seconds_gaps=trial.suggest_int("seconds_gaps", low=45, high=60),
            min_unique=trial.suggest_int("min_unique", low=7, high=24),
            min_segment=trial.suggest_int("min_segment", low=6, high=25),
            core_threshold=trial.suggest_int("core_threshold", low=8, high=20),
            start=STAGE1_DIST_DEFAULTS["start"],
            stop=STAGE1_DIST_DEFAULTS["stop"],
            num=STAGE1_DIST_DEFAULTS["num"],
            step=STAGE1_DIST_DEFAULTS["step"],
            growth=STAGE1_DIST_DEFAULTS["growth"],
            int_threshold=STAGE1_DIST_DEFAULTS["int_threshold"],
            first_center=STAGE1_DIST_DEFAULTS["first_center"],
        )

        sample_preview = list(fixed_dist.items())[:5]
        print(f"[Stage 1] Distribution sample: {sample_preview} ...")

        metrics = evaluate_params(
            params=params,
            distribution=fixed_dist,
            run_dates=run_dates,
            weight_df=weight_df,
            primary_flow=primary_flow,
            cache=cache,
            lookback_minutes=DEFAULT_LOOKBACK_MINUTES,
        )

        record = {
            "seconds_early": params.seconds_early,
            "tolerance": params.tolerance,
            "seconds_gaps": params.seconds_gaps,
            "min_unique": params.min_unique,
            "min_segment": params.min_segment,
            "core_threshold": params.core_threshold,
            "mae": metrics["mae"],
        }
        records_stage1.append(record)
        # Append immediately to CSV and update best json
        out1 = args.output_dir / "fubon_tuning_ver3_stage1.csv"
        cols1 = [
            "seconds_early",
            "tolerance",
            "seconds_gaps",
            "min_unique",
            "min_segment",
            "core_threshold",
            "mae",
        ]
        _append_csv(out1, record, cols1)
        _update_best_json("stage1", args.output_dir / "last_best_v3.json", metrics["mae"], record)
        if np.isnan(metrics["mae"]):
            return float("inf")
        return metrics["mae"]

    study1.optimize(
        objective_stage1,
        n_trials=args.stage1_trials,
        gc_after_trial=True,
        catch=(Exception,),
    )

    out1 = args.output_dir / "fubon_tuning_ver3_stage1.csv"
    df_stage1 = pd.read_csv(out1).sort_values("mae") if out1.exists() else pd.DataFrame(records_stage1)
    print("\n[Stage 1] Top configurations (by MAE):")
    if not df_stage1.empty:
        print(
            df_stage1[
                [
                    "seconds_early",
                    "tolerance",
                    "seconds_gaps",
                    "min_unique",
                    "min_segment",
                    "core_threshold",
                    "mae",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
        print("\n[Stage 1] Best MAE:", df_stage1["mae"].min())

    best_det = study1.best_params

    # ---------- Stage 2: Distribution params (freeze detection) ----------
    sampler2 = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=args.n_startup_trials,
        n_ei_candidates=64,
        seed=args.random_seed,
    )
    study2 = optuna.create_study(
        direction="minimize",
        sampler=sampler2,
        storage=args.stage2_storage,
        study_name=args.stage2_study_name,
        load_if_exists=True,
    )
    records_stage2: List[Dict[str, float]] = []

    def objective_stage2(trial: optuna.trial.Trial) -> float:
        # Match v2 ranges, plus first_center as a new tunable param
        dist_params = dict(
            start=trial.suggest_float("start", low=0.02, high=0.8),
            stop=trial.suggest_int("stop", low=50, high=100),
            num=trial.suggest_int("num", low=20, high=60),
            step=trial.suggest_float("step", low=0.25, high=0.5, step=0.05),
            growth=trial.suggest_float("growth", low=1, high=1.5),
            int_threshold=trial.suggest_int("int_threshold", low=3, high=9),
            first_center=trial.suggest_float("first_center", low=0.05, high=0.5, step=0.05),
        )

        params = TrialParams(
            seconds_early=int(best_det["seconds_early"]),
            tolerance=float(best_det["tolerance"]),
            seconds_gaps=int(best_det["seconds_gaps"]),
            min_unique=int(best_det["min_unique"]),
            min_segment=int(best_det["min_segment"]),
            core_threshold=int(best_det["core_threshold"]),
            **dist_params,
        )

        distribution = build_distribution_from_params(**dist_params)
        sample_preview = list(distribution.items())[:5]
        print(f"[Stage 2] Distribution sample: {sample_preview} ...")

        metrics = evaluate_params(
            params=params,
            distribution=distribution,
            run_dates=run_dates,
            weight_df=weight_df,
            primary_flow=primary_flow,
            cache=cache,
            lookback_minutes=DEFAULT_LOOKBACK_MINUTES,
        )

        record = {
            # Freeze detection params in the record for reproducibility
            "seconds_early": params.seconds_early,
            "tolerance": params.tolerance,
            "seconds_gaps": params.seconds_gaps,
            "min_unique": params.min_unique,
            "min_segment": params.min_segment,
            "core_threshold": params.core_threshold,
            # Tuned distribution params
            "start": params.start,
            "stop": params.stop,
            "num": params.num,
            "step": params.step,
            "growth": params.growth,
            "int_threshold": params.int_threshold,
            "first_center": params.first_center,
            "mae": metrics["mae"],
        }
        records_stage2.append(record)
        # Append immediately to CSV and update best json
        out2 = args.output_dir / "fubon_tuning_ver3_stage2.csv"
        cols2 = [
            "seconds_early",
            "tolerance",
            "seconds_gaps",
            "min_unique",
            "min_segment",
            "core_threshold",
            "start",
            "stop",
            "num",
            "step",
            "growth",
            "int_threshold",
            "first_center",
            "mae",
        ]
        _append_csv(out2, record, cols2)
        _update_best_json("stage2", args.output_dir / "last_best_v3.json", metrics["mae"], record)
        if np.isnan(metrics["mae"]):
            return float("inf")
        return metrics["mae"]

    study2.optimize(
        objective_stage2,
        n_trials=args.stage2_trials,
        gc_after_trial=True,
        catch=(Exception,),
    )

    out2 = args.output_dir / "fubon_tuning_ver3_stage2.csv"
    df_stage2 = pd.read_csv(out2).sort_values("mae") if out2.exists() else pd.DataFrame(records_stage2)
    print("\n[Stage 2] Top configurations (by MAE):")
    cols2 = [
        "seconds_early",
        "tolerance",
        "seconds_gaps",
        "min_unique",
        "min_segment",
        "core_threshold",
        "start",
        "stop",
        "num",
        "step",
        "growth",
        "int_threshold",
        "first_center",
        "mae",
    ]
    if not df_stage2.empty:
        print(df_stage2[cols2].head(10).to_string(index=False))
        print("\n[Stage 2] Best MAE:", df_stage2["mae"].min())


if __name__ == "__main__":
    main()

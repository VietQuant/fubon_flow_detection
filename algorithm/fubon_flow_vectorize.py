import datetime
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from pathlib import Path

def mmap(*args):
    return list(map(*args))

def generate_non_overlap_distribution(
    start: float = 0.5,
    stop: float = 100.0,
    num: int = 40,
    step: float = 0.5,
    decimals: int = 3,
    growth: float = 1.0,
    int_threshold: float = 3.0,
    first_center: float = None,
) -> Dict[float, List[float]]:
    if start <= 0 or stop <= 0 or stop <= start:
        raise ValueError("start and stop must be positive and stop > start.")
    if num < 2:
        raise ValueError("num must be >= 2.")
    if growth <= 0:
        raise ValueError("growth must be > 0.")

    t = np.linspace(0.0, 1.0, num)
    w = t ** growth
    log_start, log_stop = np.log(start), np.log(stop)
    raw = np.exp(log_start + (log_stop - log_start) * w)
    rounded = np.round(raw / step) * step
    rounded = np.clip(rounded, start, stop)
    lots = np.unique(rounded.astype(float))
    lots[-1] = float(round(stop, decimals))

    snapped = []
    for v in lots:
        if v >= int_threshold:
            v = round(v)
        snapped.append(float(v))
    lots = np.array(snapped, dtype=float)

    eps = 1e-9
    filtered = []
    last = None
    for v in lots:
        if last is None or v > last + eps:
            filtered.append(v)
            last = v
    if filtered[-1] < stop - eps:
        filtered.append(stop)

    if first_center is not None:
        fc = float(max(first_center, start))
        filtered = [x for x in filtered if x >= fc - eps]
        if not filtered or filtered[0] > fc + eps:
            filtered.insert(0, fc)
        else:
            filtered[0] = fc

    lots = [round(float(x), decimals) for x in filtered]
    lots[-1] = round(stop, decimals)

    clean = []
    last = None
    for v in lots:
        if last is None or v > last + eps:
            clean.append(v)
            last = v
    lots = clean

    distribution: Dict[float, List[float]] = {}
    for i, lot in enumerate(lots):
        if i == 0:
            lower = round(start, decimals)
            upper = round((lots[i] + lots[i+1]) / 2, decimals)
        elif i == len(lots) - 1:
            lower = round((lots[i-1] + lots[i]) / 2, decimals)
            upper = round(stop, decimals)
        else:
            lower = round((lots[i-1] + lots[i]) / 2, decimals)
            upper = round((lots[i] + lots[i+1]) / 2, decimals)
        distribution[lot] = [lower, upper]
    return distribution

#### For FUBON
ISO_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
CUSTOM_DATE_FORMAT = "%Y%m%d"
CUSTOM_TIME_FORMAT = "%H:%M:%S"
SECONDS_GAP = 45
SECONDS_EARLY = 16
TOLERANCE = 0.3633
UNKNOWN_SEGMENT_TAG = -1
MAX_NUMLOT = 1_000_000
NUM_LOT_BASKET = 1
MIN_UNIQUE_STOCK_PER_SIM_BLOCK = 15
MIN_SEGMENT_LENGTH_PER_CLUSTER = 7
CORE_THRESHOLD = 8
DISTRIBUTION = generate_non_overlap_distribution(
    start=0.02,
    stop=100,
    num=22,
    step=0.2,
    decimals=3,
    growth=1.1167,
    int_threshold=7,
    first_center=0.25
)

NS_PER_SECOND = 1_000_000_000

RUN_DATES = [
    "2024-09-04",
    "2024-09-05",
    "2024-09-06",
    "2024-09-09",
    "2024-09-10",
    "2024-09-11",
    "2024-09-12",
    "2024-09-13",
    "2024-09-16",
    "2024-09-17",
    "2024-09-18",
    "2024-09-19",
    "2024-09-20",
    "2024-09-23",
    "2024-09-24",
    "2024-09-25",
    "2024-09-26",
    "2024-09-27",
    "2024-09-30",
    "2025-07-28"
]


def susbet_data(df: pd.DataFrame, stocks: np.ndarray) -> pd.DataFrame:
    return df[df["stock"].isin(stocks)]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], unit="ns")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_snapshot_data(
    df: pd.DataFrame, *, minutes_ago: int, current_time: datetime.datetime, UTC: bool = True
) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0]

    snapshot_end = pd.Timestamp(current_time)
    if UTC:
        snapshot_end -= pd.Timedelta(hours=7)

    snapshot_end = snapshot_end.replace(microsecond=0)
    snapshot_start = snapshot_end - pd.Timedelta(
        minutes=minutes_ago,
        seconds=snapshot_end.second,
        microseconds=snapshot_end.microsecond,
    )

    ts_values = df["ts"].to_numpy("datetime64[ns]").astype(np.int64, copy=False)
    start_value = np.int64(snapshot_start.value)
    end_value = np.int64(snapshot_end.value)

    left = np.searchsorted(ts_values, start_value, side="left")
    right = np.searchsorted(ts_values, end_value, side="right")
    if left >= right:
        return df.iloc[0:0]
    return df.iloc[left:right]


def wrangle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "c": "matched_price",
            "bb0_p": "bid_price_1",
            "bb0_v": "bid_volume_1",
            "bo0_p": "ask_price_1",
            "bo0_v": "ask_volume_1",
            "mv": "matched_vol",
        }
    ).copy()

    matched_price = df["matched_price"].to_numpy()
    bid_price = df["bid_price_1"].to_numpy()
    ask_price = df["ask_price_1"].to_numpy()

    side = np.full(len(df), np.nan)
    side[matched_price >= ask_price] = 1
    side[matched_price <= bid_price] = -1
    df["side"] = side

    df["matched_value"] = df["matched_price"] * df["matched_vol"]
    ts = df["ts"].dt
    time_str = 10_000_000 + ts.hour * 10000 + ts.minute * 100 + ts.second
    df["time_str"] = time_str.astype(int)

    aggregated = (
        df.groupby(["time_str", "stock", "side", "quantity"], sort=False)
        .agg(
            matched_vol=("matched_vol", "sum"),
            matched_value=("matched_value", "sum"),
            ts=("ts", "last"),
        )
        .reset_index()
    )
    aggregated["num_lot"] = (aggregated["matched_vol"] / aggregated["quantity"]).round(4)
    return aggregated


def _compute_neighbour_counts(df_side: pd.DataFrame, tolerance: float, seconds_gaps: int) -> np.ndarray:
    n_rows = len(df_side)
    if n_rows == 0:
        return np.empty(0, dtype=np.int32)

    ts_values = df_side["ts"].to_numpy("datetime64[ns]").astype(np.int64)
    num_lot_arr = df_side["num_lot"].to_numpy(dtype=np.float64)
    stock_arr = df_side["stock"].to_numpy()

    group_sizes = df_side.groupby("time_str", sort=False).size().to_numpy()
    group_start_indices = np.cumsum(np.concatenate(([0], group_sizes[:-1])))
    start_idx_arr = np.repeat(group_start_indices, group_sizes)

    window_ns = np.int64(seconds_gaps) * NS_PER_SECOND
    block_start_int = ts_values[group_start_indices]
    end_idx_per_group = np.searchsorted(ts_values, block_start_int + window_ns, side="right")
    end_idx_arr = np.repeat(end_idx_per_group, group_sizes)

    lower_bounds = num_lot_arr * (1 - tolerance)
    upper_bounds = num_lot_arr * (1 + tolerance)

    counts = np.zeros(n_rows, dtype=np.int32)
    for idx in range(n_rows):
        start = start_idx_arr[idx]
        end = end_idx_arr[idx]
        if start >= end:
            continue
        candidate_num_lot = num_lot_arr[start:end]
        mask = (candidate_num_lot >= lower_bounds[idx]) & (candidate_num_lot <= upper_bounds[idx])
        if not mask.any():
            continue
        candidate_stocks = stock_arr[start:end][mask]
        if candidate_stocks.size == 0:
            continue
        counts[idx] = np.unique(candidate_stocks).size
    return counts


def find_similarity_block(
    market_snapshot: pd.DataFrame,
    tolerance: float = TOLERANCE,
    seconds_gaps: int = SECONDS_GAP,
    min_unique_stock_per_sim_block: int = MIN_UNIQUE_STOCK_PER_SIM_BLOCK,
) -> pd.DataFrame:
    if market_snapshot.empty:
        return market_snapshot.iloc[0:0]

    sides = []
    for side in (1, -1):
        df_side = market_snapshot[market_snapshot.side == side].copy()
        if df_side.empty:
            continue
        df_side.sort_values(["time_str", "ts", "stock"], inplace=True, ignore_index=True)
        df_side["block_start_time"] = df_side.groupby("time_str", sort=False)["ts"].transform("first")
        neighbour_counts = _compute_neighbour_counts(df_side, tolerance, seconds_gaps)
        df_side["no_similarity_stock"] = neighbour_counts
        sides.append(df_side)

    if not sides:
        return market_snapshot.iloc[0:0]

    similarity_df = pd.concat(sides, ignore_index=True)
    similarity_df = similarity_df[similarity_df.no_similarity_stock >= min_unique_stock_per_sim_block]
    return similarity_df


def cluster_similarity_block(
    similarity_df: pd.DataFrame,
    distribution: Dict[float, list] = DISTRIBUTION,
    min_segment_length_per_cluster: int = MIN_SEGMENT_LENGTH_PER_CLUSTER,
    fill_value: int = UNKNOWN_SEGMENT_TAG,
    seconds_early: int = SECONDS_EARLY,
) -> pd.DataFrame:
    if similarity_df.empty:
        return similarity_df

    result_frames = []
    offset = 0
    for side in (1, -1):
        df_side = similarity_df[similarity_df["side"] == side]
        if df_side.empty:
            continue
        df_side = df_side.sort_values(["time_str", "stock", "ts"], kind="mergesort")
        for lot, (left, right) in distribution.items():
            mask = (df_side["num_lot"] >= left) & (df_side["num_lot"] <= right)
            if not mask.any():
                offset += 1_000_000
                continue
            temp_df = df_side.loc[mask].copy()
            gap_flags = temp_df["ts"].diff().ge(pd.Timedelta(seconds=seconds_early))
            temp_df["segment_ids"] = gap_flags.cumsum() + offset
            offset += 1_000_000
            temp_df["lot_bucket"] = lot
            segment_lengths = (
                temp_df.groupby("segment_ids", sort=False)["segment_ids"].transform("count").astype(int)
            )
            temp_df["segment_length"] = segment_lengths
            temp_df = temp_df[segment_lengths >= min_segment_length_per_cluster]
            if not temp_df.empty:
                result_frames.append(temp_df)

    if not result_frames:
        return similarity_df.iloc[0:0]

    clustered_df = pd.concat(result_frames, ignore_index=True)
    clustered_df["segment_ids"] = clustered_df["segment_ids"].fillna(fill_value).astype(int)
    clustered_df["segment_length"] = clustered_df["segment_length"].fillna(1).astype(int)
    return clustered_df


def filter_clusters_by_core(
    clustered_df: pd.DataFrame,
    *,
    core_threshold: int = CORE_THRESHOLD,
    sentinel_tag: int = UNKNOWN_SEGMENT_TAG,
) -> pd.DataFrame:
    if clustered_df.empty:
        return clustered_df

    df = clustered_df.copy()
    if core_threshold <= 0:
        return df

    valid_mask = df["segment_ids"] != sentinel_tag
    if not valid_mask.any():
        return df

    cluster_strength = (
        df.loc[valid_mask]
        .groupby("segment_ids", sort=False)["no_similarity_stock"]
        .max()
    )
    keep_ids = cluster_strength[cluster_strength >= core_threshold].index

    if keep_ids.empty:
        return df.loc[df["segment_ids"] == sentinel_tag].copy()

    keep_mask = df["segment_ids"].isin(keep_ids) | (df["segment_ids"] == sentinel_tag)
    return df.loc[keep_mask].copy()


def final_process(
    clustered_df: pd.DataFrame, *, is_live: bool = True, current_time: datetime.datetime | None = None
) -> pd.DataFrame:
    if clustered_df.empty:
        raise ValueError("clustered_df is empty")

    clustered_df = clustered_df.sort_values(["ts"], ignore_index=True)
    latest_time = clustered_df.iloc[-1]["ts"]
    if not is_live:
        latest_time = latest_time.replace(hour=14, minute=30)
    else:
        if current_time is None:
            raise ValueError("current_time is required when is_live is True")
        latest_time = current_time
        max_time = latest_time.replace(hour=14, minute=30)
        if latest_time > max_time:
            latest_time = max_time

    print(f"latest_time: {latest_time}")
    eligible = clustered_df[clustered_df.ts <= latest_time - datetime.timedelta(seconds=SECONDS_GAP)]
    df_ans = eligible.sort_values("ts", ignore_index=True).copy()
    df_ans["initial_ts"] = df_ans["ts"]
    df_ans["ts"] = df_ans["ts"] + datetime.timedelta(seconds=SECONDS_GAP)
    return df_ans


def main() -> None:
    time_start = time.time()

    # Project-relative paths
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    RESULTS_DIR = ROOT / "results"

    weight_path = DATA_DIR / "fubon_fund_weight_long.csv"
    weight = pd.read_csv(weight_path.as_posix()).rename(columns={"volume": "quantity"})

    for run_date in RUN_DATES:
        # Keep these external data paths unchanged as requested
        orderbook = pd.read_csv(
            f"/data/tick_feature/sample_data/udp_orderbook/{run_date}.csv"
        ).rename(columns={"symbol": "stock"})
        trade_data = pd.read_csv(
            f"/data/tick_feature/sample_data/udp_tick/{run_date}.csv"
        ).rename(columns={"symbol": "stock"})

        result_lst = []

        weight_data = weight[weight.Date == run_date].dropna()
        if weight_data.empty:
            print(f"get_arbit_unwind | err: No weight data for {run_date}")
            continue
        stocks = weight_data.stock.values

        orderbook = susbet_data(df=preprocess(orderbook), stocks=stocks).reset_index(drop=True)
        trade_data = susbet_data(df=preprocess(trade_data), stocks=stocks).reset_index(drop=True)

        for hour in (9, 10, 11, 13, 14):
            for minute in range(60):
                current_time = datetime.datetime.strptime(run_date, "%Y-%m-%d").replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
                lookback_min = 5
                try:
                    orderbook_snapshot = get_snapshot_data(
                        orderbook, minutes_ago=lookback_min, current_time=current_time, UTC=True
                    )
                    trade_snapshot = get_snapshot_data(
                        trade_data, minutes_ago=lookback_min, current_time=current_time, UTC=True
                    )

                    if orderbook_snapshot.empty or trade_snapshot.empty:
                        raise ValueError(
                            f"No market snapshot data in the last {lookback_min} minutes"
                        )

                    orderbook_snapshot = orderbook_snapshot.sort_values(
                        ["ts", "stock"], kind="mergesort"
                    )
                    trade_snapshot = trade_snapshot.sort_values(
                        ["ts", "stock"], kind="mergesort"
                    )

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
                        raise ValueError(
                            f"No market snapshot data in the last {lookback_min} minutes"
                        )

                    market_snapshot = wrangle(market_snapshot)
                    similarity_df = find_similarity_block(
                        market_snapshot, min_unique_stock_per_sim_block=MIN_UNIQUE_STOCK_PER_SIM_BLOCK
                    )
                    clustered_df = cluster_similarity_block(
                        similarity_df=similarity_df,
                        distribution=DISTRIBUTION,
                        min_segment_length_per_cluster=MIN_SEGMENT_LENGTH_PER_CLUSTER,
                        seconds_early=SECONDS_EARLY
                    )
                    clustered_df = filter_clusters_by_core(
                        clustered_df,
                        core_threshold=CORE_THRESHOLD,
                    )
                    final_result = final_process(clustered_df=clustered_df, current_time=current_time)
                    result_lst.append(final_result)
                except Exception as ex:
                    print(f"get_arbit_unwind | err: {ex}")

        if not result_lst:
            raise ValueError("Final result is empty")

        df_res = pd.concat(result_lst, ignore_index=True)
        df_res = df_res.drop_duplicates(
            subset=["time_str", "stock", "initial_ts", "ts"], keep="first"
        )

        df_res["arbit_ratio"] = np.where(df_res["side"] == 1, df_res["num_lot"], 0.0)
        df_res["unwind_ratio"] = np.where(df_res["side"] == -1, df_res["num_lot"], 0.0)
        df_res["arbit_value"] = np.where(df_res["side"] == 1, df_res["matched_value"], 0.0)
        df_res["unwind_value"] = np.where(df_res["side"] == -1, df_res["matched_value"], 0.0)

        df_res.to_csv((RESULTS_DIR / f"{run_date}.csv").as_posix(), index=False)
        # df_res.to_csv(f"/home/duc/Desktop/feature_engineer/data/E1VFVN30/{run_date}.csv", index=False)


    print(f"Run time: {time.time() - time_start}")

if __name__ == "__main__":
    main()

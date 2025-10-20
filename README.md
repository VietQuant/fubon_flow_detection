# Fubon Flow Detection

A toolkit for detecting primary flows related to the Fubon ETF by combining high‑frequency market data with fund portfolio information. The project includes:

- Scrapers for daily portfolio composition and assets pages
- A data processing pipeline to produce cleaned weight tables
- A detection/aggregation script that computes intraday flows
- A two‑stage hyperparameter optimizer driven by Optuna

## Project Overview

This repository prepares and analyzes per‑minute flow signals for the Fubon ETF using:

- Scraped fund data (PCF/Assets) to derive daily weights and basket information
- External order book and trade tick data to identify synchronized multi‑stock activity
- A vectorization/detection pipeline that aggregates signals and exports daily CSVs
- An optimizer that tunes both detection and distribution parameters across dates

## Directory Structure

```
.
├─ algorithm/
│  └─ fubon_flow_vectorize.py         # Flow detection + aggregation pipeline
├─ optimizer/
│  └─ fubon_hyperparameter_tuning.py  # Two‑stage Optuna tuning
├─ scrapper/
│  ├─ fubon_daily_report_scraper.py   # Scrapes portfolio composition (PCF) pages
│  ├─ fubon_creation_basket_scraper.py# Scrapes assets/holdings pages
│  └─ exchange_rate_scraper.py        # Adds exchange rates to raw asset files
├─ data_processing/
│  └─ data_transform.py               # Converts scrapper outputs to cleaned tables
├─ evaluation/                        # Notebooks for analysis/visualization
├─ data/
│  ├─ fubon_creation_basket_all.csv   # Example cleaned artifact
│  ├─ fubon_fund_weight.csv           # Example cleaned artifact
│  ├─ fubon_fund_weight_long.csv      # Convenience copy for algorithms
│  ├─ fubon_transactions_real.pkl     # Example input used by notebooks
│  └─ fubon_weight_data/              # Scrapper outputs + cleaned products
│     ├─ portfolio_composition/       # PCF CSVs from daily_report scraper
│     ├─ raw/                         # Raw assets CSVs (used by transform)
│     ├─ creation_basket/             # (optional) alternate assets output
│     └─ cleaned/
│        └─ shift_report_date/        # Weight tables for algorithms
├─ results/
│  └─ <date>.csv                      # Detection outputs by date
└─ scripts/
   └─ run_optimizer_v3_exp1.sh        # Example shell runner for optimizer
```

## Setup Instructions

- Requirements
  - Python 3.10+ recommended
  - Packages: pandas, numpy, requests, beautifulsoup4, lxml, optuna, dask[dataframe] (optional), plotly (optional for notebooks)

- Quickstart (venv)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy requests beautifulsoup4 lxml optuna "dask[dataframe]" plotly
```

## Usage

- Scrape portfolio composition (PCF)

```bash
python scrapper/fubon_daily_report_scraper.py
# Outputs to data/fubon_weight_data/portfolio_composition by default
```

- Scrape assets/holdings (creation basket) and add exchange rates

```bash
# Scrape assets; consider pointing to raw/ so the transform picks them up
python scrapper/fubon_creation_basket_scraper.py --help  # if you add CLI, or edit default

# Add exchange rates to raw files (defaults to data/fubon_weight_data/raw)
python scrapper/exchange_rate_scraper.py
```

- Transform scrapes into cleaned tables

```bash
python data_processing/data_transform.py
# Produces:
# - data/fubon_weight_data/cleaned/fubon_portfolio_composition.csv
# - data/fubon_weight_data/cleaned/fubon_creation_basket_all.csv
# - data/fubon_weight_data/cleaned/shift_report_date/fubon_fund_weight.csv
# - data/fubon_weight_data/cleaned/shift_report_date/fubon_fund_weight_long.csv
# - data/fubon_fund_weight_long.csv (convenience copy)
```

- Run the flow detection pipeline

```bash
# Requires external market data at:
#   /data/tick_feature/sample_data/udp_orderbook/{date}.csv
#   /data/tick_feature/sample_data/udp_tick/{date}.csv
python algorithm/fubon_flow_vectorize.py
# Writes daily outputs to results/{date}.csv
```

- Run the optimizer

```bash
# Default outputs are written under results/tuning
python optimizer/fubon_hyperparameter_tuning.py \
  --stage1-trials 80 \
  --stage2-trials 60 \
  --n-startup-trials 25 \
  --random-seed 42 \
  --output-dir results/tuning \
  --stage1-storage sqlite:///results/tuning/optuna_v3_stage1.db \
  --stage2-storage sqlite:///results/tuning/optuna_v3_stage2.db
```

## Data Description

- Scrapper inputs/outputs
  - Portfolio Composition (PCF): CSVs in `data/fubon_weight_data/portfolio_composition/`
  - Assets/Holdings (raw): CSVs in `data/fubon_weight_data/raw/` (the transform expects files named like `00885_YYYYMMDD.csv`)
  - Exchange rates: Added in place to the raw files by `exchange_rate_scraper.py`

- Cleaned artifacts (produced by the transform)
  - `data/fubon_weight_data/cleaned/fubon_portfolio_composition.csv`: daily PCF summary
  - `data/fubon_weight_data/cleaned/fubon_creation_basket_all.csv`: per‑stock daily metrics
  - `data/fubon_weight_data/cleaned/shift_report_date/fubon_fund_weight.csv`: wide table (Date × stock)
  - `data/fubon_weight_data/cleaned/shift_report_date/fubon_fund_weight_long.csv`: long format weight table
  - `data/fubon_fund_weight_long.csv`: convenience copy used by algorithms

- Detection inputs/outputs
  - Weight table read by detection/optimizer: `data/fubon_fund_weight_long.csv`
  - External market data (unchanged absolute paths):
    - `/data/tick_feature/sample_data/udp_orderbook/{date}.csv`
    - `/data/tick_feature/sample_data/udp_tick/{date}.csv`
  - Detection outputs: `results/{date}.csv`

## Optimization Workflow

The optimizer performs a hierarchical, two‑stage search using Optuna:

- Stage 1 – Detection parameters
  - Tunes: `seconds_early`, `tolerance`, `seconds_gaps`, `min_unique`, `min_segment`, `core_threshold`
  - Uses a fixed baseline distribution (see `STAGE1_DIST_DEFAULTS`)
  - Writes trial records to `results/tuning/fubon_tuning_ver3_stage1.csv` and keeps best in `last_best_v3.json`

- Stage 2 – Distribution parameters
  - Freezes best detection params from Stage 1
  - Tunes: `start`, `stop`, `num`, `step`, `growth`, `int_threshold`, `first_center`
  - Writes trial records to `results/tuning/fubon_tuning_ver3_stage2.csv` and updates `last_best_v3.json`

Internally, both stages evaluate over a set of dates, run the flow pipeline, aggregate net flows, and compute MAE against the provided primary flow series (`data/transaction_data.csv`).

## Notes

- Paths
  - All internal project paths are relative to the repository root.
  - The external market data paths under `/data/tick_feature/sample_data/...` are intentionally left unchanged and must exist on your system for detection/optimization to run.

- Scrapper outputs
  - For `data_processing/data_transform.py` to consume assets correctly, ensure the raw assets files are in `data/fubon_weight_data/raw/`. If you use `fubon_creation_basket_scraper.py`, set its output to that folder or move the files there.

- Optimizer pipeline import
  - The optimizer expects a pipeline module compatible with the functions used (preprocess, wrangle, etc.). If your repo only has `algorithm/fubon_flow_vectorize.py`, update the import in `optimizer/fubon_hyperparameter_tuning.py` accordingly.

- Reproducibility
  - Use `--random-seed` and persist Optuna studies via the provided SQLite URIs under `results/tuning`.

- Notebooks
  - Notebooks in `evaluation/` use relative paths for project data/results and are optional for core workflows.
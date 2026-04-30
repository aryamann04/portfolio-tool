# Portfolio Optimizer

Constrained mean-variance portfolio optimizer with backtesting and an interactive web UI.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)

## Overview

Portfolio Optimizer applies Modern Portfolio Theory to construct optimal allocations across a user-defined universe of stocks and ETFs. It downloads historical price data via yfinance, computes annualized statistics (return, volatility, covariance), and solves a constrained optimization using scipy's SLSQP solver. Two modes are supported: minimize variance subject to a target return, or maximize return subject to a volatility cap. Results are backtested against a benchmark (SPY by default) and visualized as growth curves and an optional efficient frontier.

## Features

- **Two optimization modes** — min-variance (target return) and max-return (target volatility)
- **Global minimum variance** fallback when the target constraint is infeasible
- **Backtesting** — fits weights on a prior T-year window, evaluates realized performance on the following T years; `--today` mode fits and evaluates in-sample
- **Dividend yield integration** — optionally fetches current yields and incorporates them into effective returns
- **Data quality filtering** — drops tickers below a configurable coverage threshold before optimizing
- **Weight bounds** — per-asset min/max allocation constraints
- **Efficient frontier plot** — optional visualization of the full risk/return tradeoff curve
- **Interactive web UI** — Streamlit app wrapping the full CLI

## Project Structure

```
portfolio-tool/
├── mvp.py      # Core optimizer: data download, statistics, SLSQP solver, backtesting, CLI
└── app.py      # Streamlit web UI — calls mvp.py as a subprocess, renders results
```

## Setup

**Prerequisites:** Python 3.9+

```bash
pip install pandas numpy scipy yfinance curl_cffi matplotlib streamlit
```

## Usage

### CLI

```bash
python3 mvp.py \
  --tickers SPLV VPU VIG DVY BND BSV LQD VNQ PFF \
  --years 4.0 \
  --target-return 0.08 \
  --min-weight 0.05 \
  --max-weight 0.20
```

```bash
# Maximize return subject to a 10% volatility cap, show efficient frontier
python3 mvp.py --tickers AAPL MSFT GOOGL BND --years 3 --target-vol 0.10 --show-frontier

# Fit on the most recent 5 years (in-sample), include dividends, save plot
python3 mvp.py --tickers VIG DVY VNQ --years 5 --target-return 0.07 --today --dividend --plot-file out.png

# Custom benchmark and coverage threshold
python3 mvp.py --tickers QQQ ARKK BND --years 4 --target-vol 0.15 --benchmark QQQ --min-coverage 0.95
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | *(required)* | Space-separated list of ticker symbols |
| `--years` | *(required)* | Length of data window in years |
| `--target-return` | — | Annual target return (e.g. `0.08` = 8%); pick this **or** `--target-vol` |
| `--target-vol` | — | Annual target volatility (e.g. `0.10` = 10%) |
| `--benchmark` | `SPY` | Benchmark ticker for comparison |
| `--min-weight` | `0.0` | Minimum per-asset allocation |
| `--max-weight` | `1.0` | Maximum per-asset allocation |
| `--min-coverage` | `0.90` | Drop tickers with fewer than this fraction of trading days |
| `--today` | off | Fit and evaluate on the most recent T years (in-sample) |
| `--dividend` | off | Incorporate dividend yields into return estimates |
| `--show-frontier` | off | Plot the efficient frontier |
| `--plot-file` | — | Save the output plot to this path |

### Web UI

```bash
streamlit run app.py
```

Opens a browser UI with sidebar controls for all parameters. Click **Run Optimization** to execute and view weights, performance tables, and plots inline.

import argparse
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize

TRADING_DAYS = 252


def annualized_stats_from_prices(adj_close: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    rets = adj_close.pct_change().dropna(how="all")
    rets = rets.dropna(axis=1, how="all")
    mu_daily = rets.mean()
    cov_daily = rets.cov()
    mu_ann = mu_daily * TRADING_DAYS
    cov_ann = cov_daily * TRADING_DAYS
    vol_ann = np.sqrt(np.diag(cov_ann))
    vol_ann = pd.Series(vol_ann, index=cov_ann.index, name="ann_vol")
    corr = cov_daily.corr()
    return mu_ann, vol_ann, cov_ann, corr


def portfolio_return(mu_ann: np.ndarray, w: np.ndarray) -> float:
    return float(mu_ann @ w)


def portfolio_vol(cov_ann: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(w.T @ cov_ann @ w))


def weight_bounds(n: int, min_weight: float, max_weight: float):
    return [(min_weight, max_weight)] * n


def solve_gmv(cov_ann: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    n = cov_ann.shape[0]
    x0 = np.ones(n) / n

    def obj(w):
        return float(w.T @ cov_ann @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = weight_bounds(n, min_weight, max_weight)

    res = minimize(
        obj,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 8000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"GMV optimization failed: {res.message}")
    return res.x


def solve_min_variance(mu_ann: np.ndarray, cov_ann: np.ndarray, target_return: float, min_weight: float, max_weight: float) -> np.ndarray:
    n = len(mu_ann)
    x0 = np.ones(n) / n

    def obj(w):
        return float(w.T @ cov_ann @ w)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: (mu_ann @ w) - target_return},
    ]

    bounds = weight_bounds(n, min_weight, max_weight)

    res = minimize(
        obj,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 12000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed (min-variance): {res.message}")
    return res.x


def solve_max_return(mu_ann: np.ndarray, cov_ann: np.ndarray, target_vol: float, min_weight: float, max_weight: float) -> np.ndarray:
    x0 = solve_gmv(cov_ann, min_weight=min_weight, max_weight=max_weight).copy()

    def obj(w):
        return -float(mu_ann @ w)

    target_var = (target_vol ** 2) * (1.0 + 1e-8)

    def vol_constraint(w):
        return target_var - float(w.T @ cov_ann @ w)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": vol_constraint},
    ]

    bounds = weight_bounds(len(mu_ann), min_weight, max_weight)

    res = minimize(
        obj,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 20000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed (max-return): {res.message}")
    return res.x


def download_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        session=requests.Session(impersonate="chrome"),
        group_by="column",
    )
    if df is None or df.empty:
        raise RuntimeError("No data returned from yfinance. Check tickers/network.")

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            adj = df["Adj Close"].copy()
        elif "Close" in lvl0:
            adj = df["Close"].copy()
        else:
            raise RuntimeError("Could not find 'Adj Close' or 'Close' in downloaded data.")
    else:
        if "Adj Close" in df.columns:
            adj = df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in df.columns:
            adj = df[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise RuntimeError("Could not find 'Adj Close' or 'Close' for single ticker download.")

    adj.index = pd.to_datetime(adj.index)
    adj = adj.sort_index()
    return adj


def aligned_returns_from_adj_close(adj_close: pd.DataFrame) -> pd.DataFrame:
    prices = adj_close.dropna(axis=1, how="all")
    prices = prices.dropna(axis=0, how="any")
    return prices.pct_change().dropna()


def format_weights(tickers: list[str], w: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"ticker": tickers, "weight": w})
    out["weight"] = out["weight"].astype(float)
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    return out


def coverage_on_calendar(prices_assets: pd.DataFrame, tickers: list[str], calendar_index: pd.DatetimeIndex) -> dict:
    cov = {}
    expected = len(calendar_index)
    for t in tickers:
        if t not in prices_assets.columns:
            cov[t] = 0.0
            continue
        s = prices_assets[t].reindex(calendar_index)
        available = int(s.notna().sum())
        cov[t] = (available / expected) if expected > 0 else 0.0
    return cov


def filter_by_coverage(prices_assets: pd.DataFrame, tickers: list[str], train_idx: pd.DatetimeIndex, test_idx: pd.DatetimeIndex, min_cov: float):
    cov_train = coverage_on_calendar(prices_assets, tickers, train_idx)
    cov_test = coverage_on_calendar(prices_assets, tickers, test_idx)

    kept = []
    dropped = []
    for t in tickers:
        if cov_train.get(t, 0.0) >= min_cov and cov_test.get(t, 0.0) >= min_cov:
            kept.append(t)
        else:
            dropped.append(t)

    return kept, dropped, cov_train, cov_test


def fetch_dividend_yields(tickers: list[str]) -> pd.Series:
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t, session=requests.Session(impersonate="chrome")).info
            y = info.get("dividendYield", None)
            out[t] = float(y) / 100 if y is not None else np.nan
        except Exception:
            out[t] = np.nan
    return pd.Series(out, name="div_yield").astype(float)


def effective_yield_portfolio(div_yield: pd.Series, tickers: list[str], w: np.ndarray) -> float:
    y = div_yield.reindex(tickers).fillna(0.0).values
    return float(y @ w)


def max_return_under_bounds(mu_ann: np.ndarray, min_weight: float, max_weight: float) -> float:
    n = len(mu_ann)

    def obj(w):
        return -float(mu_ann @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = weight_bounds(n, min_weight, max_weight)
    x0 = np.ones(n) / n
    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 12000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Max-return (bounds-only) optimization failed: {res.message}")
    return float(mu_ann @ res.x)


def efficient_frontier(mu_ann: np.ndarray, cov_ann: np.ndarray, min_weight: float, max_weight: float, n_points: int = 50) -> pd.DataFrame:
    n = len(mu_ann)

    w_gmv = solve_gmv(cov_ann, min_weight=min_weight, max_weight=max_weight)
    r_gmv = float(mu_ann @ w_gmv)
    v_gmv = float(np.sqrt(w_gmv.T @ cov_ann @ w_gmv))

    r_max = max_return_under_bounds(mu_ann, min_weight=min_weight, max_weight=max_weight)
    if r_max <= r_gmv + 1e-10:
        return pd.DataFrame({"ann_vol": [v_gmv], "ann_return": [r_gmv]})

    targets = np.linspace(r_gmv, r_max, n_points)

    vols = []
    rets = []

    x0 = w_gmv.copy()
    bounds = weight_bounds(n, min_weight, max_weight)

    for tr in targets:
        def obj(w):
            return float(w.T @ cov_ann @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, tr=tr: (mu_ann @ w) - tr},
        ]

        res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 12000, "ftol": 1e-12})
        if res.success:
            w = res.x
            rets.append(float(mu_ann @ w))
            vols.append(float(np.sqrt(w.T @ cov_ann @ w)))
            x0 = w

    df = pd.DataFrame({"ann_vol": vols, "ann_return": rets})
    df = df.dropna().sort_values("ann_vol").reset_index(drop=True)
    return df


def run_pipeline(
    tickers_in: list[str],
    bench: str,
    years: float,
    min_weight: float,
    max_weight: float,
    min_coverage: float,
    today: bool,
    dividend_mode: bool,
    show_frontier: bool,
    target_return: float | None,
    target_vol: float | None,
    plot_file: str | None,
):
    end_dt = datetime.today()

    if today:
        start_dt = end_dt - timedelta(days=int(years * 365.25))
        start = start_dt.strftime("%Y-%m-%d")
        end = end_dt.strftime("%Y-%m-%d")

        prices_assets_all = download_adj_close(tickers_in, start, end)
        prices_bench_all = download_adj_close([bench], start, end)

        bench_series = prices_bench_all[bench].dropna()
        if bench_series.empty:
            raise RuntimeError("Benchmark has no data in requested range.")
        cal_all = bench_series.index

        train_idx = cal_all
        test_idx = cal_all

        kept, dropped, cov_train, _ = filter_by_coverage(prices_assets_all, tickers_in, train_idx, test_idx, min_coverage)
        if dropped:
            print("Dropped tickers due to insufficient data for required window (coverage threshold):")
            for t in sorted(dropped):
                print(f"{t}\tcoverage={cov_train.get(t, 0.0):.3f}")

        tickers = kept
        if len(tickers) == 0:
            raise RuntimeError("No tickers left after filtering. Try lowering --min-coverage or check yfinance connectivity.")

        n_assets = len(tickers)
        if n_assets * min_weight > 1.0 + 1e-12:
            raise ValueError(f"Infeasible min-weight after filtering: {n_assets} assets × {min_weight:.6f} > 1")
        if n_assets * max_weight < 1.0 - 1e-12:
            raise ValueError(f"Infeasible max-weight after filtering: {n_assets} assets × {max_weight:.6f} < 1")

        train_prices = prices_assets_all[tickers].reindex(train_idx).dropna(axis=0, how="any")
        if train_prices.shape[0] < 2:
            raise RuntimeError("Not enough overlapping data after alignment. Lower --min-coverage or use different tickers.")

        mu_ann_s, vol_ann_s, cov_ann_df, corr_df = annualized_stats_from_prices(train_prices)
        mu_ann_use = mu_ann_s.copy()

        div_y = None
        if dividend_mode:
            div_y = fetch_dividend_yields(tickers)
            mu_eff = mu_ann_s.add(div_y.fillna(0.0))
            mu_ann_use = mu_eff
            stats_table = pd.DataFrame(
                {"ann_return": mu_ann_s, "ann_vol": vol_ann_s, "div_yield": div_y, "effective_return": mu_eff}
            ).loc[tickers]
            print("\n==================== Asset Stats (annualized, incl dividends) ====================")
            print(stats_table.sort_values("effective_return", ascending=False).to_string(float_format=lambda x: f"{x:,.6f}"))
        else:
            asset_table = pd.DataFrame({"ann_return": mu_ann_s, "ann_vol": vol_ann_s}).sort_values("ann_return", ascending=False)
            print("\n==================== Asset Stats (annualized) ====================")
            print(asset_table.to_string(float_format=lambda x: f"{x:,.6f}"))

        mu_ann = mu_ann_use.values
        cov_ann = cov_ann_df.values

        frontier_path = None
        if show_frontier:
            df_frontier = efficient_frontier(mu_ann, cov_ann, min_weight=min_weight, max_weight=max_weight, n_points=50)
            if plot_file:
                base, ext = os.path.splitext(plot_file)
                frontier_path = f"{base}_frontier{ext if ext else '.png'}"
                plt.figure()
                plt.plot(df_frontier["ann_vol"].values, df_frontier["ann_return"].values, lw=2)
                plt.xlabel("Annualized Volatility")
                plt.ylabel("Annualized Return")
                plt.title(f"Efficient Frontier ({'effective' if dividend_mode else 'price-only'})\nlast {years}y (today fit)")
                plt.grid(True, linewidth=0.3)
                plt.savefig(frontier_path, bbox_inches="tight", dpi=180)
                plt.close()
            else:
                plt.figure()
                plt.plot(df_frontier["ann_vol"].values, df_frontier["ann_return"].values, lw=2)
                plt.xlabel("Annualized Volatility")
                plt.ylabel("Annualized Return")
                plt.title(f"Efficient Frontier ({'effective' if dividend_mode else 'price-only'})\nlast {years}y (today fit)")
                plt.grid(True, linewidth=0.3)

        gmv_w = solve_gmv(cov_ann, min_weight=min_weight, max_weight=max_weight)
        gmv_vol = portfolio_vol(cov_ann, gmv_w)

        if target_vol is not None and target_vol < gmv_vol - 1e-8:
            raise ValueError(f"Target vol {target_vol:.6f} infeasible; GMV vol is {gmv_vol:.6f}")

        if target_return is not None:
            w = solve_min_variance(mu_ann, cov_ann, target_return, min_weight=min_weight, max_weight=max_weight)
            mode = f"Min-variance s.t. return >= {target_return:.2%}"
        else:
            w = solve_max_return(mu_ann, cov_ann, target_vol, min_weight=min_weight, max_weight=max_weight)
            mode = f"Max-return s.t. vol <= {target_vol:.2%}"

        assets_prices = prices_assets_all[tickers].reindex(cal_all)
        bench_prices = prices_bench_all[[bench]].reindex(cal_all)

        all_prices = assets_prices.join(bench_prices, how="inner")
        assets_prices = all_prices[tickers].dropna(axis=0, how="any")
        bench_prices = all_prices[[bench]].reindex(assets_prices.index).dropna(axis=0, how="any")

        rets_assets = aligned_returns_from_adj_close(assets_prices)
        rets_bench = aligned_returns_from_adj_close(bench_prices).reindex(rets_assets.index).dropna()
        rets_assets = rets_assets.loc[rets_bench.index]

        port_daily = pd.Series(rets_assets.values @ w, index=rets_assets.index, name="Portfolio")
        bench_daily = rets_bench[bench].rename(bench)

        port_growth = (1.0 + port_daily).cumprod()
        bench_growth = (1.0 + bench_daily).cumprod()

        port_ret = portfolio_return(mu_ann, w)
        port_vol = portfolio_vol(cov_ann, w)

        port_real_ret_price = port_daily.mean() * TRADING_DAYS
        port_real_vol = port_daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
        bench_real_ret_price = bench_daily.mean() * TRADING_DAYS
        bench_real_vol = bench_daily.std(ddof=1) * np.sqrt(TRADING_DAYS)

        if dividend_mode:
            port_div_y = effective_yield_portfolio(div_y, tickers, w)
            try:
                info_b = yf.Ticker(bench, session=requests.Session(impersonate="chrome")).info
                yb = info_b.get("dividendYield", None)
                bench_div_y = float(yb) / 100 if yb is not None else 0.0
            except Exception:
                bench_div_y = 0.0
            port_real_ret_eff = port_real_ret_price + port_div_y
            bench_real_ret_eff = bench_real_ret_price + bench_div_y
        else:
            port_div_y = 0.0
            bench_div_y = 0.0
            port_real_ret_eff = port_real_ret_price
            bench_real_ret_eff = bench_real_ret_price

        print("\n==================== Inputs ====================")
        print(f"Tickers used: {tickers}")
        print(f"Window: {cal_all.min().date().isoformat()} to {cal_all.max().date().isoformat()}  (~{years} years)")
        print(f"Mode:    {mode}")
        print(f"Min weight: {min_weight:.4f} (long-only)")
        print(f"Max weight: {max_weight:.4f} (long-only)")
        print(f"Min coverage: {min_coverage:.2f}")
        print("Fit weights: TODAY (using last T years)")
        print(f"Returns used in optimization: {'effective (price + dividend yield)' if dividend_mode else 'price-only'}")

        print("\n==================== Correlation Matrix ====================")
        print(corr_df.to_string(float_format=lambda x: f"{x:,.3f}"))

        print("\n==================== Optimal Weights ====================")
        print(format_weights(tickers, w).to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

        print("\n==================== Ex-ante Portfolio (from sample moments) ====================")
        print(f"Expected ann return: {port_ret:.2%}")
        print(f"Expected ann vol:    {port_vol:.2%}")

        if dividend_mode:
            print("\n==================== Realized (in-window, buy & hold) ====================")
            print(f"Portfolio ann return (price):     {port_real_ret_price:.2%}")
            print(f"Portfolio div yield (weighted):   {port_div_y:.2%}")
            print(f"Portfolio ann return (effective): {port_real_ret_eff:.2%}")
            print(f"Portfolio ann vol:                {port_real_vol:.2%}")
            print(f"{bench} ann return (price):         {bench_real_ret_price:.2%}")
            print(f"{bench} div yield:                  {bench_div_y:.2%}")
            print(f"{bench} ann return (effective):     {bench_real_ret_eff:.2%}")
            print(f"{bench} ann vol:                   {bench_real_vol:.2%}")
        else:
            print("\n==================== Realized (in-window, buy & hold) ====================")
            print(f"Portfolio ann return: {port_real_ret_eff:.2%}")
            print(f"Portfolio ann vol:    {port_real_vol:.2%}")
            print(f"{bench} ann return:       {bench_real_ret_eff:.2%}")
            print(f"{bench} ann vol:          {bench_real_vol:.2%}")

        plt.figure()
        plt.plot(port_growth.index, port_growth.values, label="Optimized Portfolio (Adj Close total return)")
        plt.plot(bench_growth.index, bench_growth.values, label=f"{bench} (Adj Close total return)")
        plt.title(
            f"Growth of $1: Optimized Portfolio vs {bench}\n{mode}\nWeights fit on last {years}y window (today)"
        )
        plt.xlabel("Date")
        plt.ylabel("Cumulative Growth")
        plt.legend()
        plt.grid(True, linewidth=0.3)

        if plot_file:
            plt.savefig(plot_file, bbox_inches="tight", dpi=180)
            print(f"\nSaved plot to: {plot_file}")
            if show_frontier:
                base, ext = os.path.splitext(plot_file)
                frontier_path = f"{base}_frontier{ext if ext else '.png'}"
                print(f"Saved frontier plot to: {frontier_path}")
        else:
            plt.show()

        return

    mid_dt = end_dt - timedelta(days=int(years * 365.25))
    start_dt = end_dt - timedelta(days=int(2.0 * years * 365.25))

    start = start_dt.strftime("%Y-%m-%d")
    mid = mid_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    prices_assets_all = download_adj_close(tickers_in, start, end)
    prices_bench_all = download_adj_close([bench], start, end)

    bench_series = prices_bench_all[bench].dropna()
    if bench_series.empty:
        raise RuntimeError("Benchmark has no data in requested range.")

    cal_all = bench_series.index
    mid_ts_raw = pd.to_datetime(mid)
    mid_candidates = cal_all[cal_all <= mid_ts_raw]
    if len(mid_candidates) == 0:
        raise RuntimeError("Could not determine midpoint trading day on benchmark calendar.")
    mid_ts = mid_candidates.max()

    train_idx = cal_all[(cal_all >= cal_all.min()) & (cal_all <= mid_ts)]
    test_idx = cal_all[(cal_all > mid_ts) & (cal_all <= cal_all.max())]

    if len(train_idx) < 2 or len(test_idx) < 2:
        raise RuntimeError("Not enough benchmark trading days to form training/test windows.")

    kept, dropped, cov_train, cov_test = filter_by_coverage(prices_assets_all, tickers_in, train_idx, test_idx, min_coverage)

    if dropped:
        print("Dropped tickers due to insufficient data for required windows (coverage threshold):")
        for t in sorted(dropped):
            print(f"{t}\ttrain_coverage={cov_train.get(t, 0.0):.3f}\ttest_coverage={cov_test.get(t, 0.0):.3f}")

    tickers = kept
    if len(tickers) == 0:
        raise RuntimeError("No tickers left after filtering. Try lowering --min-coverage or check yfinance connectivity.")

    n_assets = len(tickers)
    if n_assets * min_weight > 1.0 + 1e-12:
        raise ValueError(f"Infeasible min-weight after filtering: {n_assets} assets × {min_weight:.6f} > 1")
    if n_assets * max_weight < 1.0 - 1e-12:
        raise ValueError(f"Infeasible max-weight after filtering: {n_assets} assets × {max_weight:.6f} < 1")

    train_prices = prices_assets_all[tickers].reindex(train_idx).dropna(axis=0, how="any")
    if train_prices.shape[0] < 2:
        raise RuntimeError("Not enough overlapping training data after alignment. Lower --min-coverage or use different tickers.")

    mu_ann_s, vol_ann_s, cov_ann_df, corr_df = annualized_stats_from_prices(train_prices)
    mu_ann_use = mu_ann_s.copy()

    div_y = None
    if dividend_mode:
        div_y = fetch_dividend_yields(tickers)
        mu_eff = mu_ann_s.add(div_y.fillna(0.0))
        mu_ann_use = mu_eff
        stats_table = pd.DataFrame(
            {"ann_return": mu_ann_s, "ann_vol": vol_ann_s, "div_yield": div_y, "effective_return": mu_eff}
        ).loc[tickers]
        print("\n==================== Training Asset Stats (annualized, incl dividends) ====================")
        print(stats_table.sort_values("effective_return", ascending=False).to_string(float_format=lambda x: f"{x:,.6f}"))
    else:
        asset_table = pd.DataFrame({"ann_return": mu_ann_s, "ann_vol": vol_ann_s}).sort_values("ann_return", ascending=False)
        print("\n==================== Training Asset Stats (annualized) ====================")
        print(asset_table.to_string(float_format=lambda x: f"{x:,.6f}"))

    mu_ann = mu_ann_use.values
    cov_ann = cov_ann_df.values

    if show_frontier:
        df_frontier = efficient_frontier(mu_ann, cov_ann, min_weight=min_weight, max_weight=max_weight, n_points=50)
        if plot_file:
            base, ext = os.path.splitext(plot_file)
            frontier_path = f"{base}_frontier{ext if ext else '.png'}"
            plt.figure()
            plt.plot(df_frontier["ann_vol"].values, df_frontier["ann_return"].values, lw=2)
            plt.xlabel("Annualized Volatility")
            plt.ylabel("Annualized Return")
            plt.title(
                f"Efficient Frontier ({'effective' if dividend_mode else 'price-only'})\n"
                f"Train: {train_idx.min().date().isoformat()} to {train_idx.max().date().isoformat()}"
            )
            plt.grid(True, linewidth=0.3)
            plt.savefig(frontier_path, bbox_inches="tight", dpi=180)
            plt.close()
        else:
            plt.figure()
            plt.plot(df_frontier["ann_vol"].values, df_frontier["ann_return"].values, lw=2)
            plt.xlabel("Annualized Volatility")
            plt.ylabel("Annualized Return")
            plt.title(
                f"Efficient Frontier ({'effective' if dividend_mode else 'price-only'})\n"
                f"Train: {train_idx.min().date().isoformat()} to {train_idx.max().date().isoformat()}"
            )
            plt.grid(True, linewidth=0.3)

    gmv_w = solve_gmv(cov_ann, min_weight=min_weight, max_weight=max_weight)
    gmv_vol = portfolio_vol(cov_ann, gmv_w)

    if target_vol is not None and target_vol < gmv_vol - 1e-8:
        raise ValueError(f"Target vol {target_vol:.6f} infeasible in training window; GMV vol is {gmv_vol:.6f}")

    if target_return is not None:
        w = solve_min_variance(mu_ann, cov_ann, target_return, min_weight=min_weight, max_weight=max_weight)
        mode = f"Min-variance s.t. return >= {target_return:.2%}"
    else:
        w = solve_max_return(mu_ann, cov_ann, target_vol, min_weight=min_weight, max_weight=max_weight)
        mode = f"Max-return s.t. vol <= {target_vol:.2%}"

    test_prices_assets = prices_assets_all[tickers].reindex(test_idx)
    test_prices_bench = prices_bench_all[[bench]].reindex(test_idx)

    test_all = test_prices_assets.join(test_prices_bench, how="inner")
    test_prices_assets = test_all[tickers].dropna(axis=0, how="any")
    test_prices_bench = test_all[[bench]].reindex(test_prices_assets.index).dropna(axis=0, how="any")

    test_rets_assets = aligned_returns_from_adj_close(test_prices_assets)
    test_rets_bench = aligned_returns_from_adj_close(test_prices_bench).reindex(test_rets_assets.index).dropna()
    test_rets_assets = test_rets_assets.loc[test_rets_bench.index]

    port_daily = pd.Series(test_rets_assets.values @ w, index=test_rets_assets.index, name="Portfolio")
    bench_daily = test_rets_bench[bench].rename(bench)

    port_growth = (1.0 + port_daily).cumprod()
    bench_growth = (1.0 + bench_daily).cumprod()

    train_port_ret = portfolio_return(mu_ann, w)
    train_port_vol = portfolio_vol(cov_ann, w)

    port_real_ret_price = port_daily.mean() * TRADING_DAYS
    port_real_vol = port_daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    bench_real_ret_price = bench_daily.mean() * TRADING_DAYS
    bench_real_vol = bench_daily.std(ddof=1) * np.sqrt(TRADING_DAYS)

    if dividend_mode:
        port_div_y = effective_yield_portfolio(div_y, tickers, w)
        try:
            info_b = yf.Ticker(bench, session=requests.Session(impersonate="chrome")).info
            yb = info_b.get("dividendYield", None)
            bench_div_y = float(yb) if yb is not None else 0.0
        except Exception:
            bench_div_y = 0.0
        port_real_ret_eff = port_real_ret_price + port_div_y
        bench_real_ret_eff = bench_real_ret_price + bench_div_y
    else:
        port_div_y = 0.0
        bench_div_y = 0.0
        port_real_ret_eff = port_real_ret_price
        bench_real_ret_eff = bench_real_ret_price

    print("\n==================== Inputs ====================")
    print(f"Tickers used: {tickers}")
    print(f"Training window: {train_idx.min().date().isoformat()} to {train_idx.max().date().isoformat()}  (~{years} years)")
    print(f"Test window:     {test_idx.min().date().isoformat()} to {test_idx.max().date().isoformat()}  (~{years} years)")
    print(f"Mode:    {mode}")
    print(f"Min weight: {min_weight:.4f} (long-only)")
    print(f"Max weight: {max_weight:.4f} (long-only)")
    print(f"Min coverage: {min_coverage:.2f}")
    print("Fit weights: AS-OF T years ago (train prior window), then backtest forward")
    print(f"Returns used in optimization: {'effective (price + dividend yield)' if dividend_mode else 'price-only'}")

    print("\n==================== Training Correlation Matrix ====================")
    print(corr_df.to_string(float_format=lambda x: f"{x:,.3f}"))

    print("\n==================== Optimal Weights (fit at t = T years ago) ====================")
    print(format_weights(tickers, w).to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

    print("\n==================== Ex-ante Portfolio (training moments) ====================")
    print(f"Expected ann return: {train_port_ret:.2%}")
    print(f"Expected ann vol:    {train_port_vol:.2%}")

    if dividend_mode:
        print("\n==================== Realized Backtest (T years ago to today) ====================")
        print(f"Portfolio ann return (price):     {port_real_ret_price:.2%}")
        print(f"Portfolio div yield (weighted):   {port_div_y:.2%}")
        print(f"Portfolio ann return (effective): {port_real_ret_eff:.2%}")
        print(f"Portfolio ann vol:                {port_real_vol:.2%}")
        print(f"{bench} ann return (price):         {bench_real_ret_price:.2%}")
        print(f"{bench} div yield:                  {bench_div_y:.2%}")
        print(f"{bench} ann return (effective):     {bench_real_ret_eff:.2%}")
        print(f"{bench} ann vol:                   {bench_real_vol:.2%}")
    else:
        print("\n==================== Realized Backtest (T years ago to today) ====================")
        print(f"Portfolio ann return: {port_real_ret_eff:.2%}")
        print(f"Portfolio ann vol:    {port_real_vol:.2%}")
        print(f"{bench} ann return:       {bench_real_ret_eff:.2%}")
        print(f"{bench} ann vol:          {bench_real_vol:.2%}")

    plt.figure()
    plt.plot(port_growth.index, port_growth.values, label="Optimized Portfolio (Adj Close total return)")
    plt.plot(bench_growth.index, bench_growth.values, label=f"{bench} (Adj Close total return)")
    plt.title(f"Growth of $1: Optimized Portfolio vs {bench}\n{mode}\nWeights fit on prior {years}y window")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth")
    plt.legend()
    plt.grid(True, linewidth=0.3)

    if plot_file:
        plt.savefig(plot_file, bbox_inches="tight", dpi=180)
        print(f"\nSaved plot to: {plot_file}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--years", type=float, required=True)
    ap.add_argument("--target-return", type=float, default=None)
    ap.add_argument("--target-vol", type=float, default=None)
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--plot-file", default=None)
    ap.add_argument("--min-weight", type=float, default=0.0)
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--dividend", action="store_true")
    ap.add_argument("--show-frontier", action="store_true")
    args = ap.parse_args()

    if (args.target_return is None) == (args.target_vol is None):
        print("Error: Provide exactly one of --target-return or --target-vol.", file=sys.stderr)
        sys.exit(2)

    tickers_in = [t.upper() for t in args.tickers]
    bench = args.benchmark.upper()

    if args.min_weight < 0:
        raise ValueError("--min-weight must be >= 0")
    if args.max_weight <= 0 or args.max_weight > 1:
        raise ValueError("--max-weight must be in (0, 1]")
    if args.min_weight > args.max_weight:
        raise ValueError("--min-weight must be <= --max-weight")
    if args.min_coverage <= 0 or args.min_coverage > 1:
        raise ValueError("--min-coverage must be in (0, 1]")

    run_pipeline(
        tickers_in=tickers_in,
        bench=bench,
        years=args.years,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        min_coverage=args.min_coverage,
        today=args.today,
        dividend_mode=args.dividend,
        show_frontier=args.show_frontier,
        target_return=args.target_return,
        target_vol=args.target_vol,
        plot_file=args.plot_file,
    )


if __name__ == "__main__":
    main()

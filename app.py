import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")


def build_command(
    script_path: str,
    tickers: list[str],
    years: float,
    mode: str,
    target_return: float | None,
    target_vol: float | None,
    benchmark: str,
    min_weight: float,
    max_weight: float,
    min_coverage: float,
    today: bool,
    dividend: bool,
    show_frontier: bool,
    plot_file: str,
) -> list[str]:
    cmd = ["python3", script_path]
    cmd += ["--tickers"] + tickers
    cmd += ["--years", str(years)]
    cmd += ["--benchmark", benchmark]
    cmd += ["--min-weight", str(min_weight)]
    cmd += ["--max-weight", str(max_weight)]
    cmd += ["--min-coverage", str(min_coverage)]
    cmd += ["--plot-file", plot_file]

    if today:
        cmd += ["--today"]
    if dividend:
        cmd += ["--dividend"]
    if show_frontier:
        cmd += ["--show-frontier"]

    if mode == "Target Return (min-variance)":
        cmd += ["--target-return", str(target_return)]
    else:
        cmd += ["--target-vol", str(target_vol)]

    return cmd


def run_optimizer(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def try_parse_weights(stdout: str) -> pd.DataFrame | None:
    m = re.search(r"=+\s*Optimal Weights\s*=+\n(.*?)(\n=+|\Z)", stdout, flags=re.S)
    if not m:
        return None
    block = m.group(1).strip()
    if not block:
        return None

    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    header = lines[0].strip().lower()
    if "ticker" not in header or "weight" not in header:
        return None

    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        t = parts[0].strip()
        try:
            w = float(parts[1])
        except Exception:
            continue
        rows.append((t, w))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["ticker", "weight"]).sort_values("weight", ascending=False).reset_index(drop=True)
    return df


st.title("Portfolio Optimizer")

with st.sidebar:
    st.header("Script")
    default_script = "mvp.py"
    script_path = st.text_input("Optimizer script path", value=default_script, help="Path to your existing CLI script (e.g. mvp.py).")

    st.divider()
    st.header("Universe")

    tickers_str = st.text_area(
        "Tickers (space or comma separated)",
        value="SPLV VPU VIG DVY BND BSV LQD VNQ PFF",
        height=80,
    )
    tickers = [t.strip().upper() for t in re.split(r"[,\s]+", tickers_str) if t.strip()]

    years = st.number_input("Years (T)", min_value=0.25, max_value=30.0, value=4.0, step=0.25)
    benchmark = st.text_input("Benchmark", value="SPY").strip().upper()

    st.divider()
    st.header("Constraints")

    min_weight = st.number_input("Min weight", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.4f")
    max_weight = st.number_input("Max weight", min_value=0.01, max_value=1.0, value=0.20, step=0.01, format="%.4f")
    min_coverage = st.slider("Min coverage", min_value=0.50, max_value=1.00, value=0.90, step=0.01)

    st.divider()
    st.header("Mode")

    mode = st.radio("Optimization mode", ["Target Return (min-variance)", "Target Vol (max-return)"], index=0)
    target_return = None
    target_vol = None
    if mode == "Target Return (min-variance)":
        target_return = st.number_input("Target return (decimal)", min_value=-0.50, max_value=1.50, value=0.08, step=0.01, format="%.4f")
    else:
        target_vol = st.number_input("Target vol (decimal)", min_value=0.01, max_value=2.00, value=0.10, step=0.01, format="%.4f")

    st.divider()
    st.header("Options")

    today = st.checkbox("--today (fit using last T years)", value=True)
    dividend = st.checkbox("--dividend (effective returns = price + div yield)", value=True)
    show_frontier = st.checkbox("--show-frontier (plot efficient frontier)", value=False)

    run_btn = st.button("Run Optimization", type="primary", use_container_width=True)

col_left, col_right = st.columns([1.99, 0.01])

with col_left:
    st.subheader("Outputs")

    if run_btn:
        if not tickers:
            st.error("Please enter at least one ticker.")
            st.stop()

        if min_weight > max_weight:
            st.error("min_weight must be <= max_weight.")
            st.stop()

        if not Path(script_path).exists():
            st.error(f"Script not found: {script_path}")
            st.stop()

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            plot_path = td_path / "result.png"

            cmd = build_command(
                script_path=script_path,
                tickers=tickers,
                years=float(years),
                mode=mode,
                target_return=float(target_return) if target_return is not None else None,
                target_vol=float(target_vol) if target_vol is not None else None,
                benchmark=benchmark,
                min_weight=float(min_weight),
                max_weight=float(max_weight),
                min_coverage=float(min_coverage),
                today=bool(today),
                dividend=bool(dividend),
                show_frontier=bool(show_frontier),
                plot_file=str(plot_path),
            )

            st.caption("Command executed")
            st.code(" ".join(shlex.quote(x) for x in cmd), language="bash")

            with st.spinner("Running..."):
                rc, out, err = run_optimizer(cmd)

            if rc != 0:
                st.error("Optimizer exited with an error.")
                if err.strip():
                    st.code(err, language="text")
                if out.strip():
                    st.code(out, language="text")
                st.stop()

            weights_df = try_parse_weights(out)
            if weights_df is not None:
                st.markdown("### Optimal Weights")
                st.dataframe(weights_df, use_container_width=True, hide_index=True)

            st.markdown("### Full Report (stdout)")
            st.code(out if out.strip() else "(no stdout)", language="text")

            if err.strip():
                with st.expander("stderr (warnings / debug)"):
                    st.code(err, language="text")

            if plot_path.exists():
                st.markdown("### Portfolio vs Benchmark")
                st.image(str(plot_path), use_container_width=True)
            else:
                st.warning("Plot file was not created. (Did the script receive --plot-file?)")

            frontier_path = td_path / "result_frontier.png"
            if show_frontier and frontier_path.exists():
                st.markdown("### Efficient Frontier")
                st.image(str(frontier_path), use_container_width=True)
            elif show_frontier:
                st.info("Frontier plot not found. Your script should save it as <plot>_frontier.<ext>.")



st.markdown("---")
st.caption(
"""
Run with `streamlit run app.py`
**Workflow**
1. Enter tickers, years, constraints, and choose a mode (target return or target vol).
2. Optional flags:
   - **--today**: fit weights on last *T* years (build portfolio “today”)
   - **(default, no --today)**: fit weights on prior *T* years and evaluate forward (reduces lookahead bias)
   - **--dividend**: uses **effective return** (price return + dividend yield) in optimization and reporting
   - **--show-frontier**: plots the efficient frontier implied by your constraints
"""
)

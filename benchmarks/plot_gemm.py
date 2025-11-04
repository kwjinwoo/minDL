#!/usr/bin/env python3
"""
plot_gemm.py — Visualize miniDL GEMM benchmarks (native vs simd)

Usage examples
--------------
# Parse two raw bench logs and write three figures to ./plots
python plot_gemm.py \
  --input bench_native.log:backend=native bench_simd.log:backend=simd \
  --outdir ./plots \
  --make bar line speedup

# Or read a CSV you've exported (columns: backend,B,M,K,N,threads,gflops,best,avg)
python plot_gemm.py --input results.csv --outdir ./plots --make bar line speedup

# If your log doesn't print backend=..., you can override with :backend=<label>
# per file as shown above.
#
# Figures written:
#   plots/gemm_bar.png      — simple bar chart comparing backends on the most common shape
#   plots/gemm_line.png     — scaling curve (square sizes where M=K=N)
#   plots/gemm_speedup.png  — speedup (baseline via --baseline, default: native)

Notes
-----
- Designed for miniDL bench output like:
    [miniDL GEMM bench]
    B=16 M=256 K=256 N=256  warmup=1 iters=5  dtype=f32  threads=8  backend=auto
    ...
      best: 0.182304 s  (2.94 GFLOP/s)
      avg : 0.192746 s  (2.79 GFLOP/s)
- If 'backend=' is missing or 'auto', you can supply a label via ":backend=<name>"
  after the file path in --input.
"""

from __future__ import annotations

import argparse
import re
from typing import List, Optional, Tuple, Dict, Any
import os
import pandas as pd
import matplotlib.pyplot as plt


HEADER_RE = re.compile(
    r"B=(?P<B>\d+)\s+M=(?P<M>\d+)\s+K=(?P<K>\d+)\s+N=(?P<N>\d+).*?threads=(?P<threads>\d+)(?:.*?backend=(?P<backend>\w+))?",
    re.IGNORECASE,
)
BEST_RE = re.compile(
    r"best:\s+(?P<best>[\d.]+)\s+s\s+\((?P<gflops>[\d.]+)\s+GFLOP/s\)",
    re.IGNORECASE,
)
AVG_RE = re.compile(
    r"avg\s*:\s+(?P<avg>[\d.]+)\s+s\s+\((?P<gflops_avg>[\d.]+)\s+GFLOP/s\)",
    re.IGNORECASE,
)


def _parse_input_token(token: str) -> Tuple[str, Dict[str, str]]:
    """
    Split 'path[:k=v[:k=v...]]' to (path, {k:v}).
    """
    parts = token.split(":")
    path = parts[0]
    meta: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            meta[k.strip()] = v.strip()
    return path, meta


def parse_log_file(path: str, override_backend: Optional[str] = None) -> pd.DataFrame:
    """
    Parse a single miniDL bench log file.
    Returns DataFrame with columns:
      backend,B,M,K,N,threads,gflops,best,avg,source
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    rows: List[Dict[str, Any]] = []

    # We look for each header block and then the nearest 'best'/'avg' following it.
    pos = 0
    while True:
        m = HEADER_RE.search(text, pos)
        if not m:
            break
        B = int(m.group("B"))
        M = int(m.group("M"))
        K = int(m.group("K"))
        N = int(m.group("N"))
        threads = int(m.group("threads"))
        backend = m.group("backend") or "auto"
        if override_backend:
            backend = override_backend

        # Find the next result lines after the header
        tail = text[m.end() :]
        mbest = BEST_RE.search(tail)
        mavg = AVG_RE.search(tail)

        row: Dict[str, Any] = dict(
            backend=backend,
            B=B,
            M=M,
            K=K,
            N=N,
            threads=threads,
            gflops=float(mbest.group("gflops")) if mbest else float("nan"),
            best=float(mbest.group("best")) if mbest else float("nan"),
            avg=float(mavg.group("avg")) if mavg else float("nan"),
            gflops_avg=float(mavg.group("gflops_avg")) if mavg else float("nan"),
            source=path,
        )
        rows.append(row)
        pos = m.end()

    if not rows:
        raise ValueError(f"No benchmark blocks found in {path}. Check the format.")

    return pd.DataFrame(rows)


def parse_csv_file(path: str) -> pd.DataFrame:
    """
    Read a CSV with at least: backend,M,K,N,threads,gflops
    Optional: B,best,avg,gflops_avg
    """
    df = pd.read_csv(path)
    required = {"backend", "M", "K", "N", "threads", "gflops"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing required columns: {missing}")
    if "B" not in df.columns:
        df["B"] = 1
    for col in ["best", "avg", "gflops_avg"]:
        if col not in df.columns:
            df[col] = float("nan")
    df["source"] = path
    return df[
        list(
            [
                "backend",
                "B",
                "M",
                "K",
                "N",
                "threads",
                "gflops",
                "best",
                "avg",
                "gflops_avg",
                "source",
            ]
        )
    ]


def load_data(inputs: List[str]) -> pd.DataFrame:
    """
    inputs tokens may be log or csv paths.
    - If extension is .csv -> parse_csv_file
    - Else -> parse_log_file
    Each token can carry overrides, e.g. file.log:backend=simd
    """
    dfs = []
    for token in inputs:
        path, meta = _parse_input_token(token)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = parse_csv_file(path)
        else:
            df = parse_log_file(path, override_backend=meta.get("backend"))
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    # Clean up backend labels like 'auto'
    return data


def most_common_config(df: pd.DataFrame) -> Tuple[int, int, int, int]:
    """
    Return (M,K,N,threads) that appears most often.
    """
    keycols = ["M", "K", "N", "threads"]
    grp = (
        df.groupby(keycols)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if grp.empty:
        raise ValueError("No data to determine a common configuration.")
    row = grp.iloc[0]
    return int(row.M), int(row.K), int(row.N), int(row.threads)


def make_bar(df: pd.DataFrame, outpath: str, title: Optional[str] = None) -> None:
    """
    Bar chart comparing backends on the most common (M,K,N,threads) configuration.
    """
    M, K, N, T = most_common_config(df)
    sub = df[(df.M == M) & (df.K == K) & (df.N == N) & (df.threads == T)]
    # If multiple entries per backend, take max GFLOP/s (best-of-N runs).
    sub = sub.sort_values("gflops", ascending=False).drop_duplicates(
        subset=["backend"], keep="first"
    )

    if sub.empty:
        raise ValueError("No rows for bar chart.")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(sub["backend"], sub["gflops"])
    ax.set_ylabel("GFLOP/s")
    ax.set_title(title or f"GEMM Performance (M={M}, K={K}, N={N}, threads={T})")

    # label bars
    for i, v in enumerate(sub["gflops"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _square_sizes(df: pd.DataFrame) -> pd.DataFrame:
    sq = df[(df.M == df.K) & (df.K == df.N)]
    # choose the most common threads value to avoid clutter
    if sq.empty:
        return sq
    common_threads = sq["threads"].mode().iat[0]
    sq = sq[sq["threads"] == common_threads]
    # Aggregate by backend+size with max gflops
    sq = (
        sq.assign(size=sq["M"])
        .sort_values(["backend", "size", "gflops"], ascending=[True, True, False])
        .drop_duplicates(subset=["backend", "size"], keep="first")
        .sort_values("size")
    )
    return sq


def make_line(df: pd.DataFrame, outpath: str, title: Optional[str] = None) -> None:
    """
    Line plot for square problem sizes (M=K=N) at the most common threads.
    """
    sq = _square_sizes(df)
    if sq.empty:
        raise ValueError("No square sizes (M=K=N) found for line plot.")
    threads = int(sq["threads"].iat[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for backend, g in sq.groupby("backend"):
        ax.plot(g["size"], g["gflops"], marker="o", linestyle="--", label=str(backend))

    ax.set_xlabel("Matrix Size (M=K=N)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(title or f"GEMM Scaling by Size (threads={threads})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def make_speedup(
    df: pd.DataFrame,
    outpath: str,
    baseline: str = "native",
    title: Optional[str] = None,
) -> None:
    """
    Speedup = backend / baseline, plotted for square sizes at common threads.
    Requires the baseline backend to exist.
    """
    sq = _square_sizes(df)
    if sq.empty:
        raise ValueError("No square sizes (M=K=N) found for speedup.")
    threads = int(sq["threads"].iat[0])

    # pivot to wide: rows=size, cols=backend, values=gflops
    wide = sq.pivot_table(
        index="size", columns="backend", values="gflops", aggfunc="max"
    )
    if baseline not in wide.columns:
        raise ValueError(
            f"Baseline '{baseline}' not present in data. Available: {list(wide.columns)}"
        )

    # compute speedups: other / baseline
    speedups = wide.div(wide[baseline], axis=0)
    speedups = speedups.drop(columns=[baseline], errors="ignore").dropna(how="all")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for col in speedups.columns:
        ax.plot(
            speedups.index,
            speedups[col],
            marker="s",
            linestyle="-",
            label=f"{col} / {baseline}",
        )

    ax.set_xlabel("Matrix Size (M=K=N)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title(title or f"Speedup over '{baseline}' (threads={threads})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # annotate points
    for col in speedups.columns:
        for x, y in zip(speedups.index, speedups[col]):
            if pd.notna(y):
                ax.text(x, y, f"{y:.1f}×", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot miniDL GEMM benchmark results.")
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="List of files. Use ':backend=NAME' to override label per file. CSV or raw logs accepted.",
    )
    p.add_argument("--outdir", default="./plots", help="Directory to write figures.")
    p.add_argument(
        "--make",
        nargs="+",
        default=["bar", "line", "speedup"],
        choices=["bar", "line", "speedup"],
        help="Which plots to generate.",
    )
    p.add_argument(
        "--baseline", default="native", help="Baseline backend for speedup plot."
    )
    p.add_argument("--title", default=None, help="Optional common title suffix.")

    args = p.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.input)

    # Save a normalized CSV alongside figures for reproducibility
    norm_csv = os.path.join(args.outdir, "gemm_normalized.csv")
    df.sort_values(["backend", "B", "M", "K", "N", "threads"]).to_csv(
        norm_csv, index=False
    )

    if "bar" in args.make:
        make_bar(df, os.path.join(args.outdir, "gemm_bar.png"), title=args.title)
    if "line" in args.make:
        make_line(df, os.path.join(args.outdir, "gemm_line.png"), title=args.title)
    if "speedup" in args.make:
        make_speedup(
            df,
            os.path.join(args.outdir, "gemm_speedup.png"),
            baseline=args.baseline,
            title=args.title,
        )

    print(f"Wrote figures to: {args.outdir}")
    print(f"Normalized CSV:   {norm_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

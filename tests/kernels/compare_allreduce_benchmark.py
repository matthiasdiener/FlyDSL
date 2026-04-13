#!/usr/bin/env python3
"""Compare two allreduce benchmark CSVs (main vs PR) and flag regressions.

Usage:
    python3 compare_benchmark.py <main.csv> <pr.csv>

Exit code 1 if any case regresses more than BOTH thresholds:
    - relative increase > MAX_REGRESSION_PCT  (default 10%)
    - absolute increase > MIN_ABS_REGRESSION_US (default 5 us)
"""
import sys
import pandas as pd

MAX_REGRESSION_PCT = 10.0
MIN_ABS_REGRESSION_US = 5.0


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <main.csv> <pr.csv>")
        sys.exit(2)

    main_csv, pr_csv = sys.argv[1], sys.argv[2]

    main_df = pd.read_csv(main_csv)
    pr_df = pd.read_csv(pr_csv)

    main_agg = main_df[main_df["rank"] == "aggregate"].copy()
    pr_agg = pr_df[pr_df["rank"] == "aggregate"].copy()

    main_agg = main_agg.set_index(["shape", "dtype"])[["avg_time_us"]]
    pr_agg = pr_agg.set_index(["shape", "dtype"])[["avg_time_us"]]

    merged = pr_agg.join(main_agg, lsuffix="_pr", rsuffix="_main")
    merged = merged.dropna()
    merged = merged[(merged["avg_time_us_main"] > 0) & (merged["avg_time_us_pr"] > 0)]

    if merged.empty:
        print("No valid comparisons available. Skipping regression check.")
        return

    merged["delta_us"] = merged["avg_time_us_pr"] - merged["avg_time_us_main"]
    merged["delta_pct"] = (merged["delta_us"] / merged["avg_time_us_main"]) * 100.0

    print("=== Allreduce Benchmark: PR vs main ===")
    for (shape, dtype), row in merged.iterrows():
        regressed = row["delta_pct"] > MAX_REGRESSION_PCT and row["delta_us"] > MIN_ABS_REGRESSION_US
        tag = "REGRESSION" if regressed else "OK"
        print(
            f"  {shape:>20s} {dtype:>4s}  "
            f"main={row['avg_time_us_main']:8.2f} us  "
            f"PR={row['avg_time_us_pr']:8.2f} us  "
            f"delta={row['delta_us']:+8.2f} us ({row['delta_pct']:+5.1f}%)  "
            f"[{tag}]"
        )

    regressions = merged[
        (merged["delta_pct"] > MAX_REGRESSION_PCT)
        & (merged["delta_us"] > MIN_ABS_REGRESSION_US)
    ]

    if not regressions.empty:
        print(
            f"\nFAILED: {len(regressions)} regression(s) exceed threshold "
            f"(>{MAX_REGRESSION_PCT}% AND >{MIN_ABS_REGRESSION_US} us)"
        )
        sys.exit(1)
    else:
        print("\nPASSED: No significant regression detected.")


if __name__ == "__main__":
    main()

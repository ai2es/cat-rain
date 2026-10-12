"""
Benchmark row group load and processing performance (pipeline-performance table in the paper).

Source: /glade/work/cbecker/cat-rain/scripts/benchmark_row_group.py, run with
/glade/work/cbecker/cat-rain/src/data.py. Here the import points at src.catrain.data, whose run_parallel and
function_pipe differ from that version (uids_final handling, filtering trajectories to devices with a home
location, and no get_daily_distance / get_max_distance_from_home measures), so timings may not match the
paper's table exactly.
Does not modify any existing source files.

Usage:
    python paper/scripts/benchmark_row_group.py -c config/process_conf.yaml
    python paper/scripts/benchmark_row_group.py -c config/process_conf.yaml --n_samples 10 --seed 42
    python paper/scripts/benchmark_row_group.py -c config/process_conf.yaml --row_groups 0 5 100 500
    python paper/scripts/benchmark_row_group.py -c config/process_conf.yaml \\
        --worker_counts 1 2 4 8 16 --n_parallel_rgs 40
"""

import argparse
import gc
import json
import sys
import time
import tracemalloc
import resource
import multiprocessing
from multiprocessing import Pool
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from skmob.core.trajectorydataframe import TrajDataFrame

from src.catrain.data import (
    available_funcs,
    data_class,
    function_pipe,
    func_w_kwargs,
    get_row_group_df,
    get_row_group_home,
    min_pings_per_date_filter,
    run_parallel,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def rss_mb():
    """Current process peak RSS in MB (Linux: /1024, macOS: /1024/1024)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024 / 1024 if sys.platform == "darwin" else ru.ru_maxrss / 1024


def timed_call(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) and return (result, elapsed_s, peak_alloc_mb).
    peak_alloc_mb is the tracemalloc peak for in-process Python allocations.
    Note: numpy/pandas C-level allocations may not be fully captured by tracemalloc;
    use the RSS values for a fuller picture.
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak_bytes / 1024 / 1024


def df_footprint_mb(dfs):
    """Sum of deep pandas memory usage across a list of DataFrames."""
    return sum(df.memory_usage(deep=True).sum() for df in dfs) / 1024 / 1024


def build_tdfs(dfs):
    """Replicate the TrajDataFrame construction from get_row_group_trajectories,
    operating on already-loaded DataFrames to isolate conversion cost from I/O."""
    return [
        TrajDataFrame(
            x.reset_index(),
            latitude="latitude",
            longitude="longitude",
            datetime="datetime",
            user_id="caid",
        )
        for x in dfs
    ]


def rg_parquet_stats(rg, meta):
    """Extract per-row-group stats from cached parquet metadata (no I/O)."""
    rows = sum(m.row_group(rg).num_rows for m in meta)
    compressed_bytes = sum(m.row_group(rg).total_byte_size for m in meta)
    return rows, compressed_bytes / 1024 / 1024


# ── Reporting ──────────────────────────────────────────────────────────────────

def hline():
    print("─" * 72)


def section(title):
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")
    print(f"  {'Metric':<42}  {'mean':>8}  {'±std':>7}  {'min':>8}  {'max':>8}  unit")
    print(f"  {'──────':<42}  {'────':>8}  {'────':>7}  {'───':>8}  {'───':>8}")


def stat_row(label, values, unit="s"):
    if not values or all(np.isnan(v) for v in values):
        return
    v = np.array([x for x in values if not np.isnan(x)])
    print(f"  {label:<42}  {v.mean():>8.3f}  {v.std():>7.3f}  {v.min():>8.3f}  {v.max():>8.3f}  {unit}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark row group load and processing.")
    parser.add_argument("-c", "--config", required=True, help="Path to config YAML")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of row groups to sample randomly (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for random sampling")
    parser.add_argument("--row_groups", type=int, nargs="+",
                        help="Explicit row group indices to benchmark (overrides --n_samples)")
    parser.add_argument("--worker_counts", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Worker counts to test in parallel scaling benchmark (default: 1 2 4 8)")
    parser.add_argument("--n_parallel_rgs", type=int, default=20,
                        help="Number of row groups to use in the parallel scaling benchmark (default: 20)")
    parser.add_argument("--save_scaling_results", type=str, default=None,
                        help="Path to save parallel scaling results as JSON for later use")
    parser.add_argument("--load_scaling_results", type=str, default=None,
                        help="Path to JSON file of prior scaling results to merge with new runs; "
                             "worker counts already present in the file are skipped")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    print("Loading metadata and geo file...")
    t_setup = time.perf_counter()
    data_class.set_attrs(config)
    setup_time = time.perf_counter() - t_setup
    print(f"set_attrs completed in {setup_time:.2f}s")

    all_rgs = data_class.row_groups
    if args.row_groups:
        sample_rgs = np.array(args.row_groups)
    else:
        rng = np.random.default_rng(args.seed)
        sample_rgs = np.sort(rng.choice(all_rgs, size=min(args.n_samples, len(all_rgs)), replace=False))

    print(f"\nTotal row groups available : {len(all_rgs):,}  ({all_rgs[0]}–{all_rgs[-1]})")
    print(f"Row groups to benchmark    : {sample_rgs.tolist()}")
    print(f"Data files                 : {len(data_class.data_files)}")
    print(f"Aggregation type           : {data_class.aggregation_type}")
    print(f"Pipeline                   : {config['pipeline']}")
    print(f"Has home loc file          : {bool(data_class.home_loc_file)}")

    # Build pipeline functions (exactly as calc_metrics.py does)
    f_kwargs = function_pipe(config)
    filters = f_kwargs["filters"]
    ind_measures = f_kwargs["individual_measures"]

    # Reverse lookup: function object → config name (for readable labels)
    func_to_name = {v: k for k, v in available_funcs.items()}

    # Storage for per-row-group results
    rows_list, compressed_mb_list = [], []
    n_uids_list, pings_mean_list, pings_p95_list = [], [], []
    df_mem_list = []
    load_time_list, load_peak_list = [], []
    tdf_time_list, tdf_peak_list = [], []
    home_time_list = []
    filter_isolated_times = {func_to_name.get(f.func, f.func.__name__): [] for f in filters}
    filter_chained_times = {func_to_name.get(f.func, f.func.__name__): [] for f in filters}
    filter_rows_remaining = {func_to_name.get(f.func, f.func.__name__): [] for f in filters}
    measure_times = {func_to_name.get(m.func, m.func.__name__): [] for m in ind_measures}
    full_time_list, full_peak_list = [], []
    rss_before_list, rss_after_list = [], []

    for rg in sample_rgs:
        print(f"\n── row group {rg} {'─' * 40}")

        # ── 1. Parquet metadata (no I/O, uses cached metadata) ────────────
        rows, comp_mb = rg_parquet_stats(rg, data_class.meta)
        rows_list.append(rows)
        compressed_mb_list.append(comp_mb)
        print(f"  metadata   : {rows:>8,} rows  |  {comp_mb:.2f} MB compressed  "
              f"({len(data_class.data_files)} files)")

        # ── 2. Load: get_row_group_df ──────────────────────────────────────
        rss_before = rss_mb()
        dfs, load_t, load_peak = timed_call(get_row_group_df, rg)
        rss_after = rss_mb()
        df_mem = df_footprint_mb(dfs)
        load_time_list.append(load_t)
        load_peak_list.append(load_peak)
        df_mem_list.append(df_mem)
        rss_before_list.append(rss_before)
        rss_after_list.append(rss_after)
        print(f"  load       : {load_t:.3f}s  |  tracemalloc peak {load_peak:.1f} MB  "
              f"|  RSS delta {rss_after - rss_before:+.1f} MB  |  df footprint {df_mem:.1f} MB")

        # ── 3. UID / ping distribution (from loaded data) ─────────────────
        combined = pd.concat(dfs).reset_index()
        # print(combined.columns)
        uid_counts = combined["caid"].value_counts()
        n_uids_list.append(uid_counts.shape[0])
        pings_mean_list.append(uid_counts.mean())
        pings_p95_list.append(float(uid_counts.quantile(0.95)))
        print(f"  uid stats  : {uid_counts.shape[0]:>6,} unique  |  "
              f"mean {uid_counts.mean():.1f} pings  |  p95 {uid_counts.quantile(0.95):.0f} pings")
        del combined

        # ── 4. TrajDataFrame conversion (I/O already done) ────────────────
        tdfs, tdf_t, tdf_peak = timed_call(build_tdfs, dfs)
        tdf_time_list.append(tdf_t)
        tdf_peak_list.append(tdf_peak)
        print(f"  tdf conv   : {tdf_t:.3f}s  |  tracemalloc peak {tdf_peak:.1f} MB")

        # ── 5. Home location load ──────────────────────────────────────────
        if data_class.home_loc_file:
            _, home_t, _ = timed_call(get_row_group_home, rg)
            home_time_list.append(home_t)
            print(f"  home load  : {home_t:.3f}s")
        else:
            home_time_list.append(np.nan)

        # ── 6. Per-filter benchmarks ───────────────────────────────────────
        # Isolated: each filter called on the original TDF so costs are
        #   independent of each other and of data reduction by prior filters.
        # Chained: filters applied in pipeline order so each filter receives
        #   the already-reduced output of the previous one, matching run_parallel.
        #   Row counts are recorded at each step to show data reduction.
        # min_pings_per_date_filter returns a UID list, not a TDF, so chained
        #   mode is skipped for it.
        tdf_bench = tdfs[0]
        if filters:
            tdf_chain = tdf_bench
            for filter_partial in filters:
                fname = func_to_name.get(filter_partial.func, filter_partial.func.__name__)
                is_uid_filter = filter_partial.func is min_pings_per_date_filter

                # Isolated
                _, ft_iso, _ = timed_call(filter_partial, tdf_bench)
                filter_isolated_times[fname].append(ft_iso)

                # Chained
                if is_uid_filter:
                    filter_chained_times[fname].append(np.nan)
                    filter_rows_remaining[fname].append(np.nan)
                    print(f"  filter {fname:<28s}: iso {ft_iso:.3f}s  "
                          f"(chained skipped — returns UID list)")
                else:
                    rows_in = len(tdf_chain)
                    tdf_chain, ft_chain, _ = timed_call(filter_partial, tdf_chain)
                    rows_out = len(tdf_chain)
                    pct = 100.0 * rows_out / rows_in if rows_in > 0 else 0.0
                    filter_chained_times[fname].append(ft_chain)
                    filter_rows_remaining[fname].append(pct)
                    print(f"  filter {fname:<28s}: iso {ft_iso:.3f}s  |  "
                          f"chain {ft_chain:.3f}s  |  "
                          f"{rows_out:,}/{rows_in:,} rows ({pct:.1f}% remaining)")

        # ── 7. Per-measure (on the already-loaded tdf, no repeated I/O) ───
        # Uses tdfs[0] because aggregation_type='full' yields a single TDF.
        # For 'daily', tdfs is a list; we benchmark on the first date's TDF.
        for measure_partial in ind_measures:
            name = func_to_name.get(measure_partial.func, measure_partial.func.__name__)
            _, mt, _ = timed_call(measure_partial, tdf_bench)
            measure_times[name].append(mt)
            print(f"  {name:<35s}: {mt:.3f}s")

        # ── 7. Full run_parallel (end-to-end including all I/O) ───────────
        # Deep-copy summary so repeated calls don't accumulate monitor counts.
        run_kwargs = {**f_kwargs, "summary": deepcopy(f_kwargs["summary"])}
        _, full_t, full_peak = timed_call(run_parallel, rg, **run_kwargs)
        full_time_list.append(full_t)
        full_peak_list.append(full_peak)
        print(f"  run_parallel : {full_t:.3f}s  |  tracemalloc peak {full_peak:.1f} MB")

    # ── Summary tables ────────────────────────────────────────────────────────

    section("ROW GROUP CHARACTERIZATION")
    stat_row("Rows per row group (all files)", rows_list, unit="rows")
    stat_row("Compressed size per row group", compressed_mb_list, unit="MB")
    stat_row("Unique UIDs per row group", n_uids_list, unit="uids")
    stat_row("Mean pings per UID", pings_mean_list, unit="pings")
    stat_row("p95 pings per UID", pings_p95_list, unit="pings")

    section("LOAD TIMING & MEMORY")
    stat_row("get_row_group_df (I/O + to_pandas)", load_time_list, unit="s")
    stat_row("TrajDataFrame conversion", tdf_time_list, unit="s")
    if data_class.home_loc_file:
        stat_row("get_row_group_home", home_time_list, unit="s")
    stat_row("DataFrame in-memory footprint", df_mem_list, unit="MB")
    stat_row("Load tracemalloc peak alloc", load_peak_list, unit="MB")
    stat_row("TDF conv tracemalloc peak alloc", tdf_peak_list, unit="MB")
    stat_row("RSS delta across load", [b - a for a, b in zip(rss_before_list, rss_after_list)], unit="MB")

    if filters:
        section("PER-FILTER TIMING  (isolated = on original TDF;  chained = pipeline order)")
        print(f"  {'Filter':<42}  {'iso mean':>8}  {'±std':>7}  {'chain mean':>10}  {'±std':>7}  {'rows remaining %':>16}")
        print(f"  {'──────':<42}  {'────────':>8}  {'────':>7}  {'──────────':>10}  {'────':>7}  {'────────────────':>16}")
        for fname in filter_isolated_times:
            iso = np.array(filter_isolated_times[fname])
            chain = np.array([x for x in filter_chained_times[fname] if not np.isnan(x)])
            rows_rem = np.array([x for x in filter_rows_remaining[fname] if not np.isnan(x)])
            chain_str = f"{chain.mean():>10.3f}  {chain.std():>7.3f}" if len(chain) else f"{'N/A':>10}  {'':>7}"
            rows_str = f"{rows_rem.mean():>14.1f}%" if len(rows_rem) else f"{'N/A':>14} "
            print(f"  {fname:<42}  {iso.mean():>8.3f}  {iso.std():>7.3f}  {chain_str}  {rows_str}")

    section("PER-MEASURE PROCESSING TIMING")
    for name, times in measure_times.items():
        if times:
            stat_row(name, times, unit="s")

    section("FULL PIPELINE (run_parallel end-to-end)")
    stat_row("run_parallel wall time", full_time_list, unit="s")
    stat_row("run_parallel tracemalloc peak", full_peak_list, unit="MB")

    # ── Parallel scaling benchmark ────────────────────────────────────────────
    # Sample a fixed set of row groups for the parallel runs.
    # Re-use sample_rgs if large enough, otherwise draw a fresh sample.
    rng2 = np.random.default_rng(args.seed + 1)
    n_par = min(args.n_parallel_rgs, len(all_rgs))
    if n_par <= len(sample_rgs):
        par_rgs = sample_rgs[:n_par]
    else:
        par_rgs = np.sort(rng2.choice(all_rgs, size=n_par, replace=False))

    print(f"\n\nRunning parallel scaling benchmark over {len(par_rgs)} fixed row groups: {par_rgs.tolist()}")
    print("(Each worker count runs the same set — results are directly comparable.)\n")

    # Load prior results if provided, then skip those worker counts
    if args.load_scaling_results:
        with open(args.load_scaling_results) as fh:
            scaling_results = json.load(fh)
        already_run = {r["n_workers"] for r in scaling_results}
        skipped = sorted(already_run & set(args.worker_counts))
        remaining = sorted(set(args.worker_counts) - already_run)
        if skipped:
            print(f"  Loaded {len(scaling_results)} prior result(s) from {args.load_scaling_results}.")
            print(f"  Skipping worker counts already present: {skipped}")
        args.worker_counts = remaining
    else:
        scaling_results = []

    for n_w in sorted(args.worker_counts):
        if n_w > len(par_rgs):
            print(f"  Skipping {n_w} workers — more workers than row groups ({len(par_rgs)}).")
            continue

        # Build a fresh partial exactly like calc_metrics.py does.
        # multiprocessing pickles f into each worker, so each gets its own copy
        # of the summary (no shared-state issues).
        fresh_kwargs = function_pipe(config)
        f = func_w_kwargs(run_parallel, fresh_kwargs)

        gc.collect()
        rss_pre = rss_mb()
        t0 = time.perf_counter()

        with Pool(processes=n_w) as pool:
            result = pool.map_async(f, par_rgs)
            results = result.get()

        wall_time = time.perf_counter() - t0
        rss_post = rss_mb()

        # RUSAGE_CHILDREN accumulates across all waited-for children since
        # process start.  We snapshot before and after to get the delta for
        # this pool run only.
        children_ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_rss_mb = (
            children_ru.ru_maxrss / 1024 / 1024
            if sys.platform == "darwin"
            else children_ru.ru_maxrss / 1024
        )

        n_success = sum(1 for r in results if r is not None)
        throughput = len(par_rgs) / wall_time  # rgs/s

        scaling_results.append({
            "n_workers": n_w,
            "wall_time_s": wall_time,
            "throughput_rgs_per_s": throughput,
            "n_success": n_success,
            "parent_rss_delta_mb": rss_post - rss_pre,
            "children_peak_rss_mb": child_rss_mb,
        })

        print(f"  {n_w:>3d} workers │ {wall_time:7.1f}s │ "
              f"{throughput:.2f} rgs/s │ {n_success}/{len(par_rgs)} succeeded │ "
              f"parent RSS delta {rss_post - rss_pre:+.0f} MB │ "
              f"children peak RSS {child_rss_mb:.0f} MB")

    # Sort combined results and optionally save
    scaling_results = sorted(scaling_results, key=lambda r: r["n_workers"])
    if args.save_scaling_results:
        with open(args.save_scaling_results, "w") as fh:
            json.dump(scaling_results, fh, indent=2)
        print(f"\n  Scaling results saved to {args.save_scaling_results}")

    # ── Scaling summary table ─────────────────────────────────────────────────
    if scaling_results:
        section("PARALLEL SCALING  (actual measurements)")
        baseline_time = scaling_results[0]["wall_time_s"]
        print(f"  {'workers':>7}  {'wall_time':>10}  {'rgs/s':>7}  "
              f"{'speedup':>8}  {'efficiency':>10}  {'parent ΔRSS':>12}  {'child peak RSS':>14}")
        print(f"  {'───────':>7}  {'─────────':>10}  {'─────':>7}  "
              f"{'───────':>8}  {'──────────':>10}  {'───────────':>12}  {'──────────────':>14}")
        for r in scaling_results:
            speedup = baseline_time / r["wall_time_s"]
            efficiency = speedup / r["n_workers"] * 100
            print(f"  {r['n_workers']:>7d}  {r['wall_time_s']:>9.1f}s  "
                  f"{r['throughput_rgs_per_s']:>7.2f}  "
                  f"{speedup:>7.2f}x  {efficiency:>9.1f}%  "
                  f"{r['parent_rss_delta_mb']:>+11.0f}M  "
                  f"{r['children_peak_rss_mb']:>13.0f}M")

        # Extrapolate to full dataset using the best-observed throughput
        best = max(scaling_results, key=lambda r: r["throughput_rgs_per_s"])
        n_total = len(all_rgs)
        print(f"\n  Row groups in full dataset : {n_total:,}")
        print(f"  Best observed throughput   : {best['throughput_rgs_per_s']:.2f} rgs/s  "
              f"({best['n_workers']} workers, measured on {len(par_rgs)} rgs)")
        print(f"  Extrapolated full-run time : "
              f"{n_total / best['throughput_rgs_per_s'] / 60:.1f} min  "
              f"(at {best['n_workers']} workers, no startup overhead)")

        # Also show ideal-scaling projections from the 1-worker baseline
        if len(scaling_results) > 1:
            serial_rate = scaling_results[0]["throughput_rgs_per_s"]
            print(f"\n  Ideal-scaling projections from {scaling_results[0]['n_workers']}-worker baseline:")
            for n_w in [4, 8, 16, 32, 64, 120]:
                if n_w > len(par_rgs):
                    continue
                ideal_time = n_total / (serial_rate * n_w) / 60
                print(f"    {n_w:>3d} workers (ideal) : {ideal_time:.1f} min")

    print(f"\nFinal process peak RSS : {rss_mb():.1f} MB")
    print(f"set_attrs overhead     : {setup_time:.2f}s")


if __name__ == "__main__":
    main()

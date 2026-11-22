"""
Sobol sensitivity analysis, split into three SLURM-friendly stages.

    python sobol_hpc.py prepare --log2-samples 12     # 1 job : draw + save Saltelli samples
    python sobol_hpc.py run                           # array  : each task simulates a strided slice of rows
    python sobol_hpc.py analyze                       # 1 job : gather shards, run SALib, write CSV + PDFs

Design notes
------------
* The Sobol sample matrix is generated ONCE (prepare) and saved to disk, so every array
  task sees exactly the same rows and the final Y vectors line up with the sample matrix
  that SALib.analyze expects.
* Row i is handled by task (i % n_tasks).  Each row gets its own deterministic seeds
  (base_seed + i*avg_over + replicate), so results do not depend on how rows are split
  across tasks/cores or on re-runs.
* Each task checkpoints to its own shard file and skips rows it has already done, so a
  job that hits its wall-time limit can simply be resubmitted.
* Inside a task, rows are spread over SLURM_CPUS_PER_TASK worker processes.
"""

import matplotlib
matplotlib.use("Agg")   # no display on compute nodes

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
import pandas as pd
import random as rand
import numpy as np

from simulate.helper_funcs import *
from simulate.rules import *
from simulate.simulate import *
from simulate.working_params import *

# publication-grade theme style (also provides plt / COLORS used in plotting below)
from simulate.style_config import *
apply_publication_style()

# imported AFTER the star-imports above so nothing in them can shadow these names
import os
import sys
import argparse
import traceback
import multiprocessing as mp
from time import perf_counter

OUTPUT_DIR = os.environ.get('SIM_OUTPUT_DIR', '.')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'results_and_data')

# -------------------
# MODEL CONFIGURATION
# -------------------

avg_over = 5

def sample_params():
    return {
        "inf_alpha": inf_alpha,
        "inf_beta": inf_beta,
        "delta": delta,
        "T_inf": T_inf,
        "T_TBD": T_TBD,
        "T_AD": T_AD,
        "T_seasonal": T_seasonal,
        "win_length": win_length,
        "win_start": win_start,
        "lambda_win": lambda_win,
        "lambda_sum": lambda_sum,
        "res_gain": res_gain,
        "res_max": res_max,
        "k_imm": k_imm,
        "theta_imm": theta_imm,
    }

# -------------------------
# set up initial population
# -------------------------

num_infected = 5

tricolor_num = 100
tricolor_cluster_sizeMIN = 1
tricolor_cluster_sizeMAX = 2

bigbrown_num = 0
bigbrown_cluster_sizeMIN = 1
bigbrown_cluster_sizeMAX = 9

Hi_list = [[tricolor_num, tricolor_cluster_sizeMIN, tricolor_cluster_sizeMAX],
           [bigbrown_num, bigbrown_cluster_sizeMIN, bigbrown_cluster_sizeMAX]]

# NOTICE : the remaining populations (Ot, Im) all start with 0 inhabitants
# NOTICE : resistance starts at 0 for every bat

# ---------------------------
# system-governing parameters
# ---------------------------

inf_alpha, inf_beta = 5, 2
delta = 0.05
T_inf = 30
T_TBD = 4.1
T_AD = 88.5/1440

res_max = 0.2
k_imm, theta_imm = 1, 1
res_gain = 0.02

T_seasonal, win_length, win_start, lambda_sum, lambda_win = latitude2_SouthMidwest()
NAME_APPEND = "lat2SMW"

time = 365*20  # total days (20 years)

# Sobol base sample size: N = 2**LOG2_SAMPLES.  Total simulated parameter sets = N*(D+2) = N*10.
#   6  ->    64 base samples ->    640 rows   (timing / test runs)
#   12 -> 4096 base samples -> 40,960 rows   (full runs)
LOG2_SAMPLES = int(os.environ.get("SOBOL_LOG2_SAMPLES", 6))

HISTORY_KEYS = ("Hi", "Ot", "In", "Im", "De")

# ------------------------
# SOBOL PROBLEM DEFINITION
# ------------------------

# keys in the simulation parameter dict, in the same order as PROBLEM["names"] / bounds
RAW_KEYS = ["inf_alpha", "inf_beta", "delta", "T_inf", "res_max", "res_gain", "k_imm", "theta_imm"]

PROBLEM = {
    "num_vars": 8,
    "names": [
        r"$\alpha_{\mathrm{inf}}$", r"$\beta_{\mathrm{inf}}$", r"$\delta$",
        r"$T_{\mathrm{inf}}$", r"$r_{\mathrm{max}}$", r"$r_{\mathrm{gain}}$", r"$k_{\mathrm{imm}}$",
        r"$\theta_{\mathrm{imm}}$"
    ],
    "bounds": [
        [1, 5],
        [2, 10],
        [0.005, 0.05],
        [10, 40],
        [0, 0.8],
        [0, 1],
        [0.1, 10],
        [0, 30],
    ],
}

OUTPUT_NAMES = ("Pmax", "Sfinal", "Mfinal", "R0")   # order of the Y columns stored in shards


# --------------------
# helpers: file layout
# --------------------

def work_dir(log2_samples):
    return os.path.join(FIGURES_DIR, f"sobol_work_{NAME_APPEND}_N{2**log2_samples}")

def params_path(wd):
    return os.path.join(wd, "param_values.npy")

def shard_path(wd, task_id):
    return os.path.join(wd, "shards", f"shard_{task_id:05d}.npz")

def atomic_savez(path, **arrays):
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ----------------
# stage 1: prepare
# ----------------

def cmd_prepare(args):
    wd = work_dir(args.log2_samples)
    os.makedirs(os.path.join(wd, "shards"), exist_ok=True)

    if os.path.exists(params_path(wd)) and not args.force:
        pv = np.load(params_path(wd))
        print(f"[prepare] {params_path(wd)} already exists ({pv.shape[0]} rows); "
              f"use --force to regenerate (this invalidates existing shards).")
        return

    num_samples = 2 ** args.log2_samples
    try:
        param_values = sobol_sample.sample(PROBLEM, num_samples, calc_second_order=False, seed=args.seed)
    except TypeError:   # older SALib without a `seed` kwarg
        np.random.seed(args.seed)
        param_values = sobol_sample.sample(PROBLEM, num_samples, calc_second_order=False)

    if args.force:
        for f in os.listdir(os.path.join(wd, "shards")):
            os.remove(os.path.join(wd, "shards", f))

    np.save(params_path(wd), param_values)
    print(f"[prepare] wrote {param_values.shape[0]} parameter sets "
          f"(N=2^{args.log2_samples}, D={PROBLEM['num_vars']}) to {params_path(wd)}")
    print(f"[prepare] each row = {avg_over} replicate simulations of {time} days")


# --------------------------
# stage 2: run  (array task)
# --------------------------

def evaluate_row(i, row, base_seed):
    """Run avg_over replicates for one Sobol row and return (i, [Pmax, Sfinal, Mfinal, R0])."""
    try:
        parameters = sample_params()
        for key, val in zip(RAW_KEYS, row):
            parameters[key] = max(1.0, val) if key == "inf_alpha" else val

        # fresh accumulators every row (never share arrays between rows)
        history_avg = {key: np.zeros(time) for key in HISTORY_KEYS}
        history = None
        for j in range(avg_over):
            seed = base_seed + i * avg_over + j
            rand.seed(seed)
            np.random.seed(seed % (2**32))
            history = simulate(make_initial_state(Hi_list, num_infected), time, parameters, False)
            for key in history_avg:
                history_avg[key] += np.array(history[key])

        for key in history_avg:
            history_avg[key] /= avg_over

        history_avg["SC"] = history["SC"]

        m = compute_metrics(history_avg, Hi_list, num_infected)
        return i, [m["P_max"], m["S_final"], m["M_final"], m["R0_empirical"]]

    except Exception:
        print(f"[run] row {i} FAILED:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        return i, [np.nan] * len(OUTPUT_NAMES)


def _worker(payload):
    i, row, base_seed = payload
    return evaluate_row(i, row, base_seed)


def cmd_run(args):
    wd = work_dir(args.log2_samples)
    param_values = np.load(params_path(wd))
    n_rows = param_values.shape[0]

    task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_tasks = args.n_tasks if args.n_tasks is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    n_workers = args.workers if args.workers is not None else int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    # strided assignment balances load if cost varies smoothly with Sobol index
    my_rows = np.arange(task_id, n_rows, n_tasks)

    # resume from an existing shard
    path = shard_path(wd, task_id)
    done = {}
    if os.path.exists(path):
        s = np.load(path)
        for idx, y in zip(s["idx"], s["Y"]):
            done[int(idx)] = y
    # retry rows that previously failed (NaN) as well as ones never attempted
    todo = [int(i) for i in my_rows if int(i) not in done or np.isnan(done[int(i)]).any()]

    print(f"[run] task {task_id}/{n_tasks}: {len(my_rows)} rows assigned, "
          f"{len(my_rows) - len(todo)} already done, {len(todo)} to run on {n_workers} worker(s)", flush=True)

    def save():
        idx = np.array(sorted(done), dtype=np.int64)
        Y = np.array([done[i] for i in idx]).reshape(len(idx), len(OUTPUT_NAMES))
        atomic_savez(path, idx=idx, Y=Y)

    payloads = [(i, param_values[i], args.seed) for i in todo]
    t0 = perf_counter()
    n_done_now = 0

    def record(res):
        nonlocal n_done_now
        i, y = res
        done[i] = np.array(y, dtype=float)
        n_done_now += 1
        if n_done_now % args.checkpoint_every == 0:
            save()
        # only task 0 prints progress (otherwise 100+ MPI ranks flood the log)
        if task_id == 0 and (n_done_now % max(1, len(todo) // 20) == 0 or n_done_now == len(todo)):
            el = perf_counter() - t0
            print(f"[run] task {task_id}: {n_done_now}/{len(todo)} rows  "
                  f"({el:.0f}s elapsed, {el / n_done_now:.1f}s/row wall)", flush=True)

    if n_workers > 1 and len(payloads) > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(n_workers, len(payloads))) as pool:
            for res in pool.imap_unordered(_worker, payloads, chunksize=1):
                record(res)
    else:
        for p in payloads:
            record(_worker(p))

    save()
    n_nan = sum(1 for v in done.values() if np.isnan(v).any())
    print(f"[run] task {task_id} finished: {len(done)} rows in shard, {n_nan} with NaN/failure", flush=True)


# ----------------
# stage 3: analyze
# ----------------

def plot_single_sobol(Si, problem, title, filename):
    names = problem["names"]
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.bar(x - width/2, Si["S1"], width, label=r"First-order ($S_1$)",
           color=COLORS["Sobol_S1"], yerr=Si["S1_conf"], capsize=3, error_kw={"lw": 0.8, "color": "#333333"})
    ax.bar(x + width/2, Si["ST"], width, label=r"Total-order ($S_T$)",
           color=COLORS["Sobol_ST"], yerr=Si["ST_conf"], capsize=3, error_kw={"lw": 0.8, "color": "#333333"})

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Sobol Sensitivity Index")
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.5)

    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def cmd_analyze(args):
    wd = work_dir(args.log2_samples)
    param_values = np.load(params_path(wd))
    n_rows = param_values.shape[0]

    Y = np.full((n_rows, len(OUTPUT_NAMES)), np.nan)
    seen = np.zeros(n_rows, dtype=bool)

    shard_dir = os.path.join(wd, "shards")
    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".npz") and ".tmp" not in f)
    for f in shard_files:
        s = np.load(os.path.join(shard_dir, f))
        Y[s["idx"]] = s["Y"]
        seen[s["idx"]] = True

    missing = np.where(~seen)[0]
    bad = np.where(seen & np.isnan(Y).any(axis=1))[0]
    print(f"[analyze] {len(shard_files)} shard(s), {seen.sum()}/{n_rows} rows present, "
          f"{len(missing)} missing, {len(bad)} failed (NaN)")

    if len(missing) or len(bad):
        if len(missing):
            print(f"[analyze] missing rows (first 20): {missing[:20].tolist()}")
        if len(bad):
            print(f"[analyze] failed rows  (first 20): {bad[:20].tolist()}")
        print("[analyze] Sobol indices need every row. Resubmit the array stage (finished rows are skipped, "
              "failed rows are retried), then rerun analyze. Aborting.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    np.savez(os.path.join(wd, "sobol_raw_outputs.npz"), param_values=param_values, Y=Y,
             output_names=np.array(OUTPUT_NAMES))

    Y_Pmax, Y_Sfinal, Y_Mfinal, Y_R0 = (Y[:, k] for k in range(4))

    Si_P = sobol.analyze(PROBLEM, Y_Pmax, calc_second_order=False, print_to_console=False)
    Si_S = sobol.analyze(PROBLEM, Y_Sfinal, calc_second_order=False, print_to_console=False)
    Si_M = sobol.analyze(PROBLEM, Y_Mfinal, calc_second_order=False, print_to_console=False)
    Si_R0 = sobol.analyze(PROBLEM, Y_R0, calc_second_order=False, print_to_console=False)

    sobol_data = {
        "Parameter": [n.replace('$', '').replace('\\', '') for n in PROBLEM["names"]],
        "R0_S1": Si_R0["S1"], "R0_S1_conf": Si_R0["S1_conf"],
        "R0_ST": Si_R0["ST"], "R0_ST_conf": Si_R0["ST_conf"],
        "Pmax_S1": Si_P["S1"], "Pmax_S1_conf": Si_P["S1_conf"],
        "Pmax_ST": Si_P["ST"], "Pmax_ST_conf": Si_P["ST_conf"],
        "Sfinal_S1": Si_S["S1"], "Sfinal_S1_conf": Si_S["S1_conf"],
        "Sfinal_ST": Si_S["ST"], "Sfinal_ST_conf": Si_S["ST_conf"],
        "Mfinal_S1": Si_M["S1"], "Mfinal_S1_conf": Si_M["S1_conf"],
        "Mfinal_ST": Si_M["ST"], "Mfinal_ST_conf": Si_M["ST_conf"],
    }

    df_sobol = pd.DataFrame(sobol_data)
    csv_path = os.path.join(FIGURES_DIR, "sobol_analysis_data.csv")
    df_sobol.to_csv(csv_path, index=False)

    plot_single_sobol(Si_R0, PROBLEM, r"Sensitivity: Empirical $R_0$", f'sobol_R0_{NAME_APPEND}.pdf')
    plot_single_sobol(Si_P, PROBLEM, r"Sensitivity: Peak Prevalence ($P_{\max}$)", f'sobol_Pmax_{NAME_APPEND}.pdf')
    plot_single_sobol(Si_S, PROBLEM, r"Sensitivity: Final Persistence ($S_{\text{final}}$)", f'sobol_Sfinal_{NAME_APPEND}.pdf')
    plot_single_sobol(Si_M, PROBLEM, r"Sensitivity: Final Mortality ($M_{\text{final}}$)", f'sobol_Mfinal_{NAME_APPEND}.pdf')
    print(f"[analyze] wrote {csv_path} and 4 PDFs to {FIGURES_DIR}")


# ------------------------------------------------
# all-in-one under MPI: mpirun python sobol_hpc.py
# (this is what the launcher script does)
# ------------------------------------------------

def cmd_all(args):
    """rank 0 prepares -> every rank simulates rows rank, rank+size, ... -> rank 0 analyzes."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except ImportError:             # no mpi4py: behaves as a single serial rank
        comm, rank, size = None, 0, 1

    def barrier():
        if comm is not None:
            comm.Barrier()

    try:
        if rank == 0:
            print(f"[all] {size} MPI rank(s), N=2^{args.log2_samples}, {time} days per simulation", flush=True)
            cmd_prepare(argparse.Namespace(log2_samples=args.log2_samples, seed=args.seed, force=False))
        barrier()

        cmd_run(argparse.Namespace(log2_samples=args.log2_samples, seed=args.seed, task_id=rank,
                                   n_tasks=size, workers=1, checkpoint_every=args.checkpoint_every))
        barrier()

        if rank == 0:
            cmd_analyze(args)
    except Exception:
        traceback.print_exc()
        if comm is not None:
            comm.Abort(1)           # don't leave the other ranks hanging at a barrier
        raise


# ---
# CLI
# ---

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log2-samples", type=int, default=LOG2_SAMPLES,
                   help="N = 2**this base samples (6 for tests, 12 for large runs). "
                        "Total rows = N*(D+2) with second-order off.")
    p.add_argument("--seed", type=int, default=int(os.environ.get("SOBOL_SEED", 12345)),
                   help="base seed for the Sobol scramble and for the per-row simulation seeds")
    sub = p.add_subparsers(dest="cmd")   # no subcommand -> "all"

    sl = sub.add_parser("all", help="prepare + run + analyze in one MPI launch (default)")
    sl.add_argument("--checkpoint-every", type=int, default=1)
    sl.set_defaults(func=cmd_all)

    sp = sub.add_parser("prepare", help="generate and save the Sobol sample matrix")
    sp.add_argument("--force", action="store_true", help="regenerate (deletes existing shards)")
    sp.set_defaults(func=cmd_prepare)

    sr = sub.add_parser("run", help="simulate this task's slice of rows")
    sr.add_argument("--task-id", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID")
    sr.add_argument("--n-tasks", type=int, default=None, help="default: $SLURM_ARRAY_TASK_COUNT")
    sr.add_argument("--workers", type=int, default=None, help="default: $SLURM_CPUS_PER_TASK")
    sr.add_argument("--checkpoint-every", type=int, default=1, help="rewrite shard every K finished rows")
    sr.set_defaults(func=cmd_run)

    sa = sub.add_parser("analyze", help="gather shards, run SALib, write CSV and figures")
    sa.set_defaults(func=cmd_analyze)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd is None:                         # launched with no subcommand (e.g. by the menu launcher)
        args = parser.parse_args(sys.argv[1:] + ["all"])
    args.func(args)
"""Statistics shared by the table/figure scripts.

* seed-level: mean ± std over seeds; paired t-test and Wilcoxon signed-rank over seeds
  (seed i of method A paired with seed i of method B — same data order and init RNG);
* sample-level: paired bootstrap over test images of the seed-averaged correctness, and
  an exact McNemar test on seed-pooled predictions (uses preds.npz);
* Holm–Bonferroni correction across the comparisons of one table.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats as st


# --------------------------------------------------------------------------- loading
# protocol: every comparison with the baselines uses seeds 0-2 for all methods (ViPER included, 2026-11-12);
# the gate comparison (ViPER vs its matched controls) is paired over seeds 0-4 (load_group(..., five=True))
FIVE_SEED_METHODS = {"viper", "abl_no_gates", "viper_lepe", "rope_mixed_lepe", "abl_static_gate", "abl_rgb_gate"}


def load_group(group: str, five: bool = False):
    """-> {method: {seed: final.json dict}} for results/<group>/<method>_s<seed>/final.json
    (seeds beyond the protocol above are ignored; five=True keeps seeds 3-4 of FIVE_SEED_METHODS for the gate comparison)"""
    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(group, "*", "final.json"))):
        name = os.path.basename(os.path.dirname(f))
        m = re.match(r"(.+)_s(\d+)$", name)
        if not m:
            continue
        if int(m.group(2)) >= (5 if five and m.group(1) in FIVE_SEED_METHODS else 3):
            continue
        run = json.load(open(f))
        nat = os.path.join(os.path.dirname(f), "multires_native_gate.json")
        if os.path.exists(nat):   # ViPER runs trained before gate_res=0 became the default: re-evaluated
            for r, v in json.load(open(nat)).items():   # with the gate branch at native resolution (top-1 only)
                if r in run.get("multires", {}):
                    run["multires"][r] = {"extrapolate": {**run["multires"][r]["extrapolate"], "top1": v}}
        nat_i = os.path.join(os.path.dirname(f), "multires_native_gate_interp.json")
        if os.path.exists(nat):   # interpolation re-evaluated the same way (never fall back to the stale gate_res runs)
            for r, e in run.get("multires", {}).items():
                e.pop("interpolate", None)
            if os.path.exists(nat_i):
                for r, v in json.load(open(nat_i)).items():
                    if r in run.get("multires", {}):
                        run["multires"][r]["interpolate"] = {"top1": v}
        out[m.group(1)][int(m.group(2))] = run
    return out


def metric(run: dict, key: str = "top1", res: str | None = None, mode: str = "extrapolate"):
    if res is None:
        return run["test"][key]
    e = run.get("multires", {}).get(str(res), {})
    e = e.get(mode) or e.get("extrapolate")
    return None if e is None else e[key]


def seed_values(runs: dict, key="top1", res=None, mode="extrapolate"):
    """-> {seed: value} skipping missing."""
    out = {}
    for s, r in runs.items():
        v = metric(r, key, res, mode)
        if v is not None:
            out[s] = v
    return out


def correctness(run_dir: str, key: str = "test_logits"):
    z = np.load(os.path.join(run_dir, "preds.npz"))
    labels = z["test_labels"].astype(np.int64)
    return (z[key].astype(np.float32).argmax(1) == labels).astype(np.float32)


# --------------------------------------------------------------------------- tests
def mean_std(vals):
    v = np.asarray(list(vals), dtype=float)
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def paired_seed_test(a: dict, b: dict):
    """paired t-test and Wilcoxon over common seeds; -> dict(diff, t_p, w_p, n)."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return dict(diff=np.nan, t_p=np.nan, w_p=np.nan, n=len(seeds))
    x, y = np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])
    d = x - y
    t_p = float(st.ttest_rel(x, y).pvalue) if np.any(d != 0) else 1.0
    try:
        w_p = float(st.wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w_p = np.nan
    return dict(diff=float(d.mean()), t_p=t_p, w_p=w_p, n=len(seeds))


def paired_bootstrap(ca: np.ndarray, cb: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """ca, cb: (n_img,) seed-averaged correctness. -> (diff in %, 95% CI, two-sided p)."""
    rng = np.random.default_rng(seed)
    d = ca - cb
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean() * 100), (float(lo), float(hi)), float(min(1.0, p))


def mcnemar(ca: np.ndarray, cb: np.ndarray):
    """exact McNemar on binary correctness vectors (pooled over seeds)."""
    b = int(((ca == 1) & (cb == 0)).sum())
    c = int(((ca == 0) & (cb == 1)).sum())
    if b + c == 0:
        return 1.0
    return float(st.binomtest(b, b + c, 0.5).pvalue)


def holm(pvals: dict):
    """Holm–Bonferroni adjusted p-values, same keys."""
    items = sorted(((p, k) for k, p in pvals.items() if p == p), key=lambda t: t[0])
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k, p in pvals.items():
        out.setdefault(k, np.nan)
    return out

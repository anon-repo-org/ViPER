#!/usr/bin/env python
"""Generate queue job files.

python scripts/make_jobs.py main        # IN-100 DeiT-S, every baseline PE, 5 seeds -> jobs/main_in100.json
python scripts/make_jobs.py viper       # ViPER + ablations, 5 seeds              -> jobs/viper_in100.json
python scripts/make_jobs.py domain      # 9 domain datasets, DeiT-Tiny scratch     -> jobs/domain.json

Priority: seed-major (all seed-0 runs first), so partial results are usable early.
"""
import json
import os
import sys

SEEDS = [0, 1, 2, 3, 4]
ABL_SEEDS = [0, 1, 2]          # baselines and backbones: 3 seeds (user decision 2026-10-30); ViPER itself keeps 5
DESIGN_SEEDS = [0, 1]          # design ablations: 2 seeds (user decision 2026-11-05); gate controls keep 5
IN100 = ("--dataset imagenet100 --backbone deit_small --epochs 100 --warmup 5 --bs 128 --workers 6")
# order inside a seed = priority (strongest / most-cited baselines first)
BASELINES = ["rope_mixed", "learned", "none", "rope_axial", "cpe", "alibi", "relbias", "sincos", "lepe",
             "fourier", "liere", "comrope_ld", "string_cayley", "spiral", "lookhere", "pape", "irpe", "dpb", "gpsa", "wavelet_rpe", "carope"]
# measured peak memory at bs 128 (GB, rounded up) -> queue only launches when it fits next to co-tenants
MEM = {"alibi": 8, "relbias": 8, "lookhere": 8, "liere": 9, "comrope_ld": 9, "string_cayley": 8, "pape": 12, "irpe": 9, "gpsa": 9, "wavelet_rpe": 9, "dpb": 9, "carope": 8}

# ViPER = score path (wavelet-gated RoPE-Mixed) + LePE value path (final configuration, user decision 2026-11-03).
# Ablations: name -> pe_kwargs changes on top of VIPER_BASE (other defaults in viper/models/pe.py::ViPER)
VIPER_BASE = {"lepe": True}
FIVE_SEEDS = {"viper", "abl_no_gates", "abl_static_gate", "abl_rgb_gate"}    # the method and its matched control get 5 seeds, other ablations 3
ABLATIONS = {
    "viper": {},
    "abl_no_gates": {"use_gate": False},                       # = RoPE-Mixed + LePE (matched control)
    "abl_no_value": {"lepe": False},                           # score path only
    "abl_static_gate": {"gate_src": "static"},                 # learned per-(layer, head, band) gates, no image
    "abl_rgb_gate": {"gate_src": "rgb"},                       # gates from learned RGB patch filters, no wavelets
    "abl_gate_qk": {"gate_k": True},
    "abl_dyadic": {"rope_init": "dyadic"},                      # dyadic octave grid instead of theta
    "abl_axial_base": {"rope_mode": "axial"},                   # fixed axial frequencies
    "abl_shared_gate": {"per_layer": False},
    "abl_haar": {"wavelet": "haar"},
    "abl_levels2": {"levels": 2},
    "abl_no_ll": {"use_ll": False},
    "abl_no_ctxconv": {"ctx_conv": False},
    "abl_bands1": {"n_bands": 1},                               # one gate per query (review R1: temperature control)
    "abl_bands2": {"n_bands": 2},
    "abl_bands8": {"n_bands": 8},
}

DOMAIN = ["eurosat", "resisc45", "bloodmnist", "dermamnist", "pathmnist", "tissuemnist", "dtd",
          "flowers102", "fgvc_aircraft"]


def job(name, cmd, mem=7, prio=50):
    return dict(name=name, cmd=cmd, mem_gb=mem, prio=prio)


def main_jobs(pes):
    out = []
    for s in ABL_SEEDS:                  # baselines: 3 seeds (user decision 2026-11-03)
        for pe in pes:
            out.append(job(f"in100_{pe}_s{s}",
                           f"python -m viper.train {IN100} --pe {pe} --seed {s} "
                           f"--out results/in100_deit_small/{pe}_s{s}", MEM.get(pe, 7), prio=10 * s))
    return out


def viper_jobs():
    out = []
    for s in SEEDS:
        for name, kw in ABLATIONS.items():
            if name not in FIVE_SEEDS and s not in DESIGN_SEEDS:
                continue
            kw = json.dumps({**VIPER_BASE, **kw}).replace('"', '\\"')
            out.append(job(f"in100_{name}_s{s}",
                           f"python -m viper.train {IN100} --pe viper --pe_kwargs \"{kw}\" --seed {s} "
                           f"--out results/in100_deit_small/{name}_s{s}", 7, prio=-1 if name in FIVE_SEEDS else 10 * s + 5))
    return out


def backbone_jobs(backbone="deit_tiny", k_baselines=4):
    """Second backbone: ViPER, its matched control and the k best baselines by ImageNet-100 *validation* top-1,
    3 seeds, same recipe as DeiT-S."""
    base = select_domain_methods(k=k_baselines + 1, always=("rope_mixed",))   # ViPER's base + top-k by val
    specs = {"viper": ("viper", VIPER_BASE), "abl_no_gates": ("viper", {**VIPER_BASE, **ABLATIONS["abl_no_gates"]})}
    specs.update({m: (m, None) for m in base if m != "viper"})
    out = []
    for s in ABL_SEEDS:
        for name, (pe, kw) in specs.items():
            extra = f" --pe_kwargs \"{json.dumps(kw).replace(chr(34), chr(92) + chr(34))}\"" if kw else ""
            out.append(job(f"{backbone}_{name}_s{s}",
                           f"python -m viper.train {IN100.replace('deit_small', backbone)} --pe {pe}{extra} --seed {s} "
                           f"--out results/in100_{backbone}/{name}_s{s}", 4 if backbone == "deit_tiny" else 12, prio=8 + s))
    return out


def ade20k_jobs(k_baselines=4, seeds=(0, 1, 2)):
    """ADE20K semantic segmentation: fine-tune the ImageNet-100 checkpoints of ViPER, its matched control and the
    same baselines as the backbone study (chosen by IN-100 validation top-1), seed s initialised from seed s."""
    base = select_domain_methods(k=k_baselines + 1, always=("rope_mixed",))
    names = ["viper", "abl_no_gates"] + [m for m in base if m != "viper"]
    # LookHere materialises 1024^2 attention matrices at 512^2: half batch x 2 accumulation steps (same effective batch)
    return [job(f"ade20k_{m}_s{s}", f"python -m viper.seg --init results/in100_deit_small/{m}_s{s} --seed {s} "
                + ("--accum 2 " if m == "lookhere" else "") + f"--out results/ade20k_deit_small/{m}_s{s}",
                18 if m == "lookhere" else 14, prio=-2 if m == "lookhere" else 7) for s in seeds for m in names]


def deit_base_jobs(seeds=(0,)):
    """Larger backbone (user decision 2026-11-04): ViPER, its matched control and STRING, 1 seed."""
    specs = {"viper": ("viper", VIPER_BASE), "abl_no_gates": ("viper", {**VIPER_BASE, **ABLATIONS["abl_no_gates"]}),
             "string_cayley": ("string_cayley", None),
             # completes Tab. 5 (user decision 2026-11-06)
             "rope_mixed": ("rope_mixed", None), "lookhere": ("lookhere", None), "lepe": ("lepe", None)}
    out = []
    for s in seeds:
        for name, (pe, kw) in specs.items():
            extra = f" --pe_kwargs \"{json.dumps(kw).replace(chr(34), chr(92) + chr(34))}\"" if kw else ""
            out.append(job(f"deit_base_{name}_s{s}", f"python -m viper.train {IN100.replace('deit_small', 'deit_base')} "
                           f"--pe {pe}{extra} --seed {s} --out results/in100_deit_base/{name}_s{s}", 18 if name == "lookhere" else 15, prio=0))
    return out


def transfer_jobs(datasets=("flowers102", "fgvc_aircraft", "dtd"), seeds=(0, 1, 2),
                  methods=("viper", "abl_no_gates", "string_cayley", "rope_mixed", "lepe", "lookhere")):
    """Transfer (user decision 2026-11-05): fine-tune the ImageNet-100 DeiT-S checkpoints (seed s from seed s) on
    small datasets with one fixed recipe for every method; the best validation epoch is evaluated on the test split.
    PE and its kwargs are copied from the source run so the architecture matches the checkpoint."""
    out = []
    for d in datasets:
        for s in seeds:
            for m in methods:
                src = f"results/in100_deit_small/{m}_s{s}"
                cfg = json.load(open(os.path.join(src, "config.json")))
                kw = cfg.get("pe_kwargs", "{}")
                extra = f" --pe_kwargs \"{kw.replace(chr(34), chr(92) + chr(34))}\"" if kw not in ("{}", "") else ""
                out.append(job(f"transfer_{d}_{m}_s{s}",
                               f"python -m viper.train --dataset {d} --backbone deit_small --pe {cfg['pe']}{extra} "
                               f"--init {src}/best.pt --epochs 100 --warmup 5 --bs 64 --lr 1e-4 --workers 6 "
                               f"--no_multires --seed {s} --out results/transfer/{d}/{m}_s{s}", 7, prio=5))
    return out


def domain_jobs(pes):
    out = []
    for s in SEEDS:
        for ds in DOMAIN:
            for pe in pes:
                out.append(job(f"dom_{ds}_{pe}_s{s}",
                               f"python -m viper.train --dataset {ds} --backbone deit_tiny --epochs 100 --warmup 5 "
                               f"--bs 64 --workers 4 --pe {pe} --seed {s} "
                               f"--out results/domain_deit_tiny/{ds}/{pe}_s{s}", 4, prio=100 + 10 * s))
    return out


def select_domain_methods(group="results/in100_deit_small", k=8, always=("viper", "none", "learned")):
    """Methods for the domain study: the always-kept references plus the best remaining methods by
    ImageNet-100 *validation* top-1 (seed-averaged over finished seeds; never test accuracy)."""
    import glob, os, re
    from collections import defaultdict
    val = defaultdict(list)
    for f in glob.glob(os.path.join(group, "*", "final.json")):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(os.path.dirname(f)))
        if m and (m.group(1) in BASELINES or m.group(1) == "viper"):
            val[m.group(1)].append(json.load(open(f))["best_val_top1"])
    ranked = sorted((m for m in val if m not in always), key=lambda m: -sum(val[m]) / len(val[m]))
    chosen = list(always) + ranked[: k - len(always)]
    print("IN-100 val top-1:", {m: round(sum(v) / len(v), 2) for m, v in sorted(val.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))})
    missing = [m for m in BASELINES + ["viper"] if m not in val]
    if missing:
        print("WARNING: no finished IN-100 run yet for", missing)
    return chosen


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "main":
        json.dump(main_jobs(BASELINES), open("jobs/main_in100.json", "w"), indent=1)
    elif what == "viper":
        json.dump(viper_jobs(), open("jobs/viper_in100.json", "w"), indent=1)
    elif what == "deit_tiny":
        json.dump(backbone_jobs(what), open(f"jobs/{what}_in100.json", "w"), indent=1)
    elif what == "deit_base":
        json.dump(deit_base_jobs(), open("jobs/deit_base_in100.json", "w"), indent=1)
    elif what == "ade20k":
        json.dump(ade20k_jobs(), open("jobs/ade20k.json", "w"), indent=1)
    elif what == "transfer":
        json.dump(transfer_jobs(), open("jobs/transfer.json", "w"), indent=1)
    elif what == "domain":            # run after IN-100 seed 0 has finished for every method
        methods = select_domain_methods()
        print("domain methods:", methods)
        json.dump(domain_jobs(methods), open("jobs/domain.json", "w"), indent=1)
    print("ok")

#!/usr/bin/env python
"""Aggregate results/<group>/<pe>_s<seed>/final.json into tables.

python scripts/aggregate.py results/pilot_in100_30ep            # markdown to stdout
python scripts/aggregate.py results/in100_deit_small --csv out.csv
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def load(group):
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(group, "*", "final.json"))):
        name = os.path.basename(os.path.dirname(f))
        m = re.match(r"(.+)_s(\d+)$", name)
        pe, seed = (m.group(1), int(m.group(2))) if m else (name, 0)
        j = json.load(open(f))
        j["seed"] = seed
        runs[pe].append(j)
    return runs


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    return f"{np.mean(vals):.2f}" + (f" ± {np.std(vals, ddof=0):.2f}" if len(vals) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("--csv")
    ap.add_argument("--mode", default="extrapolate", choices=["extrapolate", "interpolate"])
    args = ap.parse_args()
    runs = load(args.group)
    if not runs:
        print("no finished runs in", args.group); return
    res_keys = sorted({int(r) for rs in runs.values() for j in rs for r in j.get("multires", {})})
    hdr = ["PE", "n", "PE params", "val top1", "test top1", "test top5", "macro-F1"] + [f"@{r}" for r in res_keys]
    rows = []
    for pe, rs in sorted(runs.items(), key=lambda kv: -np.mean([j["test"]["top1"] for j in kv[1]])):
        row = [pe, str(len(rs)), f"{rs[0]['pe_params']:,}", ms([j["best_val_top1"] for j in rs]),
               ms([j["test"]["top1"] for j in rs]), ms([j["test"]["top5"] for j in rs]),
               ms([j["test"]["macro_f1"] for j in rs])]
        for r in res_keys:
            vals = []
            for j in rs:
                e = j.get("multires", {}).get(str(r), {})
                e = e.get(args.mode, e.get("extrapolate"))
                vals.append(e["top1"] if e else None)
            row.append(ms(vals))
        rows.append(row)
    w = [max(len(r[i]) for r in [hdr] + rows) for i in range(len(hdr))]
    print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |")
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            cw = csv.writer(f); cw.writerow(hdr); cw.writerows(rows)


if __name__ == "__main__":
    main()

"""Stream ImageNet-1k (HF `ILSVRC/imagenet-1k`, gated: accept the terms and `hf auth login` first) into packed
JPEG files for viper.data.datasets.PackedJpegs, one parquet shard at a time, so the raw ~150 GB never sits on disk.

  train: re-encoded at 256 px shorter side (JPEG q=90; smaller images kept as is)  -> ~45 GB
  val:   original bytes (used as the test set and for the 64..448 px resolution sweep)

  python -m scripts.prep_imagenet1k [--splits validation,train] [--procs 32]

Output in $VIPER_DATA/imagenet1k (default data/imagenet1k): {train,val}.bin, {train,val}_off.npy,
{train,val}_lab.npy. Resumable: progress.json records the finished shards and the byte length of each .bin.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent         # scripts/queue.py would shadow the stdlib queue module
sys.path = [q for q in sys.path if Path(q or ".").resolve() != _HERE]
sys.path.insert(0, str(_HERE.parent))

import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from viper.data.datasets import ROOT  # noqa: E402

REPO = "ILSVRC/imagenet-1k"
SHORT = 256


def shrink(b: bytes) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(b))
    w, h = im.size
    if min(w, h) <= SHORT:
        return b
    s = SHORT / min(w, h)
    im = im.convert("RGB").resize((max(SHORT, round(w * s)), max(SHORT, round(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="validation,train")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--prefetch", type=int, default=3)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    out = ROOT / "imagenet1k"; out.mkdir(parents=True, exist_ok=True)
    dl = out / "_dl"
    prog_f = out / "progress.json"
    prog = json.load(open(prog_f)) if prog_f.exists() else {}
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    pool = Pool(args.procs)

    for split in args.splits.split(","):
        name = "val" if split == "validation" else split
        shards = sorted(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
        st = prog.setdefault(name, {"done": [], "bytes": 0, "n": 0})
        binf = out / f"{name}.bin"
        offs_f, labs_f = out / f"{name}_off_part.npy", out / f"{name}_lab_part.npy"
        offs = list(np.load(offs_f)[: st["n"] + 1]) if offs_f.exists() and st["n"] else [0]
        labs = list(np.load(labs_f)[: st["n"]]) if labs_f.exists() and st["n"] else []
        with open(binf, "ab") as fb:
            fb.truncate(st["bytes"])            # drop a partially written shard
        todo = [s for s in shards if s not in st["done"]]
        print(f"{name}: {len(shards)} shards, {len(todo)} to do, {st['n']} images so far", flush=True)

        def fetch(s):
            return hf_hub_download(REPO, s, repo_type="dataset", local_dir=dl)

        with ThreadPoolExecutor(args.prefetch) as ex, open(binf, "ab") as fb:
            futs = [ex.submit(fetch, s) for s in todo[: args.prefetch]]
            for k, s in enumerate(todo):
                path = futs[k].result()
                if k + args.prefetch < len(todo):
                    futs.append(ex.submit(fetch, todo[k + args.prefetch]))
                t = pq.read_table(path, columns=["image", "label"])
                imgs = [d["bytes"] for d in t.column("image").to_pylist()]
                lab = t.column("label").to_pylist()
                if name == "train":
                    imgs = pool.map(shrink, imgs, chunksize=16)
                for b, y in zip(imgs, lab):
                    fb.write(b); offs.append(offs[-1] + len(b)); labs.append(y)
                fb.flush(); os.fsync(fb.fileno())
                os.remove(path)
                st["done"].append(s); st["bytes"] = int(offs[-1]); st["n"] = len(labs)
                np.save(offs_f, np.asarray(offs, dtype=np.int64)); np.save(labs_f, np.asarray(labs, dtype=np.int16))
                json.dump(prog, open(prog_f, "w"))
                print(f"{name} {len(st['done'])}/{len(shards)}: {st['n']} images, {st['bytes']/1e9:.1f} GB", flush=True)
        if len(st["done"]) == len(shards):
            np.save(out / f"{name}_off.npy", np.asarray(offs, dtype=np.int64))
            np.save(out / f"{name}_lab.npy", np.asarray(labs, dtype=np.int16))
            assert min(labs) >= 0 and max(labs) == 999, (min(labs), max(labs))
            print(f"{name} complete: {len(labs)} images, {len(set(labs))} classes", flush=True)


if __name__ == "__main__":
    main()

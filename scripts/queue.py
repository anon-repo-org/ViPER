#!/usr/bin/env python
"""GPU job queue for ViPER experiments.

Job files are JSON lists of {"name", "cmd", "mem_gb", "prio"} where `cmd` is a full
`python -m viper.train ... --out <dir>` command. Job files are re-read every poll, so
new jobs can be appended while the queue runs. A job is done when its log.txt contains
`FINAL test_top1=`. Jobs auto-resume from last.pt, so a crashed job (OOM from a
co-tenant etc.) is relaunched later (at most --max_launches times).

Each launched job writes <out>/pid; a restarted queue treats jobs whose pid is alive as
running, so restarting the queue never duplicates work.

  nohup python scripts/queue.py jobs/*.json --gpus 5,6,7 >> runs/queue.log 2>&1 &

Lower `prio` runs first; ties keep file order. GPUs listed in $VIPER_FORBIDDEN_GPUS
(e.g. "1,3": reserved for other users of a shared server) are refused.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = set(int(g) for g in os.environ.get("VIPER_FORBIDDEN_GPUS", "").split(",") if g)


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def gpu_free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    free = {}
    for line in out.strip().splitlines():
        i, used, total = [int(v) for v in line.split(",")]
        free[i] = total - used
    return free


def out_dir(cmd):
    o = re.search(r"--out\s+(\S+)", cmd).group(1)
    return o if os.path.isabs(o) else os.path.join(ROOT, o)


def done(out):
    p = os.path.join(out, "log.txt")
    if not os.path.exists(p):
        return False
    with open(p, "rb") as f:
        f.seek(max(0, os.path.getsize(p) - 4000))
        return re.search(rb"FINAL (test_top1|seg_miou)=", f.read()) is not None


def alive(out):
    """-> (pid, gpu) if a job launched by any queue instance is still running."""
    p = os.path.join(out, "pid")
    if not os.path.exists(p):
        return None
    try:
        pid, gpu = [int(v) for v in open(p).read().split()]
        os.kill(pid, 0)
        cmdline = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
        return (pid, gpu) if out.split(ROOT + "/")[-1] in cmdline or out in cmdline else None
    except (OSError, ValueError):
        return None


def load_jobs(patterns):
    jobs, seen = [], set()
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for i, j in enumerate(json.load(open(f))):
                if j["name"] in seen:
                    continue
                seen.add(j["name"])
                j["out"] = out_dir(j["cmd"])
                j.setdefault("mem_gb", 8)
                j.setdefault("prio", 50)
                j["order"] = len(jobs)
                jobs.append(j)
    return sorted(jobs, key=lambda j: (j["prio"], j["order"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", help="job files or globs")
    ap.add_argument("--gpus", required=True,
                    help="comma list (e.g. 5,6,7) or a file holding one; a file is re-read every poll, "
                         "so GPUs can be added/removed without restarting the queue")
    ap.add_argument("--max_per_gpu", type=int, default=2)
    ap.add_argument("--margin_gb", type=float, default=1.5)
    ap.add_argument("--cooldown", type=int, default=300)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max_launches", type=int, default=6)
    ap.add_argument("--min_disk_gb", type=float, default=40)
    args = ap.parse_args()
    def read_gpus():
        txt = open(args.gpus).read() if os.path.isfile(args.gpus) else args.gpus
        g = [int(v) for v in txt.replace("\n", ",").split(",") if v.strip()]
        bad = set(g) & FORBIDDEN
        if bad:
            log(f"ignoring reserved GPUs {sorted(bad)} (VIPER_FORBIDDEN_GPUS)")
        return [x for x in g if x not in FORBIDDEN]
    gpus = read_gpus()

    launches, procs = {}, {}
    last_launch = {g: 0.0 for g in gpus}
    jobs = load_jobs(args.jobs)
    log(f"{len(jobs)} jobs, {sum(done(j['out']) for j in jobs)} done, gpus={gpus}")
    while True:
        new_gpus = read_gpus()
        if new_gpus != gpus:
            log(f"GPU list changed: {gpus} -> {new_gpus}")
            gpus = new_gpus
            for g in gpus:
                last_launch.setdefault(g, 0.0)
        jobs = load_jobs(args.jobs)
        for name, p in list(procs.items()):
            if p.poll() is not None:
                j = next((x for x in jobs if x["name"] == name), None)
                log(f"{name} exited rc={p.returncode} (done={done(j['out']) if j else '?'})")
                del procs[name]
        running = {j["name"]: alive(j["out"]) for j in jobs}
        running = {k: v for k, v in running.items() if v}
        per_gpu = {g: sum(1 for v in running.values() if v[1] == g) for g in gpus}   # GPUs removed from the
        # list keep their running jobs until they finish; nothing new is launched there
        pending = [j for j in jobs if j["name"] not in running and not done(j["out"])
                   and launches.get(j["name"], 0) < args.max_launches]
        if not pending and not running:
            log("all jobs finished (or exhausted their launches); idling for new jobs")
        st = os.statvfs(ROOT)
        disk_gb = st.f_bavail * st.f_frsize / 1e9
        free = gpu_free_mb()
        for g in gpus:
            if not pending or disk_gb < args.min_disk_gb:
                break
            if per_gpu[g] >= args.max_per_gpu or time.time() - last_launch[g] < args.cooldown:
                continue
            # highest-priority pending job that fits on this GPU (no head-of-line blocking by big jobs)
            j = next((x for x in pending if free.get(g, 0) >= (x["mem_gb"] + args.margin_gb) * 1024), None)
            if j is None:
                continue
            os.makedirs(j["out"], exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
            cmd = j["cmd"].replace("python ", PY + " ", 1) if j["cmd"].startswith("python ") else j["cmd"]
            lf = open(os.path.join(j["out"], "stdout.txt"), "a")
            p = subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            # shell=True: the python child is the shell's only child; record the shell pid (cmdline holds --out)
            with open(os.path.join(j["out"], "pid"), "w") as f:
                f.write(f"{p.pid} {g}")
            procs[j["name"]] = p
            launches[j["name"]] = launches.get(j["name"], 0) + 1
            last_launch[g] = time.time()
            per_gpu[g] += 1
            pending.remove(j)
            free[g] = free.get(g, 0) - j["mem_gb"] * 1024
            log(f"launched {j['name']} on GPU {g} (launch {launches[j['name']]})")
        if disk_gb < args.min_disk_gb:
            log(f"low disk ({disk_gb:.0f} GB); not launching")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()

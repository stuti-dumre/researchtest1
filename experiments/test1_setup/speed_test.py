"""Test 1, S0-4: speed of LoRA training on this machine, one job alone, then two jobs at once if memory allows."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
PREP = HERE.parents[1] / "prepare"
OUT = HERE / "runs"
GB = 1024 ** 3


def start(name):
    cmd = [sys.executable, "train_lora.py", "--arm", "minim_paper", "--tier", "1000", "--steps", "20",
           "--name", str(OUT / name)]
    log = open(OUT / f"{name}_log.txt", "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=PREP, stdout=log, stderr=subprocess.STDOUT)


def wait(procs):
    """Wait for all jobs; return the peak total memory (resident set) of the jobs and their children, in GB."""
    peak = 0.0
    while any(p.poll() is None for p in procs):
        tot = 0
        for p in procs:
            try:
                pr = psutil.Process(p.pid)
                tot += sum(x.memory_info().rss for x in [pr] + pr.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
        peak = max(peak, tot / GB)
        time.sleep(1)
    return peak, [p.returncode for p in procs]


def per_step(name):
    log = pd.read_csv(OUT / name / "loss.csv").set_index("step").sec
    return (log[20] - log[1]) / 19


res = {}
free0 = psutil.virtual_memory().available / GB
t0 = time.time()
peak1, rc1 = wait([start("speed_single")])
s1 = per_step("speed_single")
res["single"] = {"exit": rc1, "peak_gb": round(peak1, 2), "sec_per_step": round(s1, 2),
                 "steps_per_min": round(60 / s1, 2), "wall_sec": round(time.time() - t0), "free_gb_before": round(free0, 1)}
print("single job:", res["single"], flush=True)

free = psutil.virtual_memory().available / GB
need = 2 * peak1 + 2
if free >= need:
    t0 = time.time()
    peak2, rc2 = wait([start("speed_pair_a"), start("speed_pair_b")])
    sa, sb = per_step("speed_pair_a"), per_step("speed_pair_b")
    res["pair"] = {"exit": rc2, "peak_gb": round(peak2, 2), "sec_per_step": [round(sa, 2), round(sb, 2)],
                   "steps_per_min_total": round(60 / sa + 60 / sb, 2), "wall_sec": round(time.time() - t0),
                   "free_gb_before": round(free, 1),
                   "speed_up_vs_single": round((60 / sa + 60 / sb) / (60 / s1), 2)}
    print("two jobs at once:", res["pair"], flush=True)
else:
    res["pair"] = {"skipped": f"free memory {free:.1f} GB < needed {need:.1f} GB (2 x single peak + 2 GB)"}
    print("two jobs at once: skipped:", res["pair"]["skipped"], flush=True)
json.dump(res, open(OUT / "speed_results.json", "w", encoding="utf-8"), indent=1)
print("saved", OUT / "speed_results.json")

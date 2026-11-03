#!/usr/bin/env python3
"""Collect every Qwen3.8-27B bench JSON on the GPU box into one table.

Groups by arm prefix and cell, averages reps, prints markdown.
"""
import glob
import json
import os
import re
from collections import defaultdict

files = glob.glob("/home/amatveev/qwen38/results/*.json")
arms = defaultdict(list)

for f in files:
    name = os.path.basename(f)[:-5]
    if not re.search(r"q38|q355_c64", name):
        continue
    cell = "c8" if ("_c8" in name or "c64" in name) else "c1"
    key = re.sub(r"_r\d+$", "", name)
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if "mean_tpot_ms" not in d:
        continue
    arms[(key, cell)].append((d["mean_tpot_ms"], d.get("output_throughput", 0)))

print(f"{'arm':28s} {'cell':4s} {'tpot ms':>18s} {'tok/s':>8s} {'n':>3s}")
for (key, cell), vals in sorted(arms.items()):
    t = [v[0] for v in vals]
    o = [v[1] for v in vals]
    ts = " / ".join(f"{x:.2f}" for x in t)
    print(f"{key:28s} {cell:4s} {ts:>18s} {sum(o)/len(o):8.1f} {len(vals):3d}")

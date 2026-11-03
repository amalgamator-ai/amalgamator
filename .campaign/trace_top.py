#!/usr/bin/env python3
"""Summarise a Kineto/Chrome trace: top kernels by self time, plus the decode
step structure, without any vLLM dependency.

usage: trace_top.py <trace.json.gz> [top_n]
"""
import gzip
import json
import sys
from collections import Counter

path = sys.argv[1]
top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 30

if path.endswith(".gz"):
    f = gzip.open(path, "rt")
else:
    f = open(path)
data = json.load(f)
events = data["traceEvents"] if isinstance(data, dict) else data

kern = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel", "gpu_user_annotation")]
if not kern:
    kern = [e for e in events if e.get("ph") == "X" and e.get("cat") != "cpu_op"]

cpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation")]

total = sum(e.get("dur", 0) for e in kern)
print(f"kernel events: {len(kern)}  total GPU-kernel time: {total/1000:.2f} ms")
if kern:
    span = max(e["ts"] + e.get("dur", 0) for e in kern) - min(e["ts"] for e in kern)
    print(f"kernel wall span: {span/1000:.2f} ms")
print()

byname = Counter()
count = Counter()
for e in kern:
    byname[e.get("name", "?")] += e.get("dur", 0)
    count[e.get("name", "?")] += 1

print(f"{'kernel':60s} {'ms':>9s} {'%':>6s} {'calls':>7s} {'us/call':>8s}")
for name, dur in byname.most_common(top_n):
    n = count[name]
    print(f"{name[:60]:60s} {dur/1000:9.2f} {100*dur/total:6.1f} {n:7d} {dur/n:8.1f}")

if cpu:
    tot_cpu = sum(e.get("dur", 0) for e in cpu)
    print(f"\nCPU ops: {len(cpu)}  total {tot_cpu/1000:.2f} ms")
    cby = Counter()
    for e in cpu:
        cby[e.get("name", "?")] += e.get("dur", 0)
    for name, dur in cby.most_common(10):
        print(f"  {name[:60]:60s} {dur/1000:9.2f} ms")

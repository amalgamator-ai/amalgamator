#!/usr/bin/env python3
"""Per-step decode composition from a Kineto trace.

Takes the tail of the trace (after the prefill burst), counts real kernels that
are not CUDA-graph replay annotations, and reports time per decode step.
"""
import gzip
import json
import sys
from collections import Counter

path, step_ms = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 11.17
f = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
events = json.load(f)["traceEvents"]

kern = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel")]
kern.sort(key=lambda e: e["ts"])
if not kern:
    raise SystemExit("no kernel events")

t0, t1 = kern[0]["ts"], max(e["ts"] + e.get("dur", 0) for e in kern)
# Decode = last 60% of the window; prefill sits at the front.
lo = t0 + 0.4 * (t1 - t0)
dec = [e for e in kern if e["ts"] >= lo]
span = max(e["ts"] + e.get("dur", 0) for e in dec) - lo
steps = span / (step_ms * 1000.0)
print(f"decode window {span/1000:.1f} ms -> {steps:.1f} steps at {step_ms} ms")

byname, calls = Counter(), Counter()
for e in dec:
    byname[e["name"]] += e["dur"]
    calls[e["name"]] += 1
tot = sum(byname.values())
print(f"kernel time in window: {tot/1000:.1f} ms ({100*tot/span:.0f}% of wall)\n")
print(f"{'kernel':54s} {'us/step':>9s} {'calls/step':>10s} {'%':>5s}")
for name, dur in byname.most_common(25):
    print(f"{name[:54]:54s} {dur/steps:9.1f} {calls[name]/steps:10.1f} {100*dur/tot:5.1f}")
print(f"\n{'TOTAL':54s} {tot/steps:9.1f}")

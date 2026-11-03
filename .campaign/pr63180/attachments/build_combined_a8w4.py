#!/usr/bin/env python3
"""Build the combined a8w4 tuned_fmoe CSV for the crit5 long-ISL A/B.

Union of the campaign's fused-shape tuning (hetero_tuned_fused_v1.csv) and
the raw hetero tuning sweep (hetero_tuned_raw.csv), restricted to the exact
signatures this server looks up at runtime (verified against the cycle-1
"no tuned FlyDSL config" miss lines):

  ('gfx950', 256, M, 5120, 640, 385, 7, Silu, bf16, fp8e4m3, fp4x2, per_1x32, 1, 0)  # ON arm (fold)
  ('gfx950', 256, M, 5120, 640, 384, 6, Silu, bf16, fp8e4m3, fp4x2, per_1x32, 1, 0)  # OFF arm (stock)
  ('gfx950', 256, M, 5120, 640, 128, 3, Silu, bf16, fp8e4m3, fp4x2, per_1x32, 1, 0)  # draft MoE (both arms)

Best row (lowest total `us`) per index key — aiter's merge would otherwise
raise on duplicate shape entries. Injected identically into both arms via
AITER_CONFIG_FMOE, so each arm runs its best tuned configs (fair A/B).
"""
import csv
from collections import defaultdict

T = "/data/amatveev/rocm10/tuned"
FILES = [f"{T}/hetero_tuned_fused_v1.csv", f"{T}/hetero_tuned_raw.csv"]
WANT = {("385", "7"), ("384", "6"), ("128", "3")}
OUT = f"{T}/combined_a8w4_5120x640_v1.csv"

def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]

hdr = None
best = {}
for path in FILES:
    h, rows = load(path)
    if hdr is None:
        hdr = h
    assert h == hdr, f"header mismatch in {path}"
    H = {n: i for i, n in enumerate(hdr)}
    is_fused_v1 = path.endswith("hetero_tuned_fused_v1.csv")
    kept = 0
    for row in rows:
        if not (row[H["gfx"]] == "gfx950"
                and row[H["cu_num"]] == "256"
                and row[H["model_dim"]] == "5120"
                and row[H["inter_dim"]] == "640"
                and row[H["q_dtype_a"]] == "torch.float8_e4m3fn"
                and row[H["q_dtype_w"]] == "torch.float4_e2m1fn_x2"
                and row[H["q_type"]] == "QuantType.per_1x32"
                and row[H["act_type"]] == "ActivationType.Silu"
                and (row[H["expert"]], row[H["topk"]]) in WANT):
            continue
        # The 385/7 fold shape goes through AITER's heterogeneous wrapper
        # (_use_fhmoe_wrappers), which only accepts the two-stage FlyDSL
        # family: kn1 'flydsl_moe1_*' and kn2 'flydsl_moe2_*' (NOT the
        # flydslv2 'flydsl_moe2_layout_*' family — the TUNE_ONLY default,
        # fhmoe-incompatible) with run_1stage=0. hetero_tuned_raw.csv's
        # 385/7 rows are layout-family (the first injection attempt armed
        # the fold OFF via the init probe), so the fused shape takes rows
        # ONLY from hetero_tuned_fused_v1.csv (tuned through the fhmoe
        # path). The 384/6 stock and 128/3 draft shapes run the plain
        # fused_moe path, which accepts any family — raw's layout rows are
        # fine there (and are the tuned best).
        if (row[H["expert"]], row[H["topk"]]) == ("385", "7"):
            if not is_fused_v1:
                continue
            kn1, kn2 = row[H["kernelName1"]], row[H["kernelName2"]]
            assert kn1.startswith("flydsl_moe1_"), f"bad kn1 {kn1}"
            assert kn2.startswith("flydsl_moe2_") and not kn2.startswith(
                "flydsl_moe2_layout_"
            ), f"bad kn2 {kn2}"
            assert row[H["run_1stage"]] == "0", "run_1stage must be 0"
        k = tuple(row[H[c]] for c in ("gfx", "cu_num", "token", "model_dim",
                                      "inter_dim", "expert", "topk", "act_type",
                                      "dtype", "q_dtype_a", "q_dtype_w", "q_type",
                                      "use_g1u1", "doweight_stage1"))
        us = float(row[H["us"]] or 1e9)
        if k not in best or us < float(best[k][H["us"]] or 1e9):
            best[k] = row
        kept += 1
    print(f"{path}: {kept} candidate rows")

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    for k in sorted(best, key=lambda k: (int(k[5]), int(k[6]), int(k[2]))):
        w.writerow(best[k])
print(f"wrote {OUT}: {len(best)} rows")

cov = defaultdict(list)
for k in best:
    cov[(k[5], k[6])].append(int(k[2]))
for shape in sorted(cov):
    ms = sorted(cov[shape])
    print(f"  shape E/topk={shape[0]}/{shape[1]}: {len(ms)} Ms: {ms}")
    # the two measured prefill Ms must be tuned for the A/B shapes; the aux
    # draft-MoE shape (128/3) only has a partial grid — its uncovered Ms stay
    # heuristic in BOTH arms (identical), which is fine.
    if shape in {("384", "6"), ("385", "7")}:
        for need in (8192, 16384):
            assert need in ms, f"MISSING M={need} for shape {shape}"
assert len(best) == len(set(best)), "duplicate keys"
print("coverage OK (8192+16384 present for 384/6 and 385/7)")

Thanks for the detailed repro and the five arms — that's a lot of work.

Our numbers next to yours, same shapes, TP4:

| ISL | ours (`bench serve`, median TTFT) | yours (`bench forward`) |
|---:|---:|---:|
| 256 | 24.9 | 17.4 |
| 384 | 26.7 | 19.3 |
| 512 | 30.4 | 20.7 |
| 768 | 50.9 | 24.2 |
| 1024 | 51.3 | 27.1 |
| 1536 | 52.3 | 32.1 |
| 2048 | 51.7 | 37.3 |
| 3072 | 52.4 | — |

Ours is ~1.4× slower at 256 and at 2048, but ~2× at 768–1024. We are not sure why. A served TTFT does include sampling and scheduler overhead that a forward replay does not, but that alone shouldn't give 2× at some shapes and 1.4× at others — and our curve is nearly flat from 768 to 3072 (50.9 → 52.4 ms while tokens quadruple), while yours scales.

The runs differ in more than the instrument:

| | ours | yours |
|---|---|---|
| GPU | MI355X | MI350X |
| image | pinned `nightly-rocm100`, late Oct | `nightly-rocm100-00b6c4c5…`, Oct 17 |
| block size | 128 | 64 |
| engram | `cpu_offload: true` | `cpu_offload: false` |
| rejection sampling | `block` | `standard` |
| frontend | `VLLM_USE_RUST_FRONTEND=1` | default |
| tuned CSV | all three shapes overridden (385/7, 384/6, 128/3) | only 385/7 overridden |

We'll align these one at a time on our box and post the numbers. If one of them looks like the obvious first variable to change, tell us and we'll start there.

# Performance

Measured in September and October 2026 on two Arc Pro B70s bound to `xe`, with a 275 W power
cap and no PCIe P2P between the cards. The runs used a Kubernetes deployment with the same
image, flags, environment and patches as `compose.yaml`. This Compose port has not been
benchmarked separately.

## The current configuration

vLLM v0.31.0, FP8 weights as W8A16, MTP at depth 2, prefix caching with a 64-token match unit.

| | Result |
| --- | --- |
| Single stream | **51.7 tok/s** |
| 16 concurrent, aggregate | **469.7 tok/s** |
| One agent, 525-token prompt, decode | 71.7 tok/s |
| One agent, 31K-token prompt, decode | 61.8 tok/s |
| One agent, 31K-token prompt, task time | 3.2 to 4.0 s, 0.29 s to first token |
| One agent, 31K-token prompt, prefix hit rate | 0.995 to 0.997 |
| 16 agents, 6,704-token prompt | **1.34 tasks/s**, 0.976 prefix hit rate |
| 12 concurrent requests at 45K tokens each | 12/12 correct, no preemptions |

The single-stream and 16-concurrent figures were measured with prefix caching off, and the
one-agent decode figures before the 64-token match unit was added. Neither change affects
decode.

## Memory

| Per card | |
| --- | --- |
| Weights | 14.21 GiB |
| KV cache (capped) | 10.75 GiB |
| Recorded graphs | 1.02 GiB |
| VRAM after startup | 28.27 and 28.11 GiB of 31.89 |
| VRAM peak under the 12 × 45K stress test | 29.39 and 29.23 GiB |

The KV pool holds 588,800 tokens across both cards, enough for 3.83 requests at the full
153,600-token window. nvtop counts about 1.6 GiB per card that vLLM cannot allocate, so the
usable ceiling is about 30.3 GiB. One TP worker peaked at 13.7 GiB of host RAM.

Startup takes about 150 s from process start to serving without MTP: 52 s of `torch.compile`
and 7 s of graph capture. MTP raises graph capture to 20 s.

## How each change moved throughput

Generation benchmark: a 70-token prompt and 800 generated tokens, three runs each, timed from
the client.

| Configuration | Single stream | 16 concurrent, aggregate |
| --- | --- | --- |
| One card, 4-bit GPTQ, graphs | 32.9 tok/s | 349.5 tok/s |
| Two cards, FP8, eager | 20.2 tok/s | 316.6 tok/s |
| Two cards, FP8, graphs | 33.4 tok/s | 390.6 tok/s |
| Two cards, FP8 as stock W8A8, graphs | 20.6 tok/s | 276.5 tok/s |
| Two cards, FP8, graphs, v0.31.0 | 34.1 tok/s | 400.6 tok/s |
| **Two cards, FP8, graphs, v0.31.0, MTP 2** | **51.7 tok/s** | **469.7 tok/s** |

Graph capture works across both cards even though vLLM logs that XPU graphs only support
single-GPU execution. It lifted single stream 65%. Most of the eager gap was kernel launch
overhead.

Without speculation, one card's 14.9 GB share of the weights at 608 GB/s puts the decode
ceiling near 41 tok/s. 34.1 tok/s is 83% of it. MTP passes the ceiling because each weight
read yields about two tokens.

## MTP depth

Measured on the previous image, vLLM 0.27.

| | No MTP | Depth 4 | Depth 2 |
| --- | --- | --- | --- |
| Single stream | 33.4 tok/s | 47.3 tok/s | **51.0 tok/s** |
| 16 concurrent, aggregate | 390.6 tok/s | 338.6 tok/s | **432.1 tok/s** |
| Acceptance per draft position | | 0.60, 0.33, 0.16, 0.08 | 0.62, 0.36 |
| Recorded graphs per card | 2.14 GiB | 5.76 GiB | 4.97 GiB |

The last two positions at depth 4 add about 0.24 tokens per pass but widen every verify batch
from 3 to 5 tokens per request.

Acceptance depends on temperature. The checkpoint's `generation_config.json` defaults the server
to 1.0, where the two positions accepted 0.61 and 0.34. At 0.2 they accepted 0.79 to 1.00 and
0.61 to 1.00. On agent tool-call traffic each two-token draft averaged 1.63 accepted tokens,
about 2.6 tokens per pass.

## Agent workload

Each task is a two-turn tool call: the model picks one of three tools, receives a canned result,
and writes an answer. 16 agents run concurrently with `--max-tokens 256`, on a 6,704-token
prompt they share.

| Configuration | Tasks per second | Prefix hit rate |
| --- | --- | --- |
| One card, 4-bit GPTQ | 0.563 | 0.909 |
| Two cards, FP8, eager | 0.984 | 0.947 |
| Two cards, FP8, graphs | 1.025 | 0.947 |
| Two cards, FP8, graphs, v0.31.0 | 1.18 | 0.947 |
| v0.31.0, MTP 2, no match unit | 0.492 | 0.71 |
| **v0.31.0, MTP 2, 64-token match unit and shared-prefix checkpoint** | **1.34** | **0.976** |

This model's attention block is 1,600 tokens, and prefix hits land only on block boundaries.
With a draft model vLLM drops the last matched block so the draft head can recompute it, which
made every MTP request re-prefill 1,600 extra tokens. Sixteen agents starting together are
prefill-bound, so MTP ran 2.4x slower there until `--prefix-match-unit=64` and
`--enable-mamba-shared-prefix-checkpoint` cut the dropped span to 64 tokens.

## Long context

Twelve concurrent requests, each a 44.7K to 44.8K-token prompt with a passphrase buried mid-way,
at temperature 0.2.

| | Without MTP | With MTP |
| --- | --- | --- |
| Passphrases quoted back correctly | 12/12 | 12/12 |
| Peak KV cache use | 87.2% | |
| Preemptions | 0 | 0 |
| Prefill | about 4,475 tok/s | |
| Decode, all twelve running | 68 tok/s aggregate | |
| Draft acceptance | | 76 to 94% |

## Reproducing

- Send one throwaway request after every restart. A cold endpoint reads up to 14% low.
- `vllm:num_preemptions_total` on `/metrics` rising means the window is too wide for the pool.
  Running out preempts and recomputes. It never fails a request.

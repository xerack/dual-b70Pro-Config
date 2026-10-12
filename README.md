# Dual Arc Pro B70 vLLM config

Docker Compose setup for serving Qwen3.8-27B (FP8 with MTP speculative decoding) from vLLM
v0.31.0 across two Intel Arc Pro B70s using tensor parallelism. It exposes an
OpenAI-compatible API on port 8000.

## Running it

```bash
cp .env.example .env
docker compose up -d
docker compose logs -f vllm
```

Set `VLLM_API_KEY` in `.env` before starting. Clients send it as a bearer token.

The `fetch-weights` service downloads the checkpoint from Hugging Face into `./models` on
first start, about 31 GB. Loading and graph capture take around 4 minutes after that. The
server is ready when `/health` returns 200.

To switch models, change `MODEL_REPO` and `MODEL_DIR` in `fetch-weights`, and the `--model`
and `--chat-template` paths in `vllm`.

## Host requirements

- Two Arc Pro B70s bound to the `xe` kernel driver, visible under `/dev/dri`.
- At least 24 GiB of free host RAM. One TP worker peaked at 13.7 GiB.
- Use `nvtop` to confirm the cards are working. `intel_gpu_top` cannot read a card bound to
  `xe`. Clock speed is not a load signal, because a Level Zero context holds the card at its
  top frequency while idle. Watch utilization and VRAM growth as the KV cache fills.

## The patches

The image is pinned by digest, and both patches match that build only. They run at every
container start, are idempotent, and stop the server from starting if an anchor is missing.

- `patch_vllm_worker_affinity.py` gives each TP worker its own `ZE_AFFINITY_MASK` as it
  spawns, so each worker sees exactly one card. Exposing both cards to one process fails on
  boards without PCIe P2P and duplicates host memory under `xe`.
- `patch_fp8_stack.py` runs the checkpoint's block-FP8 layers as W8A16, and sends small
  batches to a dedicated Xe2 kernel when the image provides one.

Both come from
[SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
at `1a76136`, under the MIT license in `patches/LICENSE`. The affinity patch is edited to
work on v0.31.0: it keeps two of the original hunks and fails on a missing file.

## Container settings

These are the cookbook's dual-B70 contract. Leaving any of them out stops startup or hangs
the first all-reduce.

| Setting | Why |
| --- | --- |
| `ipc: host` | Required by the cookbook's TP2 recipe. |
| `cap_add: SYS_PTRACE` | oneCCL exchanges memory handles between workers through pidfd. With `CCL_ZE_IPC_EXCHANGE=sockets` instead, startup hung at the first all-reduce. |
| `working_dir: /` | The image's `/workspace/vllm` source tree would shadow the installed package the patches edit. |
| `CCL_SYCL_*_SIMPLE_THRESHOLD`, `CCL_SYCL_ALLTOALL_TMP_BUF` | Keep oneCCL on algorithms that never open peer device memory, which fails between B70s without P2P. See [intel/llm-scaler#594](https://github.com/intel/llm-scaler/issues/594). |
| `ONEAPI_DEVICE_SELECTOR=level_zero:*` | Without it the runtime can pick the CPU OpenCL device and serve from it slowly, with nothing in the logs. A wildcard, because each worker sees one card at index 0. |
| `ZE_AFFINITY_MASK=0,1`, `B70_WORKER_AFFINITY=1` | The starting mask, which the affinity patch narrows per worker. |

Do not set `PYTORCH_ALLOC_CONF=expandable_segments:True`. With the oneCCL 2022.1.2 in this
image, a buffer from that allocator segfaults the first all-reduce.

## Serving flags

Quantization, `--max-num-seqs`, graph capture and the memory settings constrain each other.
Getting one wrong stops the engine during startup.

| Flag | Why |
| --- | --- |
| `--served-model-name=Qwen/Qwen3.8-27B` | Clients use the upstream name, so the local path can change. |
| `--language-model-only` | Skips the checkpoint's 16-bit vision tower, returning about 0.9 GiB to the KV cache. |
| No `--quantization` | The checkpoint declares FP8 itself. |
| `--kv-cache-dtype=fp8` | Halves KV cache memory. |
| `--max-model-len=153600` | Shares one KV pool with concurrency. The startup log's "Maximum concurrency" line says how many requests this long fit. Running out preempts and recomputes. |
| `--enable-prefix-caching` | Skips re-prefilling the system prompt and tool definitions agents resend every turn. |
| `--prefix-match-unit=64` | Prefix hits land on 64-token boundaries inside the 1,600-token hybrid block, so MTP drops 64 tokens of a hit. Must divide the "kv lcm block sizes" value in the log. |
| `--enable-mamba-shared-prefix-checkpoint` | Saves recurrent state where MTP's prefix lookup resumes. Without it, prompts that diverge mid-block miss the cache entirely. |
| `--gpu-memory-utilization=0.90` | Only the startup free-memory check. The next flag sets the real budget. |
| `--kv-cache-memory-bytes=11542724608` | 10.75 GiB of KV cache per card. Graph capture adds 1.02 GiB with MTP. Each card holds 28.3 of 31.89 GiB after startup and peaked at 29.4 GiB under stress. |
| `--max-num-seqs=64` | Graph capture refuses to start above the recurrent-state blocks one card holds (133). |
| `--speculative-config` (MTP, 2 tokens) | Doubles single-request decode speed. |
| `--max-num-batched-tokens=8192` | At the default of 2048, long prompts split into chunks small enough to be launch-bound. Costs activation memory, which the vision tower's removal pays for. |
| `--reasoning-parser=qwen3` | Splits chain-of-thought out of `content`. The XML tool parser eats the closing `</think>`, so without this the reasoning merges back in. |
| `--tool-call-parser=qwen3_xml` | The model emits XML tool calls. The `hermes` parser expects JSON and leaves calls as text. |

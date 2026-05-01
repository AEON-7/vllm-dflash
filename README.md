[![☕ Tips](https://img.shields.io/badge/%E2%98%95_Tips-Support_the_work-ff5e5b?style=flat)](https://github.com/AEON-7/AEON-7#-support-the-work)

<p align="center">
  <strong>DFlash vLLM for DGX Spark</strong><br>
  <em>Plug &amp; Play Block-Diffusion Speculative Decoding, optionally stacked with TurboQuant KV compression</em>
</p>

<p align="center">
  <code>docker pull ghcr.io/aeon-7/vllm-dflash:latest</code>
</p>

<p align="center">
  <a href="#-important--read-first">Read First</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#performance-dgx-spark-gb10">Performance</a> &bull;
  <a href="#turboquant-optional">TurboQuant</a> &bull;
  <a href="#which-config-should-i-use">Which config?</a> &bull;
  <a href="#configuration">Config</a> &bull;
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

## ⚠ IMPORTANT — read first

### Hardware requirements

| | |
|---|---|
| **GPU** | NVIDIA **Blackwell** with compute capability **SM120 / SM121** — DGX Spark (GB10), B200/GB200, RTX 5090 |
| **Architecture** | **aarch64** (ARM64) for DGX Spark; x86_64 should work for B200 but is untested |
| **Unified / GPU memory** | **≥ 64 GB** free (DGX Spark has 128 GB unified) |
| **Disk** | **≥ 50 GB** free: 18 GB image + 20 GB model + 4 GB drafter + cache |
| **NVIDIA driver** | **≥ 580.x** (GB10 support) |
| **Docker** | **≥ 25.x** with `nvidia-container-toolkit` installed |

**Will NOT work on:** H100/H200 (Hopper), A100 (Ampere), L40/L4, RTX 40-series, or any pre-Blackwell hardware. NVFP4 tensor cores are Blackwell-exclusive.

### Critical caveats

1. **First boot takes 5–10 minutes** after the model is on disk (weight load + CUDA graph capture + FlashInfer NVFP4 autotune). Subsequent boots with cached JIT artifacts are ~2 min.
2. **First inference request takes ~30 seconds** even after the server reports healthy — vLLM does one-time CUDA graph specialization on the first real input. **Always warm up 1–3 requests** before measuring latency.
3. **Model download is 20 GB**; DFlash drafter auto-downloads another ~4 GB on first container start.
4. **DFlash acceptance is content-dependent** — expect 60 tok/s on code/reasoning, 30 tok/s on free prose. Random-token adversarial inputs will show no speedup.
5. **Set `VLLM_API_KEY` as a persistent env var**, not a one-shot `$(openssl rand ...)` expansion inside `docker run` — you need the same value to call the API later.
6. **TurboQuant is optional.** The default image does *not* include it. See [TurboQuant (optional)](#turboquant-optional) for when to enable it.
7. **Config sensitivity**: the default tuned settings (`MAX_NUM_SEQS=16`, `MAX_NUM_BATCHED_TOKENS=32768`) are sized for DGX Spark's 128 GB. On smaller Blackwell cards (e.g. RTX 5090 32 GB) drop `MAX_MODEL_LEN` and `GPU_MEMORY_UTILIZATION` proportionally.

### Before you file an issue

Check the [Troubleshooting](#troubleshooting) section — most problems are one of: queue saturation at high concurrency, CUDA OOM from oversized `MAX_MODEL_LEN`, or model-name mismatch between the mount and the API call.

---

## Overview

A pre-built vLLM container tuned for **NVIDIA DGX Spark (GB10 Blackwell)**, serving
[`AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4`](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4)
— a 27B hybrid linear-attention + full-attention model, NVFP4-quantized, vision-capable:

- **DFlash** block-diffusion speculative decoding (k=15) — 2–5× faster decode than vanilla vLLM depending on prompt class
- **NVFP4** quantization with AWQ calibration — native Blackwell FP4 tensor cores
- **OpenAI-compatible** `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`
- **Optional TurboQuant** KV-cache compression for long-context / high-concurrency — see [TurboQuant](#turboquant-optional)

All benchmarks below are on DGX Spark GB10 (128 GB unified memory, 273 GB/s LPDDR5X).

---

## Quick Start

Three copy-paste steps. **Plan for ~40 minutes end-to-end** the first time — 20 min for model download (depends on your connection), 5–7 min for cold boot, a few seconds per warmup request.

### 1. Download the model (one-time, ~20 GB)

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4 \
    --local-dir /models/DFlash-Qwen3.5-27B-Uncensored-NVFP4
```

### 2. Launch the container

```bash
# Generate + remember an API key (save it — you'll need it for every request)
export VLLM_API_KEY=$(openssl rand -hex 32)
echo "VLLM_API_KEY=$VLLM_API_KEY" >> ~/.bashrc    # optional: persist across shells

docker run -d --name vllm-dflash \
    --gpus all --network host --ipc host --ulimit memlock=-1:-1 \
    -v /models/DFlash-Qwen3.5-27B-Uncensored-NVFP4:/models/target:ro \
    -e MODEL_PATH=/models/target \
    -e SERVED_MODEL_NAME=qwen35-dflash \
    -e DFLASH_DRAFTER=z-lab/Qwen3.5-27B-DFlash \
    -e DFLASH_NUM_SPEC_TOKENS=15 \
    -e MAX_MODEL_LEN=65536 \
    -e MAX_NUM_SEQS=16 \
    -e MAX_NUM_BATCHED_TOKENS=32768 \
    -e GPU_MEMORY_UTILIZATION=0.85 \
    -e ATTENTION_BACKEND=flash_attn \
    -e VLLM_API_KEY=$VLLM_API_KEY \
    ghcr.io/aeon-7/vllm-dflash:latest
```

The drafter (`z-lab/Qwen3.5-27B-DFlash`) auto-downloads on first run (~4 GB). Output tokens are addressed by the `SERVED_MODEL_NAME` you set above (`qwen35-dflash` here).

### 3. Wait for readiness, warm up, test

```bash
# Wait for healthy (cold start ~5–7 min — watch docker logs -f vllm-dflash if curious)
until curl -sf http://localhost:8000/health; do sleep 10; done
echo "server up"

# Warm up (first request does CUDA graph specialization — expect ~30 s)
for i in 1 2 3; do
  curl -sf -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $VLLM_API_KEY" \
    -d '{"model":"qwen35-dflash","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0}' \
    > /dev/null && echo "warmup $i ok"
done

# Real test
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $VLLM_API_KEY" \
    -d '{
        "model": "qwen35-dflash",
        "messages": [{"role": "user", "content": "Write a binary search in Python with type hints."}],
        "max_tokens": 256,
        "temperature": 0
    }'
```

You're running a 27B multimodal model with 2–5× speculative-decoding speedup on a 128 GB Spark.

---

## Performance (DGX Spark GB10)

All numbers below are on **unmodified `ghcr.io/aeon-7/vllm-dflash:latest`** running
`AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4`, DFlash `k=15`, 65K context, with the
recommended configuration above. Measurements use **natural-language prompts** with
`temperature=0` for full determinism. See [BENCHMARKS.md](BENCHMARKS.md) for the
reproducible script.

### Single-stream throughput by prompt style

Post-warmup, 3 runs, variance <0.3%.

| Prompt style | Tok/s | TPOT p50 | Notes |
|:---|:---:|:---:|:---|
| **Code** (algorithm + docstrings) | **64.0** | 15.2 ms | Highly patterned — DFlash excels |
| **Reasoning** (math step-by-step) | **54.0** | 18.4 ms | Structured, predictable |
| **Dialogue** (chat continuation) | **38.4** | 26.0 ms | Natural conversational |
| **Prose** (free-form essay) | **29.5** | 33.6 ms | Creative text — DFlash hardest to apply |

DFlash acceptance length (tokens accepted per 15-token draft) ranges from ~2.0 on prose
to ~5.5 on code. Per-position acceptance decays from ~78% at position 0 to <3% by position 8.

### Concurrency scaling (natural prompts)

**Code** (best case for DFlash):

| Concurrency | Aggregate tok/s | Median per-req | TTFT p50 | TPOT p50 |
|:---:|:---:|:---:|:---:|:---:|
| c=1 | **64.0** | 64.0 tok/s | 239 ms | 15.2 ms |
| c=4 | **181.5** | 45.4 tok/s | 408 ms | 21.2 ms |
| c=8 | **262.8** | 32.9 tok/s | 564 ms | 29.4 ms |
| c=16 | **327.9** | 20.5 tok/s | 884 ms | 47.1 ms |

**Prose** (worst case):

| Concurrency | Aggregate tok/s | Median per-req | TTFT p50 | TPOT p50 |
|:---:|:---:|:---:|:---:|:---:|
| c=1 | **29.5** | 29.5 tok/s | 225 ms | 33.6 ms |
| c=4 | **83.5** | 21.1 tok/s | 432 ms | 46.8 ms |
| c=8 | **122.4** | 15.3 tok/s | 557 ms | 64.4 ms |
| c=16 | **151.8** | 9.5 tok/s | 860 ms | 104 ms |

**At c=16 the container serves 328 tok/s on coding / 152 tok/s on prose, with TTFT
below 900 ms.**

### Summary metrics

| Metric | Value |
|---|---|
| Peak single-stream | **64.0 tok/s** (code) |
| Peak aggregate (c=16) | **327.9 tok/s** (code), 151.8 tok/s (prose) |
| TPOT p50 range | 15 ms (code, c=1) → 104 ms (prose, c=16) |
| TTFT p50 range | 225 ms (c=1) → 884 ms (c=16) |
| Model size | ~20 GB (NVFP4) |
| KV headroom | 70 GiB free after weights + graphs |
| Max context | 65K default (model supports up to 262K) |

---

## Which config should I use?

| Your workload looks like… | Recommended | Why |
|---|---|---|
| **Single interactive chat user**, short-to-medium context | **Baseline** (this image, defaults above) | 3% overhead isn't worth the extra moving part |
| **4–16 concurrent users**, typical chat <16K | **Baseline** | Tuned config already handles this well |
| **Agent fleet: 32–128 concurrent**, tool-calling under 16K | **TurboQuant hybrid** | KV capacity is the bottleneck, not decode throughput |
| **Long-context work >32K, single-session** | **Depends** — measure first | TurboQuant saves memory but -11% decode at 32K |
| **Getting OOM** at current concurrency / context | **TurboQuant hybrid** | 3.76× KV compression recovers the headroom |
| **Latency-critical single-prompt serving** | **Baseline** | Every 3% decode matters when p50 is 15 ms |

**Short version**: TurboQuant is a **capacity unlock**, not a throughput unlock. Enable it when you're memory-bound (many concurrent sessions or long contexts); skip it when you're compute-bound (few sessions, short prompts).

---

## TurboQuant (optional)

For long-context or high-concurrency workloads, the container can be extended with
[0xSero/turboquant](https://github.com/0xSero/turboquant) KV-cache compression
(4-bit keys, 3-bit values, Hadamard-rotation + Lloyd-Max codebooks — paper:
[arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

TurboQuant is **not enabled in the default image**. To use it, build the extension
Dockerfile in [`turboquant/`](turboquant/) which pip-installs the plugin and wires
it in via a Python `.pth` bootstrap.

### Overhead vs baseline

Measured on the same model + tuned config. TurboQuant overhead is **~3% across all
modes, concurrencies, and prompt styles** — essentially free on short-to-medium outputs.

#### Code prompts

| Concurrency | TQ off | TQ capture_only | TQ hybrid | Δ hybrid vs off |
|:---:|:---:|:---:|:---:|:---:|
| c=1  | 64.02 | 61.50 | 61.71 | **-3.61%** |
| c=4  | 181.47 | 175.71 | 175.79 | **-3.13%** |
| c=8  | 262.77 | 255.19 | 252.78 | **-3.80%** |
| c=16 | 327.89 | 314.93 | 318.36 | **-2.91%** |

#### Prose prompts

| Concurrency | TQ off | TQ capture_only | TQ hybrid | Δ hybrid vs off |
|:---:|:---:|:---:|:---:|:---:|
| c=1  | 29.46 | 28.14 | 28.49 | **-3.29%** |
| c=4  | 83.53 | 80.28 | 80.72 | **-3.36%** |
| c=8  | 122.41 | 117.67 | 119.17 | **-2.65%** |
| c=16 | 151.81 | 147.43 | 148.80 | **-1.98%** |

### Long-context behaviour

TurboQuant's hybrid-mode decode cost is **flat until the 128-token ring buffer
overflows, then grows with context length** because each decode step has to
dequantize more compressed history. Short-to-medium contexts see no penalty;
decode slows measurably at 32K+.

| Context tokens | TQ off decode | TQ hybrid decode | Δ |
|:---:|:---:|:---:|:---:|
| 4,000 | 31.81 tok/s | 33.35 tok/s | **+4.85%** |
| 16,000 | 23.92 tok/s | 24.20 tok/s | **+1.18%** |
| 32,000 | 19.43 tok/s | 17.22 tok/s | **-11.38%** |

### When to enable TurboQuant

- **Multi-session long-context serving** — the real win is KV capacity, letting
  you hold more simultaneous sessions at full context (not visible in c=1
  microbenchmarks)
- **Agentic workloads** with long rolling context where freeing compressed
  history recovers VRAM for the next request
- **Any use case hitting OOM on long contexts** under default KV

### When NOT to enable TurboQuant

- Short-context single-user chat (<16K) — the decode overhead isn't worth the
  complexity when there's no capacity pressure
- Pure latency-critical 32K+ single-request paths — you'll eat the ~11% decode
  cost without the capacity payoff

### Modes

| `TQ_MODE` | What it does |
|---|---|
| `off` | Plugin installed but dormant — zero overhead |
| `capture_only` | Captures K/V into compressed store; attention still uses paged cache |
| `hybrid` | Attention reads from compressed history beyond a 128-token ring buffer |
| `full_tq` | (experimental) TQ handles prefill too |

### Enabling

Build and run the TurboQuant variant:

```bash
cd turboquant
docker build -t vllm-dflash-tq:latest .

docker run -d --name vllm-dflash-tq \
    --gpus all --network host --ipc host --ulimit memlock=-1:-1 \
    -v /models/DFlash-Qwen3.5-27B-Uncensored-NVFP4:/models/target:ro \
    -e MODEL_PATH=/models/target \
    -e DFLASH_DRAFTER=z-lab/Qwen3.5-27B-DFlash \
    -e DFLASH_NUM_SPEC_TOKENS=15 \
    -e MAX_MODEL_LEN=65536 \
    -e MAX_NUM_SEQS=16 \
    -e MAX_NUM_BATCHED_TOKENS=32768 \
    -e GPU_MEMORY_UTILIZATION=0.85 \
    -e ATTENTION_BACKEND=flash_attn \
    -e ENABLE_TURBOQUANT=1 \
    -e TQ_MODE=hybrid \
    -e TQ_KEY_BITS=4 \
    -e TQ_VALUE_BITS=3 \
    vllm-dflash-tq:latest
```

### Compatibility note

0xSero/turboquant currently requires a small patch to be CUDA-graph-safe
([PR #12](https://github.com/0xSero/turboquant/pull/12)). The Dockerfile in
`turboquant/` applies that patch automatically. Once the PR is merged upstream,
the patch step will be removed.

---

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | *required* | Local path to target model |
| `DFLASH_DRAFTER` | *required* | HF repo or path of the DFlash drafter |
| `DFLASH_NUM_SPEC_TOKENS` | `15` | Speculative token count per draft |
| `MAX_MODEL_LEN` | `65536` | Maximum sequence length (model supports up to 262144) |
| `MAX_NUM_SEQS` | `16` | Concurrent sequences (default was 4; 16 is the Spark sweet spot) |
| `MAX_NUM_BATCHED_TOKENS` | `32768` | Scheduler token budget (default was 8192; 32768 unblocks c=8+) |
| `GPU_MEMORY_UTILIZATION` | `0.85` | VRAM fraction; keep at 0.85 on Spark to avoid swap |
| `ATTENTION_BACKEND` | `flash_attn` | Try `TRITON_ATTN` if you hit FA kernel bugs |
| `VLLM_API_KEY` | unset | Bearer token required for all endpoints when set. **Generate + save this value** — you need the same string for every client call |
| `SERVED_MODEL_NAME` | `basename($MODEL_PATH)` | Name clients pass as `"model":` in requests. **Set this explicitly** to avoid confusion |
| `EXTRA_ARGS` | unset | Passed verbatim to `vllm serve` |

### TurboQuant-specific (when `ENABLE_TURBOQUANT=1`)

| Variable | Default | Description |
|---|---|---|
| `TQ_MODE` | `hybrid` | `off` / `capture_only` / `hybrid` / `full_tq` |
| `TQ_KEY_BITS` | `4` | Key quantization bits (3–4 typical) |
| `TQ_VALUE_BITS` | `3` | Value quantization bits (2–4; 2 loses quality) |
| `TQ_RING_CAPACITY` | `128` | Exact-precision tokens at tail of context |
| `TQ_INITIAL_LAYERS` | `4` | First N layers get `key_bits+1` for quality |

### DGX Spark tuning recap

The three env vars that matter most on GB10:

```
MAX_NUM_SEQS=16                  # was 4 — unlocks c=8+ without queue saturation
MAX_NUM_BATCHED_TOKENS=32768     # was 8192 — matches scheduler's spec-decode headroom
GPU_MEMORY_UTILIZATION=0.85      # safe headroom; don't push higher on 128 GB unified
```

At defaults, c=8 hit TTFT p50 of **14.7 seconds** due to queue saturation.
With the tuned config, c=8 drops to **817 ms** and c=16 becomes usable.

---

## Troubleshooting

<details>
<summary><strong>Container restarts or hangs on startup</strong></summary>

First boot takes 5–7 minutes on DGX Spark:
- ~2 min: weight load
- ~1 min: DFlash drafter download (first run only; cached to volume after)
- ~2 min: CUDA graph capture + FlashInfer NVFP4 GEMM autotune

```bash
docker logs -f vllm-dflash
```

Look for `Application startup complete`. If you see a `Traceback`, grab the full error text and file an issue.
</details>

<details>
<summary><strong>API returns 404 / "model not found"</strong></summary>

The model name in your request must match `SERVED_MODEL_NAME` (default: basename of `MODEL_PATH`). If you mounted at `/models/target`, the served name is `target` unless you set `SERVED_MODEL_NAME` explicitly. The Quick Start sets `SERVED_MODEL_NAME=qwen35-dflash` — use that exact string in `"model":` fields.

Check what's actually served:
```bash
curl -sf http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY" | jq
```
</details>

<details>
<summary><strong>First request takes 30+ seconds even though /health is OK</strong></summary>

Expected. vLLM does one-time CUDA graph specialization on the first real input. Always fire 1–3 warmup requests before benchmarking. See the Quick Start step 3 loop.
</details>

<details>
<summary><strong>TTFT blows up to 10+ seconds at concurrency</strong></summary>

You're queue-bound. The legacy default was `MAX_NUM_SEQS=4` / `MAX_NUM_BATCHED_TOKENS=8192`, which saturates at c=8 with DFlash spec-decode. The Quick Start uses the tuned values (`MAX_NUM_SEQS=16`, `MAX_NUM_BATCHED_TOKENS=32768`); make sure your `docker run` has them.
</details>

<details>
<summary><strong>CUDA out of memory</strong></summary>

Lower `GPU_MEMORY_UTILIZATION` to 0.80 or drop `MAX_MODEL_LEN` (e.g., 65536 → 32768 → 16384). Spark's 128 GB is unified — the GPU shares it with the host, so leave 15–20 GB headroom. If you need more concurrent sessions at long context, enable [TurboQuant](#turboquant-optional) for ~3.76× KV compression.
</details>

<details>
<summary><strong>"Cannot copy between CPU and CUDA tensors" when enabling TurboQuant</strong></summary>

You're running an unpatched `0xSero/turboquant`. Use the extension Dockerfile in [`turboquant/`](turboquant/) — it pins to the fix-branch hosting [PR #12](https://github.com/0xSero/turboquant/pull/12) which makes the QJL quantizer CUDA-graph-safe.
</details>

<details>
<summary><strong>DFlash acceptance rate looks low</strong></summary>

DFlash is content-sensitive. Expected accepted-tokens-per-draft (k=15):

| Prompt type | Accepted/draft | Effective speedup |
|---|---|---|
| Code / reasoning | ~5+ | ~2.1× |
| Dialogue | ~3 | ~1.5× |
| Free prose | ~2 | ~1.3× |
| Random / adversarial | ~1.5 | ~1.0× |

See the [BENCHMARKS.md](BENCHMARKS.md) acceptance-profile table for exact per-position numbers.
</details>

<details>
<summary><strong>Image won't pull or fails on my GPU</strong></summary>

The image requires a Blackwell GPU with SM120 or SM121 (see [Hardware requirements](#hardware-requirements) at the top). If `nvidia-smi` shows compute capability 8.x (Ampere) or 9.0 (Hopper), **this image will not run**. NVFP4 tensor cores are Blackwell-exclusive.
</details>

---

## How it works

### DFlash — block-diffusion speculative decoding

DFlash speeds up generation by **speculating multiple tokens per step** using a
small draft model, then verifying them against the target model in a single forward
pass. The drafter here is a 5-layer Qwen3 variant fine-tuned to predict the next
15 tokens from the target's intermediate hidden states at layers (1, 16, 31, 46, 61).

Key properties:
- **Lossless** — every accepted token matches what greedy decoding would produce
- **Memory-bandwidth-bound-friendly** — a single target-model pass verifies many candidate tokens
- **Content-adaptive** — structured text (code, math) wins more than free prose

See paper: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036).

### NVFP4 on Blackwell

NVIDIA's FP4 format (E2M1) is a native tensor-core datatype on Blackwell (B200, GB10,
RTX 50×0). Unlike older INT4/GPTQ which introduce visible degradation, NVFP4 with
AWQ_FULL calibration is effectively lossless. Our image autodetects NVFP4 checkpoints
and routes through FlashInfer CUTLASS kernels.

Weights + activations are quantized; KV cache stays in BF16 by default (use
TurboQuant to compress KV as well).

### Hybrid architecture of Qwen3.5-27B

The model has 64 transformer layers arranged in a hybrid pattern:
- **48 linear-attention layers** (Gated DeltaNet / Mamba-style recurrent state)
- **16 full-attention layers** (classical attention with KV cache)
- **1 MTP head** (used as the DFlash drafter anchor)

DFlash's `target_layer_ids=[1,16,31,46,61]` are the hidden-state checkpoints the
drafter consumes. TurboQuant compresses **only the 16 full-attention layers' KV
cache**; linear-attention layers have no K/V to compress (their recurrent state
is already compact).

### Why dense 27B beats 122B MoE on DGX Spark

DGX Spark is memory-bandwidth-bound (273 GB/s LPDDR5X unified). MoE experts require
scatter/gather across the unified memory, which defeats the bandwidth budget. A
dense 27B moves a predictable 20 GB of weights per token — ideal for the Spark's
memory architecture. On coding/reasoning benchmarks it rivals or beats larger MoE
variants that would OOM or thrash on this hardware.

---

## Credits

- **DFlash**: Zheng et al., ICLR 2026 ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036))
- **TurboQuant**: Zandieh et al., ICLR 2026 ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874));
  this container uses [0xSero/turboquant](https://github.com/0xSero/turboquant) as the plugin
- **Model**: [AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4)
- **Drafter**: [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash)
- **vLLM**: [vllm-project/vllm](https://github.com/vllm-project/vllm) 0.19.1

## License

GPL-3.0 (inherited from 0xSero/turboquant when the TurboQuant extension is enabled;
the base DFlash container is MIT). See `LICENSE`.

---

## ☕ Support the work

If this release has been useful, tips are deeply appreciated — they go directly toward more compute, more models, and more open releases.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <strong>₿ Bitcoin (BTC)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/btc.png" alt="BTC QR" width="200"/><br/>
      <sub><code>bc1q09xmzn00q4z3c5raene0f3pzn9d9pvawfm0py4</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>Ξ Ethereum (ETH)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/eth.png" alt="ETH QR" width="200"/><br/>
      <sub><code>0x1512667F6D61454ad531d2E45C0a5d1fd82D0500</code></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>◎ Solana (SOL)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/sol.png" alt="SOL QR" width="200"/><br/>
      <sub><code>DgQsjHdAnT5PNLQTNpJdpLS3tYGpVcsHQCkpoiAKsw8t</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>ⓜ Monero (XMR)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/xmr.png" alt="XMR QR" width="200"/><br/>
      <sub><code>836XrSKw4R76vNi3QPJ5Fa9ugcyvE2cWmKSPv3AhpTNNKvqP8v5ba9JRL4Vh7UnFNjDz3E2GXZDVVenu3rkZaNdUFhjAvgd</code></sub>
    </td>
  </tr>
</table>

> **Ethereum L2s (Base, Arbitrum, Optimism, Polygon, etc.) and EVM-compatible tokens** can be sent to the same Ethereum address.

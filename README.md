<p align="center">
  <strong>DFlash vLLM for DGX Spark</strong><br>
  <em>Plug &amp; Play Block-Diffusion Speculative Decoding, optionally stacked with TurboQuant KV compression</em>
</p>

<p align="center">
  <code>docker pull ghcr.io/aeon-7/vllm-dflash:latest</code>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#performance-dgx-spark-gb10">Performance</a> &bull;
  <a href="#turboquant-optional">TurboQuant</a> &bull;
  <a href="#configuration">Config</a> &bull;
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

## Overview

A pre-built vLLM container tuned for **NVIDIA DGX Spark (GB10 Blackwell, SM121)**, serving
[`AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4`](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4)
(27B hybrid linear-attention + full-attention model, NVFP4 quantized, vision-capable) with:

- **DFlash** block-diffusion speculative decoding (inline drafter, k=15) — ~2–5× faster decode than vanilla vLLM depending on prompt class
- **NVFP4** quantization with AWQ calibration — native Blackwell FP4 tensor cores
- **OpenAI-compatible** `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`
- **Optional TurboQuant** KV-cache compression for long-context / high-concurrency workloads — see [TurboQuant](#turboquant-optional)

Everything below is measured on DGX Spark GB10 (128 GB unified memory, 273 GB/s LPDDR5X).

---

## Quick Start

Three steps, ~10 minutes to first token.

### 1. Download the model

```bash
pip install huggingface_hub[cli]
huggingface-cli download AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4 \
    --local-dir /models/DFlash-Qwen3.5-27B-Uncensored-NVFP4
```

Size: ~20 GB.

### 2. Launch the container

```bash
docker run -d --name vllm-dflash \
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
    -e VLLM_API_KEY=$(openssl rand -hex 32) \
    ghcr.io/aeon-7/vllm-dflash:latest
```

The drafter (`z-lab/Qwen3.5-27B-DFlash`) auto-downloads on first run.

### 3. Test

```bash
# Wait for health (cold start ~5–7 min)
until curl -sf http://localhost:8000/health; do sleep 5; done

# Send a completion
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $VLLM_API_KEY" \
    -d '{
        "model": "DFlash-Qwen3.5-27B-Uncensored-NVFP4",
        "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
        "max_tokens": 64,
        "temperature": 0
    }'
```

That's it. You're running a 27B multimodal model with 2–5× speculative-decoding speedup on a 128 GB Spark.

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
| `ATTENTION_BACKEND` | `flash_attn` | Use `TRITON_ATTN` if you have issues with FA |
| `VLLM_API_KEY` | unset | Bearer token required for all endpoints when set |
| `SERVED_MODEL_NAME` | derived | Name shown in `/v1/models` |
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
- ~2 min: weight load (fastsafetensors-free path)
- ~1 min: DFlash drafter download (first time only, cached after)
- ~2 min: CUDA graph capture + FlashInfer fp4_gemm autotune

Watch the logs:
```bash
docker logs -f vllm-dflash
```
Look for `Application startup complete`. If you see `Traceback`, file an issue
with the full error text.
</details>

<details>
<summary><strong>TTFT is 10+ seconds</strong></summary>

You're probably running with `MAX_NUM_SEQS=4` (the old default) and hitting concurrency
above 4. Restart with `MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=32768`.
</details>

<details>
<summary><strong>CUDA out of memory</strong></summary>

Lower `GPU_MEMORY_UTILIZATION` to 0.80 or `MAX_MODEL_LEN` to 32768. On Spark the
128 GB is unified — the GPU shares it with the host, so leaving 15–20 GB headroom
is wise.
</details>

<details>
<summary><strong>"Cannot copy between CPU and CUDA tensors" when enabling TurboQuant</strong></summary>

You're running an unpatched 0xSero/turboquant. Use the extension Dockerfile in
[`turboquant/`](turboquant/), which applies [PR #12](https://github.com/0xSero/turboquant/pull/12)
during image build.
</details>

<details>
<summary><strong>DFlash acceptance rate looks low on my prompts</strong></summary>

DFlash is content-sensitive. Acceptance scales with prompt predictability:
- Code / reasoning: ~5+ tokens/draft accepted
- Dialogue: ~3 tokens/draft
- Free prose: ~2 tokens/draft
- Random tokens: ~1.5 tokens/draft (adversarial)

This is expected; see the [BENCHMARKS.md](BENCHMARKS.md) acceptance-profile table.
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

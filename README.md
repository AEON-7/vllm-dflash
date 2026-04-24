<p align="center">
  <strong>DFlash vLLM for DGX Spark</strong><br>
  <em>Plug & Play Block-Diffusion Speculative Decoding</em>
</p>

<p align="center">
  <code>docker pull ghcr.io/aeon-7/vllm-dflash:latest</code>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#available-models">Models</a> &bull;
  <a href="#container-reference">Container Docs</a> &bull;
  <a href="#how-dflash-works">How It Works</a>
</p>

---

## Performance (DGX Spark GB10)

All numbers below are on **unmodified `ghcr.io/aeon-7/vllm-dflash:latest`**
running `AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4`, DFlash `k=15`, 65K
context, the tuned configuration shown further down.  Measurements use
**natural-language prompts** with `temperature=0` for full determinism; see
[BENCHMARKS.md](BENCHMARKS.md) for the exact script, prompts, and raw data.

### Single-Stream Throughput by Prompt Style

Post-warmup steady-state (3-run average, variance <0.3%).

| Prompt style | Tok/s | TPOT p50 | Notes |
|:---|:---:|:---:|:---|
| **Code** (algorithm + docstrings + type hints) | **64.0** | 15.2 ms | Highly patterned — DFlash excels |
| **Reasoning** (math step-by-step) | **54.0** | 18.4 ms | Structured, predictable tokens |
| **Dialogue** (chat continuation) | **38.4** | 26.0 ms | Natural conversational |
| **Prose** (free-form essay) | **29.5** | 33.6 ms | Creative text — acceptance length drops |

DFlash acceptance length (tokens accepted per 15-token draft) ranges from
~2.0 on prose to ~5.5 on code.  Per-position acceptance decays from ~78% at
position 0 to <3% by position 8.

### Concurrency Scaling — Spark-Tuned Config, Natural Prompts

**Code** (best case for DFlash):

| Concurrency | Aggregate tok/s | Median per-request | TTFT p50 | TPOT p50 |
|:---:|:---:|:---:|:---:|:---:|
| c=1 | **64.0** | 64.0 tok/s | 239 ms | 15.2 ms |
| c=4 | **181.5** | 45.4 tok/s | 408 ms | 21.2 ms |
| c=8 | **262.8** | 32.9 tok/s | 564 ms | 29.4 ms |
| c=16 | **327.9** | 20.5 tok/s | 884 ms | 47.1 ms |

**Prose** (worst case for DFlash):

| Concurrency | Aggregate tok/s | Median per-request | TTFT p50 | TPOT p50 |
|:---:|:---:|:---:|:---:|:---:|
| c=1 | **29.5** | 29.5 tok/s | 225 ms | 33.6 ms |
| c=4 | **83.5** | 21.1 tok/s | 432 ms | 46.8 ms |
| c=8 | **122.4** | 15.3 tok/s | 557 ms | 64.4 ms |
| c=16 | **151.8** | 9.5 tok/s | 860 ms | 104 ms |

**At c=16 the container serves 328 tok/s on coding workloads, 152 tok/s on
free-form prose** — both while holding TTFT p50 under 900 ms.

### Why the Spark-Tuned Config Matters

The default launch settings on this image (`MAX_NUM_SEQS=4`,
`MAX_NUM_BATCHED_TOKENS=8192`) hit queue saturation at c=8: under random-token
load, TTFT p50 went to **14,688 ms** while aggregate throughput plateaued at
~70 tok/s.  The tuned configuration fixes both:

| | Default | Spark-tuned |
|---|---|---|
| `MAX_NUM_SEQS` | 4 | **16** |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | **32768** |
| `GPU_MEMORY_UTILIZATION` | 0.80 | **0.85** |
| c=8 aggregate tok/s (code) | queue-saturated | **262.8** |
| c=16 | not reachable | **327.9** |

Single-stream throughput is effectively unchanged (+1.9% at c=1) — the value
of the tuned config is entirely in multi-session scaling.

### Recommended Configuration

```bash
docker run -d --name vllm-dflash \
    --gpus all --network host --ipc host --ulimit memlock=-1:-1 \
    -v /path/to/model:/models/target:ro \
    -v /path/to/drafter:/models/dflash-drafter:ro \
    -e MODEL_PATH=/models/target \
    -e DFLASH_DRAFTER=/models/dflash-drafter \
    -e DFLASH_NUM_SPEC_TOKENS=15 \
    -e MAX_MODEL_LEN=65536 \
    -e MAX_NUM_SEQS=16 \
    -e MAX_NUM_BATCHED_TOKENS=32768 \
    -e GPU_MEMORY_UTILIZATION=0.85 \
    -e ATTENTION_BACKEND=flash_attn \
    ghcr.io/aeon-7/vllm-dflash:latest
```

### Summary Metrics

| Metric (Spark-tuned) | Value |
|---|---|
| **Peak single-stream** | **64.0 tok/s** (code) |
| **Peak aggregate (c=16)** | **327.9 tok/s** (code), 151.8 tok/s (prose) |
| **TPOT p50 range** | 15 ms (code, c=1) → 104 ms (prose, c=16) |
| **TTFT p50 range** | 225 ms (c=1) → 884 ms (c=16) |
| **Model Size** | ~20 GB (NVFP4) / ~52 GB (BF16) |
| **KV cache headroom** | 70 GiB free after weights + graphs |
| **Max Context** | 65K default (model supports up to 262K) |

See [BENCHMARKS.md](BENCHMARKS.md) for full methodology and the
reproducible benchmark script.

---

## Quick Links

| | |
|---|---|
| **[Quick Start Guide](#quick-start)** | Download model + launch in 5 minutes |
| **[How DFlash Works](#how-dflash-works)** | Block-diffusion speculative decoding explained |
| **[Why Dense Over MoE?](#why-dense-over-moe-on-dgx-spark)** | Why 27B dense beats 122B MoE on DGX Spark |
| **[Why NVFP4?](#why-nvfp4-on-blackwell)** | Free performance boost on Blackwell GPUs |
| **[Container Reference](#container-reference)** | All environment variables and usage patterns |
| **[DFlash Paper](https://arxiv.org/abs/2602.06036)** | Original research paper |
| **[Docker Image](https://github.com/users/AEON-7/packages/container/package/vllm-dflash)** | `ghcr.io/aeon-7/vllm-dflash:latest` |

---

## Available Models

> More models coming soon. The DFlash container works with any vLLM-compatible model — DFlash speculative decoding is enabled by setting `DFLASH_DRAFTER` to a compatible drafter model.

| Model | Params | Precision | Size | Vision | Best For | Guide |
|---|:---:|:---:|:---:|:---:|---|:---:|
| [Qwen3.5-27B Uncensored NVFP4](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4) | 27B | NVFP4 | ~20 GB | Yes | **DGX Spark / Blackwell** — recommended | **[Start Here](#qwen35-27b-uncensored-nvfp4)** |
| [Qwen3.5-27B Uncensored BF16](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored) | 27B | BF16 | ~52 GB | Yes | Non-Blackwell / research | [Guide](#qwen35-27b-uncensored-bf16) |

---

## Quick Start

### Qwen3.5-27B Uncensored (NVFP4)

> **Recommended for DGX Spark and all Blackwell GPUs.** NVFP4 is a native Blackwell tensor core datatype — effectively lossless quantization with 3x memory reduction. [Why?](#why-nvfp4-on-blackwell)

#### 1. Download the model

```bash
# Install the HuggingFace CLI if you don't have it
pip install -U huggingface-hub

huggingface-cli download AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4 \
  --local-dir ~/models/DFlash-Qwen3.5-27B-Uncensored-NVFP4
```

#### 2. Create your environment file

```bash
cat > .env.dflash << 'EOF'
# Authentication
HF_TOKEN=hf_your_token_here
VLLM_API_KEY=$(openssl rand -hex 32)

# Model path (where you downloaded the model)
MODEL_HOST_PATH=~/models/DFlash-Qwen3.5-27B-Uncensored-NVFP4

# DFlash speculative decoding (drafter auto-downloads on first run)
DFLASH_DRAFTER=z-lab/Qwen3.5-27B-DFlash
DFLASH_NUM_SPEC_TOKENS=15

# DGX Spark optimal settings (64K context, 4 concurrent sequences)
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=4
GPU_MEMORY_UTILIZATION=0.85
MAX_NUM_BATCHED_TOKENS=65536
EOF

# Generate a real API key and inject it
sed -i "s|\$(openssl rand -hex 32)|$(openssl rand -hex 32)|" .env.dflash
echo "Your API key: $(grep VLLM_API_KEY .env.dflash | cut -d= -f2)"
```

#### 3. Save `docker-compose.dflash.yml`

```yaml
services:
  vllm-dflash:
    image: ghcr.io/aeon-7/vllm-dflash:latest
    container_name: vllm-dflash
    restart: unless-stopped
    network_mode: host
    ipc: host
    volumes:
      - ${MODEL_HOST_PATH}:/models/DFlash-Qwen3.5-27B-Uncensored-NVFP4
      - dflash-drafter-cache:/models/drafter-cache
    environment:
      - MODEL_PATH=/models/DFlash-Qwen3.5-27B-Uncensored-NVFP4
      - SERVED_MODEL_NAME=DFlash-Qwen3.5-27B-Uncensored
      - DFLASH_DRAFTER=${DFLASH_DRAFTER}
      - DFLASH_NUM_SPEC_TOKENS=${DFLASH_NUM_SPEC_TOKENS}
      - GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
      - MAX_MODEL_LEN=${MAX_MODEL_LEN}
      - MAX_NUM_SEQS=${MAX_NUM_SEQS}
      - MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
      - NVIDIA_VISIBLE_DEVICES=all
      - TORCH_MATMUL_PRECISION=high
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - HF_TOKEN=${HF_TOKEN}
      - VLLM_API_KEY=${VLLM_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  dflash-drafter-cache:
```

#### 4. Launch

```bash
docker compose --env-file .env.dflash -f docker-compose.dflash.yml up -d

# Watch startup (~5 min for weight loading + CUDA graph compilation)
docker compose -f docker-compose.dflash.yml logs -f
```

You'll see DFlash drafter auto-download on first run, then model loading, torch.compile, FP4 GEMM autotuning, and CUDA graph capture. The server is ready when you see `Application startup complete`.

#### 5. Test

```bash
# Text generation
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(grep VLLM_API_KEY .env.dflash | cut -d= -f2)" \
  -d '{
    "model": "DFlash-Qwen3.5-27B-Uncensored",
    "messages": [{"role": "user", "content": "Explain quantum entanglement simply."}],
    "max_tokens": 200
  }'

# Vision (image understanding)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(grep VLLM_API_KEY .env.dflash | cut -d= -f2)" \
  -d '{
    "model": "DFlash-Qwen3.5-27B-Uncensored",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
      {"type": "text", "text": "What do you see?"}
    ]}],
    "max_tokens": 200
  }'
```

The API is fully **OpenAI-compatible** — use it with any OpenAI SDK, LangChain, LlamaIndex, Open WebUI, or other client by pointing the base URL to `http://<your-ip>:8000/v1`.

---

### Qwen3.5-27B Uncensored (BF16)

> **For non-Blackwell GPUs or research workflows that need full-precision weights.** If you have a Blackwell GPU, use the [NVFP4 version](#qwen35-27b-uncensored-nvfp4) instead — it's a [free performance boost](#why-nvfp4-on-blackwell).

#### 1. Download the model

```bash
huggingface-cli download AEON-7/DFlash-Qwen3.5-27B-Uncensored \
  --local-dir ~/models/DFlash-Qwen3.5-27B-Uncensored
```

#### 2. Create your environment file

```bash
cat > .env.dflash << 'EOF'
# Authentication
HF_TOKEN=hf_your_token_here
VLLM_API_KEY=$(openssl rand -hex 32)

# Model path
MODEL_HOST_PATH=~/models/DFlash-Qwen3.5-27B-Uncensored

# DFlash speculative decoding
DFLASH_DRAFTER=z-lab/Qwen3.5-27B-DFlash
DFLASH_NUM_SPEC_TOKENS=15

# DGX Spark BF16 settings (needs more memory than NVFP4)
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=2
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_BATCHED_TOKENS=65536
EOF

sed -i "s|\$(openssl rand -hex 32)|$(openssl rand -hex 32)|" .env.dflash
echo "Your API key: $(grep VLLM_API_KEY .env.dflash | cut -d= -f2)"
```

#### 3. Save `docker-compose.dflash-bf16.yml`

```yaml
services:
  vllm-dflash-bf16:
    image: ghcr.io/aeon-7/vllm-dflash:latest
    container_name: vllm-dflash-bf16
    restart: unless-stopped
    network_mode: host
    ipc: host
    volumes:
      - ${MODEL_HOST_PATH}:/models/DFlash-Qwen3.5-27B-Uncensored
      - dflash-drafter-cache:/models/drafter-cache
    environment:
      - MODEL_PATH=/models/DFlash-Qwen3.5-27B-Uncensored
      - SERVED_MODEL_NAME=DFlash-Qwen3.5-27B-Uncensored
      - DFLASH_DRAFTER=${DFLASH_DRAFTER}
      - DFLASH_NUM_SPEC_TOKENS=${DFLASH_NUM_SPEC_TOKENS}
      - GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
      - MAX_MODEL_LEN=${MAX_MODEL_LEN}
      - MAX_NUM_SEQS=${MAX_NUM_SEQS}
      - MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
      - NVIDIA_VISIBLE_DEVICES=all
      - TORCH_MATMUL_PRECISION=high
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - HF_TOKEN=${HF_TOKEN}
      - VLLM_API_KEY=${VLLM_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  dflash-drafter-cache:
```

#### 4. Launch & Test

```bash
docker compose --env-file .env.dflash -f docker-compose.dflash-bf16.yml up -d
docker compose -f docker-compose.dflash-bf16.yml logs -f
```

Test commands are identical to the [NVFP4 section above](#5-test).

---

## Container Reference

### `ghcr.io/aeon-7/vllm-dflash:latest`

A pre-built vLLM container optimized for **NVIDIA DGX Spark (GB10 Blackwell, SM121)** with DFlash block-diffusion speculative decoding. Built on top of the AEON-7 vLLM Spark base image with all Blackwell-specific patches (FlashInfer CUTLASS, SM121 compatibility, torch.compile, FP4 GEMM autotuning).

The container's entrypoint automatically:
- Downloads the DFlash drafter model from HuggingFace (cached across restarts)
- Detects NVFP4 vs BF16 models and sets quantization flags
- Configures speculative decoding with sensible defaults
- Enables vision + text multimodal support
- Starts vLLM with OpenAI-compatible API on port 8000

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models/target` | Path to model weights inside the container, or a HuggingFace repo ID to auto-download |
| `DFLASH_DRAFTER` | *(empty = off)* | HF repo ID (e.g. `z-lab/Qwen3.5-27B-DFlash`) or local path. Set to `off` or leave empty to disable speculative decoding |
| `DFLASH_NUM_SPEC_TOKENS` | `15` | Speculative tokens per draft step. `15` for best single-stream, `5` for high concurrency |
| `SERVED_MODEL_NAME` | *(from MODEL_PATH)* | Model name exposed via the API |
| `MAX_MODEL_LEN` | `65536` | Maximum sequence length (model supports up to 256K — see [tuning guide](#dgx-spark-tuning-guide)) |
| `MAX_NUM_SEQS` | `4` | Maximum concurrent sequences |
| `GPU_MEMORY_UTILIZATION` | `0.85` | GPU memory fraction (increase to 0.90 for BF16) |
| `MAX_NUM_BATCHED_TOKENS` | `65536` | Maximum tokens batched per iteration |
| `KV_CACHE_DTYPE` | `auto` (DFlash) / `fp8_e4m3` (no DFlash) | KV cache precision. DFlash requires `auto` (BF16) due to non-causal attention |
| `ATTENTION_BACKEND` | `flash_attn` | Attention implementation |
| `QUANTIZATION` | `auto` | `auto` detects from model config. `modelopt` for NVFP4, `none` for BF16 |
| `VLLM_API_KEY` | *(empty)* | API key for Bearer token authentication |
| `HF_TOKEN` | *(empty)* | HuggingFace token for downloading gated models |
| `EXTRA_ARGS` | *(empty)* | Additional `vllm serve` arguments (space-separated) |

### Usage Patterns

#### With Docker Compose (recommended)

See the [Quick Start](#quick-start) section for complete compose files.

#### With `docker run`

```bash
# NVFP4 model with DFlash
docker run -d --runtime nvidia --network host --ipc host \
  -v ~/models/DFlash-Qwen3.5-27B-Uncensored-NVFP4:/models/target \
  -v dflash-cache:/models/drafter-cache \
  -e MODEL_PATH=/models/target \
  -e DFLASH_DRAFTER=z-lab/Qwen3.5-27B-DFlash \
  -e DFLASH_NUM_SPEC_TOKENS=15 \
  -e VLLM_API_KEY=your-secret-key \
  -e HF_TOKEN=hf_your_token \
  ghcr.io/aeon-7/vllm-dflash:latest
```

#### Without DFlash (plain vLLM)

The container works as a standard vLLM server when DFlash is disabled:

```bash
docker run -d --runtime nvidia --network host --ipc host \
  -v ~/models/any-model:/models/target \
  -e MODEL_PATH=/models/target \
  ghcr.io/aeon-7/vllm-dflash:latest
```

#### With a HuggingFace model (auto-download)

Set `MODEL_PATH` to a HuggingFace repo ID and the container downloads it automatically:

```bash
docker run -d --runtime nvidia --network host --ipc host \
  -e MODEL_PATH=AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4 \
  -e DFLASH_DRAFTER=z-lab/Qwen3.5-27B-DFlash \
  -e HF_TOKEN=hf_your_token \
  ghcr.io/aeon-7/vllm-dflash:latest
```

#### Using with other models

The DFlash container is not limited to the Qwen3.5-27B models listed above. It works with any vLLM-compatible model. DFlash speculative decoding requires a compatible drafter — currently the [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) drafter is available for Qwen3.5-27B. When running models without a DFlash drafter, simply omit the `DFLASH_DRAFTER` variable and the container runs standard vLLM inference.

```bash
# Example: run any NVFP4 model without DFlash
docker run -d --runtime nvidia --network host --ipc host \
  -v ~/models/some-other-model:/models/target \
  -e MODEL_PATH=/models/target \
  -e QUANTIZATION=modelopt \
  ghcr.io/aeon-7/vllm-dflash:latest

# Example: pass extra vLLM arguments
docker run -d --runtime nvidia --network host --ipc host \
  -v ~/models/target:/models/target \
  -e MODEL_PATH=/models/target \
  -e EXTRA_ARGS="--tensor-parallel-size 2 --enforce-eager" \
  ghcr.io/aeon-7/vllm-dflash:latest
```

### DGX Spark Tuning Guide

The container defaults are tuned for the DGX Spark GB10 (128 GB unified memory, 273 GB/s bandwidth) with **64K context and 4 concurrent sequences**. Note that vLLM pre-allocates the full KV cache budget at startup, so `MAX_MODEL_LEN` directly impacts memory usage. Increase context or concurrency carefully — the model supports up to 256K, but higher values leave less headroom on the Spark's unified memory.

| Workload | `MAX_MODEL_LEN` | `NUM_SPEC_TOKENS` | `MAX_NUM_SEQS` | `GPU_MEM_UTIL` | Expected tok/s |
|---|:---:|:---:|:---:|:---:|:---:|
| **Default (NVFP4)** | 65536 | 15 | 4 | 0.85 | 33-39 |
| **High concurrency** | 32768 | 5 | 8 | 0.85 | 85-92 total |
| **Long context** | 131072 | 15 | 2-3 | 0.85 | 33 |
| **BF16 model** | 65536 | 15 | 2 | 0.90 | 33 |
| **No DFlash** (baseline) | 65536 | — | 4 | 0.85 | 12 |

### Agentic Workloads

When using this model as a backend for agentic frameworks (OpenClaw, LangGraph, CrewAI, AutoGen, etc.) where a primary agent spawns multiple sub-agents in parallel, keep the following in mind:

- **Increase `MAX_NUM_SEQS` to 6-8** — agents spawn concurrent tool calls and sub-agents that each hold an active sequence. Pair with a lower `MAX_MODEL_LEN` (32K) to fit more sequences in memory
- **Limit sub-agent context size** — configure your gateway or agent framework to cap sub-agent context windows at 16K-32K tokens. This prevents a single runaway agent from consuming most of the KV cache and starving other sequences
- **Aggressively reclaim finished sequences** — configure sub-agents to spin down promptly after completing their task rather than holding the connection open. This frees KV cache for other work
- **Monitor KV cache usage** — check `GPU KV cache usage` in the server logs. If it consistently exceeds 80%, reduce `MAX_NUM_SEQS` or tighten sub-agent context limits
- **Primary agent gets full context** — reserve the full 128K context for your orchestrating agent; sub-agents rarely need more than 16-32K

Example gateway configuration (e.g. OpenClaw):
```
MAX_MODEL_LEN:          32768  (lower context = more concurrent agents)
MAX_NUM_SEQS:           8      (matches max concurrent agents)
Sub-agent context:      16K    (prevents KV cache overflow)
Primary agent context:  32K    (orchestrator gets full context)
Agent idle timeout:     30s    (aggressive reclaim)
```

---

## How DFlash Works

### The Bandwidth Bottleneck

On the DGX Spark, the fundamental bottleneck is **memory bandwidth**. At 273 GB/s, loading 20 GB of NVFP4 weights per token limits inference to ~12 tok/s. Every dense model hits this wall — it's physics, not software.

### Block-Diffusion Speculative Decoding

DFlash ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) breaks through this bottleneck using a **2B block-diffusion drafter** that works fundamentally differently from traditional speculative decoding:

**Traditional speculative decoding:**
```
Drafter: token1 → token2 → token3 → token4  (sequential, N forward passes)
Target:  verify all 4 tokens                  (1 forward pass)
```

**DFlash block-diffusion:**
```
Drafter: [token1, token2, token3, ..., token15]  (parallel, 1 diffusion step)
Target:  verify all 15 tokens                     (1 forward pass)
```

The key insight: the drafter generates **all speculative tokens simultaneously** in a single diffusion forward pass, not sequentially. Drafting cost is roughly constant regardless of how many tokens you propose. This is why 15 speculative tokens performs best — you're not paying 15x the draft cost.

### What Happens Per Step

1. **Draft** — The 2B DFlash model runs one diffusion step, producing 15 candidate tokens in parallel (~constant cost)
2. **Verify** — The 27B target model checks all 15 tokens in a single forward pass (one memory bandwidth pass over the weights)
3. **Accept** — On average, 3-4 tokens are accepted per verification pass
4. **Net result** — You pay for one weight-loading pass but produce multiple tokens, amortizing the bandwidth cost

### Why This Matters on DGX Spark

| | Without DFlash | With DFlash |
|---|---|---|
| **Weight passes per token** | 1 | ~0.3 (amortized) |
| **Single-stream throughput** | 12.2 tok/s | 33.2 tok/s |
| **Effective bandwidth utilization** | 1 token per pass | ~3.5 tokens per pass |
| **User experience** | Sluggish, noticeable delay | Responsive, fluid |

DFlash turns the DGX Spark from "it can run a 27B model" into "it runs a 27B model *well*."

---

## Why Dense Over MoE on DGX Spark

Qwen3.5 comes in two architectures: the **122B-A10B MoE** (256 experts, ~10B active per token) and the **27B dense** model (all parameters active on every token).

### Dense Model Advantages

- **Higher quality per FLOP** — All 27B parameters contribute to every token. MoE models route to a sparse expert subset, which means some experts are undertrained and routing decisions introduce noise.
- **No routing overhead** — MoE models spend compute on expert selection, load balancing, and all-to-all communication.
- **Predictable latency** — No variance from different experts being selected per token. Every forward pass costs the same.
- **Simpler deployment** — No expert parallelism, no load imbalance, fits on a single GPU with NVFP4.

### The Old Tradeoff

Dense models move all their parameters through memory per token. On a bandwidth-limited device like DGX Spark (273 GB/s), a 27B dense model was slow — **12 tok/s**. MoE only moves ~10B active parameters, so it could be faster despite the larger total size.

### DFlash Changes the Equation

DFlash amortizes the bandwidth cost of the dense model across ~3.5 tokens per forward pass:

| | 27B Dense (no DFlash) | 27B Dense + DFlash | 122B MoE |
|---|:---:|:---:|:---:|
| **Active params/token** | 27B | 27B | ~10B |
| **Effective params/token** | 27B | ~8B (amortized) | ~10B |
| **Single-stream tok/s** | 12.2 | **33.2** | ~15 |
| **Quality** | Higher (dense) | Higher (dense) | Lower (sparse routing) |

**DFlash makes the 27B dense model faster than the 122B MoE on DGX Spark** while delivering better quality per parameter. Dense models are practical again.

---

## Why NVFP4 on Blackwell

If you have an **NVIDIA Blackwell GPU** (B200, GB200, GB10/DGX Spark, or later), always use NVFP4 over BF16.

### What is NVFP4?

NVFP4 (FP4 E2M1) is a **native Blackwell tensor core datatype**. Unlike older INT4/GPTQ quantization that introduces visible degradation, NVFP4 with AWQ_FULL calibration is effectively lossless:

- **AWQ_FULL calibration** — Exhaustive grid search (10 scaling factors per layer) plus clipping optimization
- **Selective quantization** — Vision encoder, embeddings, layer norms, and lm_head remain in full BF16
- **Hardware-native** — Blackwell SM 12.x tensor cores execute FP4 matrix multiplies natively via FlashInfer CUTLASS, not through dequantize-then-compute

### The Numbers

| | BF16 | NVFP4 |
|---|:---:|:---:|
| **Model size** | ~52 GB | ~20 GB |
| **Memory for KV cache** | Less headroom | 3x more headroom |
| **Concurrent sequences** | 4 (comfortable) | 8+ (comfortable) |
| **Throughput** | Same (bandwidth-limited) | Same or faster (less bandwidth) |
| **Quality** | Full precision | Effectively lossless |

NVFP4 is a **free upgrade** — same quality, 3x less memory, more headroom for longer context and concurrent requests. The BF16 version exists for non-Blackwell hardware and research workflows.

---

## Hybrid Architecture

Qwen3.5-27B uses a **hybrid attention architecture** mixing two attention types across 64 layers:

| Layer Type | Count | Purpose |
|---|:---:|---|
| **Linear attention (GDN)** | 48 | Gated Delta Network — O(1) per-token state, efficient long-context processing |
| **Full attention** | 16 | Standard multi-head attention every 4th layer for global context capture |

This gives near-linear scaling with sequence length while maintaining full-attention quality at regular intervals. The model also includes a **27-layer ViT vision encoder** (460M params) for image understanding.

---

## Optimized for DGX Spark GB10

This container is specifically built and tested for the **NVIDIA DGX Spark** personal AI supercomputer:

| Spec | Value |
|---|---|
| **GPU** | NVIDIA GB10 (Blackwell, SM 12.1) |
| **Memory** | 128 GB unified (CPU+GPU shared) |
| **Bandwidth** | 273 GB/s |
| **CUDA Compute** | SM 12.1 |
| **TDP** | 200W |

The base image includes:
- FlashInfer compiled from source for SM121
- CUTLASS FP4 GEMM with autotuning
- SM121 compatibility patches for vLLM
- torch.compile with AOT caching
- GDN Triton allocator fixes for Blackwell
- NVFP4 NaN guard for CUTLASS MoE

While the container may work on other Blackwell GPUs (B200, GB200), it has been validated and benchmarked specifically on DGX Spark GB10.

---

## Credits

- **Base model**: [Qwen Team](https://huggingface.co/Qwen) — Qwen3.5-27B
- **DFlash speculative decoding**: [z-lab](https://huggingface.co/z-lab) — [arXiv 2602.06036](https://arxiv.org/abs/2602.06036)
- **Abliteration**: [llm-abliteration](https://github.com/VoidedMirror/llm-abliteration)
- **Container & optimization**: [AEON-7](https://huggingface.co/AEON-7)

## License

Apache 2.0

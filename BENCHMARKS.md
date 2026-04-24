# DFlash-Qwen3.5-27B on DGX Spark GB10 — Benchmark Results

Detailed measurements for `ghcr.io/aeon-7/vllm-dflash:latest` running
`AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4` on a single DGX Spark (NVIDIA GB10,
sm_121a, 128 GB unified memory, driver 580.142, CUDA 13.2).

All measurements use **DFlash spec-decode with k=15** and the Spark-tuned
configuration documented in the README.  All numbers are produced by the
`benchmark/bench_natural.py` script in this repository and are fully
reproducible with the exact prompts listed below.

## Test environment

| Component | Version |
|---|---|
| Hardware | DGX Spark (GB10, sm_121a, 128 GB LPDDR5X) |
| OS | Ubuntu 24.04 aarch64 |
| NVIDIA driver | 580.142 |
| CUDA runtime | 13.2 |
| vLLM | 0.19.1rc1.dev110+gb55d830ec |
| PyTorch | 2.12.0.dev20260408+cu130 |
| Transformers | 5.5.0 |
| FlashInfer | 0.6.7 |
| Image | `ghcr.io/aeon-7/vllm-dflash:latest` (unmodified) |

## Methodology

- **Deterministic prompts**: `temperature=0`, fixed prompts (below), same
  prompt repeated across all concurrency levels.
- **Streaming measurement**: TTFT is time to first streamed content token;
  TPOT is `(total_duration - ttft) / (output_tokens - 1)`.
- **Warmup**: 3 serial requests before each prompt class, results discarded.
  Post-warmup variance across runs <0.3% — the numbers are steady-state,
  not JIT-sensitive.
- **Per-level samples**: 16 requests at each concurrency level, fired
  concurrently via `asyncio.Semaphore`.
- **Token accounting**: `completion_tokens` from the streamed OpenAI `usage`
  block, not a tokenizer re-count.

The `bench_natural.py` script under `benchmark/` is the ground truth;
everything below was produced by running it against a live container with
the Spark-tuned launch settings.

### Prompts used (exact strings)

```python
PROMPTS = {
    "code": (
        "Write a complete Python implementation of quicksort with comments, "
        "type hints, and 3 edge case tests."
    ),
    "reasoning": (
        "Let's think step by step. If a train leaves Paris at 9am going "
        "80 km/h and another leaves Lyon (450 km away) at 10am going "
        "100 km/h towards Paris, when and where do they meet? Show all working."
    ),
    "prose": (
        "Write a detailed 500-word essay about the history and cultural "
        "impact of jazz music in the 20th century."
    ),
    "dialogue": (
        "Continue this conversation naturally. Keep each speaker's turn short. "
        "Alice: 'Have you been to that new coffee place downtown?' Bob:"
    ),
}
```

## Results

### 1. Single-stream, natural prompts (c=1)

Steady-state tokens/sec at concurrency=1, post-warmup, 3 runs, median reported:

| Prompt class | Tok/s | TPOT p50 | TTFT p50 | Accept length* |
|:---|:---:|:---:|:---:|:---:|
| Code | **64.0** | 15.2 ms | 239 ms | ~5.5 |
| Reasoning | **54.0** | 18.4 ms | — | ~5.0 |
| Dialogue | **38.4** | 26.0 ms | — | ~3.3 |
| Prose | **29.5** | 33.6 ms | 225 ms | ~2.0 |

*Approximate acceptance length — accepted tokens per 15-token draft, derived
from the acceptance-rate curve under that prompt class.

### 2. Concurrency scaling — Code (DFlash sweet spot)

| Concurrency | N requests | Wall-clock | Aggregate tok/s | Median per-req | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| c=1 | 16 | 128.0 s | **64.02** | 64.02 | 239 ms | 244 ms | 15.18 ms | 15.22 ms |
| c=4 | 16 | 45.1 s | **181.47** | 45.41 | 408 ms | 460 ms | 21.24 ms | 21.28 ms |
| c=8 | 16 | 31.2 s | **262.77** | 32.88 | 564 ms | 571 ms | 29.38 ms | 29.46 ms |
| c=16 | 16 | 25.0 s | **327.89** | 20.52 | 884 ms | 884 ms | 47.10 ms | 47.26 ms |

### 3. Concurrency scaling — Prose (DFlash worst case)

| Concurrency | N requests | Wall-clock | Aggregate tok/s | Median per-req | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| c=1 | 16 | 278.0 s | **29.46** | 29.47 | 225 ms | 228 ms | 33.57 ms | 33.59 ms |
| c=4 | 16 | 98.1 s | **83.53** | 21.07 | 432 ms | 477 ms | 46.77 ms | 47.26 ms |
| c=8 | 16 | 66.9 s | **122.41** | 15.31 | 557 ms | 572 ms | 64.35 ms | 64.74 ms |
| c=16 | 16 | 54.0 s | **151.81** | 9.49 | 860 ms | 861 ms | 103.86 ms | 103.97 ms |

### 4. DFlash acceptance profile (k=15)

Per-position acceptance rate, measured across 16,317 drafts / 244,755 draft
tokens on random-token input (which is adversarial — natural prompts yield
higher numbers):

| Position | Accept % |
|:---:|:---:|
| 0 | 78.1% |
| 1 | 53.7% |
| 2 | 32.9% |
| 3 | 18.6% |
| 4 | 12.2% |
| 5 | 7.4% |
| 6 | 4.3% |
| 7 | 3.6% |
| 8 | 2.3% |
| 9–14 | <2% |

Mean acceptance length on random input: **3.19 tokens per 15-token draft**.
On natural text, effective acceptance length scales roughly linearly with
the inverse of prompt entropy — prose ~2.0, dialogue ~3.3, code ~5.5.

## Key findings

1. **DFlash is content-sensitive, not JIT-sensitive.** Variance across runs
   is <0.3% post-warmup. Performance differences between prompt classes
   come from prompt predictability, not engine state.

2. **The `MAX_NUM_SEQS=4` default is the main operational bottleneck.**
   Under defaults at c=8, TTFT p50 on random-token load was 14,688 ms —
   queue-saturated. The Spark-tuned config (`MAX_NUM_SEQS=16` +
   `MAX_NUM_BATCHED_TOKENS=32768`) drops this to 564–883 ms and pushes
   aggregate throughput 43–176% higher depending on prompt class and
   concurrency.

3. **c=16 is a new operating point.** At 328 tok/s aggregate on code and
   152 tok/s on prose, c=16 is a real serving configuration — not possible
   at all under defaults.

4. **TPOT stays linear with concurrency.** 15 → 47 ms (code) and 34 → 104 ms
   (prose) from c=1 to c=16 — roughly 3× per-token latency for 16×
   concurrency, which is the expected interactive-serving shape.

5. **k=15 is the right draft length.** Position 8+ accepts <3%, but the
   per-token compute cost is low enough that dropping to k=10 produces no
   measurable gain.

## Reproducing

The full benchmark is a single Python script using `httpx.AsyncClient`:
[`benchmark/bench_natural.py`](benchmark/bench_natural.py).

```bash
# 1. Start the container with the Spark-tuned config (see main README).
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

# 2. Wait for health (cold start ~5–7 min on DGX Spark)
until curl -sf http://localhost:8000/health; do sleep 5; done

# 3. Install benchmark deps on the host
pip install httpx

# 4. Run the benchmark (all four prompt classes, c=1,4,8,16)
for p in code reasoning dialogue prose; do
  python3 benchmark/bench_natural.py \
    --host localhost --port 8000 \
    --model qwen35-aeon7-dflash \
    --prompt $p \
    --concurrencies 1 4 8 16 \
    --requests-per-level 16 \
    --max-tokens 512 \
    --output results/${p}.csv
done
```

Expected runtime: ~25 min total for all four prompt classes.  The script
writes per-concurrency CSVs with `output_tps_aggregate`, `median_tokens_per_req_tps`,
TTFT p50/p95, and TPOT p50/p95.

## Hardware note

These results are **specific to DGX Spark GB10** (sm_121a, 273 GB/s LPDDR5X
unified memory). On B200 or H200 the absolute numbers will be higher; the
relative scaling curve will be similar but the c=16 saturation point will
move up.  The *relative* improvement from Spark-tuned vs default config is
expected to hold across Blackwell GPUs because the bottleneck is the
scheduler queue budget, not GPU compute.

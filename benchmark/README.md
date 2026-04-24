# bench_natural.py

Deterministic natural-prompt concurrency benchmark for `ghcr.io/aeon-7/vllm-dflash`.
Produces the numbers reported in [../BENCHMARKS.md](../BENCHMARKS.md).

## Usage

```bash
pip install httpx

python3 bench_natural.py \
    --host localhost --port 8000 \
    --model qwen35-aeon7-dflash \
    --prompt code \
    --concurrencies 1 4 8 16 \
    --requests-per-level 16 \
    --max-tokens 512 \
    --output code.csv
```

`--prompt` can be `code`, `reasoning`, `dialogue`, or `prose`.  The exact
prompt strings are embedded in the script so results are bit-exact
reproducible.

## What it measures

- **Aggregate output tok/s**: total output tokens / wall-clock duration at
  a given concurrency level.
- **Median per-request tok/s**: median of individual streaming session rates.
- **TTFT p50/p95**: time from request send to first streamed content token.
- **TPOT p50/p95**: per-token decode time, excluding TTFT.

Streaming is via OpenAI-style SSE with `stream_options.include_usage=True`,
so `completion_tokens` comes straight from the server's usage block rather
than a host-side re-tokenize.

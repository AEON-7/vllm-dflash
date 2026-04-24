#!/usr/bin/env python3
"""
Natural-prompt concurrency benchmark for vllm-dflash.

Fires N concurrent requests using a realistic fixed prompt (code or prose),
streams the responses, and reports tok/s (output + total), TTFT, TPOT.

Deterministic: temperature=0, same prompt, same seed.  Multiple runs per
concurrency level are averaged.

Usage:
    python3 bench_natural.py --host localhost --port 8000 \
        --model qwen35-aeon7-dflash \
        --prompt code \
        --concurrencies 1 4 8 16 \
        --max-tokens 512 \
        --output results.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


PROMPTS = {
    "code": (
        "Write a complete Python implementation of quicksort with comments, "
        "type hints, and 3 edge case tests."
    ),
    "reasoning": (
        "Let's think step by step. If a train leaves Paris at 9am going 80 km/h "
        "and another leaves Lyon (450 km away) at 10am going 100 km/h towards "
        "Paris, when and where do they meet? Show all working."
    ),
    "prose": (
        "Write a detailed 500-word essay about the history and cultural impact "
        "of jazz music in the 20th century."
    ),
    "dialogue": (
        "Continue this conversation naturally. Keep each speaker's turn short. "
        "Alice: 'Have you been to that new coffee place downtown?' Bob:"
    ),
}


@dataclass
class RequestResult:
    ttft_s: float                # time to first token
    total_s: float               # full duration
    output_tokens: int           # completion_tokens from usage
    # Derived:
    @property
    def decode_s(self) -> float:
        return max(self.total_s - self.ttft_s, 1e-6)

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return (self.decode_s / (self.output_tokens - 1)) * 1000.0

    @property
    def tokens_per_sec(self) -> float:
        return self.output_tokens / self.total_s


async def run_one(
    client: httpx.AsyncClient,
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.perf_counter()
        first_token_time: float | None = None
        output_tokens = 0
        completion_tokens = 0
        async with client.stream(
            "POST",
            f"{base}/v1/chat/completions",
            json=body,
            timeout=600.0,
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data: "):
                    continue
                payload = raw[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # First content chunk marks first-token time
                if first_token_time is None:
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            first_token_time = time.perf_counter()
                            break
                # Usage comes in the final chunk with include_usage=True
                usage = event.get("usage") or {}
                if usage.get("completion_tokens"):
                    completion_tokens = usage["completion_tokens"]
                # Count rough output tokens as a fallback
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        output_tokens += 1
        end = time.perf_counter()
        ttft = (first_token_time or end) - start
        total = end - start
        return RequestResult(
            ttft_s=ttft,
            total_s=total,
            output_tokens=completion_tokens or output_tokens,
        )


async def run_concurrency_level(
    base: str,
    model: str,
    prompt: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
) -> list[RequestResult]:
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            run_one(client, base, model, prompt, max_tokens, sem)
            for _ in range(num_requests)
        ]
        return await asyncio.gather(*tasks)


def summarize(results: list[RequestResult], wallclock_s: float) -> dict:
    total_out = sum(r.output_tokens for r in results)
    ttfts_ms = sorted(r.ttft_s * 1000 for r in results)
    tpots_ms = sorted(r.tpot_ms for r in results)

    def pct(xs, p):
        if not xs:
            return 0.0
        k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
        return xs[k]

    return {
        "n_requests": len(results),
        "wallclock_s": round(wallclock_s, 2),
        "total_output_tokens": total_out,
        "output_tps_aggregate": round(total_out / wallclock_s, 2),
        "median_tokens_per_req_tps": round(statistics.median(r.tokens_per_sec for r in results), 2),
        "ttft_ms_p50": round(pct(ttfts_ms, 50), 1),
        "ttft_ms_p95": round(pct(ttfts_ms, 95), 1),
        "tpot_ms_p50": round(pct(tpots_ms, 50), 2),
        "tpot_ms_p95": round(pct(tpots_ms, 95), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="code", choices=list(PROMPTS),
                    help="Which canned prompt to fire.")
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--requests-per-level", type=int, default=None,
                    help="Requests per concurrency level (default: 4x concurrency).")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=3, help="Warmup requests before measurement.")
    ap.add_argument("--output", type=Path, default=Path("bench_natural.csv"))
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    prompt = PROMPTS[args.prompt]

    print(f"=== Natural-prompt concurrency bench: {args.prompt} ===")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Model:  {args.model}")
    print(f"  Base:   {base}")
    print()

    # Warmup
    print(f"Warmup: {args.warmup} serial requests…")
    asyncio.run(run_concurrency_level(base, args.model, prompt, 1, args.warmup, min(128, args.max_tokens)))
    print("Warmup done.")
    print()

    rows = []
    for c in args.concurrencies:
        n = args.requests_per_level or max(c * 4, 8)
        print(f"-- c={c}, n={n} --", flush=True)
        t0 = time.perf_counter()
        results = asyncio.run(
            run_concurrency_level(base, args.model, prompt, c, n, args.max_tokens)
        )
        wall = time.perf_counter() - t0
        summary = summarize(results, wall)
        summary["concurrency"] = c
        summary["prompt_class"] = args.prompt
        rows.append(summary)
        print(
            f"   aggregate out: {summary['output_tps_aggregate']:>7} tok/s  "
            f"median per-req: {summary['median_tokens_per_req_tps']:>6} tok/s  "
            f"TTFT p50/p95: {summary['ttft_ms_p50']:>6}/{summary['ttft_ms_p95']:>6} ms  "
            f"TPOT p50: {summary['tpot_ms_p50']:>5} ms"
        )

    # Write CSV
    fieldnames = [
        "prompt_class", "concurrency", "n_requests", "wallclock_s",
        "total_output_tokens", "output_tps_aggregate",
        "median_tokens_per_req_tps",
        "ttft_ms_p50", "ttft_ms_p95", "tpot_ms_p50", "tpot_ms_p95",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print()
    print(f"Written {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

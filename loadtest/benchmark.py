"""Streaming-aware vLLM benchmark that records evidence rather than anecdotes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx


@dataclass
class Result:
    ok: bool
    status_code: int
    ttft_seconds: float | None
    total_seconds: float
    completion_tokens: int | None
    output_tokens_per_second: float | None
    error: str | None = None


async def one_request(client, base_url, api_key, model, prompt, max_tokens) -> Result:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at = None
    completion_tokens = None
    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"authorization": f"Bearer {api_key}"},
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                return Result(False, response.status_code, None, time.perf_counter() - started, None, None, body.decode(errors="replace")[:500])
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line.removeprefix("data: "))
                choices = event.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content") and first_token_at is None:
                    first_token_at = time.perf_counter()
                if event.get("usage"):
                    completion_tokens = event["usage"].get("completion_tokens")
        finished = time.perf_counter()
        ttft = first_token_at - started if first_token_at else None
        generation_seconds = finished - first_token_at if first_token_at else None
        tps = completion_tokens / generation_seconds if completion_tokens and generation_seconds else None
        return Result(True, 200, ttft, finished - started, completion_tokens, tps)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return Result(False, 0, None, time.perf_counter() - started, None, None, repr(exc))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))]


async def scenario(args, concurrency: int) -> dict:
    limits = httpx.Limits(max_connections=max(20, concurrency * 2))
    timeout = httpx.Timeout(args.timeout, connect=10)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        started = time.perf_counter()
        results = []
        for _ in range(0, args.requests, concurrency):
            batch_size = min(concurrency, args.requests - len(results))
            batch = [one_request(client, args.base_url, args.api_key, args.model, args.prompt, args.max_tokens) for _ in range(batch_size)]
            results.extend(await asyncio.gather(*batch))
        wall_seconds = time.perf_counter() - started

    successes = [r for r in results if r.ok]
    ttfts = [r.ttft_seconds for r in successes if r.ttft_seconds is not None]
    latencies = [r.total_seconds for r in successes]
    tps = [r.output_tokens_per_second for r in successes if r.output_tokens_per_second]
    total_tokens = sum(r.completion_tokens or 0 for r in successes)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "successes": len(successes),
        "failures": len(results) - len(successes),
        "wall_seconds": wall_seconds,
        "request_throughput_per_second": len(successes) / wall_seconds,
        "aggregate_output_tokens_per_second": total_tokens / wall_seconds,
        "ttft_seconds": {"median": statistics.median(ttfts) if ttfts else None, "p95": percentile(ttfts, 0.95)},
        "latency_seconds": {"median": statistics.median(latencies) if latencies else None, "p95": percentile(latencies, 0.95)},
        "per_request_output_tokens_per_second": {"median": statistics.median(tps) if tps else None, "p95": percentile(tps, 0.95)},
        "raw": [asdict(result) for result in results],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "local-development-only"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--prompt", default="Explain continuous batching and KV cache in five concise bullet points.")
    parser.add_argument("--output-dir", default="benchmark-results")
    args = parser.parse_args()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "scenarios": [],
    }
    for concurrency in [int(value) for value in args.concurrency.split(",")]:
        print(f"running concurrency={concurrency}", flush=True)
        report["scenarios"].append(await scenario(args, concurrency))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"benchmark-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    asyncio.run(main())


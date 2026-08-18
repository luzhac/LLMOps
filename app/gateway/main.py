"""Thin operational gateway in front of vLLM's OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", case_sensitive=False)

    upstream_url: str = "http://vllm:8000"
    api_key: str
    upstream_api_key: str = "local-upstream-key"
    request_timeout_seconds: float = 300.0
    connect_timeout_seconds: float = 5.0
    max_concurrent_requests: int = 8


settings = Settings()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gateway")
semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

REQUESTS = Counter(
    "llm_gateway_requests_total",
    "Gateway requests",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "llm_gateway_request_duration_seconds",
    "End-to-end gateway latency",
    ["path"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
IN_FLIGHT = Gauge("llm_gateway_in_flight_requests", "Requests currently in flight")
REJECTIONS = Counter(
    "llm_gateway_rejections_total", "Rejected requests", ["reason"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(
        settings.request_timeout_seconds, connect=settings.connect_timeout_seconds
    )
    app.state.client = httpx.AsyncClient(timeout=timeout)
    yield
    await app.state.client.aclose()


app = FastAPI(title="Trade Balance LLM Gateway", version="0.1.0", lifespan=lifespan)


def authorised(value: str | None) -> bool:
    if not value or not value.startswith("Bearer "):
        return False
    return hmac.compare_digest(value.removeprefix("Bearer "), settings.api_key)


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    IN_FLIGHT.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        duration = time.perf_counter() - started
        IN_FLIGHT.dec()
        REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        LATENCY.labels(request.url.path).observe(duration)
        logger.info(
            json.dumps(
                {
                    "event": "request_complete",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "duration_seconds": round(duration, 4),
                }
            )
        )


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> Response:
    try:
        response = await app.state.client.get(f"{settings.upstream_url}/health")
        if response.status_code < 500:
            return JSONResponse({"status": "ready"})
    except httpx.HTTPError:
        pass
    return JSONResponse({"status": "upstream_unavailable"}, status_code=503)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def upstream_stream(request: Request, body: bytes, path: str) -> AsyncIterator[bytes]:
    headers = {
        "authorization": f"Bearer {settings.upstream_api_key}",
        "content-type": request.headers.get("content-type", "application/json"),
        "x-request-id": request.headers.get("x-request-id", str(uuid.uuid4())),
    }
    async with app.state.client.stream(
        request.method, f"{settings.upstream_url}{path}", content=body, headers=headers
    ) as upstream:
        if upstream.status_code >= 400:
            error_body = await upstream.aread()
            raise HTTPException(upstream.status_code, error_body.decode(errors="replace"))
        async for chunk in upstream.aiter_raw():
            yield chunk


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_v1(
    path: str,
    request: Request,
    authorization: str | None = Header(default=None),
):
    if not authorised(authorization):
        REJECTIONS.labels("authentication").inc()
        raise HTTPException(status_code=401, detail="invalid API key")

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.05)
    except TimeoutError as exc:
        REJECTIONS.labels("concurrency_limit").inc()
        raise HTTPException(status_code=429, detail="gateway concurrency limit reached") from exc

    body = await request.body()
    try:
        wants_stream = False
        if body:
            try:
                wants_stream = bool(json.loads(body).get("stream", False))
            except (json.JSONDecodeError, AttributeError):
                pass

        upstream_path = f"/v1/{path}"
        if wants_stream:
            async def release_after_stream() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream_stream(request, body, upstream_path):
                        yield chunk
                finally:
                    semaphore.release()

            return StreamingResponse(
                release_after_stream(), media_type="text/event-stream"
            )

        headers = {
            "authorization": f"Bearer {settings.upstream_api_key}",
            "content-type": request.headers.get("content-type", "application/json"),
        }
        response = await app.state.client.request(
            request.method,
            f"{settings.upstream_url}{upstream_path}",
            content=body,
            headers=headers,
        )
        return Response(
            response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )
    finally:
        if not wants_stream:
            semaphore.release()


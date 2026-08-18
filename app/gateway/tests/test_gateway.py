import importlib
import os

import httpx
from fastapi.testclient import TestClient


os.environ["GATEWAY_API_KEY"] = "test-key"
gateway = importlib.import_module("app.gateway.main")


def test_liveness():
    with TestClient(gateway.app) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_authentication_is_required():
    with TestClient(gateway.app) as client:
        response = client.post("/v1/chat/completions", json={})
    assert response.status_code == 401


def test_non_streaming_proxy(monkeypatch):
    async def handler(request: httpx.Request):
        assert request.headers["authorization"] == "Bearer local-upstream-key"
        return httpx.Response(200, json={"id": "example", "choices": []})

    transport = httpx.MockTransport(handler)
    with TestClient(gateway.app) as client:
        old_client = gateway.app.state.client
        gateway.app.state.client = httpx.AsyncClient(transport=transport)
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={"model": "example", "messages": [], "stream": False},
        )
        gateway.app.state.client = old_client

    assert response.status_code == 200
    assert response.json()["id"] == "example"


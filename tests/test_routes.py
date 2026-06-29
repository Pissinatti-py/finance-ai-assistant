"""HTTP + WebSocket endpoint tests.

Uses a bare FastAPI app (no lifespan) so neither Bedrock nor vector_db is
needed. stream_agent and add_chunks are mocked per test. The /chat WebSocket
tests use Starlette's sync TestClient (websocket support), so they are plain
(non-async) functions.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.websockets import WebSocketDisconnect

from app.api.routes import router

_app = FastAPI()
_app.include_router(router)

API_KEY = "test-api-key"  # matches AI_API_KEY set in conftest.py


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as c:
        yield c


# ── /health ──────────────────────────────────────────────────────────────────


async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── /chat (WebSocket) ──────────────────────────────────────────────────────────


def _ws_client() -> TestClient:
    return TestClient(_app)


def test_chat_ws_missing_api_key_rejected():
    with pytest.raises(WebSocketDisconnect):
        with _ws_client().websocket_connect("/chat") as ws:
            ws.receive_json()


def test_chat_ws_wrong_api_key_rejected():
    with pytest.raises(WebSocketDisconnect):
        with _ws_client().websocket_connect("/chat", headers={"X-API-Key": "wrong"}) as ws:
            ws.receive_json()


def test_chat_ws_streams_tokens_then_done():
    async def fake_stream(question, lang, session_id=None):
        yield {"type": "token", "content": "Hyper"}
        yield {"type": "token", "content": "tension"}
        yield {
            "type": "done",
            "answer": "Hypertension",
            "cached": False,
            "sources": ["art-001"],
            "session_id": "s1",
        }

    with patch("app.api.routes.stream_agent", fake_stream):
        with _ws_client().websocket_connect("/chat", headers={"X-API-Key": API_KEY}) as ws:
            ws.send_json({"question": "What is hypertension?"})
            assert ws.receive_json() == {"type": "token", "content": "Hyper"}
            assert ws.receive_json() == {"type": "token", "content": "tension"}
            done = ws.receive_json()
    assert done["type"] == "done"
    assert done["answer"] == "Hypertension"
    assert done["sources"] == ["art-001"]
    assert done["session_id"] == "s1"


def test_chat_ws_passes_lang_and_session():
    captured: dict = {}

    async def fake_stream(question, lang, session_id=None):
        captured.update(question=question, lang=lang, session_id=session_id)
        yield {"type": "done", "answer": "ok", "cached": False, "sources": [], "session_id": session_id}

    with patch("app.api.routes.stream_agent", fake_stream):
        with _ws_client().websocket_connect("/chat", headers={"X-API-Key": API_KEY}) as ws:
            ws.send_json({"question": "q", "lang": "en", "session_id": "abc"})
            ws.receive_json()
    assert captured == {"question": "q", "lang": "en", "session_id": "abc"}


def test_chat_ws_malformed_payload_reports_error():
    with patch("app.api.routes.stream_agent", AsyncMock()):
        with _ws_client().websocket_connect("/chat", headers={"X-API-Key": API_KEY}) as ws:
            ws.send_json({"lang": "en"})  # missing required "question"
            msg = ws.receive_json()
    assert msg["type"] == "error"


def test_chat_ws_oversized_question_reports_error():
    with patch("app.api.routes.stream_agent", AsyncMock()):
        with _ws_client().websocket_connect("/chat", headers={"X-API-Key": API_KEY}) as ws:
            ws.send_json({"question": "x" * 5000})  # exceeds max_length=4000
            msg = ws.receive_json()
    assert msg["type"] == "error"


# ── /ingest ───────────────────────────────────────────────────────────────────


async def test_ingest_missing_api_key_returns_403(client):
    resp = await client.post("/ingest", json={"content": "some text"})
    assert resp.status_code == 403


async def test_ingest_wrong_api_key_returns_401(client):
    resp = await client.post(
        "/ingest",
        json={"content": "text"},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


async def test_ingest_returns_chunks_stored(client):
    with patch("app.api.routes.add_chunks", AsyncMock(return_value=4)):
        resp = await client.post(
            "/ingest",
            json={"content": "long medical text...", "source": "CRM 2024"},
            headers={"X-API-Key": API_KEY},
        )
    assert resp.status_code == 200
    assert resp.json() == {"chunks_stored": 4}


async def test_ingest_passes_source_and_metadata(client):
    captured: dict = {}

    async def _capture(content, source, metadata):
        captured.update(source=source, metadata=metadata)
        return 1

    with patch("app.api.routes.add_chunks", _capture):
        await client.post(
            "/ingest",
            json={"content": "x", "source": "doc_A", "metadata": {"year": 2024}},
            headers={"X-API-Key": API_KEY},
        )
    assert captured["source"] == "doc_A"
    assert captured["metadata"] == {"year": 2024}


async def test_ingest_oversized_content_rejected(client):
    resp = await client.post(
        "/ingest",
        json={"content": "x" * 100_001},  # exceeds max_length=100_000
        headers={"X-API-Key": API_KEY},
    )
    assert resp.status_code == 422


async def test_ingest_empty_content_rejected(client):
    resp = await client.post(
        "/ingest",
        json={"content": "   "},  # stripped to empty → fails min_length=1
        headers={"X-API-Key": API_KEY},
    )
    assert resp.status_code == 422

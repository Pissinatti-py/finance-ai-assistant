import logging
import secrets

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Security,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.security import APIKeyHeader

from app.agent.runner import stream_agent
from app.api.schemas import ChatRequest, IngestRequest, IngestResponse
from app.config import settings
from app.rag.store import add_chunks

logger = logging.getLogger(__name__)

router = APIRouter()
# auto_error=False so we distinguish missing (403) from wrong (401) deterministically.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_key(key: str | None = Security(_api_key_header)) -> None:
    """
    Validate the ``X-API-Key`` header on protected endpoints.

    :param key: The header value, or None if absent.
    :type key: str | None
    :raises HTTPException: 403 if the header is missing, 401 if it is wrong.
    :return: Nothing; raises on failure.
    :rtype: None
    """
    if key is None:
        raise HTTPException(status_code=403, detail="Missing API key")
    if not secrets.compare_digest(key, settings.ai_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


@router.websocket("/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """
    Real-time chat over WebSocket.

    Auth: an ``X-API-Key`` header or ``?api_key=`` query param matching
    ``AI_API_KEY``; otherwise the handshake is rejected. Once connected, the
    client sends a JSON :class:`ChatRequest` per turn and receives a stream of
    ``{"type": "token", "content": str}`` events followed by a final
    ``{"type": "done", ...}`` event. Malformed payloads and runtime failures
    are reported as ``{"type": "error", "detail": str}`` without closing.

    :param websocket: The incoming WebSocket connection.
    :type websocket: WebSocket
    :return: Nothing.
    :rtype: None
    """
    key = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    if key is None or not secrets.compare_digest(key, settings.ai_api_key):
        logger.warning("ws /chat handshake rejected: invalid api key")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logger.info("ws /chat connected")
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                request = ChatRequest(**payload)
            except Exception as exc:  # malformed turn — report and keep the socket open
                logger.warning(f"ws /chat invalid payload: {exc}")
                await websocket.send_json({"type": "error", "detail": str(exc)})
                continue
            try:
                async for event in stream_agent(request.question, request.lang, session_id=request.session_id):
                    await websocket.send_json(event)
            except Exception as exc:  # agent/backend failure — report and keep the socket open
                logger.exception("ws /chat agent failure")
                await websocket.send_json({"type": "error", "detail": str(exc)})
    except WebSocketDisconnect:
        logger.info("ws /chat disconnected")
        return


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, _=Depends(_verify_key)):
    """
    Ingest a document into the RAG knowledge base.

    :param request: The ingest request (content, optional source and metadata).
    :type request: IngestRequest
    :return: The number of chunks stored.
    :rtype: IngestResponse
    """
    n = await add_chunks(request.content, request.source, request.metadata)
    return IngestResponse(chunks_stored=n)


@router.get("/health")
async def health():
    """
    Liveness probe.

    :return: ``{"status": "ok"}`` when the service is running.
    :rtype: dict
    """
    return {"status": "ok"}

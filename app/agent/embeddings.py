"""Local embeddings via fastembed (ONNX, CPU) — no cloud, no credentials.

fastembed is synchronous, so the adapter offloads encoding to a worker thread
and exposes the same async interface the cache and RAG layers already call
(``aembed_query`` / ``aembed_documents``). Swapping providers (Voyage, Bedrock,
OpenAI) only means giving ``client`` another object with those two methods.
"""

import asyncio

from fastembed import TextEmbedding

from app.config import settings


class FastEmbedClient:
    """Async wrapper around a fastembed :class:`TextEmbedding` model."""

    def __init__(self, model_name: str) -> None:
        """
        Load the embedding model (downloads/caches weights on first use).

        :param model_name: The fastembed model id (e.g. ``BAAI/bge-small-en-v1.5``).
        :type model_name: str
        """
        self._model = TextEmbedding(model_name=model_name)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to vectors synchronously (runs in a worker thread)."""
        return [vec.tolist() for vec in self._model.embed(texts)]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of documents.

        :param texts: The texts to embed.
        :type texts: list[str]
        :return: One embedding vector per input text.
        :rtype: list[list[float]]
        """
        return await asyncio.to_thread(self._embed, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """
        Embed a single query.

        :param text: The text to embed.
        :type text: str
        :return: The embedding vector.
        :rtype: list[float]
        """
        vecs = await asyncio.to_thread(self._embed, [text])
        return vecs[0]


# ponytail: single shared instance — both cache and rag import from here
client: FastEmbedClient | None = None


def init_embeddings() -> None:
    """
    Initialize the shared fastembed client.

    Called once from the FastAPI lifespan (and from the seed script). Sets the
    module-level ``client`` that the cache and RAG layers read at call time.

    :return: Nothing.
    :rtype: None
    """
    global client
    client = FastEmbedClient(settings.embedding_model)

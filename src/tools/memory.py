from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.tools import tool

from ..config import get_settings

logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(self, path: Optional[str] = None) -> None:
        cfg = get_settings()
        db_path = path or cfg.chroma_path
        self._client = chromadb.PersistentClient(path=db_path)
        model_name = cfg.chroma_embedding_model
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self._collection = self._client.get_or_create_collection(
            name="research",
            embedding_function=embedding_function,
        )

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[str]:
        if chunk_size <= 0:
            return [text]
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size - 1)

        chunks: List[str] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start = end - chunk_overlap
        return chunks

    def store_research(
        self,
        text: str,
        source: str = "research",
        source_url: str = "unknown",
    ) -> str:
        if not text.strip():
            return ""
        chunks = self.chunk_text(text)
        if not chunks:
            return ""
        doc_id = str(uuid.uuid4())
        ids = [f"{doc_id}-{idx}" for idx in range(len(chunks))]
        self._collection.add(
            ids=ids,
            documents=chunks,
            metadatas=[
                {"source": source, "source_url": source_url, "chunk_index": idx}
                for idx in range(len(chunks))
            ],
        )
        return doc_id

    def retrieve_knowledge(self, query: str, k: int = 3) -> List[str]:
        if not query.strip():
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
        )
        documents = results.get("documents", [[]])
        return [doc for doc in documents[0] if doc]

    def retrieve_knowledge_with_sources(
        self,
        query: str,
        k: int = 3,
    ) -> List[Dict[str, str]]:
        if not query.strip():
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        combined: List[Dict[str, str]] = []
        for doc, meta in zip(documents, metadatas, strict=False):
            source_url = str(meta.get("source_url", "unknown")) if meta else "unknown"
            combined.append({"text": str(doc), "source_url": source_url})
        return combined


_VECTOR_DB: Optional[VectorDB] = None


def get_vector_db() -> VectorDB:
    global _VECTOR_DB
    if _VECTOR_DB is None:
        _VECTOR_DB = VectorDB()
    return _VECTOR_DB


@tool("store_research")
def store_research(
    text: str,
    source: str = "research",
    source_url: str = "unknown",
) -> str:
    """Embed and store research text in the vector database."""
    try:
        return get_vector_db().store_research(
            text=text,
            source=source,
            source_url=source_url,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to store research: %s", exc)
        return ""


@tool("retrieve_knowledge")
def retrieve_knowledge(query: str, k: int = 3) -> List[Dict[str, str]]:
    """Retrieve semantically similar research entries with source URLs."""
    try:
        return get_vector_db().retrieve_knowledge_with_sources(query=query, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to retrieve knowledge: %s", exc)
        return []

"""Ingestion: profile document -> chunks -> cached embeddings -> vector store."""

from careergraph.ingestion.chunker import ProfileChunker
from careergraph.ingestion.embedder import (
    EmbeddingBackend,
    EmbeddingCache,
    SentenceTransformerEmbedder,
    build_embedder,
)
from careergraph.ingestion.pipeline import IngestionPipeline, IngestionReport, load_profile
from careergraph.ingestion.vector_store import VectorStore, build_vector_store

__all__ = [
    "EmbeddingBackend",
    "EmbeddingCache",
    "IngestionPipeline",
    "IngestionReport",
    "ProfileChunker",
    "SentenceTransformerEmbedder",
    "VectorStore",
    "build_embedder",
    "build_vector_store",
    "load_profile",
]

"""Composition root: builds the object graph from :class:`Settings`."""

from __future__ import annotations

from dataclasses import dataclass

from repo_sage.answer.engine import AnswerEngine
from repo_sage.chunking.code import CodeChunker
from repo_sage.config import Settings
from repo_sage.embeddings.base import Embedder
from repo_sage.embeddings.factory import build_embedder
from repo_sage.indexer import Indexer
from repo_sage.llm.base import LLM
from repo_sage.llm.factory import build_llm
from repo_sage.retrieval.rerank import Reranker, build_reranker
from repo_sage.retrieval.retriever import HybridRetriever
from repo_sage.store.base import Store
from repo_sage.store.factory import build_store


@dataclass(slots=True)
class Services:
    settings: Settings
    store: Store
    embedder: Embedder
    llm: LLM
    reranker: Reranker
    indexer: Indexer
    retriever: HybridRetriever
    engine: AnswerEngine

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        store: Store | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
    ) -> Services:
        store = store or build_store(settings)
        embedder = embedder or build_embedder(settings)
        llm = llm or build_llm(settings)
        reranker = reranker or build_reranker(settings.reranker, settings.cross_encoder_model)
        chunker = CodeChunker(
            max_lines=settings.chunk_max_lines,
            window=settings.chunk_window_lines,
            overlap=settings.chunk_window_overlap,
        )
        indexer = Indexer(
            store=store,
            embedder=embedder,
            data_dir=settings.data_dir,
            chunker=chunker,
            max_file_bytes=settings.max_file_bytes,
        )
        retriever = HybridRetriever(
            store=store, embedder=embedder, reranker=reranker, candidate_k=settings.candidate_k
        )
        engine = AnswerEngine(retriever=retriever, llm=llm, top_k=settings.top_k)
        return cls(settings, store, embedder, llm, reranker, indexer, retriever, engine)

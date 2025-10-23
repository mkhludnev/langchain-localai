from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import Field, PrivateAttr, model_validator


class LocalAIRerank(BaseDocumentCompressor):
    """Document compressor that uses LocalAI Rerank API (supports sync and async)."""

    model: str = Field(default="jina-reranker-v1-base-en")
    top_n: Optional[int] = Field(default=3, ge=1)
    openai_api_key: Optional[str] = Field(default=None)
    openai_api_base: str = Field(default="")

    # Private HTTP clients
    _sync_client: Optional[httpx.Client] = PrivateAttr(default=None)
    _async_client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["openai_api_key"] = get_from_dict_or_env(
                data, "openai_api_key", "OPENAI_API_KEY"
            )
            data["openai_api_base"] = get_from_dict_or_env(
                data, "openai_api_base", "OPENAI_API_BASE", default=""
            )
        return data

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            headers = {"Content-Type": "application/json"}
            if self.openai_api_key:
                headers["Authorization"] = f"Bearer {self.openai_api_key}"
            self._sync_client = httpx.Client(headers=headers)
        return self._sync_client

    async def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            headers = {"Content-Type": "application/json"}
            if self.openai_api_key:
                headers["Authorization"] = f"Bearer {self.openai_api_key}"
            self._async_client = httpx.AsyncClient(headers=headers)
        return self._async_client

    def _rerank_sync(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        resolved_model = model or self.model
        resolved_top_n = top_n if top_n is not None and top_n > 0 else self.top_n

        data = {
            "query": query,
            "documents": docs,
            "model": resolved_model,
            "top_n": resolved_top_n,
        }

        client = self._get_sync_client()
        url = f"{self.openai_api_base.rstrip('/')}/v1/rerank"
        response = client.post(url, json=data)
        response.raise_for_status()
        resp = response.json()

        if "results" not in resp:
            raise RuntimeError(resp.get("detail", "Unknown error from rerank API"))

        return [
            {"index": r["index"], "relevance_score": r["relevance_score"]}
            for r in resp["results"]
        ]

    async def _rerank_async(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        resolved_model = model or self.model
        resolved_top_n = top_n if top_n is not None and top_n > 0 else self.top_n

        data = {
            "query": query,
            "documents": docs,
            "model": resolved_model,
            "top_n": resolved_top_n,
        }

        client = await self._get_async_client()
        url = f"{self.openai_api_base.rstrip('/')}/v1/rerank"
        response = await client.post(url, json=data)
        response.raise_for_status()
        resp = response.json()

        if "results" not in resp:
            raise RuntimeError(resp.get("detail", "Unknown error from rerank API"))

        return [
            {"index": r["index"], "relevance_score": r["relevance_score"]}
            for r in resp["results"]
        ]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        results = self._rerank_sync(documents, query)
        return self._build_compressed_docs(documents, results)

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        results = await self._rerank_async(documents, query)
        return self._build_compressed_docs(documents, results)

    def _build_compressed_docs(
        self, documents: Sequence[Document], results: List[Dict[str, Any]]
    ) -> List[Document]:
        compressed = []
        for res in results:
            original_doc = documents[res["index"]]
            new_doc = Document(
                page_content=original_doc.page_content,
                metadata=deepcopy(original_doc.metadata),
            )
            new_doc.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(new_doc)
        return compressed

    def close(self) -> None:
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

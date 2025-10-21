from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import requests
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import get_from_dict_or_env, pre_init
from pydantic import ConfigDict


class LocalAIRerank(BaseDocumentCompressor):
    """Document compressor that uses `LocalAI Rerank API`."""

    session: Any = None
    """Requests session to communicate with API."""
    top_n: Optional[int] = 3
    """Number of documents to return."""
    model: str = "jina-reranker-v1-base-en"
    """Model to use for reranking."""
    openai_api_key: Optional[str] = None
    """TODO"""
    openai_api_base: Optional[str] = None
    """TODO"""

    model_config = ConfigDict(
        protected_namespaces=(),
        extra="forbid",
    )

    @pre_init
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        session = requests.Session()
        session.headers.update(
            {
                **(
                    {"Authorization": f"Bearer {values['openai_api_key']}"}
                    if values["openai_api_key"]
                    else {}
                ),
                "Content-type": "application/json",
            }
        )
        values["session"] = session
        return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        model: Optional[str] = None,
        top_n: Optional[int] = -1,
        max_chunks_per_doc: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
            max_chunks_per_doc : The maximum number of chunks derived from a document.
        """  # noqa: E501
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = model or self.model
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        data = {
            "query": query,
            "documents": docs,
            "model": model,
            "top_n": top_n,
        }

        if not self.openai_api_base:
            raise ValueError("openai_api_base or env OPENAI_API_KEY must be set")
        else:
            resp = self.session.post(
                self.openai_api_base
                + ("/" if not self.openai_api_base.endswith("/") else "")
                + "v1/rerank",
                json=data,
            ).json()

        if "results" not in resp:
            raise RuntimeError(resp["detail"])

        results = resp["results"]
        result_dicts = []
        for res in results:
            result_dicts.append(
                {
                    "index": res["index"],
                    "relevance_score": res["relevance_score"],
                }
            )
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Jina's Rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed

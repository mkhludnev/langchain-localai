import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from langchain_localai import LocalAIRerank


def test_localai_rerank_base_url() -> None:
    reranker = LocalAIRerank(
        openai_api_key="random-string", openai_api_base="http://localhost:8080"
    )
    assert reranker.openai_api_base == "http://localhost:8080"


def test_localai_rerank_sync_client_headers() -> None:
    reranker = LocalAIRerank(openai_api_key="secret", openai_api_base="http://x")
    client = reranker._get_sync_client()
    # httpx stores headers in .headers and the Authorization header should be set
    assert client.headers.get("Authorization") == "Bearer secret"
    assert client.headers.get("Content-Type") == "application/json"
    # cleanup
    reranker.close()


@pytest.mark.asyncio
async def test_localai_rerank_async_client_headers() -> None:
    reranker = LocalAIRerank(openai_api_key="secret", openai_api_base="http://x")
    client = await reranker._get_async_client()
    assert client.headers.get("Authorization") == "Bearer secret"
    assert client.headers.get("Content-Type") == "application/json"
    # cleanup
    await reranker.aclose()


def test_localai_rerank_top_n_validation() -> None:
    # top_n has ge=1 constraint; creating with 0 should raise ValidationError
    with pytest.raises(ValidationError):
        LocalAIRerank(top_n=0)  # type: ignore[arg-type]


def test_localai_rerank_empty_documents_compress_returns_empty() -> None:
    reranker = LocalAIRerank(openai_api_key="k", openai_api_base="http://x")
    # compress_documents should handle empty input without network calls
    # and return empty list
    res = reranker.compress_documents([], "query")
    assert res == []


@pytest.mark.asyncio
async def test_localai_rerank_async_empty_documents_returns_empty() -> None:
    reranker = LocalAIRerank(openai_api_key="k", openai_api_base="http://x")
    res = await reranker.acompress_documents([], "query")
    assert res == []


@pytest.mark.requires("openai")
def test_build_compressed_docs_adds_relevance_score() -> None:
    reranker = LocalAIRerank(openai_api_key="k", openai_api_base="http://x")
    docs = [
        Document(page_content="doc0", metadata={"id": 0}),
        Document(page_content="doc1", metadata={"id": 1}),
    ]
    # pretend the reranker returned only the second document as top result
    results = [{"index": 1, "relevance_score": 0.9123}]
    compressed = reranker._build_compressed_docs(docs, results)
    assert len(compressed) == 1
    out = compressed[0]
    assert out.page_content == "doc1"
    assert out.metadata["id"] == 1
    assert "relevance_score" in out.metadata
    assert out.metadata["relevance_score"] == 0.9123

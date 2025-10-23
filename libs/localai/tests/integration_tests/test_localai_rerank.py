import pytest
from langchain_core.documents import Document

from langchain_localai import LocalAIRerank


@pytest.mark.vcr
def test_localai_rerank_sync() -> None:
    reranker = LocalAIRerank(
        openai_api_key="foo",
        model="bge-reranker-v2-m3",
        openai_api_base="https://foo.bar/",
    )

    docs = [
        Document(page_content="foo bar"),
        Document(page_content="moo foo"),
        Document(page_content="completely unrelated content"),
    ]

    results = reranker.compress_documents(docs, query="foo")
    # Basic structure checks
    assert isinstance(results, (list, tuple))
    assert len(results) >= 1

    # Each returned document should be a Document and have a numeric relevance_score
    scores = []
    for doc in results:
        assert isinstance(doc, Document)
        assert "relevance_score" in doc.metadata
        score = doc.metadata["relevance_score"]
        assert isinstance(score, (float, int))
        scores.append(score)

    # Scores should be sorted in non-increasing order (highest relevance first)
    if len(scores) > 1:
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    reranker.close()


@pytest.mark.vcr
async def test_localai_rerank_async() -> None:
    reranker = LocalAIRerank(
        openai_api_key="foo",
        model="bge-reranker-v2-m3",
        openai_api_base="https://foo.bar/",
    )

    docs = [
        Document(page_content="foo bar"),
        Document(page_content="moo foo"),
        Document(page_content="completely unrelated content"),
    ]

    results = await reranker.acompress_documents(docs, query="foo")
    # Basic structure checks
    assert isinstance(results, (list, tuple))
    assert len(results) >= 1

    # Each returned document should be a Document and have a numeric relevance_score
    scores = []
    for doc in results:
        assert isinstance(doc, Document)
        assert "relevance_score" in doc.metadata
        score = doc.metadata["relevance_score"]
        assert isinstance(score, (float, int))
        scores.append(score)

    # Scores should be sorted in non-increasing order (highest relevance first)
    if len(scores) > 1:
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    await reranker.aclose()


# @pytest.mark.vcr
@pytest.mark.skip  # odd but https://github.com/mudler/LocalAI/issues/6700
def test_localai_rerank_internal_top_n_sync() -> None:
    reranker = LocalAIRerank(
        openai_api_key="foo",
        model="bge-reranker-v2-m3",
        openai_api_base="https://foo.bar/",
    )

    raw_docs = ["foo bar", "moo foo", "another document"]
    # Use the internal API to control top_n
    res = reranker._rerank_sync(raw_docs, query="foo", top_n=1)
    assert isinstance(res, list)
    assert len(res) == 1
    entry = res[0]
    assert "index" in entry and "relevance_score" in entry
    assert isinstance(entry["index"], int)
    assert isinstance(entry["relevance_score"], (float, int))

    reranker.close()


# @pytest.mark.vcr
@pytest.mark.skip  # odd but https://github.com/mudler/LocalAI/issues/6700
async def test_localai_rerank_internal_top_n_async() -> None:
    reranker = LocalAIRerank(
        openai_api_key="foo",
        model="bge-reranker-v2-m3",
        openai_api_base="https://foo.bar/",
    )

    raw_docs = ["foo bar", "moo foo", "another document"]
    res = await reranker._rerank_async(raw_docs, query="foo", top_n=2)
    assert isinstance(res, list)
    assert len(res) == 2
    for entry in res:
        assert "index" in entry and "relevance_score" in entry
        assert isinstance(entry["index"], int)
        assert isinstance(entry["relevance_score"], (float, int))

    await reranker.aclose()

# langchain-localai

This package contains the LangChain integration with LocalAI

## Installation

```bash
pip install -U langchain-localai
```

## Embeddings

`LocalAIEmbeddings` class exposes embeddings from LocalAI.

```python
from langchain_localai import LocalAIEmbeddings

embeddings = LocalAIEmbeddings(
    openai_api_base="http://localhost:8080",
    model="embedding-model-name")
embeddings.embed_query("What is the meaning of life?")
```

## Reranker

`LocalAIRerank` class exposes reranker from LocalAI.

```python
from langchain_localai import LocalAIRerank
from langchain_core.documents import Document

reranker = LocalAIRerank(
        openai_api_key="foo",
        model="bge-reranker-v2-m3",
        openai_api_base="http://localhost:8080",
    )
reranked_docs = reranker.compress_documents(documents=[
    Document(page_content="bar"),Document(page_content="baz")], query="foo")
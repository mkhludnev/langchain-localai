# langchain-localai

This package contains the LangChain integration with LocalAI

## Installation

```bash
pip install -U langchain-localai
```

## Embeddings

`LocalAIEmbeddings` class exposes embeddings from LocalAI.

```python
from libs.localai.langchain_localai import LocalAIEmbeddings

embeddings = LocalAIEmbeddings(
    openai_api_base="http://localhost:8080",
    model="embedding-model-name")
embeddings.embed_query("What is the meaning of life?")
```

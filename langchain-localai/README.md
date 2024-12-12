# langchain-localai

This package contains the LangChain integration with LocalAI

## Installation

```bash
pip install -U langchain-localai
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatLocalAI` class exposes chat models from LocalAI.

```python
from langchain_localai import ChatLocalAI

llm = ChatLocalAI()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`LocalAIEmbeddings` class exposes embeddings from LocalAI.

```python
from langchain_localai import LocalAIEmbeddings

embeddings = LocalAIEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`LocalAILLM` class exposes LLMs from LocalAI.

```python
from langchain_localai import LocalAILLM

llm = LocalAILLM()
llm.invoke("The meaning of life is")
```

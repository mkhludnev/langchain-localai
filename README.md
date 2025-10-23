from xml.dom.minidom import Document

## langchain-localai

This project provides Langchain [integration package](https://python.langchain.com/docs/contributing/how_to/integrations/) for [LocalAI Embeddings](https://localai.io/features/embeddings/) and for [LocalAI Reranker](https://localai.io/features/reranker/)..

It's a descendant of Langchain's [community package](https://python.langchain.com/docs/integrations/text_embedding/localai/). 
The main advantage is that:
 - it depends on OpenAI SDK v1 and 
 - provides faster bulk embedding requests to LocalAI server.
 - async support
 - reranker support

### Installation

```bash
pip install langchain-localai
```

### Quick example (conceptual)

```python

from langchain_localai import LocalAIEmbeddings

embeddings = LocalAIEmbeddings(
    openai_api_base="https://localhost:8080", model="embedding-model-name",
    # if you are behind an explicit proxy, pass proxy URL via
    # openai_proxy="http://proxy.yourcompany.com:8080",
)

query_result = embeddings.embed_query("Sample query")
doc_result = embeddings.embed_documents( ["First document", "Second document"])


from langchain_localai import LocalAIRerank
from langchain_core.documents import Document

reranker = LocalAIRerank(
        openai_api_key="foo",
        model="bge-reranker-v2-m3",
        openai_api_base="http://localhost:8080",
    )
reranked_docs = reranker.compress_documents(documents=[
    Document(page_content="bar"),Document(page_content="baz")], query="foo")
```

###  Who should use this
Teams deploying LocalAI locally or within private networks and users who require reliable, performant embeddings and reranker (cross-encoder) integration with [LangChain](https://docs.langchain.com/oss/python/langchain/overview).

### Contributing
Please open issues with reproducible steps and metrics (latency, throughput) or submit pull requests to improve batching, resilience, or tests.

### License
See LICENSE file in the repository.

### Contact
Repository owner: @mkhludnev
Open an issue or discussion for questions.


### References

 - Initial [discussion](https://github.com/langchain-ai/langchain/pull/22399#issuecomment-2537387476).  
 - package [README.md](./libs/localai/README.md)
 - [documentation](./libs/localai/docs/localai_embeddings.ipynb)
 - PyPi [repository](https://pypi.org/project/langchain-localai)
 - LocalAI embeddings: https://localai.io/features/embeddings/
 - LocalAI Reranker (aka cross-encoder): https://localai.io/features/reranker/
 - [LangChain](https://docs.langchain.com/oss/python/langchain/overview).

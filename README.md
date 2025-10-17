## langchain-localai

This project provides Langchain [integration package](https://python.langchain.com/docs/contributing/how_to/integrations/) for [LocalAI embeddings](https://localai.io/features/embeddings/).

It's descendant of Langchain [community package](https://python.langchain.com/docs/integrations/text_embedding/localai/). 
The main advantage is that it depends on OpenAI SDK v1 that provides faster bulk embeddings requests with LocalAI.

### Installation

pip install langchain-localai

### Quick example (conceptual)

```python
from langchain_localai import LocalAIEmbeddings


from langchain_localai import LocalAIEmbeddings

embeddings = LocalAIEmbeddings(
    openai_api_base="http://localhost:8080", model="embedding-model-name",
    # if you are behind an explicit proxy, pass proxy URL via
    # openai_proxy="http://proxy.yourcompany.com:8080",
)

query_result = embeddings.embed_query("Sample query")
doc_result = embeddings.embed_documents( ["First document", "Second document"])
```

###  Who should use this
Teams deploying LocalAI locally or within private networks and users who require reliable, performant embeddings integration with LangChain.

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

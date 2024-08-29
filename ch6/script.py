# %%
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from transformers import AutoTokenizer
from typing import Optional
import os
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

HF_TOKEN: Optional[str] = os.environ["HF_TOKEN"]
model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
)
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

llm = HuggingFaceLLM(
    model_name=model_name,
    model_kwargs={
        "token": HF_TOKEN,
        "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
        # "quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    tokenizer_name=model_name,
    tokenizer_kwargs={"token": HF_TOKEN},
    stopping_ids=stopping_ids,
)
emb_model_name = "BAAI/bge-small-en-v1.5"
embed_model = HuggingFaceEmbedding(model_name=emb_model_name)
Settings.embed_model = embed_model
Settings.llm = llm
# %% [markdown]
# ## Learning about query mechanics – an overview
#
# First, we need to understand the typical steps in the query process: **retrieval**, **postprocessing**, and
# **response synthesis**
# 
# First, we will focus on **retrievers**.
# %% [markdown]
# ### Understanding the basic retrievers

# Retrieval mechanisms are a central element in any RAG system. Although they work in different
# ways, all types of retrievers are based on the same principle: they browse an index and select the
# relevant nodes to build the necessary context. Each index type offers several retrieval modes, each
# providing different features and customization options. Keep in mind that while all retrievers return
# `NodeWithScore`, not all of them associate a specific node score

# %%
# Assuming that we have already dealt with document ingestion, the following code builds an index and then builds a
# retriever based on the structure of the index:
from llama_index.core import SummaryIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader("files").load_data()
summary_index = SummaryIndex.from_documents(documents)
retriever = summary_index.as_retriever(
    retriever_mode='embedding'
)
result = retriever.retrieve("Tell me about ancient Rome")
print(result[0].text)
# %%
# The second option is direct instantiation, as shown in the following example:

from llama_index.core.retrievers import SummaryIndexEmbeddingRetriever
documents = SimpleDirectoryReader("files").load_data()
summary_index = SummaryIndex.from_documents(documents)
retriever = SummaryIndexEmbeddingRetriever(
    index=summary_index
)
result = retriever.retrieve("Tell me about ancient Rome")
print(result[0].text)

# %% [markdown]

# ### The VectorStoreIndex retrievers

# #### VectorIndexRetriever

# The default retriever that’s used by VectorStoreIndex is VectorIndexRetriever. It can
# easily be constructed using the following command:

# ```py
# VectorStoreIndex.as_retriever()
# ```

# This retriever operates by converting queries into vectors and then performing similarity-based searches
# in the vector space. Several parameters can be customized for different use cases:
#
# * `similarity_top_k`: This defines the number of top (k) results returned by the retriever.
# This determines how many of the most similar results are returned for each query. For example,
# if we want a broader search, we can change the default value, which is 2.
#
# * `vector_store_query_mode`: This sets the query mode of the vector store. Different
# variants of external vector stores, such as Pinecone, OpenSearch , and others, support different query modes.
# This is the mechanism by which we can make best use of their search capabilities.
#
# * `filters`: Remember that in Chapter 3, in the Nodes section, we saw how to add metadata to
# our nodes? Well, we can use this metadata to narrow down the search scope of the retriever.
# We will see a practical example of this in this chapter, where we will use metadata filters to
# implement a simple system for filtering nodes returned by an index.
#
# * `alpha`: This one is useful when using a hybrid search mode (a combination of sparse and
# dense search). We will discuss the difference between sparse and dense search in more detail
# later in this chapter.
#
# * `sparse_top_k`: The number of top results for the sparse search. This is relevant in hybrid
# search modes. The previous mention applies here also.

# * `doc_ids`: Similar to metadata filters, but slightly coarser, doc_ids can be used to restrict
# the search to a specific subset of documents. For example, suppose the organization uses a
# common knowledge base that is shared by all departments. At the same time, however, the
# organization has a clear naming convention for documents. If the department’s name or code is
# found in the document name, we could use this parameter to limit a user’s query to documents
# in their department only.
#
# * `node_ids`: This parameter is similar to doc_ids but refers to node IDs within the index.
# This can give us even more granular control over the information that’s returned by the retriever.

# * `vector_store_kwargs`: This parameter can pass additional arguments that are specific
# to each vector store so that they can be sent at query time.

# #### VectorIndexAutoRetriever

# All the parameters we discussed earlier regarding VectorIndexRetriever are very useful when we
# know exactly what we are looking for and understand the structure of the data very well. Unfortunately,
# in some situations, we will be dealing with complex structures or ambiguities in the indexed data.
# VectorIndexAutoRetriever is a more advanced form of retriever that can use an LLM to
# automatically set query parameters in a vector store based on a natural language description of the
# content and supporting metadata. This is particularly useful when users are unfamiliar with the
# structure of the data or do not know how to formulate an effective query.

# ### The SummaryIndex retrievers

# #### SummaryIndexRetriever

# ```py
# SummaryIndex.as_retriever(retriever_mode = 'default')
# ```

# #### SummaryIndexEmbeddingRetriever
# ```py
# SummaryIndex.as_retriever(retriever_mode='embedding')
# ```

# #### SummaryIndexEmbeddingRetriever
# ```py
# SummaryIndex.as_retriever(retriever_mode='embedding')
# ```

# #### SummaryIndexLLMRetriever
# ```py
# SummaryIndex.as_retriever(retriever_mode='llm')
# ```

# ### The DocumentSummaryIndex retrievers

# #### DocumentSummaryIndexLLMRetriever
# ```py
# DocumentSummaryIndex.as_retriever(retriever_mode='llm')
# ```

# #### DocumentSummaryIndexEmbeddingRetriever

# ```py
# DocumentSummaryIndex.as_retriever(retriever_mode='embedding')
# ```

# ### The TreeIndex retrievers

# #### TreeSelectLeafRetriever

# ```py
# TreeIndex.as_retriever(retriever_mode='select_leaf')
# ```

# #### TreeSelectLeafEmbeddingRetriever

# ```py
# TreeIndex.as_retriever(retriever_mode='select_leaf_embedding')
# ```

# #### TreeAllLeafRetriever

# ```py
# TreeIndex.as_retriever(retriever_mode='all_leaf')
# ```

# #### TreeRootRetriever

# ```py
# TreeIndex.as_retriever(retriever_mode='root')
#

# ### The KeywordTableIndex retrievers

# #### KeywordTableGPTRetriever

# ```py
# KeywordTableIndex.as_retriever(retriever_mode='default')
# ```

# #### KeywordTableSimpleRetriever

# ```py
# KeywordTableIndex.as_retriever(retriever_mode='simple')
# ```

# #### KeywordTableRAKERetriever

# ```py
# KeywordTableIndex.as_retriever(retriever_mode='rake')
# ```
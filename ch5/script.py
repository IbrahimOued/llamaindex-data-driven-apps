# %% [markdown]
# ### A simple usage example for the VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from transformers import AutoTokenizer
from typing import Optional
import torch
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

HF_TOKEN: Optional[str] = os.environ("HF_TOKEN")

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

documents = SimpleDirectoryReader("./files").load_data()
index = VectorStoreIndex.from_documents(documents)
print("Index created successfully!")
# %% [markdown]
# ### A brief introduction to embeddings
# %%
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embedding_model = HuggingFaceEmbedding(
    model_name="WhereIsAI/UAE-Large-V1"
)
embeddings = embedding_model.get_text_embedding(
    "The quick brown fox jumps over the lazy cat!"
)
print(embeddings[:15])
# %% [markdown]
# ### Understanding the StorageContext

# The StorageContext serves as the unifying custodian over configurable storage components used
# during indexing and querying. Its key components are as follows:

# * The Document store (docstore): This manages the storage of documents. The data is locally
# stored in a file named docstore.json.
#
# * The Index Store (index_store): This manages the storage of Index structures. Indexes are
# stored locally in a file called index_store.json.
#
# * Vector Stores (vector_stores): This is a dictionary managing multiple vector stores, each
# potentially serving a different purpose. The vector stores are stored locally in vector_store.
# json.
#
# * The Graph Store (graph_store): This manages the storage of graph data structures. A file
# named graph_store.json is automatically created by LlamaIndex for storing the graphs.
# %%
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Create a Chroma client and collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("example_collection")

# Set up the ChromaVectorStore and StorageContext
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
from llama_index import (VectorDirectoryReader, StorageContext)

db = chromadb.PersistentClient(path="./chroma_database")
chroma_collection=db.get_or_create_collection("my_chroma_store")

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context
)

results = chroma_collection.get()
print(results)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)
# %%

# %%
import torch
import os
from typing import List, Optional
from llama_index.core import set_global_tokenizer, VectorStoreIndex, SimpleDirectoryReader, Settings
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage
# %%
HF_TOKEN: Optional[str] = os.environ["HF_TOKEN"]

model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
# %%
# Setup Tokenizer and Stopping ids

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
# %%
# Setup HuggingFaceLLM

# Optional quantization to 4bit
# import torch
# from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

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

# %%
# Call complete with a prompt
response = llm.complete("Who is Imam Ahmad?")
# %%
response.text 

# %%
# Call with a list of messages

messages = [
    ChatMessage(role="system", content="You are CEO of MetaAI"),
    ChatMessage(role="user", content="Introduce Llama3 to the world."),
]
# %%
response = llm.chat(messages)
# %%
## Building a RAG with Llama3

# load data
documents = SimpleDirectoryReader('./files').load_data()
# setup embedding model
emb_model_name = "BAAI/bge-small-en-v1.5"
embed_model = HuggingFaceEmbedding(model_name=emb_model_name)

# Set Default LLM and Embedding Model
# bge embedding model
Settings.embed_model = embed_model

# Llama-3-8B-Instruct model
Settings.llm = llm

# Create index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("summarize each document in a few sentences")
# %%
response.response
# %%

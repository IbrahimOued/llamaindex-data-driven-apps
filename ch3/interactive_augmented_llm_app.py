from llama_index.core import SummaryIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from transformers import AutoTokenizer
from typing import Optional
import torch
import os

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

loader = WikipediaReader()
documents = loader.load_data(pages=["Messi Lionel"])
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)
index = SummaryIndex(nodes)
query_engine = index.as_query_engine()

print("Ask me anything about Lionel Messi!")
while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break
    response = query_engine.query(question)
    print(response.response)
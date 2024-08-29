# %%
# Ingesting from web page
from llama_index.readers.web import SimpleWebPageReader
urls = ["https://docs.llamaindex.ai"]
documents = SimpleWebPageReader().load_data(urls)
for doc in documents:
    print(doc.text)
# %%
# Ingesting from datbase
from llama_index.readers.database import DatabaseReader
# create a small sqlite database
import sqlite3
conn = sqlite3.connect("./example.db")
c = conn.cursor()
c.execute(
    """
    CREATE TABLE products (
        id integer PRIMARY KEY,
        name text NOT NULL,
        price real
    )
    """
)
c.execute(
    """
    INSERT INTO products (name, price) VALUES
    ('A', 10),
    ('B', 20),
    ('C', 30)
    """
)
conn.commit()
conn.close()
# %%
reader = DatabaseReader(
    uri="sqlite:///example.db"
)
query = "SELECT * FROM products"
documents = reader.load_data(query=query)
for doc in documents:
    print(doc.text)
# %% [markdown]
# Bulk-ingesting data from sources with multiple file formats
# %%
# Use SimpleDirectoryReader to ingest multiple data formats
# from llama_index.core import SimpleDirectoryReader
# reader = SimpleDirectoryReader(
#     input_dir="./files",
#     recursive=True
# )
# documents = reader.load_data()
# for doc in documents:
#     print(doc.metadata)

# # You can also pass in a list of specific files to load, like this:
# files = ["file1.pdf", "file2.docx", "file3.txt"]
# reader = SimpleDirectoryReader(files)
# list_documents = reader.load_data()
# %%
# Parsing like a pro with LlamaParse but need an API key

# Parsing the documents into nodes
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, CodeSplitter
from llama_index.readers.file import FlatReader
from pathlib import Path
# reader = FlatReader()

# documents = reader.load_data("./files/insights.txt")

# for node in nodes:
#     print(f"Metadata {node.metadata} \nText: {node.text}")
# %% [markdown]
# Alright. Let’s see what’s in store in the text splitter category.

# **SentenceSplitter**
#
# This one splits text while maintaining sentence boundaries, providing nodes containing groups
# of sentences. You saw an example of using this parser in Chapter 3, Kickstarting Your Journey with
# LlamaIndex, in the Automatically extracting nodes from documents using splitters section.
# %%
reader = FlatReader()

sentences_documents = reader.load_data(Path("./files/sentences.txt"))
splitter = SentenceSplitter(
    chunk_size = 70,
    chunk_overlap = 2,
    separator = " ",
)
nodes = splitter.get_nodes_from_documents(sentences_documents)
for node in nodes:
    print(f"Metadata: {node.metadata} \nText: {node.text}")
# %% [markdown]
# **TokenSplitter**
#
# This splitter breaks down text while respecting sentence boundaries to create suitable nodes for further
# natural language processing. It operates at the token level.
# A typical usage in code would look like this:
# %%
sentences_documents = reader.load_data(Path("./files/tokens.txt"))
splitter = TokenTextSplitter(
    chunk_size = 70,
    chunk_overlap = 2,
    separator = " ",
    backup_separators = [".", "!", "?"]
)
nodes = splitter.get_nodes_from_documents(sentences_documents)
for node in nodes:
    print(f"Metadata: {node.metadata} \nText: {node.text}")
# %% [markdown]
# **CodeSplitter**
#
# This smart splitter knows how to interpret source code. It splits text based on programming language
# and is ideal for managing technical documentation or source code. Before running the example, make
# sure you install the necessary libraries:
# %%
code_documents = reader.load_data(Path("./files/code.txt"))
code_splitter = CodeSplitter.from_defaults(
    language = 'python',
    chunk_lines=5,
    chunk_lines_overlap=2,
    max_chars=150
)
nodes = code_splitter.get_nodes_from_documents(code_documents[0])
for node in nodes:
    print(f"Metadata: {node.metadata} \nText: {node.text}")
# %%
# Estimate your maximal costs before running the actual extractors

from llama_index.core import Settings
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.llms.mock import MockLLM
from llama_index.core.schema import TextNode
from llama_index.core.callbacks import (CallbackManager, TokenCountingHandler)

llm = MockLLM(max_tokens=256)
counter = TokenCountingHandler(verbose=False)
callback_manager = CallbackManager([counter])
Settings.llm = llm
Settings.callback_manager = CallbackManager([counter])
sample_text = (
    "LlamaIndex is a powerful tool for extracting insights from documents. "
    "It is easy to use and provides a wide range of functionalities."
)

nodes = [TextNode(text=sample_text)]
extractor = QuestionsAnsweredExtractor(
    show_progress=True,
)
Questions_metadata = extractor.extract(nodes)
print(f"Prompt Tokens: {counter.prompt_llm_token_count}")
print(f"Completion Tokens: {counter.completion_llm_token_count}")
print(f"Total Token Count: {counter.total_llm_token_count}")
# %% [markdown]
# ### Scrubbing personal data and other sensitive information
# %%
from llama_index.core.postprocessor import NERPIINodePostprocessor
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import NodeWithScore, TextNode
original = (
    "Dear Jane Doe. Your address has been recorded in "
    "our database. Please confirm it is valid: 8804 Vista "
    "Serro Dr. Cabo Robles, California(CA)."
)
node = TextNode(text=original)
processor = NERPIINodePostprocessor()
clean_nodes = processor.postprocess_nodes(
    [NodeWithScore(node=node)]
)
print(clean_nodes[0].node.get_text())
# %%

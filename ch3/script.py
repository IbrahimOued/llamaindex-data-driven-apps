# %% [markdown]
# # Chapter 3: Kickstarting your journey with Llamaindex
# %% [markdown]
# ## Documents
# %%
from llama_index.core import Document
text = "The quick brown fox jumps over the lazy dog."
doc = Document(
    text=text,
    metadata={'author': 'John Doe','category': 'others'},
    id_='1'
)
print(doc)
# %% [markdown]
# Here are some of the most important attributes of a Document object:
# 
# * `text`: This attribute stores the text content of the document
# 
# * `metadata`: This attribute is a dictionary that can be used to include additional information
# about the document, such as the file name or categories. The keys in the metadata dictionary
# must be strings and the values can be strings, floats, or integers
# 
# * `id_`: This is a unique ID for each Document. You can set this manually if you want, but if you
# don’t specify an ID, LlamaIndex will automatically generate one for you

# %% [markdown]

# Here is a basic example of automated data ingestion using one of the predefined LlamaHub data
# loaders. Before you can run the example, make sure you install the libraries mentioned in the technical
# requirements section and complete all the necessary environment preparations mentioned in Chapter 2
# if you haven’t already:
# %%
from llama_index.readers.wikipedia import WikipediaReader
loader = WikipediaReader()
documents = loader.load_data(pages=['Pythagorean theorem', 'Artificial_intelligence'])
print(f"loaded {len(documents)} documents")
# %% [markdown]
# ## Nodes

# While Documents represent the raw data and can be used as such, Nodes are smaller chunks of content
# extracted from the Documents. The goal is to break down Documents into smaller, more manageable
# pieces of text. This serves a few purposes:

# * **Allows our proprietary knowledge to fit within the model’s prompt limits**: Imagine that if we
# had an internal procedure that is 50 pages long, we would definitely run into size limit problems
# when trying to feed that in the context of our prompt. However, most likely, in practice, we
# wouldn’t need to feed the entire procedure in one prompt. Therefore, selecting just the relevant
# Nodes can solve this problem.
#
# * **Creates semantic units of data centered around specific information**: This can make it easier
# to work with and analyze the data, as it is organized into smaller, more focused units.
# 
# * **Allows the creation of relationships between Nodes**: This means that Nodes can be linked
# together based on their relationships, creating a network of interconnected data. This can
# be useful for understanding the connections and dependencies between different pieces of
# information within the Documents.

# Here’s a list of some important attributes of the TextNode class:
# * `text`: The chunk of text derived from an original Document.
#
# * `start_char_idx` and `end_char_idx` are optional integer values that can store the starting
# and ending character positions of the text within the Document. This could be helpful when
# the text is part of a larger Document, and you need to pinpoint the exact location.
#
# * `text_template` and `metadata_template` are template fields that define how the text
# and metadata are formatted. They help produce a more structured and readable representation
# of TextNode.
# * `metadata_seperator`: This is a string field that defines the separator between metadata
# fields. When multiple metadata items are included, this separator is used to maintain readability
# and structure.
# * Any useful `metadata` such as the parent Document ID, relationships to other Nodes, and
# optional tags. This metadata can be used for storing additional context when necessary. We’ll
# talk about it in more detail in Chapter 4, Ingesting Data into Our RAG Workflow

# manually creating the Node objects
# %%
from llama_index.core import Document
from llama_index.core.schema import TextNode
doc = Document(text="This is a sample document text")
n1 = TextNode(text=doc.text[0:16], doc_id=doc.id_)
n2 = TextNode(text=doc.text[17:30], doc_id=doc.id_)
print(n1)
print(n2)
# %% [markdown]
# ### Automatically extracting Nodes from Documents using splitters
# Because Document chunking is very important in an RAG workflow, LlamaIndex comes with built-in
# tools for this purpose. One such tool is `TokenTextSplitter`
# %%
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
doc = Document(text=("This is sentence 1. This is sentence 2. "
                     "Sentence 3 here."),
               metadata={'author': 'John Doe', 'category': 'others'})

splitter = TokenTextSplitter(
    chunk_size=12,
    chunk_overlap=0,
    separator=" "
)

nodes = splitter.get_nodes_from_documents([doc])

for node in nodes:
    print(node.text)
    print(node.metadata)

# %% [markdown]
# ### Nodes don’t like to be alone – they crave relationships

# Now that we’ve covered some basic examples of how to create simple Nodes, how about adding some
# relationships between them?
# Here’s an example that manually creates a simple relationship between two Nodes:
# %%
from llama_index.core import Document
from llama_index.core.schema import (
    TextNode,
    NodeRelationship,
    RelatedNodeInfo
)
doc = Document(text="First sentence. Second Sentence")
n1 = TextNode(text="First sentence", node_id=doc.doc_id)
n2 = TextNode(text="Second sentence", node_id=doc.doc_id)
n1.relationships[NodeRelationship.NEXT] = n2.node_id
n2.relationships[NodeRelationship.PREVIOUS] = n1.node_id
print(n1.relationships)
print(n2.relationships)
# %% [markdown]

# There are other types of relationships that we could define. In addition to simple relationships such
# as previous or next, Nodes can be connected using the following:
# 
# * `SOURCE`: The source relationship represents the original source Document that a node was
# extracted or parsed from. When you parse a Document into multiple Nodes, you can track
# which Document each node originated from using the source relationship.
# 
# * `PARENT`: The parent relationship indicates a hierarchical structure where the node with this
# relationship is one level higher than the associated node. In a tree structure, a parent node
# would have one or more children. This relationship is used to navigate or manage nested data
# structures where you might have a main node and subordinate Nodes representing sections,
# paragraphs, or other subdivisions of the main node.
# 
# * `CHILD`: This is the opposite of PARENT. A node with the child relationship is a subordinate of
# another node – the parent. Child Nodes can be seen as the leaves or branches in a tree structure
# stemming from their parent node.

# %% [markdown]

# ## Indexes

# Our third important concept – the index – refers to a specific data structure used to organize a
# collection of Nodes for optimized storage and retrieval.

# LlamaIndex supports different types of indexes, each with its strengths and trade-offs. Here is a list
# of some of the available index types:
# 
# * `SummaryIndex`: This is very similar to a box for recipes – it keeps your Nodes in order, so
# you can access them one by one. It takes in a set of documents, chunks them up into Nodes,
# and then concatenates them into a list. It’s great for reading through a big Document.
# * `DocumentSummaryIndex`: This constructs a concise summary for each document, mapping
# these summaries back to their respective nodes. It facilitates efficient information retrieval by
# using these summaries to quickly identify relevant documents.

# * `VectorStoreIndex`: This is one of the more sophisticated types of indexes and probably
# the workhorse in most RAG applications. It converts text into vector embeddings and uses
# math to group similar Nodes, helping locate Nodes that are alike.

# * `TreeIndex`: The perfect solution for those who love order. This index behaves similarly to
# putting smaller boxes inside bigger ones, organizing Nodes by levels in a tree-like structure.
# Inside, each parent node stores summaries of the children nodes. These are generated by
# the LLM, using a general summarization prompt. This particular index can be very useful
# for summarization.

# * `KeywordTableIndex`: Imagine you need to find a dish by the ingredients you have. The
# keyword index connects important words to the Nodes they’re in. It makes finding any node
# easy by looking up keywords.
# * `KnowledgeGraphIndex`: This is useful when you need to link facts in a big network of
# data stored as a knowledge graph. This one is good for answering tricky questions about lots
# of connected information.
#
# * `ComposableGraph`: This allows you to create complex index structures in which Document-
# level indexes are indexed in higher-level collections. That’s right: you can even build an index
# of indexes if you want to access the data from multiple Documents in a larger collection
# of Documents.

# All the index types in LlamaIndex share some common core features:
#
# * Building the index: Each index type can be constructed by passing in a set of Nodes during
# initialization. This builds the underlying index structure.
#
# * Inserting new Nodes: After an index is built, new Nodes can be manually inserted. This adds
# to the existing index structure.
# 
# * Querying the index: Once built, indexes provide a query interface to retrieve relevant Nodes
# based on a specific query. The retrieval logic varies by index type.
# %%
# For now, let’s consider a simple example to illustrate the creation of SummaryIndex:
from llama_index.core import SummaryIndex, Document
from llama_index.core.schema import TextNode
nodes = [
    TextNode(text="Lionel Messi is a football player from Argentina."),
    TextNode(text="He has won the Ballon d'Or trophy 7 times."),
    TextNode(text="Lionel Messi's hometown is Rosario."),
    TextNode(text="He was born on June 24, 1987.")
]
index = SummaryIndex(nodes)

# %%

# et’s use the Lionel Messi index we just created as an example. Say you ask, “What is Messi’s hometown?”
# See the following:
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from typing import List, Optional
import torch
import os

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

query_engine = index.as_query_engine()
response = query_engine.query("What is Messi's hometown?")
print(response)
# %%

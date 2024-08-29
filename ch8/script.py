# %% [markdown]
# # Building Chatbots and Agents with LlamaIndex

# ## Understanding chatbots and agents

# ### Discovering ChatEngine

# In its simplest form, a chat engine can be initialized just as easily, based on an index:
# ```python
# chat_engine = index.as_chat_engine()
# response = chat_engine.chat("Hi, how are you?")
# ```

# Once initialized, a chat engine can be queried using various methods:
# * `chat()`: This method initiates a synchronous chat session, processing the user’s message and
# returning the response immediately.
# * `achat()`: This method is similar to chat() but executes the query asynchronously, allowing
# multiple requests to be processed simultaneously. This can be useful, for example, in a web or
# mobile application where we want to avoid blocking the main thread during server queries.
# `stream_chat()`: This method opens a streaming chat session, where responses can be
# returned as they are generated, for more dynamic interaction. This is particularly useful for
# long or complex responses that require significant processing time, allowing the user to start
# seeing parts of the response before all processing is complete.
# * `astream_chat()`: This method is an asynchronous version of stream_chat() that
# allows us to handle streaming interactions in an asynchronous context.

# Another option is to initiate a Read-Eval-Print (REPL) loop with ChatEngine:
# ```python
# chat_engine.chat_repl()
# ```

# To reset a chat conversation, you can use the following command:
# ```python
# chat_engine.reset()
# ```

# ## Understanding the different chat modes

# When initializing a chat engine, we can use the chat_mode argument to invoke various chat engine
# types predefined in LlamaIndex. I will show you how each of these engines works.

# ### Understanding how chat memory works

# #### Understanding how chat memory works

# The ChatMemoryBuffer class is a specialized memory buffer that’s designed to store chat history
# efficiently while also managing the token limit imposed by different LLMs.

# here are two different storage options for the chat store:
# * The default `SimpleChatStore`, which stores the conversation in memory
# * The more advanced `RedisChatStore`, which stores the chat history in a Redis database,
# eliminating the need to manually persist and load the chat history
# %%

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from transformers import AutoTokenizer
from typing import Optional
import os
import torch

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
# %%
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# we can try to load the previous conversation. If there is no
# previous conversation save file, we simply initialize an empty chat_store:
try:
    chat_store = SimpleChatStore.from_persist_path(
        persist_path="./memory_json"
    )
except FileNotFoundError:
    chat_store = SimpleChatStore()

# It’s now time to initialize our memory buffer by using chat_store as an argument. Although not
# needed here, for a more detailed illustration, we will also customize token_limit and chat_
# store_key:
memory = ChatMemoryBuffer.from_defaults(
    token_limit=2000,
    chat_store=chat_store,
    chat_store_key="user_X"
)
# OK; we have all the necessary pieces. Let’s put them together into a SimpleChatEngine class and
# create a chat loop:
chat_engine = SimpleChatEngine.from_defaults(
    memory=memory
)
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = chat_engine.chat(user_input)
    print(f"Bot: {response}")
# Once the user types exit and we break the loop, we use the persist() method to store the current
# conversation for future sessions:
chat_store.persist(persist_path="./chat_memory.json")
# %% [markdown]
# #### Simple mode
# ```python
# from llama_index.core.chat_engine import SimpleChatEngine
# chat_engine = SimpleChatEngine.from_defaults()
# chat_engine.chat_repl()
# ```
# If we want, we can customize the LLM using the llm argument:
# ```
# ```py
# chat_engine = SimpleChatEngine.from_defaults(llm=llm)
# ```

# #### Context mode
# ContextChatEngine is designed to enhance chat interactions by leveraging our proprietary
# knowledge. It works by retrieving relevant text from an index based on the user’s input, integrating
# this retrieved information into the system prompt to provide context, and then generating a response
# with the help of the LLM.

# ```python
# from llama_index.core.chat_engine import SimpleChatEngine
# chat_engine = SimpleChatEngine.from_defaults()
# chat_engine.chat_repl()
# ```
# If we want, we can customize the LLM using the llm argument:
# ```
# ```py
# chat_engine = SimpleChatEngine.from_defaults(llm=llm)
# ```
# There are several parameters that we can customize for this chat engine:
# * `retriever`: The actual retriever that’s used to retrieve relevant text from the index based on
# the user’s message. When the chat engine is initialized directly from the index, it will use the
# default retriever for that particular index type
# * `llm`: An instance of an LLM, which will be used for generating responses
# * `memory`: A ChatMemoryBuffer object, which is used to store and manage the chat history
# * `chat_history`: This is an optional list of ChatMessage instances representing the history
# of the conversation. It can be used to maintain continuity in a conversation. This history
# includes all messages that have been exchanged in the chat session, including both user and
# chatbot messages. For instance, it can be used to continue a conversation from a certain point.
# A ChatMessage object contains three attributes:
#  role: This defaults to user
#  content: The actual message
#  Any optional arguments provided via additional_kwargs
# * `prefix_messages`: A list of ChatMessage instances that may be used as predefined
# messages or prompts before the actual user message. This can be useful for setting a particular
# tone or context for the chat
# * `node_postprocessors`: An optional list of BaseNodePostprocessor instances
# for further processing the nodes retrieved by the retriever. This can be used to implement
# guardrails, scrub sensitive information from the context, or make any other adjustments to
# the retrieved nodes if required
# * `context_template`: A string template that can be used to format the prompt that feeds
# the context to the LLM
# * `callback_manager`: An optional CallbackManager instance for managing callbacks
# during the chat process. This is useful for tracing and debugging purposes
# * `system_prompt`: An optional string that’s used as a system prompt, providing initial context
# or instructions for the chatbot
# * `service_context`: An optional ServiceContext instance, which can be used to make
# additional customizations to the chat engine

# To implement ContextChatEngine, we must load our data and build an index, then optionally
# configure the chat engine with different parameters as needed.
# Here’s a quick example based on our sample data files, which can be found in the ch8/files
# subfolder in this book’s GitHub repository:
# %%
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
docs = SimpleDirectoryReader(input_dir="./files").load_data()
index = VectorStoreIndex.from_documents(docs)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    system_prompt=(
        "You're a chatbot, able to talk about "
        "general topics, as well as answering specific "
        "questions about ancient Rome."
    ),
)
chat_engine.chat_repl()

# Alternatively, we could have defined it standalone, providing a retriever as an argument, like this:
# retriever = index.as_retriever(retriever_mode='default')
# chat_engine = ContextChatEngine.from_defaults(
#     retriever=retriever
# )
# %% [markdown]
# #### Condense question mode
# CondenseQuestionChatEngine streamlines the user interaction by first condensing the
# conversation and the latest user message into a standalone question with the help of the LLM. This
# standalone question, which tries to capture the essential elements of th

# The main benefit of using this approach is that it maintains the conversation focused on the topic,
# preserving the essential points of the entire dialogue throughout every interaction. And it always
# responds in the context of our proprietary data.

# The fact that the final response comes from our retrieved proprietary data and not directly from the
# LLM can also be a disadvantage sometimes. This chat mode may struggle with more general questions,
# such as inquiries about previous interactions, due to its reliance on querying the knowledge base for
# every response.

# The key parameters of CondenseQuestionChatEngine:
# * `query_engine`: This is a `BaseQueryEngine` instance that’s used to query the condensed
# question. Any type of query engine may be used here, including complex constructs with
# routing functionality
# * `condense_question_prompt`: This is a `BasePromptTemplate` instance that’s used
# for condensing the conversation and user message into a single, standalone question
# * `Memory`: A ChatMemoryBuffer instance that’s used to manage and store the chat history
# * `llm`: A language model instance for generating the condensed question
# * `verbose`: A Boolean flag for printing verbose logs during operation
# * `callback_manager`: An optional `CallbackManager` instance for managing callbacks
# %%
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage
documents = SimpleDirectoryReader("./files").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine=index.as_query_engine()
chat_history = [
    ChatMessage(
        role="user",
        content="Arch of Constantine is a famous"
        "building in Rome"
    ),
    ChatMessage(
        role="user",
        content="The Pantheon should not be "
        "regarded as a famous building"
    ),
]

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    chat_history=chat_history
)
response = chat_engine.chat(
    "What are two of the most famous structures in ancient Rome?"
)
print(response)
# %% [markdown]
# #### Condense and context mode

# CondensePlusContextChatEngine offers an even more comprehensive chat interaction by
# combining the benefits of condensed questions and context retrieval.
# While the previous chat engine we discussed is more straightforward and focuses on simplifying the
# conversation into a question for response generation, CondensePlusContextChatEngine takes
# an extra step to enrich the conversation with additional context from the indexed data, leading to more
# detailed and context-aware responses. The trade-off here is an increase in response generation time due
# to the additional step performed

# the key parameters of CondensePlusContextChatEngine:
# * `retriever`: Used to fetch context based on the condensed question
# * `llm`: The LLM that’s used to generate the condensed question and the final response
# * `memory`: A ChatMemoryBuffer instance for storing and managing chat history
# * `context_prompt`: A prompt template for formatting the context in the system prompt
# * `condense_prompt`: A prompt for condensing the conversation into a standalone question
# * `system_prompt`: A prompt with instructions for the chatbot
# * `skip_condense`: A Boolean flag to bypass the condensation step if desired
# * `node_postprocessors`: An optional list of BaseNodePostprocessors for additional
# processing of retrieved nodes
# * `callback_manager`: As usual, this can be used for managing callbacks
# * `verbose`: A Boolean flag for enabling verbose logging during operation

# ```python
# index.as_chat_engine(chat_mode="condense_plus_context")
# ```
# ## Implementing agentic strategies in our apps

# You probably realize, however, that **the chat engine models we have discussed so far can only answer
# questions and cannot execute functions or interact in ways other than read-only with backend data**.
# For these use cases, we need agents.

# The major difference between an agent and a simple chat engine is that an agent operates based on a
# reasoning loop and has several tools at its disposal. After all, who would be Bond without the gadgets
# that Q always provides?
# Unlike a simple chatbot, which can – at best – answer questions, either directly with the help of an
# LLM or by extracting proprietary data from a knowledge base, agents are much more powerful and
# can handle far more complex scenarios. This gives them a lot more utility in a business context, where
# human interactions augmented by AI are becoming increasingly prevalent.
# Let’s understand the core components of an agent: the tools and the reasoning loop.

# ### Building tools and ToolSpec classes for our agents

# A tool can also be a wrapper for any kind of user-defined function, capable of reading or writing data,
# calling functions from external APIs, or executing any kind of code. This means that tools come in
# two different flavors:
# * `QueryEngineTool`: This can encapsulate any existing query engine. This is the kind we
# covered during Chapter 6 and it can only provide read-only access to our data
# * `FunctionTool`: This enables any user-defined function to be transformed into a tool. This
# is a universal type of tool as it allows any type of operation to be executed

# Because we have already seen examples of how QueryEngineTool works, let’s focus on
# FunctionTool instead.
# %%
from llama_index.core.tools import FunctionTool
def calculate_average(*values):
    """
    calculate the average of the provided values
    """
    return sum(values) / len(values)

average_tool = FunctionTool.from_defaults(fn=calculate_average)
# %% [markdown]
# To enable agents to assimilate our functions as tools, **they must contain descriptive docstrings**, just like
# in the previous example. **LlamaIndex relies on these docstrings to provide agents with an understanding
# of the purpose and proper usage of a particular tool wrapping a user-defined function**.

# This description will be used by the reasoning loop of an agent to determine which particular tool is
# fit for solving a specific task, allowing the agent to decide the execution path.
# However, competent agents are usually able to handle more than just one tool.
# For this purpose, LlamaIndex also provides the `ToolSpec` class

# Let’s take the DatabaseToolSpec class available on LlamaHub as an example.
# ```py
# from llama_index.tools.database import DatabaseToolSpec
# db_tools = DatabaseToolSpec(<db_specific_configuration>)
# tool_list = db_tools.to_tool_list()
# ```

# ### Understanding reasoning loops

# Having so many specialized tools already available for our agents is a great advantage. But unfortunately,
# a box full of some of the best-quality instruments is not always enough. Our agents also need to know
# when to use each of these tools.
# Specifically, the RAG applications we build need to decide – as autonomously as possible – which tool
# to use, depending on the specific user query and the dataset they are operating on. Any hard-coded
# solution will only deliver good results in a limited number of scenarios. This is where reasoning loops
# come in.
# The reasoning loop is a fundamental aspect of agents, enabling them to intelligently decide which tools
# to use in different scenarios. This aspect is important because, in complex, real-world applications,
# the requirements can vary significantly and a static approach would limit the agent’s effectiveness

# ### OpenAIAgent
# %% [markdown]
# ```py
# from llama_index.tools.database import DatabaseToolSpec
# from llama_index.core.tools import FunctionTool
# from llama_index.agent.openai import OpenAIAgent
# from llama_index.llms.openai import OpenAI
#
# def write_text_to_file(text, filename):
#     """
#     Writes the text to a file with the specified filename.
#     Args:
#         text (str): The text to be written to the file.
#         filename (str): File name to write the text into.
#     Returns: None
#     """
#     with open(filename, 'w') as file:
#         file.write(text)
#
# save_tool = FunctionTool.from_defaults(fn=write_text_to_file)
# db_tools = DatabaseToolSpec(uri="sqlite:///files//database//employees.db")
# tools = [save_tool]+db_tools.to_tool_list()
#
# llm = OpenAI(model="gpt-4")
# agent = OpenAIAgent.from_tools(
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_function_calls=20
# )
#
# response = agent.chat(
#     "For each IT department employee with a salary lower "
#     "than the average organization salary, write an email,"
#     "announcing a 10% raise and then save all emails into "
#     "a file called 'emails.txt'")
# print(response)
# ```
# %% [markdown]
# ### ReActAgent

# In contrast to OpenAIAgent, ReActAgent uses more generic text completion endpoints that can be
# driven by any LLM. It operates based on a ReAct loop within a chat mode built on top of a set of tools.
#
# This loop involves deciding whether to use any of the available tools, potentially using it and observing
# its output, and then deciding whether to repeat the process or provide a final response. This flexibility
# allows it to choose between using tools or relying solely on the LLM
#
# Unlike OpenAIAgent, which uses a function calling API with a model capable of selecting and chaining
# together multiple tools, the ReActAgent class’s logic must be fully encoded through its prompts.
# ReActAgent uses a predefined loop with a maximum number of iterations, along with strategic
# prompting, to mimic a reasoning loop. Nevertheless, with strategic prompt engineering, ReActAgent
# can achieve effective tool orchestration and chained execution, similar to the output of the OpenAI
# Function API.
# The key difference is that whereas the logic of the OpenAI Function API is embedded in the model,
# ReActAgent relies on the structure of its prompts to induce the desired tool selection behavior. This
# approach offers considerable flexibility as it can adapt to different language model backends, allowing
# for different implementations and applications.
# %%
from llama_index.core.agent.react import ReActAgent
from llama_index.tools.database import DatabaseToolSpec
from llama_index.core.tools import FunctionTool

def write_text_to_file(text, filename):
    """
    Writes the text to a file with the specified filename.
    Args:
        text (str): The text to be written to the file.
        filename (str): File name to write the text into.
    Returns: None
    """
    with open(filename, 'w') as file:
        file.write(text)

save_tool = FunctionTool.from_defaults(fn=write_text_to_file)
db_tools = DatabaseToolSpec(uri="sqlite:///files//database//employees.db")
tools = [save_tool]+db_tools.to_tool_list()


agent = ReActAgent.from_tools(tools, max_iterations=100)
response = agent.chat(
    "For each IT department employee with a salary lower "
    "than the average organization salary, write an email,"
    "announcing a 10% raise and then save all emails into "
    "a file called 'emails.txt'")
print(response)
# %% [markdown]
# #### How do we interact with agents?
# There are two main methods that we can use to interact with an agent: `chat()` and `query()`. The
# **first method utilizes stored conversation history to provide context-informed responses**, making it
# suitable for ongoing dialogues.

# On the other hand, the former method operates in a stateless mode, treating each call independently
# without reference to past interactions. This is better suited for standalone requests.

# #### Enhancing our agents with the help of utility tools

# To improve the capabilities of the existing tools, LlamaIndex also provides two very useful so-called
# utility tools – `OnDemandLoaderTool` and `LoadAndSearchToolSpec`. They are universal and
# can be used with any type of agent to augment the standard tool functionality in certain scenarios.

# One common issue when interacting with an API is that we might receive a very long response in
# return. Our agents may not always be able to handle such large outputs.
# Problems may arise because they may overflow the context window of the LLM or sometimes, key
# context may be diluted by a large amount of data, decreasing the accuracy of the agent’s reasoning logic.
# A good way to understand this issue is by looking at our previous example for OpenAIAgent.
# In that case, we used a collection of tools called DatabaseToolSpec to retrieve data from our
# sample Employees table. If you’ve run that particular agent with the Verbose parameter set to True,
# then you’ve probably noticed that the outputs produced by the load_data tool are in the form of
# LlamaIndex document objects,

# This means that whenever the agent calls the load_data tool, using a SQL query to interrogate
# the database, instead of simply receiving the output of the query, it gets a whole document in return
# – along with a bunch of additional data, such as the ID of the document, metadata fields, hashes,
# and so on. The agent has to extract the actual query results from that data using the LLM, hence the
# aforementioned potential issues.
# 
# So, what if we want to extract only the result of the query, without all the additional data on top of it?
# That is the job of LoadAndSearchToolSpec.
# %% [markdown]
# #### Understanding `OnDemandLoaderTool`
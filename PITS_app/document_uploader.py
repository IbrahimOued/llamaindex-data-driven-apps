from global_settings import STORAGE_PATH, CACHE_FILE
from logging_functions import log_action
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.ingestion import IngestionPipeline, IngestionCache
from llama_index.text_splitter import TokenTextSplitter
from llama_index.extractors import SummaryExtractor
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def ingest_documents():
    documents = SimpleDirectoryReader(
        STORAGE_PATH,
        filename_as_id=True,
    ).load_data()
    for doc in documents:
        print(doc.id_)
        log_action(
            f"File '{doc.id_}' ingested",
            action_type="UPLOAD",
        )

    try:
        cache_hashed = IngestionCache.from_persist_path(CACHE_FILE)
        print("Cache file found, Running using cache...")
    except:
        cache_hashed = ""
        print("No cache file found, Running without cache...")

    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(
                chunk_size=1024,
                chunk_overlap=20,
            ),
            SummaryExtractor(summaries=['self']),
            HuggingFaceEmbedding()
        ],
        cacje=cache_hashed,
    )
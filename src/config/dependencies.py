from rag.embeddings.embedding_factory import EmbeddingFactory
from models.model_factory import ModelFactory
from rag.vector_stores.store_factory import StoreFactory
import os

# Dependency to initialize the FAISS or Pinecone store
def get_store():
    store =  StoreFactory(os.getenv("STORE_TYPE", "faiss"), EmbeddingFactory(os.getenv("EMBEDDING_MODEL")).create_embedding()).get_store()
    store.load_vector_store()
    return store

# Dependency to initialize the model
def get_model():
    return ModelFactory(os.getenv("MODEL_TYPE")).create_model()

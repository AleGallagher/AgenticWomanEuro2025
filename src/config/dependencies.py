from rag.embeddings.embedding_factory import EmbeddingFactory
from models.model_factory import ModelFactory
from rag.vector_stores.store_factory import StoreFactory
from rag.retriever import Retriever
from fastapi import Depends
import os

# Dependency to initialize the embedding model
def get_embedding_model():
    return EmbeddingFactory(os.getenv("EMBEDDING_MODEL")).create_embedding()

# Dependency to initialize the FAISS or Pinecone store
def get_store(embedding_model=Depends(get_embedding_model)):
    store =  StoreFactory(os.getenv("STORE_TYPE", "faiss"), embedding_model).get_store()
    store.load_vector_store()
    return store

# Dependency to initialize the model
def get_model():
    return ModelFactory(os.getenv("MODEL_TYPE")).create_model()

# Dependency to initialize the retriever
def get_retriever(
    embedding_model=Depends(get_embedding_model),
    store=Depends(get_store),
    model=Depends(get_model),
):
    return Retriever(embedding_model, store, model)
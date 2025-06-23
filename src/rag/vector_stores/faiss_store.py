import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from .base_store import BaseStore

class FAISSStore(BaseStore):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatL2(len(embedding_model.embed_query("test"))) 
        self.docstore = InMemoryDocstore({})
        self.vector_store = FAISS(index = self.index, embedding_function = embedding_model, docstore = self.docstore, index_to_docstore_id={})

    def add_documents(self, chunks):
        self.vector_store.add_documents(documents=chunks)

    def search(self, query, top_k=5):
        results = self.vector_store.similarity_search(query, k=top_k)
        return results

    def delete(self, ids):
        raise NotImplementedError("Deletion is not supported in FAISS.")
    
    def save_data_base(self, database_name):
        self.vector_store.save_local(database_name)
    
    def get_vector_store(self):
        return self.vector_store
    
    def load_vector_store(self):
        database_name = r"f:\Python\AgenticEuro2025\src\rag\euro2025"
        self.vector_store = FAISS.load_local(database_name, self.embedding_model, allow_dangerous_deserialization=True)
        self.index = self.vector_store.index
        self.docstore = self.vector_store.docstore


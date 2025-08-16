import os

from langchain_pinecone import PineconeVectorStore

from .base_store import BaseStore

class PineconeStore(BaseStore):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vector_store = PineconeVectorStore.from_existing_index(index_name=os.getenv("PINECONE_INDEX"), embedding=embedding_model)

    def add_documents(self, chunks):
        self.vector_store.add_documents(chunks)

    def search(self, query, top_k=5):
        results = self.vector_store.similarity_search(query, k=top_k)
        return [res.page_content for res in results]

    def delete(self, ids):
        self.index.delete(ids=ids)

    def get_vector_store(self):
        return self.vector_store
    
    def save_data_base(self, database_name):
        pass

    def load_vector_store(self):
        pass
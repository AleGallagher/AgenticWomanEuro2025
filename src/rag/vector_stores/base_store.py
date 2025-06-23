from abc import ABC, abstractmethod

class BaseStore(ABC):
    @abstractmethod
    def add_documents(self, chunks):
        """
        Add embeddings to the store.
        """
        pass

    @abstractmethod
    def search(self, query_embedding, top_k=5):
        """
        Search for the most similar embeddings.
        """
        pass

    @abstractmethod
    def delete(self, ids):
        """
        Delete embeddings by their IDs.
        """
        pass

    @abstractmethod
    def get_vector_store(self):
        """
        Get the vector store object.
        """
        pass

    @abstractmethod
    def save_data_base(self):
        """
        Get the vector store object.
        """
        pass

    @abstractmethod
    def load_vector_store(self):
        """
        Get the vector store object.
        """
        pass
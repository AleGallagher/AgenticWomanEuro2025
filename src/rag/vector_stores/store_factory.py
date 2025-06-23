class StoreFactory:
    def __init__(self,  store_type: str, embedding_model):
        self.store_type = store_type
        self.embedding_model = embedding_model
        self.store_registry = {
            "faiss": self._create_faiss_store,
            "pinecone": self._create_pinecone_store,
        }

    def get_store(self):
        """Create a store based on the type."""
        if self.store_type not in self.store_registry:
            raise ValueError(f"Unknown model type: {self.store_type}")
        return self.store_registry[self.store_type]()
    
    def _create_pinecone_store(self):
        """Create an PineconeStore store."""
        from rag.vector_stores.pinecone_store import PineconeStore
        return PineconeStore(self.embedding_model)

    def _create_faiss_store(self):
        """Create an FAISSStore store."""
        from rag.vector_stores.faiss_store import FAISSStore      
        # Create the Ollama model
        return FAISSStore(self.embedding_model)
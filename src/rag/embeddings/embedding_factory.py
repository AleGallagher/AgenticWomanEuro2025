import json
import os
from pathlib import Path

class EmbeddingFactory:
    """Factory class for creating embedding model"""

    def __init__(self,  embedding_type: str, config_path: str = "../../config/config.json"):
        self.embedding_type = embedding_type
        self.config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.json"
        self.embedding_registry = {
            "openai": self._create_openai_embedding,
            "ollama": self._create_ollama_embedding,
        }

    def create_embedding(self):
        """Create an embedding model based on the type."""
        if self.embedding_type not in self.embedding_registry:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        return self.embedding_registry[self.embedding_type]()

    def _create_openai_embedding(self):
        """Create an OpenAI embedding model."""
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY"))

    def _create_ollama_embedding(self):
        """Create an Ollama embedding model."""
        from langchain_ollama import OllamaEmbeddings

        # Create the Ollama embedding model
        return OllamaEmbeddings(**self._get_config())

    def _get_config(self):
        """Get the config from the config file."""
        with open(self.config_path, "r") as f:
            config = json.load(f)
        embedding_models = config.get("embedding_models", [])
        for embedding_config in embedding_models:
            if embedding_config.get("type") == self.embedding_type:
                return embedding_config["params"]
        raise ValueError(f"No embedding model config found for type: {self.embedding_type}")

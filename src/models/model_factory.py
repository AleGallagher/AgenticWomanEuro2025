import json
from pathlib import Path

class ModelFactory:
    """Factory class for creating model"""

    def __init__(self,  model_type: str, config_path: str = "../config/config.json"):
        self.model_type = model_type
        self.config_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
        self.model_registry = {
            "openai": self._create_openai_model,
            "ollama": self._create_ollama_model,
        }
    
    def create_model(self):
        """Create a model based on the type."""
        if self.model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return self.model_registry[self.model_type]()

    def _create_openai_model(self):
        """Create an OpenAI model."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**self._get_config())

    def _create_ollama_model(self):
        """Create an Ollama model."""
        from langchain_ollama import ChatOllama

        # Create the Ollama model
        return ChatOllama(**self._get_config())
    
    def _get_config(self):
        """Get the config from the config file."""
        with open(self.config_path, "r") as f:
            config = json.load(f)
        models = config.get("models", [])
        for model in models:
            if model.get("type") == self.model_type:
                return model["params"]
        raise ValueError(f"No model config found for type: {self.model_type}")
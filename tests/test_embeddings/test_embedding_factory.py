import unittest
from unittest.mock import patch, mock_open
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from rag.embeddings.embedding_factory import EmbeddingFactory

class TestEmbeddingFactory(unittest.TestCase):

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_create_openai_embedding(self, mock_openai_embeddings):
        """Test creating an OpenAI embedding."""
        mock_openai_embeddings.return_value = "MockOpenAIEmbedding"
        factory = EmbeddingFactory(embedding_type="openai")
        embedding = factory.create_embedding()
        self.assertEqual(embedding, "MockOpenAIEmbedding")
        mock_openai_embeddings.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data='{"embedding_models": [{"type": "ollama", "params": {"model": "test-model"}}]}')
    @patch("langchain_ollama.OllamaEmbeddings")
    def test_create_ollama_embedding(self, mock_ollama_embeddings, mock_file):
        """Test creating an Ollama embedding."""
        mock_ollama_embeddings.return_value = "MockOllamaEmbedding"
        factory = EmbeddingFactory(embedding_type="ollama")
        embedding = factory.create_embedding()
        self.assertEqual(embedding, "MockOllamaEmbedding")
        mock_ollama_embeddings.assert_called_once_with(model="test-model")

    def test_invalid_embedding_type(self,):
        """Test error handling for an invalid embedding type."""
        factory = EmbeddingFactory(embedding_type="invalid_type")
        with self.assertRaises(ValueError) as context:
            factory.create_embedding()
        self.assertEqual(str(context.exception), "Unknown embedding type: invalid_type")

    @patch("builtins.open", new_callable=mock_open, read_data='{"embedding_models": [{"type": "openai", "params": {"api_key": "test-key"}}]}')
    def test_get_config_valid(self, mock_file):
        """Test retrieving a valid configuration."""
        factory = EmbeddingFactory(embedding_type="openai")
        config = factory._get_config()
        self.assertEqual(config, {"api_key": "test-key"})

if __name__ == "__main__":
    unittest.main()
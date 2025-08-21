import os
import sys
import unittest
from unittest.mock import mock_open, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from rag.embeddings.embedding_factory import EmbeddingFactory

class TestEmbeddingFactory(unittest.TestCase):

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_create_openai_embedding(self, mock_openai_embeddings):
        # GIVEN
        mock_openai_embeddings.return_value = "MockOpenAIEmbedding"
        factory = EmbeddingFactory(embedding_type="openai")

        # WHEN
        embedding = factory.create_embedding()

        # THEN
        self.assertEqual(embedding, "MockOpenAIEmbedding")
        mock_openai_embeddings.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data='{"embedding_models": [{"type": "ollama", "params": {"model": "test-model"}}]}')
    @patch("langchain_ollama.OllamaEmbeddings")
    def test_create_ollama_embedding(self, mock_ollama_embeddings, mock_file):
        # GIVEN
        mock_ollama_embeddings.return_value = "MockOllamaEmbedding"
        factory = EmbeddingFactory(embedding_type="ollama")

        # WHEN
        embedding = factory.create_embedding()

        # THEN
        self.assertEqual(embedding, "MockOllamaEmbedding")
        mock_ollama_embeddings.assert_called_once_with(model="test-model")

    def test_invalid_embedding_type(self,):
        # GIVEN
        factory = EmbeddingFactory(embedding_type="invalid_type")
        with self.assertRaises(ValueError) as context:
            factory.create_embedding()
        # THEN
        self.assertEqual(str(context.exception), "Unknown embedding type: invalid_type")

    @patch("builtins.open", new_callable=mock_open, read_data='{"embedding_models": [{"type": "openai", "params": {"api_key": "test-key"}}]}')
    def test_get_config_valid(self, mock_file):
        # GIVEN
        factory = EmbeddingFactory(embedding_type="openai")
        # WHEN
        config = factory._get_config()
        # THEN
        self.assertEqual(config, {"api_key": "test-key"})

if __name__ == "__main__":
    unittest.main()
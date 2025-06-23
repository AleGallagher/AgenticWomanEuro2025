import unittest
from unittest.mock import patch, mock_open
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from models.model_factory import ModelFactory

class TestModelFactory(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data='{"models": [{"type": "openai", "params": {"api_key": "test-key"}}]}')
    @patch("langchain_openai.ChatOpenAI")
    def test_create_openai_model(self, mock_chat_openai, mock_file):
        """Test creating an OpenAI model."""
        mock_chat_openai.return_value = "MockOpenAIModel"
        factory = ModelFactory(model_type="openai")
        model = factory.create_model()

        self.assertEqual(model, "MockOpenAIModel")
        mock_chat_openai.assert_called_once_with(api_key='test-key')
        mock_file.assert_called_once_with(factory.config_path, "r")

    @patch("builtins.open", new_callable=mock_open, read_data='{"models": [{"type": "ollama", "params": {"model": "test-model"}}]}')
    @patch("langchain_ollama.ChatOllama")
    def test_create_ollama_model(self, mock_chat_ollama, mock_file):
        """Test creating an Ollama model."""
        mock_chat_ollama.return_value = "MockOllamaModel"
        factory = ModelFactory(model_type="ollama")
        model = factory.create_model()

        self.assertEqual(model, "MockOllamaModel")
        mock_chat_ollama.assert_called_once_with(model="test-model")
        mock_file.assert_called_once_with(factory.config_path, "r")

    @patch("builtins.open", new_callable=mock_open, read_data='{"models": [{"type": "openai", "params": {"api_key": "test-key"}}]}')
    def test_invalid_model_type(self, mock_file):
        """Test error handling for an invalid model type."""
        factory = ModelFactory(model_type="invalid_type")
        with self.assertRaises(ValueError) as context:
            factory.create_model()
        self.assertEqual(str(context.exception), "Unknown model type: invalid_type")

    @patch("builtins.open", new_callable=mock_open, read_data='{"models": [{"type": "openai", "params": {"api_key": "test-key"}}]}')
    def test_get_config_valid(self, mock_file):
        """Test retrieving a valid configuration."""
        factory = ModelFactory(model_type="openai")
        config = factory._get_config()
        self.assertEqual(config, {"api_key": "test-key"})

    @patch("builtins.open", new_callable=mock_open, read_data='{"models": [{"type": "openai", "params": {"api_key": "test-key"}}]}')
    def test_get_config_invalid(self, mock_file):
        """Test error handling for a missing configuration."""
        factory = ModelFactory(model_type="ollama")
        with self.assertRaises(ValueError) as context:
            factory._get_config()
        self.assertEqual(str(context.exception), "No model config found for type: ollama")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_config_file_not_found(self, mock_file):
        """Test error handling when the config file is missing."""
        factory = ModelFactory(model_type="openai")
        with self.assertRaises(FileNotFoundError):
            factory._get_config()

if __name__ == "__main__":
    unittest.main()
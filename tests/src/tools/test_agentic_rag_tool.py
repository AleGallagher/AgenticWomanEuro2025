import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from langchain_core.messages import HumanMessage

from rag.embeddings.embedding_factory import EmbeddingFactory
from rag.vector_stores.faiss_store import FAISSStore
from tools.agentic_rag_tool import agentic_rag

class TestAgenticRagStream(unittest.TestCase):
    @patch("tools.agentic_rag_tool.AgenticRAG")
    def test_agentic_rag_stream_valid_response(self, mock_agentic_rag):
        # GIVEN
        mock_graph = MagicMock()
        mock_graph.return_value = {
            "messages": [HumanMessage(content="Mock response")]
        }
        mock_agentic_rag.return_value = mock_graph
        question = "What can you say about Spain?"
        language = "English"
        embedding_model = EmbeddingFactory("ollama").create_embedding()
        store = FAISSStore(embedding_model=embedding_model)

        # WHEN
        result = agentic_rag.invoke({"vector_store" : store , "question": question, "language": language})

        # THEN
        self.assertEqual(result, "Mock response")
        mock_agentic_rag.assert_called_once_with(store)
        expected_initial_state = {
            "messages": [HumanMessage(content="What can you say about Spain?")],
            "question_language": language
        }
        mock_graph.assert_called_once_with(expected_initial_state)


if __name__ == "__main__":
    unittest.main()
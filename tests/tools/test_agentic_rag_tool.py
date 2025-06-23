import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from tools.agentic_rag_tool import agentic_rag_stream
from langchain_core.messages import HumanMessage

class TestAgenticRagStream(unittest.TestCase):
    @patch("tools.agentic_rag_tool.AgenticRAG")
    def test_agentic_rag_stream_valid_response(self, mock_agentic_rag):
        """Test agentic_rag_stream with valid inputs."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [HumanMessage(content="Mock response")]
        }
        mock_agentic_rag.return_value.graph = mock_graph

        vector_store = MagicMock()
        question = "What can you say about Spain?"
        language = "English"

        result = agentic_rag_stream.invoke({"vector_store" : vector_store , "question": question, "language": language})
        self.assertEqual(result, "Mock response")
        mock_graph.invoke.assert_called_once_with({
            "messages": [HumanMessage(content=question)],
            "question_language": language,
            "agent_action": "",
            "rewrite_count": 0,
        })

if __name__ == "__main__":
    unittest.main()
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from tools.sql_tool import get_sql_tool
from langchain_core.messages import HumanMessage

class TestSQLTool(unittest.TestCase):
    @patch("tools.sql_tool.SQLAgent")
    def test_get_sql_tool(self, mock_sql_agent):
        """Test the get_sql_tool function."""
        # GIVEN
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
                "messages": [HumanMessage(content="MockResult")]
        }
        mock_sql_agent.return_value.graph = mock_graph
        model = "test-model"
        agent_input = "Who is the coach of Spain?"
        question_language = "English"

        # THEN
        result = get_sql_tool(
            {            
                "model": model,
                "agent_input": agent_input,
                "question_language": question_language,
            }
        )

        # ASSERT
        self.assertEqual(result, "MockResult")
        mock_graph.invoke.assert_called_once_with({
            "input": agent_input,
            "question_language": question_language,
        })

if __name__ == "__main__":
    unittest.main()
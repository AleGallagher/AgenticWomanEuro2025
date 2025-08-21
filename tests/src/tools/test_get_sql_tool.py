import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from langchain_core.messages import HumanMessage

from tools.sql_tool import get_sql_tool

class TestSQLTool(unittest.TestCase):
  
    @patch("tools.sql_tool.SQLAgent")
    def test_get_sql_tool(self, mock_sql_agent):
        """Test the get_sql_tool function."""
        # GIVEN
        instance_mock = MagicMock()
        instance_mock.return_value = {
            "messages": [HumanMessage(content="MockResult")]
        }
        mock_sql_agent.return_value = instance_mock
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(api_key="test-key", model="gpt-3.5-turbo")
        agent_input = "Who is the coach of Spain?"
        question_language = "English"

        # WHEN 
        result = get_sql_tool.invoke({"model": model, "question": agent_input, "question_language": question_language})

        # THEN
        self.assertEqual(result, "MockResult")
        mock_sql_agent.assert_called_once_with(model)


if __name__ == "__main__":
    unittest.main()
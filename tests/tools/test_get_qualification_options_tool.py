import unittest
from unittest.mock import patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from tools.qualification_tool import get_qualification_options

class TestQualificationTool(unittest.TestCase):
    @patch("tools.qualification_tool.handle_qualification_question")
    def test_get_qualification_options(self, mock_handle_qualification_question):
        """Test the get_qualification_options function."""
        # GIVEN
        mock_handle_qualification_question.return_value = "MockResult"
        model = "test-model"
        agent_input = "What need Spain to qualify?"
        question_language = "English"

        # THEN
        result = get_qualification_options(
            {"model" : model, "agent_input" : agent_input, "question_language" : question_language}
        )

        # ASSERT
        self.assertEqual(result, "MockResult")
        mock_handle_qualification_question.assert_called_once_with(model, agent_input, question_language)

if __name__ == "__main__":
    unittest.main()
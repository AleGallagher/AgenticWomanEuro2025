import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from tools.qualification_tool import get_qualification_options

class TestQualificationTool(unittest.TestCase):
    @patch("tools.qualification_tool.handle_qualification_question")
    def test_get_qualification_options(self, mock_handle_qualification_question):
        # GIVEN
        result_mock = MagicMock()
        result_mock.content.strip.return_value = "MockResult"
        mock_handle_qualification_question.return_value = result_mock
        question = "What need Spain to qualify?"
        question_language = "English"
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(api_key="test-key", model="gpt-3.5-turbo")

        # THEN
        result = get_qualification_options.invoke(
            {"model" : model, "question" : question, "question_language" : question_language}
        )

        # ASSERT
        self.assertEqual(result, "MockResult")
        mock_handle_qualification_question.assert_called_once_with(llm=model, question=question, question_language=question_language)

if __name__ == "__main__":
    unittest.main()
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from unittest.mock import patch, MagicMock
from rag.agentic_rag import AgenticRAG
from langchain_core.messages import HumanMessage

class TestAgenticRAG(unittest.TestCase):
    @patch("rag.agentic_rag.ChatOpenAI")
    @patch("rag.agentic_rag.StrOutputParser")
    @patch("rag.agentic_rag.ChatPromptTemplate")
    def test_generate(self, mock_chat_prompt_template, mock_str_output_parser, mock_chat_openai):
        # GIVEN
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_prompt_template = MagicMock()
        mock_chat_prompt_template.from_template.return_value = mock_prompt_template
        mock_output_parser = MagicMock()
        mock_str_output_parser.return_value = mock_output_parser

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"content": "Mock response"}
        mock_prompt_template.__or__.return_value = mock_llm
        mock_llm.__or__.return_value = mock_output_parser
        mock_output_parser.__or__.return_value = mock_chain

        state = {
            "messages": [
                HumanMessage(content="What is the schedule for the Women's Eurocup 2025?"),
                HumanMessage(content="Mock context about the Women's Eurocup 2025.")
            ],
            "question_language": "en"
        }

        # THEN
        agentic_rag = AgenticRAG(vector_store=MagicMock())
        response = agentic_rag.graph.invoke(state)

        # ASSERT
        mock_chain.invoke.assert_called_once_with({
            "question": "What is the schedule for the Women's Eurocup 2025?",
            "language": "en"
        })
        self.assertEqual(response, {"messages": [{"content": "Mock response"}]})

if __name__ == "__main__":
    unittest.main()
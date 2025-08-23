import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag.agentic_rag import AgenticRAG

class TestAgenticRAG(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.get_vector_store.return_value = MagicMock()

    @patch('rag.agentic_rag.ChatOpenAI')
    def test_grade_documents_relevant(self, mock_openai):
        # GIVEN
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        mock_grade_result = MagicMock()
        mock_grade_result.confidence_score = 1.0
        with patch('rag.agentic_rag.PromptTemplate') as mock_prompt_template:
            mock_prompt = MagicMock()
            mock_prompt_template.return_value = mock_prompt
            
            # Mock the pipe operation
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_grade_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            agentic_rag = AgenticRAG(self.mock_vector_store)
            
            state = {
                "messages": [
                    HumanMessage(content="What can you say about Spain?"),
                    ToolMessage(content="Spain has won multiple tournaments...", tool_call_id="call_123")
                ]
            }
            
            # WHEN
            result = agentic_rag._grade_documents(state)
            
            # THEN
            self.assertEqual(result, "generate")

    @patch('rag.agentic_rag.ChatOpenAI')
    def test_grade_documents_irrelevant(self, mock_openai):
        # GIVEN
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        mock_grade_result = MagicMock()
        mock_grade_result.confidence_score = 0.3
        with patch('rag.agentic_rag.PromptTemplate') as mock_prompt_template:
            mock_prompt = MagicMock()
            mock_prompt_template.return_value = mock_prompt
            
            # Mock the pipe operation
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_grade_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            agentic_rag = AgenticRAG(self.mock_vector_store)
            
            state = {
                "messages": [
                    HumanMessage(content="What can you say about Spain?"),
                    ToolMessage(content="Spain has won multiple tournaments...", tool_call_id="call_123")
                ]
            }
            
            # WHEN
            result = agentic_rag._grade_documents(state)
            
            # THEN
            self.assertEqual(result, "rewrite")

    @patch('rag.agentic_rag.ChatOpenAI')
    def test_extract_metadata(self, mock_openai):
        # GIVEN
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        # Mock the metadata response
        mock_metadata_result = MagicMock()
        mock_metadata_result.countries = ["Spain"]
        
        # Mock the structured output chain
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = mock_metadata_result
        mock_llm.with_structured_output.return_value = mock_structured_llm
        state = {
        "messages": [HumanMessage(content="What can you say about Spain?")]
        }
        agentic_rag = AgenticRAG(self.mock_vector_store)
        result = agentic_rag._extract_metadata(state)
    
        # WHEN
        self.assertIn("question_metadata", result)
        self.assertEqual(result["question_metadata"].countries, ["Spain"])

        # THEN
        mock_llm.with_structured_output.assert_called_once()
        mock_structured_llm.invoke.assert_called_once()
        call_args = mock_structured_llm.invoke.call_args[0][0]
        self.assertIn("What can you say about Spain?", call_args)

    @patch('rag.agentic_rag.ChatOpenAI')
    def test_not_found(self, mock_openai):
        # GIVEN
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        # Mock the structured output chain
        mock_llm.invoke.return_value = "no data found"
        state = {
        "messages": [HumanMessage(content="What can you say about Spain?")],
        "question_language": "English",
        }
        agentic_rag = AgenticRAG(self.mock_vector_store)

        # WHEN
        result = agentic_rag._not_found(state)

        # THEN
        self.assertEqual(result["messages"][0], "no data found")
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        self.assertIn("What can you say about Spain?", call_args)
        self.assertIn("English", call_args)

    @patch('rag.agentic_rag.ChatOpenAI')
    def test_rewrite_question_not_found(self, mock_openai):
        # GIVEN
        state = {
        "messages": [HumanMessage(content="What can you say about Spain?")],
        "question_language": "English",
        "rewrite_count": 5,
        }
        agentic_rag = AgenticRAG(self.mock_vector_store)

        # WHEN
        result = agentic_rag._rewrite_question(state)

        # THEN
        self.assertEqual(result["agent_action"], "NOT_FOUND")

    @patch('rag.agentic_rag.ChatOpenAI')
    def test_rewrite_question_agent(self, mock_openai):
        # GIVEN
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        # Mock the structured output chain
        mock_llm.invoke.return_value = AIMessage(content="Spain has won multiple tournaments...")
        state = {
            "messages": [HumanMessage(content="What can you say about Spain?")],
            "question_language": "English",
            "rewrite_count": 0,
        }
        agentic_rag = AgenticRAG(self.mock_vector_store)

        # WHEN
        result = agentic_rag._rewrite_question(state)
    
        # THEN
        self.assertEqual(result["agent_action"], "agent")
        self.assertEqual(result["messages"][-1].content, "Spain has won multiple tournaments...")

    @patch('rag.agentic_rag.itemgetter')
    @patch('rag.agentic_rag.StrOutputParser')
    @patch('rag.agentic_rag.ChatPromptTemplate')
    @patch('rag.agentic_rag.ChatOpenAI')
    def test_generate_response(self, mock_openai, mock_chat_prompt, mock_str_parser, mock_itemgetter):
        # GIVEN
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        mock_prompt_template = MagicMock()
        mock_chat_prompt.from_template.return_value = mock_prompt_template

        mock_parser_instance = MagicMock()
        expected_response = AIMessage(content="Spain has a strong football team with excellent players.")
        mock_parser_instance.invoke.return_value = expected_response
        mock_str_parser.return_value = mock_parser_instance

        # Mock itemgetter functions
        mock_language_getter = MagicMock()
        mock_question_getter = MagicMock()
        mock_itemgetter.side_effect = lambda key: {
            "language": mock_language_getter,
            "question": mock_question_getter,
            "context": lambda x: "mocked context"
        }[key]

        # Build the operator pipeline:
        # dict | ChatPromptTemplate.from_template(prompt)  -> calls mock_prompt_template.__ror__(dict)
        # result | self.llm                                 -> calls result.__or__(mock_llm)
        # result2 | StrOutputParserInstance                 -> calls result2.__or__(mock_parser_instance)
        mock_step1 = MagicMock()
        mock_step2 = MagicMock()

        mock_prompt_template.__ror__ = MagicMock(return_value=mock_step1)   # dict | prompt -> step1
        mock_step1.__or__ = MagicMock(return_value=mock_step2)             # step1 | self.llm -> step2
        mock_step2.__or__ = MagicMock(return_value=mock_parser_instance)   # step2 | StrOutputParser() -> parser instance
        state = {
            "messages": [HumanMessage(content="What can you say about Spain?"), AIMessage(content="What can you say about Spain?")],
            "question_language": "English",
            "rewrite_count": 0,
        }
        agentic_rag = AgenticRAG(self.mock_vector_store)

        # WHEN
        result = agentic_rag._generate_response(state)

        # THEN
        self.assertEqual(result["messages"][-1].content, "Spain has a strong football team with excellent players.")

if __name__ == "__main__":
    unittest.main()
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from agents.main_agent import MainAgent, State

class TestMainAgent(unittest.TestCase):
    
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_vector_store = MagicMock()
        
        with patch('agents.main_agent.DatabaseService'):
            self.agent = MainAgent(self.mock_model, self.mock_vector_store)
    
    @patch('agents.main_agent.ChatOpenAI')
    def test_call_with_valid_football_question(self, mock_chat_openai):
        # GIVEN
        # Mock language detection
        mock_lang_llm = MagicMock()
        mock_lang_llm.invoke.return_value.content.strip.return_value = "English"

        # Mock football question validation
        mock_validation_llm = MagicMock()
        mock_validation_llm.invoke.return_value.content.strip.return_value = "YES"
        
        # Mock agent response (no tool calls)
        mock_agent_response = AIMessage(content="This is a response about football")
        mock_agent_response.tool_calls = []
        self.mock_model.bind_tools.return_value.invoke.return_value = mock_agent_response

        mock_chat_openai.side_effect = [mock_lang_llm, mock_validation_llm]

        # Create state and config
        state = State(
            messages=[HumanMessage(content="Who won the Euro 2025?")],
            question_language="",
            selected_tool="",
            user_id="test_user",
            country="test_country",
            is_valid_question=True
        )
        config = {"configurable": {"thread_id": "test_thread"}}
        
        # WHEN
        result = self.agent(state, config)
        
        # THEN
        self.assertIsNotNone(result)
        self.assertIn("messages", result)
        self.assertEqual(result.get("messages")[-1].content, "This is a response about football")


    @patch('agents.main_agent.ChatOpenAI')
    def test_call_with_invalid_question(self, mock_chat_openai):
        # GIVEN
        # Mock language detection
        mock_lang_llm = MagicMock()
        mock_lang_llm.invoke.return_value.content.strip.return_value = "English"
        
        # Mock validation to return NO
        mock_validation_llm = MagicMock()
        mock_validation_llm.invoke.return_value.content.strip.return_value = "NO"
        
        mock_chat_openai.return_value = mock_validation_llm
        
        # Create state and config
        state = State(
            messages=[HumanMessage(content="What is the weather today?")],
            question_language="",
            selected_tool="",
            user_id="test_user", 
            country="test_country",
            is_valid_question=True
        )
        config = {"configurable": {"thread_id": "test_thread"}}
        
        # WHEN
        result = self.agent(state, config)
        
        # THEN
        self.assertIsNotNone(result)
        self.assertEqual(result.get("messages")[-1].content, "I'm a football statistics assistant specialized in UEFA Euro championships. I can only help with football-related questions about players, teams, matches, statistics, and tournaments. Please ask me something about football!")
    

    @patch('agents.main_agent.ChatOpenAI')
    def test_call_with_tool_execution(self, mock_chat_openai):
        """Test __call__ method when tools are called."""
        # Mock language detection
        mock_lang_llm = MagicMock()
        mock_lang_llm.invoke.return_value.content.strip.return_value = "Spanish"
        
        # Mock validation
        mock_validation_llm = MagicMock()
        mock_validation_llm.invoke.return_value.content.strip.return_value = "YES"
        
        # Mock translation
        mock_translation_llm = MagicMock()
        mock_translation_llm.invoke.return_value.content.strip.return_value = "Who won Euro 2025?"
        
        # Mock agent response with tool calls
        mock_tool_call = {
            "name": "agentic_rag",
            "args": {"question": "¿Quién ganó la Euro 2025?"},
            "id": "call_123"
        }
        mock_tool_agent_response = AIMessage(content="")
        mock_tool_agent_response.tool_calls = [mock_tool_call]
        
        # Mock final agent response after tool execution
        mock_final_response = AIMessage(content="España ha ganado la Euro 2025")
        mock_final_response.tool_calls = []
        
        self.mock_model.bind_tools.return_value.invoke.side_effect = [
            mock_tool_agent_response,
            mock_final_response,
        ]
        mock_chat_openai.side_effect = [mock_lang_llm, mock_validation_llm, mock_translation_llm]
        
        # Mock tool execution
        with patch.object(self.agent, '_get_dict_tools') as mock_get_tools:
            # GIVEN
            mock_tool = MagicMock()
            mock_tool.invoke.return_value = "España ha ganado la Euro 2025"
            mock_get_tools.return_value = {"agentic_rag": mock_tool}
            
            # Create state and config
            state = State(
                messages=[HumanMessage(content="¿Quién ganó la Euro 2025?")],
                question_language="",
                selected_tool="",
                user_id="test_user",
                country="test_country", 
                is_valid_question=True
            )
            config = {"configurable": {"thread_id": "test_thread"}}
            
            # WHEN
            result = self.agent(state, config)
            
            # THEN
            self.assertIsNotNone(result)
            self.assertEqual(result.get("messages")[-1].content, "España ha ganado la Euro 2025")

    
    @patch('agents.main_agent.ChatOpenAI')
    def test_call_language_detection(self, mock_chat_openai):
        # GIVEN
        # Mock language detection to return French
        mock_lang_llm = MagicMock()
        mock_lang_llm.invoke.return_value.content.strip.return_value = "French"
        
        # Mock validation
        mock_validation_llm = MagicMock() 
        mock_validation_llm.invoke.return_value.content.strip.return_value = "YES"
        
        # Mock agent response (no tool calls)
        mock_agent_response = AIMessage(content="Response about football")
        mock_agent_response.tool_calls = []
        self.mock_model.bind_tools.return_value.invoke.return_value = mock_agent_response
        
        mock_chat_openai.return_value = mock_lang_llm
        
        # Create state and config
        state = State(
            messages=[HumanMessage(content="Qui a gagné l'Euro 2024?")],
            question_language="",
            selected_tool="", 
            user_id="test_user",
            country="test_country",
            is_valid_question=True
        )
        config = {"configurable": {"thread_id": "test_thread"}}
        
        # WHEN
        result = self.agent(state, config)

        # THEN
        self.assertIsNotNone(result)
        self.assertEqual(result.get("question_language"), "French")

if __name__ == "__main__":
    unittest.main()
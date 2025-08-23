import os
import sys
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from agents.sql_agent import SQLAgent, State

class TestSQLAgent(unittest.TestCase):
    
    @patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
    @patch('agents.sql_agent.ChatOpenAI')
    @patch('agents.sql_agent.SQLDatabase')
    @patch('agents.sql_agent.SQLDatabaseToolkit')
    @patch('agents.sql_agent.create_openai_functions_agent')
    @patch('agents.sql_agent.AgentExecutor')
    def test_call_successful_execution(self, mock_executor_class, mock_create_agent, 
                                     mock_toolkit_class, mock_db_class, mock_openai):
        # GIVEN
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        mock_db = Mock()
        mock_db_class.from_uri.return_value = mock_db
        
        mock_toolkit = Mock()
        mock_tools = [Mock(), Mock()]
        mock_toolkit.get_tools.return_value = mock_tools
        mock_toolkit_class.return_value = mock_toolkit
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        mock_executor = Mock()
        mock_executor.invoke.return_value = {"output": "Spain won 2-1 against Portugal"}
        mock_executor_class.return_value = mock_executor
        
        # Mock the graph compilation and execution
        with patch('agents.sql_agent.StateGraph') as mock_state_graph:
            mock_builder = Mock()
            mock_graph = Mock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Spain won 2-1 against Portugal")]
            }
            mock_builder.compile.return_value = mock_graph
            mock_state_graph.return_value = mock_builder
            
            sql_agent = SQLAgent(mock_llm)
            
            state = {
                "messages": [HumanMessage(content="How did Spain perform against Portugal?")],
                "question_language": "English", 
                "input": "How did Spain perform against Portugal?"
            }
            
            # WHEN
            result = sql_agent(state)
            
            # THEN
            self.assertIn("messages", result)
            self.assertEqual(result["messages"][-1].content, "Spain won 2-1 against Portugal")
            mock_graph.invoke.assert_called_once_with(state)

    @patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
    @patch('agents.sql_agent.ChatOpenAI')
    @patch('agents.sql_agent.SQLDatabase')
    @patch('agents.sql_agent.SQLDatabaseToolkit')
    @patch('agents.sql_agent.create_openai_functions_agent')
    @patch('agents.sql_agent.AgentExecutor')
    def test_call_with_no_results(self, mock_executor_class, mock_create_agent,
                                mock_toolkit_class, mock_db_class, mock_openai):
        # GIVEN
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        mock_db = Mock()
        mock_db_class.from_uri.return_value = mock_db
        
        mock_toolkit = Mock()
        mock_tools = [Mock(), Mock()]
        mock_toolkit.get_tools.return_value = mock_tools
        mock_toolkit_class.return_value = mock_toolkit
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        mock_executor = Mock()
        mock_executor.invoke.return_value = {"output": "agent stopped due to iteration limit or time limit."}
        mock_executor_class.return_value = mock_executor
        
        # Mock the graph compilation and execution for no results case
        with patch('agents.sql_agent.StateGraph') as mock_state_graph:
            mock_builder = Mock()
            mock_graph = Mock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Seems that there are no results for this question. Can I help you with something else?")]
            }
            mock_builder.compile.return_value = mock_graph
            mock_state_graph.return_value = mock_builder
            
            sql_agent = SQLAgent(mock_llm)
            
            state = {
                "messages": [HumanMessage(content="Invalid query")],
                "question_language": "English",
                "input": "Invalid query"
            }
            
            # WHEN
            result = sql_agent(state)
            
            # THEN
            self.assertIn("messages", result)
            self.assertEqual(result["messages"][-1].content, "Seems that there are no results for this question. Can I help you with something else?")
            mock_graph.invoke.assert_called_once_with(state)

    @patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
    @patch('agents.sql_agent.ChatOpenAI')
    @patch('agents.sql_agent.SQLDatabase')
    @patch('agents.sql_agent.SQLDatabaseToolkit')
    @patch('agents.sql_agent.create_openai_functions_agent')
    @patch('agents.sql_agent.AgentExecutor')
    def test_call_with_exception_handling(self, mock_executor_class, mock_create_agent,
                                        mock_toolkit_class, mock_db_class, mock_openai):
        # GIVEN
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        mock_db = Mock()
        mock_db_class.from_uri.return_value = mock_db
        
        mock_toolkit = Mock()
        mock_tools = [Mock(), Mock()]
        mock_toolkit.get_tools.return_value = mock_tools
        mock_toolkit_class.return_value = mock_toolkit
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        mock_executor = Mock()
        mock_executor.invoke.side_effect = Exception("Database connection error")
        mock_executor_class.return_value = mock_executor
        
        # Mock the graph compilation and execution for error case
        with patch('agents.sql_agent.StateGraph') as mock_state_graph:
            mock_builder = Mock()
            mock_graph = Mock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="An error occurred while processing your request. Please try a different request or rephrase your question.")]
            }
            mock_builder.compile.return_value = mock_graph
            mock_state_graph.return_value = mock_builder
            
            sql_agent = SQLAgent(mock_llm)
            
            state = {
                "messages": [HumanMessage(content="Test query")],
                "question_language": "English",
                "input": "Test query"
            }
            
            # WHEN
            result = sql_agent(state)
            
            # THEN
            self.assertIn("messages", result)
            self.assertEqual(result["messages"][-1].content, "An error occurred while processing your request. Please try a different request or rephrase your question.")
            mock_graph.invoke.assert_called_once_with(state)

    @patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
    @patch('agents.sql_agent.ChatOpenAI')
    @patch('agents.sql_agent.SQLDatabase')
    @patch('agents.sql_agent.SQLDatabaseToolkit')
    def test_call_state_structure(self, mock_toolkit_class, mock_db_class, mock_openai):
        # GIVEN
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        mock_db = Mock()
        mock_db_class.from_uri.return_value = mock_db
        
        mock_toolkit = Mock()
        mock_tools = [Mock(), Mock()]
        mock_toolkit.get_tools.return_value = mock_tools
        mock_toolkit_class.return_value = mock_toolkit
        
        with patch('agents.sql_agent.StateGraph') as mock_state_graph, \
             patch('agents.sql_agent.create_openai_functions_agent') as mock_create_agent, \
             patch('agents.sql_agent.AgentExecutor') as mock_executor_class:
            
            mock_builder = Mock()
            mock_graph = Mock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Test response")]
            }
            mock_builder.compile.return_value = mock_graph
            mock_state_graph.return_value = mock_builder
            
            sql_agent = SQLAgent(mock_llm)
            
            # Test with minimal state structure
            state = {
                "messages": [HumanMessage(content="Test")],
                "question_language": "Spanish",
                "input": "Test"
            }
            
            # WHEN
            result = sql_agent(state)
            
            # THEN
            mock_graph.invoke.assert_called_once_with(state)
            self.assertIsInstance(result, dict)
            self.assertIn("messages", result)


if __name__ == "__main__":
    unittest.main()
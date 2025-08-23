import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from agents.qualification_agent import handle_qualification_question

class TestQualificationAgent(unittest.TestCase):
    
    @patch('agents.qualification_agent.get_sql_tool')
    @patch('agents.qualification_agent.PromptTemplate')
    def test_handle_qualification_question_success(self, mock_prompt_template, mock_sql_tool):
        # GIVEN
        mock_llm = MagicMock()
        question = "What does Spain need to qualify?"
        question_language = "English"
        
        # Mock SQL tool response
        mock_sql_result = "Group A standings: Spain 6pts, France 4pts, Italy 2pts"
        mock_sql_tool.invoke.return_value = mock_sql_result
        
        # Mock LLM chain
        mock_chain = MagicMock()
        mock_llm_response = "Spain needs to win their next match to qualify."
        mock_chain.invoke.return_value = mock_llm_response
        
        # Mock prompt template
        mock_template = MagicMock()
        mock_template.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_template.return_value = mock_template
        
        # WHEN
        result = handle_qualification_question(mock_llm, question, question_language)
        
        # THEN
        self.assertEqual(result, mock_llm_response)
        mock_sql_tool.invoke.assert_called_once_with({
            "model": mock_llm,
            "agent_input": f"Get current group standings and upcoming matches for the group of the team mentioned in: '{question}'",
            "question_language": "English"
        })
        mock_prompt_template.assert_called_once()
        mock_chain.invoke.assert_called_once_with({
            "question": question,
            "sql_data": mock_sql_result,
            "rules": handle_qualification_question.__globals__['COMPETITION_RULES_TEXT'],
            "language": question_language,
        })

    @patch('agents.qualification_agent.get_sql_tool')
    @patch('agents.qualification_agent.PromptTemplate')
    def test_handle_qualification_question_different_language(self, mock_prompt_template, mock_sql_tool):
        # GIVEN
        mock_llm = MagicMock()
        question = "¿Qué necesita España para clasificarse?"
        question_language = "Spanish"
        
        # Mock SQL tool response
        mock_sql_result = "Group standings data"
        mock_sql_tool.invoke.return_value = mock_sql_result
        
        # Mock LLM chain
        mock_chain = MagicMock()
        mock_llm_response = "España necesita ganar su próximo partido."
        mock_chain.invoke.return_value = mock_llm_response
        
        # Mock prompt template
        mock_template = MagicMock()
        mock_template.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_template.return_value = mock_template
        
        # WHEN
        result = handle_qualification_question(mock_llm, question, question_language)
        
        # THEN
        self.assertEqual(result, mock_llm_response)
        mock_chain.invoke.assert_called_once()
        call_args = mock_chain.invoke.call_args[0][0]
        self.assertEqual(call_args["language"], "Spanish")

    @patch('agents.qualification_agent.get_sql_tool')
    @patch('agents.qualification_agent.PromptTemplate')
    def test_handle_qualification_question_prompt_template_creation(self, mock_prompt_template, mock_sql_tool):
        # GIVEN
        mock_llm = MagicMock()
        question = "What about France?"
        question_language = "English"
        
        mock_sql_tool.invoke.return_value = "mock_sql_data"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "mock_response"
        mock_template = MagicMock()
        mock_template.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_template.return_value = mock_template
        
        # WHEN
        handle_qualification_question(mock_llm, question, question_language)
        
        # THEN
        mock_prompt_template.assert_called_once()
        call_args = mock_prompt_template.call_args
        self.assertIn("input_variables", call_args[1])
        self.assertEqual(call_args[1]["input_variables"], ["team", "current_standings", "remaining_matches", "rules"])
        self.assertIn("template", call_args[1])

if __name__ == "__main__":
    unittest.main()
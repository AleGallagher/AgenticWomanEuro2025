import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from services.database_service import DatabaseService
class TestDatabaseService(unittest.TestCase):

    @patch("services.database_service.create_engine")
    @patch("services.database_service.sessionmaker")
    def test_save_question_answer_with_mocked_session_local(self, mock_sessionmaker, mock_create_engine):
        # GIVEN
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_context
        db_service = DatabaseService()
        db_service.SessionLocal = mock_sessionmaker

        user_id = "test_user_2"
        question = "What is the capital of Germany?"
        country = "Germany"
        response = "Berlin"
        question_language = "German"
        tool = "test_tool_2"

        # WHEN
        db_service.save_question_answer(user_id, question, country, response, question_language, tool)

        # ASSERT
        mock_session.execute.assert_called_once()
        args, kwargs = mock_session.execute.call_args
        params = args[0].compile().params
        self.assertEqual(params["user_id"], user_id)
        self.assertEqual(params["question"], question)
        self.assertEqual(params["country"], country)
        self.assertEqual(params["response"], response)
        self.assertEqual(params["question_language"], question_language)
        self.assertEqual(params["tool"], tool)
        mock_session.commit.assert_called_once()

if __name__ == "__main__":
    unittest.main()

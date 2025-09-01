import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from services.database_service import DatabaseService


class TestDatabaseService(unittest.TestCase):

    @patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
    @patch("services.database_service.create_async_engine")
    @patch("services.database_service.async_sessionmaker")
    def test_save_question_answer_with_mocked_session_local(self, mock_sessionmaker, mock_create_engine):
        # GIVEN
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        
        # Create async context manager mock
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_sessionmaker.return_value = mock_session_context

        db_service = DatabaseService()
        db_service.AsyncSessionLocal = mock_sessionmaker
        db_service._initialized = True

        user_id = "test_user_2"
        question = "What is the capital of Germany?"
        original_question = "What is the capital of Germany?"
        country = "Germany"
        response = "Berlin"
        question_language = "German"
        tool = "test_tool_2"

        # WHEN
        asyncio.run(db_service.save_question_answer(user_id, question, original_question, country, response, question_language, tool))

        # ASSERT
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

if __name__ == "__main__":
    unittest.main()

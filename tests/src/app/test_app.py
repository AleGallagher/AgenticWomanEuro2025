import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
with patch('config.dependencies.get_model'), \
     patch('config.dependencies.get_store'), \
     patch('agents.main_agent.MainAgent'), \
     patch('services.telegram_service.TelegramService'):
    from app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        # GIVEN & WHEN
        response = self.client.get("/")
        
        # THEN
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"greeting": "Hello UEFA Women's EURO 2025"})

    @patch('app.agent')
    def test_send_message_success(self, mock_agent):
        # GIVEN
        mock_result = {
            "messages": [
                HumanMessage(content="Test question"),
                AIMessage(content="Test response")
            ]
        }
        mock_agent.return_value = mock_result
        
        message_data = {
            "question": "What is UEFA Euro 2025?",
            "session_id": "test_session_123",
            "country": "Spain"
        }
        
        # WHEN
        response = self.client.post("/message", json=message_data)
        
        # THEN
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"output": "Test response"})
        mock_agent.assert_called_once()
        # Verify agent was called with correct parameters
        call_args = mock_agent.call_args
        self.assertEqual(call_args[1]["state"]["user_id"], "test_session_123")
        self.assertEqual(call_args[1]["state"]["country"], "Spain")
        self.assertEqual(call_args[1]["state"]["messages"][0].content, "What is UEFA Euro 2025?")
        self.assertEqual(call_args[1]["config"]["configurable"]["thread_id"], "test_session_123")

    def test_send_message_empty_question(self):
        # GIVEN
        message_data = {
            "question": "   ",
            "session_id": "test_session_123",
            "country": "Spain"
        }
        
        # WHEN
        response = self.client.post("/message", json=message_data)
        
        # THEN
        self.assertEqual(response.status_code, 400)

    def test_send_message_empty_session_id(self):
        # GIVEN
        message_data = {
            "question": "What is UEFA Euro 2025?",
            "session_id": "   ",
            "country": "Spain"
        }
        
        # WHEN
        response = self.client.post("/message", json=message_data)
        
        # THEN
        self.assertEqual(response.status_code, 400)

    @patch('app.agent')
    def test_send_message_agent_exception(self, mock_agent):
        # GIVEN
        mock_agent.side_effect = Exception("Agent error")
        
        message_data = {
            "question": "What is UEFA Euro 2025?",
            "session_id": "test_session_123",
            "country": "Spain"
        }
        
        # WHEN
        response = self.client.post("/message", json=message_data)
        
        # THEN
        expected_error_message = "Sorry, I cannot answer this question now. Please try a different request or rephrase your question."
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"output": expected_error_message})

    @patch('app.telegram_service')
    def test_send_feedback_success(self, mock_telegram_service):
        # GIVEN
        mock_telegram_service.send_feedback = MagicMock()
        feedback_data = {
            "feedback": "Great application, very helpful!"
        }
        
        # WHEN
        response = self.client.post("/feedback", json=feedback_data)
        
        # THEN
        self.assertEqual(response.status_code, 200)
        mock_telegram_service.send_feedback.assert_called_once_with("Great application, very helpful!")

    def test_send_feedback_empty_feedback(self):
        # GIVEN
        feedback_data = {
            "feedback": "   "
        }
        
        # WHEN
        response = self.client.post("/feedback", json=feedback_data)
        
        # THEN
        self.assertEqual(response.status_code, 400)
        self.assertIn("The 'feedback' field cannot be empty.", response.json()["detail"])

    @patch('app.agent')
    def test_send_message_with_none_country(self, mock_agent):
        # GIVEN
        mock_result = {
            "messages": [
                HumanMessage(content="Test question"),
                AIMessage(content="Test response")
            ]
        }
        mock_agent.return_value = mock_result
        
        message_data = {
            "question": "What is UEFA Euro 2025?",
            "session_id": "test_session_123",
            "country": None
        }
        
        # WHEN
        response = self.client.post("/message", json=message_data)
        
        # THEN
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"output": "Test response"})
        call_args = mock_agent.call_args
        self.assertIsNone(call_args[1]["state"]["country"])


if __name__ == "__main__":
    unittest.main()
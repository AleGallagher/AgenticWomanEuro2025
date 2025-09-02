import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

@pytest.mark.asyncio
async def test_feedback():
    with patch('routers.feed_back.TelegramService') as mock_agent_class:
        mock_instance = MagicMock()
        mock_agent_class.return_value = mock_instance
        from fastapi.testclient import TestClient

        from app import app

        client = TestClient(app)
        
        message_data = {
            "feedback": "Great application, very helpful!"
        }
        
        response = client.post("/feedback", json=message_data)
        
        assert response.status_code == 200
        mock_instance.send_feedback.assert_called_once_with("Great application, very helpful!")

@pytest.mark.asyncio
async def test_feedback_empty_message():
    with patch('routers.feed_back.TelegramService') as mock_agent_class:
        mock_instance = MagicMock()
        mock_agent_class.return_value = mock_instance
        from fastapi.testclient import TestClient

        from app import app

        client = TestClient(app)
        
        message_data = {
            "feedback": "   "
        }
        
        response = client.post("/feedback", json=message_data)
        
        assert response.status_code == 400
        assert "The 'feedback' field cannot be empty." in response.json()["detail"]


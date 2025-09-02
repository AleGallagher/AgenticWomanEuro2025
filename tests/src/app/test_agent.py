import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

@pytest.mark.asyncio
@patch.dict(os.environ, {'MODEL_TYPE': 'ollama'})
@patch.dict(os.environ, {'EMBEDDING_MODEL': 'ollama'})
@patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
async def test_agent():
    mock_result = {
        "messages": [
            HumanMessage(content="What is UEFA Euro 2025?"),
            AIMessage(content="Test response")
        ]
    }
    
    with patch('routers.agent.MainAgent') as mock_telegram_service:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.return_value = mock_result
        mock_telegram_service.return_value = mock_agent_instance
        from fastapi.testclient import TestClient

        from app import app

        client = TestClient(app)
        
        message_data = {
            "question": "What is UEFA Euro 2025?",
            "session_id": "test_session_123",
            "country": "Spain"
        }
        
        response = client.post("/message", json=message_data)
        
        assert response.status_code == 200
        assert response.json() == {"output": "Test response"}
        mock_agent_instance.assert_called_once()
        
        call_args = mock_agent_instance.call_args
        assert call_args[1]["state"]["user_id"] == "test_session_123"
        assert call_args[1]["state"]["country"] == "Spain"
        assert call_args[1]["state"]["messages"][0].content == "What is UEFA Euro 2025?"
        assert call_args[1]["config"]["configurable"]["thread_id"] == "test_session_123"

@pytest.mark.asyncio
@patch.dict(os.environ, {'MODEL_TYPE': 'ollama'})
@patch.dict(os.environ, {'EMBEDDING_MODEL': 'ollama'})
@patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
async def test_agent_message_empty_question():
    with patch('routers.agent.MainAgent') as mock_telegram_service:
        mock_agent_instance = AsyncMock()
        mock_telegram_service.return_value = mock_agent_instance
        from fastapi.testclient import TestClient

        from app import app

        client = TestClient(app)
        
        message_data = {
            "question": "   ",
            "session_id": "test_session_123",
            "country": "Spain"
        }
        
        response = client.post("/message", json=message_data)
        
        assert response.status_code == 400
        assert "The 'question' field cannot be empty." in response.json()["error"]

@pytest.mark.asyncio
@patch.dict(os.environ, {'MODEL_TYPE': 'ollama'})
@patch.dict(os.environ, {'EMBEDDING_MODEL': 'ollama'})
@patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
async def test_agent_message_empty_session_id():
    with patch('routers.agent.MainAgent') as mock_telegram_service:
        mock_agent_instance = AsyncMock()
        mock_telegram_service.return_value = mock_agent_instance
        from fastapi.testclient import TestClient

        from app import app

        client = TestClient(app)
        
        message_data = {
            "question": "What is UEFA Euro 2025?",
            "session_id": " ",
            "country": "Spain"
        }
        
        response = client.post("/message", json=message_data)
        
        assert response.status_code == 400
        assert "The 'session_id' field cannot be empty." in response.json()["error"]

@pytest.mark.asyncio
@patch.dict(os.environ, {'MODEL_TYPE': 'ollama'})
@patch.dict(os.environ, {'EMBEDDING_MODEL': 'ollama'})
@patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
async def test_agent_message_exception():
    with patch('routers.agent.MainAgent') as mock_telegram_service:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.side_effect = Exception("Agent error")
        mock_telegram_service.return_value = mock_agent_instance
        from fastapi.testclient import TestClient

        from app import app

        client = TestClient(app)
        
        message_data = {
            "question": "What is UEFA Euro 2025?",
            "session_id": "test_session_123",
            "country": "Spain"
        }
        
        response = client.post("/message", json=message_data)
        
        assert response.status_code == 200
        assert "Sorry, I cannot answer this question now. Please try a different request or rephrase your question." in response.json()["output"]


@pytest.mark.asyncio
@patch.dict(os.environ, {'MODEL_TYPE': 'ollama'})
@patch.dict(os.environ, {'EMBEDDING_MODEL': 'ollama'})
@patch.dict(os.environ, {'POSTGRES_HOST': 'postgresql://test:test@localhost:5432/test'})
async def test_send_message_with_none_country():
    mock_result = {
        "messages": [
            HumanMessage(content="What is UEFA Euro 2025?"),
            AIMessage(content="Test response")
        ]
    }
    
    with patch('routers.agent.MainAgent') as mock_telegram_service:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.return_value = mock_result
        mock_telegram_service.return_value = mock_agent_instance
        from fastapi.testclient import TestClient

        from app import app

        client = TestClient(app)
        
        message_data = {
            "question": "What is UEFA Euro 2025?",
            "session_id": "test_session_123"
        }
        
        response = client.post("/message", json=message_data)
        
        assert response.status_code == 200
        assert "Test response" in response.json()["output"]
        assert mock_agent_instance.call_args[1]["state"]["country"] is None
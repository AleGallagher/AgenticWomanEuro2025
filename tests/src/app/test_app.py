import os
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

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

if __name__ == "__main__":
    unittest.main()
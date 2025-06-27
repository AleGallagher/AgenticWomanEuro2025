import os
from fastapi import HTTPException
import requests

class TelegramService:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def send_feedback(self, body):
        """
        Sends feedback to a Telegram chat.
        """
        telegram_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": f"New Feedback Received:\n\n{body}"
        }
        try:
            response = requests.post(telegram_url, json=payload)
            response.raise_for_status()
            print("Feedback sent to Telegram successfully!")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail="Failed to send feedback to Telegram.")
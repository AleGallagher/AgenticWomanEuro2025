from sqlalchemy import create_engine, MetaData, Table, Column, String, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
import uuid
import os
import time

class DatabaseService:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", os.getenv("POSTGRES_HOST"))
        self.engine = create_engine(self.database_url)
        self.metadata = MetaData()

        self.question_answer_table = Table(
            "question_answer",
            self.metadata,
            Column("id", String, primary_key=True),
            Column("country", String, nullable=True),
            Column("user_id", String, nullable=False),
            Column("question", Text, nullable=False),
            Column("original_question", Text, nullable=False),
            Column("response", Text, nullable=False),
            Column("question_language", String, nullable=False),
            Column("tool", String, nullable=False)
        )

        self.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def save_question_answer(self, user_id: str, question: str, original_question: str, country, response: str, question_language: str, tool: str):
        """
        Store the question and response in the database.

        Args:
            user_id (str): The unique identifier for the user.
            question (str): The question asked by the user.
            original_question (str): The original question asked by the user.
            country (str): The country associated with the user.
            response (str): The response generated for the question.
            question_language (str): The language of the question.
            tool (str): The tool used to generate the response.
            retries (int): Number of retry attempts in case of failure.
            delay (int): Delay (in seconds) between retries.
        """
        attempt = 0
        retries = 3
        delay = 2
        while attempt < retries:
            try:
                with self.SessionLocal() as session:
                    session.execute(
                        self.question_answer_table.insert().values(
                            id=str(uuid.uuid4()),
                            country=country,
                            user_id=user_id,
                            question=question,
                            original_question=original_question,
                            response=response,
                            question_language=question_language,
                            tool=tool,
                        )
                    )
                    session.commit()
                    return
            except OperationalError as e:
                attempt += 1
                print(f"Attempt {attempt} failed: {e}")
                if attempt < retries:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("All retry attempts failed.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

import asyncio
import os
import uuid

from sqlalchemy import Column, MetaData, String, Table, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import sessionmaker


class DatabaseService:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", os.getenv("POSTGRES_HOST"))
        if self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")

        self.engine = create_async_engine(self.database_url)
        self.metadata = MetaData()

        self.question_answer_table = Table(
            "question_answer",
            self.metadata,
            Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
            Column("country", String, nullable=True),
            Column("user_id", String, nullable=False),
            Column("question", Text, nullable=False),
            Column("original_question", Text, nullable=False),
            Column("response", Text, nullable=False),
            Column("question_language", String, nullable=False),
            Column("tool", String, nullable=False)
        )

        # Create async session factory
        self.AsyncSessionLocal = async_sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        self._initialized = False

    async def initialize_tables(self):
        """Initialize database tables (call once at startup)"""
        if not self._initialized:
            async with self.engine.begin() as conn:
                await conn.run_sync(self.metadata.create_all)
            self._initialized = True

    async def save_question_answer(self, user_id: str, question: str, original_question: str, country, response: str, question_language: str, tool: str):
        """
        Store the question and response in the database.

        Args:
            id (str): The unique identifier for the question-answer pair.
            country (str): The country associated with the user.
            user_id (str): The unique identifier for the user.
            question (str): The question asked by the user.
            original_question (str): The original question asked by the user.
            response (str): The response generated for the question.
            question_language (str): The language of the question.
            tool (str): The tool used to generate the response.
        """
        if not self._initialized:
            await self.initialize_tables()
        attempt = 0
        retries = 3
        delay = 2
        while attempt < retries:
            try:
                async with self.AsyncSessionLocal() as session:
                    await session.execute(
                        self.question_answer_table.insert().values(
                            id=uuid.uuid4(),
                            country=country,
                            user_id=user_id,
                            question=question,
                            original_question=original_question,
                            response=response,
                            question_language=question_language,
                            tool=tool,
                        )
                    )
                    await session.commit()
                    return
            except OperationalError as e:
                attempt += 1
                print(f"Attempt {attempt} failed: {e}")
                if attempt < retries:
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print("All retry attempts failed.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    async def close(self):
        """Close the database engine"""
        await self.engine.dispose()
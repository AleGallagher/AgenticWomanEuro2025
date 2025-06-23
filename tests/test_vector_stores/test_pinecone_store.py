import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from rag.vector_stores.pinecone_store import PineconeStore

class TestPineconeStore(unittest.TestCase):
    @patch("rag.vector_stores.pinecone_store.PineconeVectorStore")
    def setUp(self, MockPineconeVectorStore):
        # Mock the PineconeVectorStore instance
        self.mock_vector_store = MockPineconeVectorStore.from_existing_index.return_value
        self.mock_embedding_model = MagicMock()

        # Set a dummy environment variable for PINECONE_INDEX
        os.environ["PINECONE_INDEX"] = "test-index"

        # Create PineconeStore instance
        self.store = PineconeStore(embedding_model=self.mock_embedding_model)

    def test_add_documents(self):
        chunks = [{"id": "1", "content": "test content"}]

        self.store.add_documents(chunks)

        self.mock_vector_store.add_documents.assert_called_once_with(chunks)

    def test_search(self):
        query = "test query"
        mock_results = [
            MagicMock(page_content="result 1"),
            MagicMock(page_content="result 2")
        ]
        self.mock_vector_store.similarity_search.return_value = mock_results

        results = self.store.search(query, top_k=2)

        self.mock_vector_store.similarity_search.assert_called_once_with(query, k=2)
        self.assertEqual(results, ["result 1", "result 2"])

    def test_delete(self):
        ids = ["1", "2"]

        self.store.delete(ids)

        self.mock_vector_store.delete.assert_called_once_with(ids=ids)

    def test_get_vector_store(self):
        vector_store = self.store.get_vector_store()

        self.assertEqual(vector_store, self.mock_vector_store)

    def test_save_data_base(self):
        # save_data_base is currently a no-op, so just check it doesn't raise
        try:
            self.store.save_data_base("dummy_name")
        except Exception as e:
            self.fail(f"save_data_base raised Exception unexpectedly: {e}")

    def test_load_vector_store(self):
        # load_vector_store is currently a no-op, so just check it doesn't raise
        try:
            self.store.load_vector_store()
        except Exception as e:
            self.fail(f"load_vector_store raised Exception unexpectedly: {e}")

if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from rag.vector_stores.store_factory import StoreFactory

class TestStoreFactory(unittest.TestCase):
    @patch("rag.vector_stores.faiss_store.FAISSStore")
    def test_get_store_faiss(self, store):
        # GIVEN
        store.return_value = "FAISSStore"
        mock_embedding_model = MagicMock()
        factory = StoreFactory(store_type="faiss", embedding_model=mock_embedding_model)

        # WHEN
        store = factory.get_store()

        # THEN
        self.assertEqual(store, "FAISSStore")

    @patch("rag.vector_stores.pinecone_store.PineconeStore")
    def test_get_store_pinecone(self, store):
        # GIVEN
        store.return_value = "PineconeStore"
        mock_embedding_model = MagicMock()
        factory = StoreFactory(store_type="pinecone", embedding_model=mock_embedding_model)

        # WHEN
        store = factory.get_store()

        # THEN
        self.assertEqual(store, "PineconeStore")

    def test_invalid_store_type(self):
        # GIVEN
        mock_embedding_model = MagicMock()
        factory = StoreFactory(store_type="asd", embedding_model=mock_embedding_model)

        # THEN
        with self.assertRaises(ValueError) as context:
            factory.get_store()
        self.assertEqual(str(context.exception), "Unknown model type: asd")
if __name__ == "__main__":
    unittest.main()

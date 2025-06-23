import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from rag.vector_stores.store_factory import StoreFactory

class TestStoreFactory(unittest.TestCase):
    @patch("rag.vector_stores.faiss_store.FAISSStore")
    def test_get_store_faiss(self, store):
        # Arrange
        store.return_value = "FAISSStore"
        mock_embedding_model = MagicMock()
        factory = StoreFactory(store_type="faiss", embedding_model=mock_embedding_model)

        # Act
        store = factory.get_store()

        # Assert
        self.assertEqual(store, "FAISSStore")

    def test_invalid_store_type(self):
        # Arrange
        mock_embedding_model = MagicMock()
        factory = StoreFactory(store_type="asd", embedding_model=mock_embedding_model)

        # Act
        with self.assertRaises(ValueError) as context:
            factory.get_store()
        self.assertEqual(str(context.exception), "Unknown model type: asd")
if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch, mock_open
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from unittest.mock import patch, MagicMock
from rag.vector_stores.faiss_store import FAISSStore

class TestFAISSStore(unittest.TestCase):
    @patch("rag.vector_stores.faiss_store.faiss.IndexFlatL2")
    @patch("rag.vector_stores.faiss_store.FAISS")
    def test_initialization(self, mock_faiss_vector_store, mock_faiss_index):
        """Test the initialization of FAISSStore."""
        # Mock the embedding model
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]

        # Mock FAISS index
        mock_index = MagicMock()
        mock_faiss_index.return_value = mock_index

        # Initialize FAISSStore
        store = FAISSStore(embedding_model=mock_embedding_model)

        # Assertions
        mock_faiss_index.assert_called_once_with(3)  # Verify the index was initialized with the correct dimension
        self.assertEqual(store.embedding_model, mock_embedding_model)  # Verify the embedding model is set
        self.assertEqual(store.index, mock_index)  # Verify the index is set
        mock_faiss_vector_store.assert_called_once_with(
            index=mock_index,
            embedding_function=mock_embedding_model,
            docstore=store.docstore,
            index_to_docstore_id={}
        )  # Verify the FAISS vector store was initialized

    @patch("rag.vector_stores.faiss_store.FAISS")
    def test_add_documents(self, mock_faiss_vector_store):
        """Test the add_documents method."""
        # Mock FAISS vector store
        mock_vector_store_instance = MagicMock()
        mock_faiss_vector_store.return_value = mock_vector_store_instance

        # Initialize FAISSStore
        store = FAISSStore(embedding_model=MagicMock())

        # Call add_documents
        chunks = [{"content": "test document"}]
        store.add_documents(chunks)

        # Assertions
        mock_vector_store_instance.add_documents.assert_called_once_with(documents=chunks)

    @patch("rag.vector_stores.faiss_store.FAISS")
    def test_search(self, mock_faiss_vector_store):
        """Test the search method."""
        # Mock FAISS vector store
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.similarity_search.return_value = ["result1", "result2"]
        mock_faiss_vector_store.return_value = mock_vector_store_instance

        # Initialize FAISSStore
        store = FAISSStore(embedding_model=MagicMock())

        # Call search
        results = store.search(query="test query", top_k=2)

        # Assertions
        self.assertEqual(results, ["result1", "result2"])  # Verify the search results
        mock_vector_store_instance.similarity_search.assert_called_once_with("test query", k=2)

    @patch("rag.vector_stores.faiss_store.FAISS")
    def test_save_data_base(self, mock_faiss_vector_store):
        """Test the save_data_base method."""
        # Mock FAISS vector store
        mock_vector_store_instance = MagicMock()
        mock_faiss_vector_store.return_value = mock_vector_store_instance

        # Initialize FAISSStore
        store = FAISSStore(embedding_model=MagicMock())

        # Call save_data_base
        store.save_data_base("test_db")

        # Assertions
        mock_vector_store_instance.save_local.assert_called_once_with("test_db")

    def test_delete(self):
        """Test the delete method."""
        # Initialize FAISSStore
        store = FAISSStore(embedding_model=MagicMock())

        # Call delete and verify it raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            store.delete(["id1", "id2"])

if __name__ == "__main__":
    unittest.main()
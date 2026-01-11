import unittest
import pickle
import os
import faiss
import pandas as pd

VECTOR_STORE_DIR = os.path.join("..", "vector_store")

class TestTask2(unittest.TestCase):

    def setUp(self):
        # Load artifacts
        with open(os.path.join(VECTOR_STORE_DIR, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        with open(os.path.join(VECTOR_STORE_DIR, "chunk_metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        self.index = faiss.read_index(os.path.join(VECTOR_STORE_DIR, "faiss_index.bin"))

        self.df_meta = pd.DataFrame(self.metadata)

    def test_chunks_not_empty(self):
        """Check chunks list is not empty"""
        self.assertTrue(len(self.chunks) > 0, "Chunks list is empty!")

    def test_metadata_not_empty(self):
        """Check metadata list is not empty"""
        self.assertTrue(len(self.metadata) > 0, "Metadata list is empty!")

    def test_faiss_index(self):
        """Check FAISS index is not empty"""
        self.assertTrue(self.index.ntotal > 0, "FAISS index is empty!")

    def test_metadata_columns(self):
        """Check metadata contains required keys"""
        required_keys = ['complaint_id', 'product', 'chunk_index', 'total_chunks']
        for key in required_keys:
            self.assertIn(key, self.df_meta.columns, f"Missing metadata key: {key}")

    def test_chunks_match_metadata(self):
        """Check length of chunks and metadata are equal"""
        self.assertEqual(len(self.chunks), len(self.metadata),
                         "Number of chunks and metadata entries do not match!")

if __name__ == "__main__":
    unittest.main()

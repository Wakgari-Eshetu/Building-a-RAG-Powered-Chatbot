import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, TOP_K, TEXT_COLUMN

class Retriever:
    def __init__(self, parquet_path="../data/raw/complaint_embeddings.parquet"):
        # Load only the columns we need
        self.df = pd.read_parquet(parquet_path, columns=[TEXT_COLUMN, "embedding"])

        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Convert embeddings to float32 for FAISS
        embeddings = np.vstack(self.df["embedding"].values).astype("float32")

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, question: str, k: int = TOP_K):
        # Embed the question
        query_embedding = self.embedding_model.encode(
            [question], convert_to_numpy=True
        ).astype("float32")

        # Retrieve top-k similar chunks
        distances, indices = self.index.search(query_embedding, k)
        retrieved_chunks = self.df.iloc[indices[0]][TEXT_COLUMN].tolist()
        return retrieved_chunks

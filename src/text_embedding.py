# src/text_embedding.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "filtered_complaints.csv")
df = pd.read_csv(DATA_PATH)
print("Loaded cleaned dataset:", df.shape)


SAMPLE_SIZE = 15000  # total complaints for sampling

# Proportional stratified sampling by Product
stratified_sample = df.groupby('Product', group_keys=False).apply(
    lambda x: x.sample(frac=SAMPLE_SIZE/len(df), random_state=42)
)
print("Stratified sample shape:", stratified_sample.shape)
print("Product distribution:\n",stratified_sample['Product'].value_counts())

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into overlapping chunks."""
    text = str(text)
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

# Number of chunks per complaint
stratified_sample['num_chunks'] = stratified_sample['Consumer complaint narrative'].apply(
    lambda x: len(chunk_text(x, chunk_size=500, chunk_overlap=50))
)

print("Total chunks to generate (approx):", stratified_sample['num_chunks'].sum())


#Embedding Model Setup

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 384-dim embeddings

#Chunking and Embedding

metadata_fields = ['Complaint ID', 'Product', 'Issue', 'Sub-issue', 'Company']

all_embeddings = []
all_metadatas = []
all_chunks = []

for idx, row in tqdm(stratified_sample.iterrows(), total=stratified_sample.shape[0], desc="Processing complaints"):
    narrative = row['Consumer complaint narrative']
    chunks = chunk_text(narrative, chunk_size=500, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk)
        all_embeddings.append(embedding)

        metadata = {field: row[field] for field in metadata_fields}
        metadata['chunk_index'] = i
        metadata['total_chunks'] = len(chunks)
        all_metadatas.append(metadata)
        all_chunks.append(chunk)

print("Total chunks generated:", len(all_chunks))

# Create FAISS Vector Store

embedding_matrix = np.array(all_embeddings).astype('float32')
dimension = embedding_matrix.shape[1]

# Flat L2 index
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)
print("FAISS index created with", index.ntotal, "vectors")

#Save Vector Store and Metadata

# Save Vector Store and Metadata (FIXED PATH)

VECTOR_STORE_DIR = os.path.join(BASE_DIR, "..", "vector_store")
VECTOR_STORE_DIR = os.path.abspath(VECTOR_STORE_DIR)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

print("Saving vector store to:", VECTOR_STORE_DIR)

faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "faiss_index.bin"))

with open(os.path.join(VECTOR_STORE_DIR, "chunk_metadata.pkl"), "wb") as f:
    pickle.dump(all_metadatas, f)

with open(os.path.join(VECTOR_STORE_DIR, "chunks.pkl"), "wb") as f:
    pickle.dump(all_chunks, f)

print("Vector store and metadata saved successfully")

DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
DATA_DIR = os.path.abspath(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

stratified_sample.to_pickle(
    os.path.join(DATA_DIR, "stratified_sample.pkl")
)

print("Stratified sample saved to data/processed/stratified_sample.pkl")
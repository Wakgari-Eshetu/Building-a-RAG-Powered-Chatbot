# Intelligent Complaint Analysis – RAG-Powered Chatbot

## Overview

CrediTrust Financial serves East African markets and receives thousands of customer complaints monthly across **Credit Cards, Personal Loans, Savings Accounts, and Money Transfers**.  

This project builds a **Retrieval-Augmented Generation (RAG) chatbot** that transforms unstructured complaints into actionable insights, allowing internal teams to ask questions in plain English and get evidence-backed answers quickly.

---

## Features

- **Task 1 – EDA & Preprocessing**
  - Explore complaint distribution across products.
  - Visualize narrative lengths.
  - Filter and clean narratives.
  - Output: `data/processed/filtered_complaints.csv`

- **Task 2 – Text Chunking & Embedding**
  - Stratified sampling of complaints (~10K–15K).
  - Chunking of long narratives.
  - Vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
  - FAISS vector store with metadata.
  - Output: `vector_store/faiss_index.bin`, `chunks.pkl`, `chunk_metadata.pkl`

- **Visualization**
  - Product distribution and narrative lengths.
  - Chunks per complaint and per product.

- **Unit Tests**
  - Validate Task 1 dataset filtering, required columns, and no empty narratives.
  - Validate Task 2 chunking, metadata, and FAISS vector store.

---

## Installation

```bash
git clone <repo-url>
cd rag-complaint-chatbot
pip install -r requirements.txt

rag-complaint-chatbot/
├── data/                # Raw & processed complaint datasets
├── vector_store/        # FAISS index, chunks, metadata
├── notebooks/           # Jupyter notebooks for Task1 & Task2
├── src/                 # Scripts for preprocessing & embedding
├── tests/               # Unit tests for Task1 & Task2
├── app.py               # Gradio/Streamlit chatbot
├── requirements.txt
└── README.md

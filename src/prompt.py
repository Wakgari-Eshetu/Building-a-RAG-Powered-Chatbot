# src/prompt.py

def build_prompt(context: str, question: str) -> str:
    return f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.

Use ONLY the information provided in the context below.
If the context does not contain enough information to answer the question,
clearly state that you do not have enough information.

Context:
{context}

Question:
{question}

Answer:
""".strip()

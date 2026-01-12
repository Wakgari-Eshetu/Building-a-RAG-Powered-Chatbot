from transformers import pipeline
from src.config import LLM_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE


class Generator:
    def __init__(self):
        self.llm = pipeline(
            "text-generation",
            model=LLM_MODEL_NAME,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )

    def generate(self, prompt: str) -> str:
        response = self.llm(prompt)[0]["generated_text"]
        return response[len(prompt):].strip()

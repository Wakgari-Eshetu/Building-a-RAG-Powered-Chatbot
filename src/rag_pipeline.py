from src.retriever import Retriever
from src.prompt import build_prompt
from src.generator import Generator


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def run(self, question: str):
        # 1. Retrieve
        chunks = self.retriever.retrieve(question)

        # 2. Combine context
        context = "\n\n".join(chunks)

        # 3. Build prompt
        prompt = build_prompt(context=context, question=question)

        # 4. Generate answer
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": chunks[:2]  # show 1–2 sources for evaluation
        }

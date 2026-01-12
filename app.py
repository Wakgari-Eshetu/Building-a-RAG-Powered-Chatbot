import gradio as gr
from src.rag_pipeline import RAGPipeline  # Your working RAG pipeline


rag = RAGPipeline()


def answer_question(question):
    
    if not question.strip():
        return "Please enter a question.", ""

    # Run the RAG pipeline
    output = rag.run(question)  # Should return {"answer": str, "sources": [str]}
    answer = output["answer"]
    
    # Combine sources with separators for clarity
    sources = "\n\n---\n\n".join(output["sources"])
    
    return answer, sources


with gr.Blocks() as demo:
    gr.Markdown("## 📢 CrediTrust Complaint Assistant")
    gr.Markdown("Ask questions about customer complaints and see the sources used for answers.")
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="Type your question here...",
            lines=2
        )
        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")
    
    answer_output = gr.Textbox(
        label="AI-generated Answer",
        placeholder="The answer will appear here",
        lines=6
    )
    
    sources_output = gr.Textbox(
        label="Retrieved Sources",
        placeholder="The sources used to generate the answer will appear here",
        lines=8
    )
    
    # Connect Ask button
    ask_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    
    # Connect Clear button
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[answer_output, sources_output]
    )

# Launch the app
demo.launch(
    share=True  # Optional: set True to get a public link for demonstration
)

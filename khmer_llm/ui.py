"""Minimal customer-facing web page: Khmer question in, answer + sources out (Gradio)."""
from .runtime import ensure_package
from .stages.rag import KhmerRAG


def launch_ui(rag: KhmerRAG, share: bool = True):
    """share=True prints a public link (works on Kaggle / Colab)."""
    ensure_package("gradio")
    import gradio as gr

    def answer(question):
        if not question or not question.strip():
            return "", ""
        r = rag.ask(question.strip(), verbose=False)
        srcs = "\n\n".join(f"[{n}] {s['source']}\n{s['passage'][:300]}" for n, s in enumerate(r["sources"], 1))
        return r["answer"], srcs

    with gr.Blocks(title="Khmer Q&A") as demo:
        gr.Markdown("## សួរសំណួរជាភាសាខ្មែរ · Ask a question in Khmer")
        q = gr.Textbox(label="សំណួរ (Question)", lines=2)
        btn = gr.Button("សួរ (Ask)")
        a = gr.Textbox(label="ចម្លើយ (Answer)", lines=6)
        s = gr.Textbox(label="ប្រភព (Sources)", lines=8)
        btn.click(answer, q, [a, s])
        q.submit(answer, q, [a, s])
    demo.launch(share=share)

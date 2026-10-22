"""Minimal web page: Khmer customer message in, suggested replies (or one answer) + sources out (Gradio)."""
from .chat import STANCES
from .runtime import ensure_package
from .stages.rag import KhmerRAG

STANCE_LABELS = {"ok": "✅ ok (បាន)", "neutral": "⏳ neutral (ចាំមើលសិន)", "no": "❌ no (មិនបាន)"}


def launch_ui(rag: KhmerRAG, share: bool = True):
    """share=True prints a public link (works on Kaggle / Colab)."""
    ensure_package("gradio")
    import gradio as gr

    def respond(message, mode):
        if not message or not message.strip():
            return [""] * (len(STANCES) + 2)
        if mode == "plain":
            r = rag.ask(message.strip(), verbose=False)
            replies = [""] * len(STANCES)
            plain = r["answer"]
        else:
            r = rag.suggest_replies(message.strip(), verbose=False)
            replies = [r["replies"][s] for s in STANCES]
            plain = ""
        srcs = "\n\n".join(f"[{n}] {s['source']}\n{s['passage'][:300]}" for n, s in enumerate(r["sources"], 1))
        return [*replies, plain, srcs]

    with gr.Blocks(title="Khmer Q&A") as demo:
        gr.Markdown("## សារអតិថិជន · Customer message (Khmer)")
        msg = gr.Textbox(label="សារ (Message)", lines=2)
        mode = gr.Radio(["3 replies", "plain"], value="3 replies", label="Mode")
        btn = gr.Button("ឆ្លើយ (Reply)")
        with gr.Row():
            reply_boxes = [gr.Textbox(label=STANCE_LABELS.get(s, s), lines=3) for s in STANCES]
        plain = gr.Textbox(label="ចម្លើយ (Answer)", lines=4)
        srcs = gr.Textbox(label="ប្រភព (Sources)", lines=6)
        outputs = [*reply_boxes, plain, srcs]
        btn.click(respond, [msg, mode], outputs)
        msg.submit(respond, [msg, mode], outputs)
    demo.launch(share=share)

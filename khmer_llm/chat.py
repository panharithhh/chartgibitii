"""Chat template and inference helpers shared by SFT, RAG, the CLI and the web UI.

Template:  [<|system|> sys] [<|context|> ctx] <|user|> question <|assistant|> answer <eos>
"""
from typing import Optional

import torch

from .runtime import Runtime, autocast_ctx
from .text import clean_multiline

SFT_SYSTEM_PROMPT = ""   # optional Khmer system prompt prepended to every example


def neutralize_markers(text: str) -> str:
    """Stops chat markers typed by users or found in documents from becoming real role tokens."""
    return text.replace("<|", "< |")


def build_chat_example(tok, instruction: str, response: Optional[str] = None, context: str = "",
                       max_len: Optional[int] = None, system: str = ""):
    """
    Returns (ids, loss_mask) with mask = 1 only on answer tokens + EOS, or None if it cannot fit.
    Over-long examples lose the end of the context first, then the end of the answer.
    """
    enc = lambda text: tok.encode(neutralize_markers(text))
    head = (tok.marker("<|system|>") + enc(system)) if system else []
    ctx = (tok.marker("<|context|>") + enc(context)) if context else []
    tail = tok.marker("<|user|>") + enc(instruction) + tok.marker("<|assistant|>")
    resp = (enc(response) + [tok.eos_id]) if response is not None else []
    if max_len is not None:
        over = len(head) + len(ctx) + len(tail) + len(resp) - max_len
        if over > 0 and ctx:
            ctx = ctx[:len(ctx) - over] if len(ctx) - over > len(tok.marker("<|context|>")) + 8 else []
            over = len(head) + len(ctx) + len(tail) + len(resp) - max_len
        if over > 0 and response is not None:
            if len(resp) - over < 32:
                return None
            resp = resp[:len(resp) - over]
            over = 0
        if over > 0:
            return None
    prompt = head + ctx + tail
    return prompt + resp, [0] * len(prompt) + [1] * len(resp)


def generate(model, tok, prompt: str, rt: Runtime, max_new_tokens: int = 80,
             temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95) -> str:
    """Plain text continuation (base model)."""
    ids = [tok.eos_id] + tok.encode(prompt)   # EOS = document start, as in training data
    idx = torch.tensor([ids], dtype=torch.long, device=rt.device)
    out = model.generate(idx, max_new_tokens, temperature, top_k, top_p, eos_id=tok.eos_id,
                         amp=autocast_ctx(rt))
    return tok.decode(out[0, 1:].tolist())


def chat(model, tok, question: str, rt: Runtime, context: str = "", max_new_tokens: int = 200,
         temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.1) -> str:
    """Answers a Khmer question with the chat template (optionally grounded on `context`)."""
    question, context = clean_multiline(question), clean_multiline(context)
    L = model.cfg.context_length
    budget = L - max_new_tokens - 1
    built = build_chat_example(tok, question, None, context, max_len=budget, system=SFT_SYSTEM_PROMPT)
    if built is not None:
        prompt = [tok.eos_id] + built[0]
    else:   # question alone is longer than the window: keep its end
        user, asst = tok.marker("<|user|>"), tok.marker("<|assistant|>")
        keep = budget - len(user) - len(asst)
        prompt = [tok.eos_id] + user + tok.encode(neutralize_markers(question))[-keep:] + asst
    idx = torch.tensor([prompt], dtype=torch.long, device=rt.device)
    out = model.generate(idx, max_new_tokens, temperature, top_k, top_p, eos_id=tok.eos_id,
                         amp=autocast_ctx(rt), repetition_penalty=repetition_penalty)
    new = [i for i in out[0, len(prompt):].tolist() if i != tok.eos_id]
    return tok.decode(new).strip()

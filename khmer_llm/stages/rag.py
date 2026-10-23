"""Stage 3 — RAG: Khmer question in -> retrieve passages -> chat model answers from them, with sources."""
import json
import os
import re
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..chat import STANCES, build_chat_example, chat, system_text
from ..config import Config, with_architecture_of
from ..data.corpus import open_text
from ..data.bpe import load_tokenizer
from ..data.tokenizer import KhmerTokenizer
from ..model.lm import KimiMoELM, build_model
from ..runtime import Runtime, ensure_package, link_previous_outputs, setup_runtime
from ..text import chunk_text, clean_multiline, normalize_khmer
from ..train.checkpoint import atomic_torch_save, first_existing

RAG_DEMO_QUESTIONS = [
    "តើរាជធានីនៃប្រទេសកម្ពុជាគឺជាអ្វី?",            # What is the capital of Cambodia?
    "តើអង្គរវត្តមានទីតាំងនៅខេត្តណា?",               # Which province is Angkor Wat in?
    "តើពាក្យ «ចំណេះដឹង» មានន័យដូចម្តេច?",           # What does the word "knowledge" mean?
]


def rag_fallback_path(cfg: Config) -> Optional[str]:
    """Knowledge base when no rag_docs_dir: the corpus, or the tokenizer sample if the corpus was deleted."""
    return first_existing([cfg.corpus_path, cfg.tokenizer_sample_path])


def load_rag_documents(cfg: Config) -> List[Tuple[str, str]]:
    """(source, text) pairs from rag_docs_dir (.txt/.md/.jsonl), else consecutive corpus lines."""
    docs = []
    if cfg.rag_docs_dir and os.path.isdir(cfg.rag_docs_dir):
        for root, _, files in os.walk(cfg.rag_docs_dir):
            for name in sorted(files):
                path = os.path.join(root, name)
                rel = os.path.relpath(path, cfg.rag_docs_dir)
                try:
                    if name.endswith((".txt", ".md")):
                        with open(path, encoding="utf-8", errors="ignore") as f:
                            docs.append((rel, f.read()))
                    elif name.endswith(".jsonl"):
                        with open(path, encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f):
                                if not line.strip():
                                    continue
                                row = json.loads(line)
                                text = row.get("text") or row.get("content") or row.get("passage") or ""
                                title = row.get("title") or ""
                                if text:
                                    docs.append((f"{rel}:{row.get('source', i + 1)}",
                                                 f"{title}\n{text}" if title else text))
                except Exception as e:
                    print(f"[!] skipped {rel}: {e}")
        print(f"[📚] Loaded {len(docs):,} documents from {cfg.rag_docs_dir}")
    else:
        if cfg.rag_docs_dir:
            print(f"[!] rag_docs_dir {cfg.rag_docs_dir} not found; using the harvested corpus instead.")
        group, start = [], 0
        with open_text(rag_fallback_path(cfg)) as f:
            for i, line in enumerate(f):
                if i >= cfg.rag_max_corpus_lines:
                    break
                group.append(line.strip())
                if len(group) == 20:           # consecutive lines usually come from the same document
                    docs.append((f"corpus:{start + 1}-{i + 1}", "\n".join(group)))
                    group, start = [], i + 1
        if group:
            docs.append((f"corpus:{start + 1}-{start + len(group)}", "\n".join(group)))
        print(f"[📚] Using {len(docs):,} corpus blocks as the knowledge base (set Config.rag_docs_dir for your own docs)")
    return docs


class BM25Index:
    """Okapi BM25 over token ids (SentencePiece pieces work well for unsegmented Khmer)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.W = k1, b, None

    def build(self, token_lists: List[List[int]], vocab_size: int) -> "BM25Index":
        ensure_package("scipy")
        from scipy import sparse
        rows, cols, vals, lengths = [], [], [], np.zeros(len(token_lists), dtype=np.float32)
        for d, ids in enumerate(token_lists):
            if len(ids) == 0:
                continue
            terms, tf = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
            rows.append(np.full(len(terms), d, dtype=np.int64))
            cols.append(terms)
            vals.append(tf.astype(np.float32))
            lengths[d] = len(ids)
        n = len(token_lists)
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        tf = np.concatenate(vals) if vals else np.zeros(0, np.float32)
        df = np.bincount(cols, minlength=vocab_size).astype(np.float32)
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = max(1.0, float(lengths.mean())) if n else 1.0
        norm = self.k1 * (1 - self.b + self.b * lengths[rows] / avgdl)
        w = tf * (self.k1 + 1) / (tf + norm) * idf[cols]
        self.W = sparse.csc_matrix((w, (rows, cols)), shape=(n, vocab_size), dtype=np.float32)
        return self

    def search(self, query_ids: List[int], k: int) -> List[int]:
        q = np.unique(np.asarray([i for i in query_ids if 0 <= i < self.W.shape[1]], dtype=np.int64))
        if len(q) == 0:
            return []
        scores = np.asarray(self.W[:, q].sum(axis=1)).ravel()
        k = min(k, int((scores > 0).sum()))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])].tolist()

    def state(self) -> dict:
        return dict(k1=self.k1, b=self.b, data=self.W.data, indices=self.W.indices,
                    indptr=self.W.indptr, shape=self.W.shape)

    @classmethod
    def from_state(cls, s: dict) -> "BM25Index":
        ensure_package("scipy")
        from scipy import sparse
        idx = cls(s["k1"], s["b"])
        idx.W = sparse.csc_matrix((s["data"], s["indices"], s["indptr"]), shape=tuple(s["shape"]))
        return idx


class DenseEncoder:
    """Multilingual sentence embeddings via plain transformers (mean pooling + L2 norm for e5)."""

    def __init__(self, model_name: str, device: str):
        ensure_package("transformers")
        from transformers import AutoModel, AutoTokenizer
        self.name, self.device = model_name, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        if device.startswith("cuda"):
            self.model.half()
        low = model_name.lower()
        self.cls_pooling = "bge" in low
        self.q_prefix, self.p_prefix = ("query: ", "passage: ") if "e5" in low else ("", "")

    @torch.no_grad()
    def encode(self, texts: List[str], is_query: bool, batch_size: int = 64, max_length: int = 512) -> torch.Tensor:
        prefix = self.q_prefix if is_query else self.p_prefix
        order = np.argsort([-len(t) for t in texts], kind="stable")   # length-sorted batches = less padding
        out = torch.empty(len(texts), self.model.config.hidden_size, dtype=torch.float16)
        for i in range(0, len(texts), batch_size):
            ids = order[i:i + batch_size]
            enc = self.tokenizer([prefix + texts[j] for j in ids], padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            if self.cls_pooling:
                e = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[torch.as_tensor(ids)] = F.normalize(e.float(), dim=-1).cpu().half()
        return out


NGRAM_BUCKETS = 1 << 20
NON_WORD_RE = re.compile(r"[^\wក-៿᧠-᧿]+")   # \w alone misses Khmer vowel signs


def char_trigrams(text: str) -> np.ndarray:
    """Hashed character trigrams with spaces/punctuation removed: robust to how Khmer is spaced."""
    s = NON_WORD_RE.sub("", normalize_khmer(text).lower())
    cp = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    if len(cp) < 3:
        return cp % NGRAM_BUCKETS
    return ((cp[:-2] * 1000003 + cp[1:-1]) * 1000003 + cp[2:]) % NGRAM_BUCKETS


def lexical_terms(tok: KhmerTokenizer, text: str, ids: Optional[List[int]] = None) -> np.ndarray:
    """BM25 terms = canonical SentencePiece ids + (offset) hashed char trigrams."""
    ids = tok.encode(text) if ids is None else ids
    pieces = np.asarray(tok.retrieval_terms(ids), dtype=np.int64)
    return np.concatenate([pieces, tok.vocab_size + char_trigrams(text)])


def rrf_fuse(ranked_lists: List[List[int]], top_k: int, k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion of several ranked id lists."""
    scores: Dict[int, float] = {}
    for ranks in ranked_lists:
        for r, doc in enumerate(ranks):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


def _rag_fingerprint(cfg: Config, tok: KhmerTokenizer) -> dict:
    src = cfg.rag_docs_dir if cfg.rag_docs_dir and os.path.isdir(cfg.rag_docs_dir) else (rag_fallback_path(cfg) or "")
    if os.path.isdir(src):
        stats = [os.stat(os.path.join(r, f)) for r, _, fs in os.walk(src) for f in fs]
        mtime, size = max((s.st_mtime for s in stats), default=0), sum(s.st_size for s in stats)
    else:
        mtime = os.path.getmtime(src) if os.path.exists(src) else 0
        size = os.path.getsize(src) if os.path.exists(src) else 0
    return dict(src=src, mtime=mtime, size=size, max_lines=cfg.rag_max_corpus_lines, chunk=cfg.rag_chunk_chars,
                overlap=cfg.rag_chunk_overlap_chars, dense=cfg.rag_dense_model, tokenizer=tok.fingerprint,
                lexical="pieces+trigrams-v1")


class KhmerRAG:
    """Khmer question in -> retrieve passages -> chat model answers from them, with sources."""

    def __init__(self, cfg: Config, rt: Runtime, model: KimiMoELM, tok: KhmerTokenizer, chunks: List[str],
                 sources: List[str], bm25: BM25Index, encoder: Optional[DenseEncoder] = None,
                 dense: Optional[torch.Tensor] = None):
        self.cfg, self.rt, self.model, self.tok = cfg, rt, model, tok
        self.chunks, self.sources, self.bm25, self.encoder = chunks, sources, bm25, encoder
        self.dense = None
        if dense is not None:
            self.dense = dense.to(rt.device) if rt.device_type == "cuda" else dense.float()

    @classmethod
    def build(cls, cfg: Config, rt: Runtime, model: KimiMoELM, tok: KhmerTokenizer,
              prefer_existing_index: bool = False) -> "KhmerRAG":
        """Loads the cached index when it matches, else (re)builds it.
        prefer_existing_index: with no rag_docs_dir given, reuse whatever index exists (ask / ui)."""
        fp = _rag_fingerprint(cfg, tok)
        saved = torch.load(cfg.rag_index_path, map_location="cpu", weights_only=False) \
            if os.path.exists(cfg.rag_index_path) else None
        saved_fp = (saved or {}).get("fingerprint") or {}
        reuse = saved is not None and (saved_fp == fp or (prefer_existing_index and not cfg.rag_docs_dir
                                                          and saved_fp.get("tokenizer") == tok.fingerprint))
        encoder = _load_encoder(saved_fp.get("dense") if reuse else cfg.rag_dense_model, rt)
        if reuse:
            print(f"[✓] RAG index loaded: {len(saved['chunks']):,} passages from {saved_fp.get('src')}")
            dense = saved.get("dense") if encoder is not None else None
            if encoder is not None and dense is None:
                dense = encoder.encode(saved["chunks"], is_query=False)
            return cls(cfg, rt, model, tok, saved["chunks"], saved["sources"],
                       BM25Index.from_state(saved["bm25"]), encoder, dense)

        print("\n[🔎] Building RAG index...")
        chunks, sources, seen = [], [], set()
        for src, text in load_rag_documents(cfg):
            for c in chunk_text(text, cfg.rag_chunk_chars, cfg.rag_chunk_overlap_chars):
                if len(c) >= 20 and c not in seen:
                    seen.add(c)
                    chunks.append(c)
                    sources.append(src)
        if not chunks:
            raise RuntimeError("RAG knowledge base is empty: put .txt/.md/.jsonl files in Config.rag_docs_dir.")
        terms = [lexical_terms(tok, c, ids) for c, ids in zip(chunks, tok.encode_batch(chunks))]
        bm25 = BM25Index().build(terms, tok.vocab_size + NGRAM_BUCKETS)
        dense = None
        if encoder is not None:
            t0 = time.time()
            dense = encoder.encode(chunks, is_query=False)
            print(f"     dense embeddings: {tuple(dense.shape)} in {time.time() - t0:.0f}s")
        atomic_torch_save(dict(fingerprint=fp, chunks=chunks, sources=sources, bm25=bm25.state(), dense=dense),
                          cfg.rag_index_path)
        kind = "BM25" + (f" + dense ({cfg.rag_dense_model})" if dense is not None else "")
        print(f"[✓] RAG index: {len(chunks):,} passages | {kind} -> {cfg.rag_index_path}")
        return cls(cfg, rt, model, tok, chunks, sources, bm25, encoder, dense)

    @classmethod
    def load(cls, cfg: Optional[Config] = None) -> "KhmerRAG":
        """Rebuilds the whole Q&A stack from files in work_dir (notebook / CLI / web UI).
        Model shape and vocab come from the saved weights, so CLI defaults never mismatch them."""
        cfg = cfg or Config()
        link_previous_outputs(cfg)
        rt = setup_runtime(allow_ddp=False)
        model = load_chat_model(cfg, rt)
        cfg = with_architecture_of(cfg, asdict(model.cfg))
        return cls.build(cfg, rt, model, load_tokenizer(cfg), prefer_existing_index=True)

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[int]:
        question = clean_multiline(question)
        k = top_k or self.cfg.rag_top_k
        n_cand = max(50, 10 * k)
        ranked = [self.bm25.search(lexical_terms(self.tok, question), n_cand)]
        if self.encoder is not None and self.dense is not None:
            q = self.encoder.encode([question], is_query=True)[0].to(self.dense.device, self.dense.dtype)
            sims = self.dense @ q
            ranked.append(torch.topk(sims.float(), min(n_cand, len(sims))).indices.tolist())
        return rrf_fuse(ranked, k)

    def build_context(self, question: str, hits: List[int], max_new_tokens: int,
                      system: str = "") -> Tuple[str, List[int]]:
        """Numbered passages cut to fit the window with the question and answer; also returns the hits used."""
        q_len = len(build_chat_example(self.tok, question, None, "", system=system)[0])
        budget = min(self.cfg.rag_max_context_tokens, self.model.cfg.context_length - max_new_tokens - q_len - 8)
        parts, used_hits, used = [], [], 0
        for n, i in enumerate(hits, 1):
            piece = f"[{n}] {self.chunks[i]}"
            ids = self.tok.encode(piece)
            if used + len(ids) > budget:
                if not parts and budget > 16:   # first passage alone is too long: keep its beginning
                    parts.append(self.tok.decode(ids[:budget]))
                    used_hits.append(i)
                break
            parts.append(piece)
            used_hits.append(i)
            used += len(ids) + 1
        return "\n".join(parts), used_hits

    def ask(self, question: str, top_k: Optional[int] = None, max_new_tokens: Optional[int] = None,
            temperature: float = 0.3, verbose: bool = True, stance: Optional[str] = None) -> dict:
        """One answer. stance="ok" / "neutral" / "no" picks a reply style learned from shop-reply data."""
        return self.suggest_replies(question, stances=(stance,), top_k=top_k, max_new_tokens=max_new_tokens,
                                    temperature=temperature, verbose=verbose, _single=True)

    def suggest_replies(self, question: str, stances=STANCES, top_k: Optional[int] = None,
                        max_new_tokens: Optional[int] = None, temperature: float = 0.3, verbose: bool = True,
                        _single: bool = False) -> dict:
        """One reply per style for the same question and retrieved passages, e.g. for a seller to pick from."""
        question = clean_multiline(question)
        max_new = max_new_tokens or self.cfg.rag_max_new_tokens
        hits = self.retrieve(question, top_k)
        longest = max((system_text(s) for s in stances), key=len)
        context, used_hits = self.build_context(question, hits, max_new, system=longest)
        replies = {s: chat(self.model, self.tok, question, self.rt, context=context, max_new_tokens=max_new,
                           temperature=temperature, top_p=0.9, stance=s) for s in stances}
        sources = [dict(source=self.sources[i], passage=self.chunks[i]) for i in used_hits]
        result = dict(question=question, sources=sources,
                      **(dict(answer=next(iter(replies.values()))) if _single else dict(replies=replies)))
        if verbose:
            print(f"\n❓ {question}")
            for s, reply in replies.items():
                print(f"💬 {'' if s is None else f'[{s}] '}{reply}")
            for n, src in enumerate(sources, 1):
                print(f"   [{n}] {src['source']}: {src['passage'][:120]}...")
        return result


def _load_encoder(name: Optional[str], rt: Runtime) -> Optional[DenseEncoder]:
    if not name:
        return None
    try:
        return DenseEncoder(name, rt.device)
    except Exception as e:
        print(f"[!] Dense retriever unavailable ({type(e).__name__}: {str(e)[:150]}); using BM25 only.")
        return None


def load_chat_model(cfg: Config, rt: Runtime) -> KimiMoELM:
    """SFT weights if present (else the base model), built with the architecture saved alongside them."""
    path = first_existing([cfg.sft_final_path, cfg.sft_ckpt_path])
    if path is None:
        path = first_existing([cfg.pretrained_ckpt, cfg.final_path, cfg.ckpt_path])
        if path is None:
            raise FileNotFoundError("No model weights found: run the 'pretrain' (and 'sft') stages first.")
        print(f"[!] No SFT model found; RAG uses the base model {path} (answers will be weak).")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(with_architecture_of(cfg, ckpt.get("config") or {}), rt)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    print(f"[✓] Q&A model loaded from {path}")
    return model


def run_rag(cfg: Config, rt: Runtime, tok: KhmerTokenizer) -> KhmerRAG:
    model = load_chat_model(cfg, rt)
    rag = KhmerRAG.build(with_architecture_of(cfg, asdict(model.cfg)), rt, model, tok)
    print("\n" + "=" * 78)
    print("💬 RAG DEMO — Khmer question in, grounded answer + sources out")
    print("=" * 78)
    for q in RAG_DEMO_QUESTIONS:
        rag.ask(q)
    print('\nAsk your own:  RESULTS["rag"].ask("<Khmer question>")')
    return rag

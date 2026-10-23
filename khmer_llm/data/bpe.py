"""Byte-level BPE tokenizer (HF `tokenizers`, Llama-3 / GPT-4 style) with a Khmer-aware split regex,
plus the factory that picks a tokenizer by Config.tokenizer_type.

Tokenizer types:
  "unigram"  SentencePiece Unigram + byte fallback (default)
  "sp_bpe"   SentencePiece BPE + byte fallback (Llama-2 / Gemma / SeaLLM style; no pre-split needed)
  "byte_bpe" byte-level BPE over regex chunks (Llama-3 / Qwen / GPT-4 style) with KHMER_SPLIT_PATTERN
"""
import os
import zlib
from typing import Iterable, List, Optional

from ..config import Config
from ..runtime import ensure_package
from .tokenizer import CHAT_SYMBOLS, KhmerTokenizer, _iter_gz_lines


def _cls(*ranges) -> str:
    return "".join(chr(a) if a == b else f"{chr(a)}-{chr(b)}" for a, b in ranges)


KHMER_ALL = _cls((0x1780, 0x17FF), (0x19E0, 0x19FF))
KHMER_WORD = _cls((0x1780, 0x17D3), (0x17D7, 0x17D7), (0x17DC, 0x17DD), (0x19E0, 0x19FF))   # letters, vowels, signs, coeng, ៗ
KHMER_DIGITS = _cls((0x17E0, 0x17E9), (0x17F0, 0x17F9))
KHMER_PUNCT = _cls((0x17D4, 0x17D6), (0x17D8, 0x17DB))   # ។ ៕ ៖ ៘ ៙ ៚ ៛

# Khmer split pattern. Khmer vowel signs, coeng (subscript marker) and other signs are Unicode category
# M, not L, so GPT-4's \p{L}+ cuts every syllable apart ("ខ្ញុំ" -> "ខ", "្ញ", "ុំ"). Here a whole
# Khmer run (consonants + vowels + signs + coeng + ៗ) is one chunk; Khmer has no spaces between
# words, so a chunk is a phrase and BPE learns syllables/words inside it, like SentencePiece does.
# Numerals are capped at 3 digits and punctuation stands alone; other scripts keep combining marks.
KHMER_SPLIT_PATTERN = (
    rf" ?[{KHMER_WORD}]+"
    rf"| ?[{KHMER_DIGITS}]{{1,3}}"
    rf"| ?[{KHMER_PUNCT}]"
    r"|'(?i:[sdmt]|ll|ve|re)"
    rf"|[^\r\n\p{{L}}\p{{N}}\p{{M}}]?(?:(?![{KHMER_ALL}])[\p{{L}}\p{{M}}])+"
    r"|\p{N}{1,3}"
    rf"| ?(?:(?![{KHMER_ALL}])[^\s\p{{L}}\p{{N}}\p{{M}}])+[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)
# GPT-4 (cl100k) pattern, for comparison only: it breaks Khmer clusters (see above).
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIALS = ["<s>", "</s>", "<pad>"] + CHAT_SYMBOLS   # ids at the top of the vocab, never produced by encode()


def _pre_tokenizer(pattern: str):
    from tokenizers import Regex, pre_tokenizers
    return pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
                                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])


def train_byte_bpe(lines: Iterable[str], path: str, vocab_size: int, pattern: str = KHMER_SPLIT_PATTERN) -> int:
    """Trains byte-level BPE with vocab_size - len(SPECIALS) entries and saves tokenizer.json at `path`."""
    ensure_package("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = _pre_tokenizer(pattern)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size - len(SPECIALS), min_frequency=2, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    tk.train_from_iterator(lines, trainer=trainer)
    tk.save(path + ".tmp")
    os.replace(path + ".tmp", path)
    return tk.get_vocab_size()


class KhmerByteBPETokenizer:
    """Byte-level BPE with the same interface as KhmerTokenizer. Specials (EOS, pad, chat markers) are
    managed here at the top ids, so text that contains "</s>" or "<|user|>" can never produce them."""

    def __init__(self, cfg: Config, corpus_path: Optional[str] = None):
        ensure_package("tokenizers")
        from tokenizers import Tokenizer
        self.model_file = cfg.tokenizer_prefix + ".json"
        self.vocab_size = cfg.vocab_size
        if not os.path.exists(self.model_file):
            assert corpus_path is not None, "byte-BPE tokenizer missing and no corpus to train on"
            print(f"\n[🔧] Training byte-level BPE tokenizer ({cfg.vocab_size:,} vocab, Khmer regex)...")
            lines = _iter_gz_lines(corpus_path) if corpus_path.endswith(".gz") else \
                (l.rstrip("\n") for l in open(corpus_path, encoding="utf-8") if l.strip())
            n = train_byte_bpe(lines, self.model_file, cfg.vocab_size)
            print(f"[✓] Saved {self.model_file} ({n:,} learned tokens + {len(SPECIALS)} specials)")
        self.hf = Tokenizer.from_file(self.model_file)
        n_learned = self.hf.get_vocab_size()
        first_special = self.vocab_size - len(SPECIALS)
        assert n_learned <= first_special, f"tokenizer has {n_learned} tokens; vocab_size {self.vocab_size} too small"
        self.special_ids = {s: first_special + i for i, s in enumerate(SPECIALS)}
        self.bos_id, self.eos_id, self.pad_id = (self.special_ids[s] for s in ("<s>", "</s>", "<pad>"))
        self.unk_id = None
        self.marker_ids = {s: self.special_ids[s] for s in CHAT_SYMBOLS}
        with open(self.model_file, "rb") as f:
            self.fingerprint = f"{zlib.crc32(f.read()):08x}"
        vocab = self.hf.get_vocab()                         # byte-level strings; "Ġ" = leading space
        self._canonical = {i: vocab[t[1:]] for t, i in vocab.items() if t.startswith("Ġ") and t[1:] in vocab}
        self._retrieval_skip = set(self.special_ids.values()) | {vocab.get("Ġ", -1)}

    def encode(self, text: str) -> List[int]:
        return self.hf.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self.hf.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: List[int]) -> str:
        return self.hf.decode([i for i in ids if i < self.hf.get_vocab_size()], skip_special_tokens=False)

    def marker(self, symbol: str) -> List[int]:
        return [self.marker_ids[symbol]]

    def retrieval_terms(self, ids: List[int]) -> List[int]:
        """BM25 terms: no specials / bare space, and " word" folded to "word"."""
        return [self._canonical.get(i, i) for i in ids if i not in self._retrieval_skip]


TOKENIZER_TYPES = ("unigram", "sp_bpe", "byte_bpe")


def load_tokenizer(cfg: Config, corpus_path: Optional[str] = None):
    """Loads (or trains from corpus_path) the tokenizer selected by cfg.tokenizer_type."""
    if cfg.tokenizer_type in ("unigram", "sp_bpe"):
        return KhmerTokenizer(cfg, corpus_path)
    if cfg.tokenizer_type == "byte_bpe":
        return KhmerByteBPETokenizer(cfg, corpus_path)
    raise ValueError(f"unknown tokenizer_type {cfg.tokenizer_type!r}; use one of {TOKENIZER_TYPES}")

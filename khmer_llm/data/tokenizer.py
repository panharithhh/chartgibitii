"""SentencePiece tokenizer (Unigram, or BPE with tokenizer_type="sp_bpe"), byte fallback, atomic chat markers."""
import gzip
import os
import zlib
from typing import List, Optional

import sentencepiece as spm

from ..config import Config

CHAT_SYMBOLS = ["<|system|>", "<|context|>", "<|user|>", "<|assistant|>"]


def _iter_gz_lines(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.rstrip("\n")


class KhmerTokenizer:
    def __init__(self, cfg: Config, corpus_path: Optional[str] = None):
        self.model_file = cfg.tokenizer_prefix + ".model"
        if not os.path.exists(self.model_file):
            assert corpus_path is not None, "tokenizer missing and no corpus to train on"
            model_type = "bpe" if cfg.tokenizer_type == "sp_bpe" else "unigram"
            print(f"\n[🔧] Training SentencePiece {model_type} tokenizer ({cfg.vocab_size:,} vocab)...")
            source = dict(input=corpus_path) if not corpus_path.endswith(".gz") else \
                dict(sentence_iterator=_iter_gz_lines(corpus_path))   # SentencePiece cannot read .gz itself
            spm.SentencePieceTrainer.train(
                **source,
                model_prefix=cfg.tokenizer_prefix,
                vocab_size=cfg.vocab_size,
                model_type=model_type,
                character_coverage=0.9995,
                byte_fallback=True,
                split_digits=True,
                unk_id=0, bos_id=1, eos_id=2, pad_id=3,
                user_defined_symbols=CHAT_SYMBOLS,   # atomic chat markers for SFT / RAG
                input_sentence_size=cfg.tokenizer_sample_lines,
                shuffle_input_sentence=True,
                max_sentence_length=16384,          # Khmer = 3 bytes/char in UTF-8
                num_threads=os.cpu_count() or 4,
            )
            print(f"[✓] Saved {self.model_file}")
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)
        self.vocab_size = self.sp.get_piece_size()
        self.unk_id = self.sp.unk_id()
        self.bos_id, self.eos_id, self.pad_id = self.sp.bos_id(), self.sp.eos_id(), self.sp.pad_id()
        self.marker_ids = {s: self.sp.piece_to_id(s) for s in CHAT_SYMBOLS}
        missing = [s for s, i in self.marker_ids.items() if i == self.unk_id]
        if missing:   # tokenizer from an older run: markers still work, just as several pieces
            print(f"[!] Tokenizer has no atomic chat markers {missing}; they will be encoded as text.")
        # identifies the trained model (not just its path), so caches built with another tokenizer are rebuilt
        self.fingerprint = f"{zlib.crc32(self.sp.serialized_model_proto()):08x}"
        skip = {self.unk_id, self.bos_id, self.eos_id, self.pad_id, self.sp.piece_to_id("▁")}
        skip.update(i for i in self.marker_ids.values() if i != self.unk_id)
        self._retrieval_skip = skip
        # "▁word" (word after a space / at text start) -> "word", so retrieval matches either position
        self._canonical = list(range(self.vocab_size))
        for i in range(self.vocab_size):
            piece = self.sp.id_to_piece(i)
            if piece.startswith("▁") and len(piece) > 1:
                j = self.sp.piece_to_id(piece[1:])
                if j != self.unk_id:
                    self._canonical[i] = j

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        try:
            return self.sp.encode(texts, out_type=int, num_threads=os.cpu_count() or 4)
        except TypeError:   # old sentencepiece without num_threads
            return [self.sp.encode(t, out_type=int) for t in texts]

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    def marker(self, symbol: str) -> List[int]:
        i = self.marker_ids[symbol]
        return [i] if i != self.unk_id else self.encode(symbol)

    def retrieval_terms(self, ids: List[int]) -> List[int]:
        """BM25 terms: ids without control ids / chat markers / bare '▁', with '▁word' folded to 'word'."""
        return [self._canonical[i] for i in ids if i not in self._retrieval_skip]

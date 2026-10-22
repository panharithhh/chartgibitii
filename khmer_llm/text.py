"""Khmer text cleaning, filtering, line splitting and RAG chunking."""
import re
import unicodedata
from typing import List

KHMER_RE = re.compile(r"[ក-៿᧠-᧿]")
ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t 　]+")
SENT_END_RE = re.compile(r"(?<=[។៕?!])")
CHUNK_SPLIT_RE = re.compile(r"(?<=[។៕?!])\s*|\n+")
KHMER_SIGN_RE = re.compile(r"[ា-៓៝]")   # dependent vowels / signs: never start a piece
COENG = "្"                                        # subscript marker: never end a piece
MAX_LINE_CHARS = 2000


def normalize_khmer(text: str) -> str:
    """NFC, drop zero-width chars and control chars (newlines kept)."""
    text = unicodedata.normalize("NFC", str(text))
    return CTRL_RE.sub(" ", ZERO_WIDTH_RE.sub("", text))


def clean_multiline(text) -> str:
    """normalize_khmer + collapse spaces on each line; keeps line breaks."""
    if not isinstance(text, str):
        return ""
    text = normalize_khmer(text)
    return "\n".join(SPACE_RE.sub(" ", line).strip() for line in text.split("\n")).strip()


def khmer_ratio(s: str) -> float:
    return len(KHMER_RE.findall(s)) / max(1, len(s))


def is_valid_khmer(s: str) -> bool:
    if len(s) < 10:
        return False
    kh = len(KHMER_RE.findall(s))
    return kh >= 5 and kh / len(s) >= 0.25


def text_to_lines(text: str) -> List[str]:
    """Clean a document and split it into paragraph-level lines (keeps '។' punctuation)."""
    out = []
    for para in normalize_khmer(text).split("\n"):
        para = SPACE_RE.sub(" ", para).strip()
        if not para:
            continue
        chunks = [para]
        if len(para) > MAX_LINE_CHARS:
            chunks, cur = [], ""
            for sent in SENT_END_RE.split(para):
                if cur and len(cur) + len(sent) > MAX_LINE_CHARS:
                    chunks.append(cur.strip())
                    cur = ""
                cur += sent
            if cur.strip():
                chunks.append(cur.strip())
        out.extend(c[:MAX_LINE_CHARS] for c in chunks if is_valid_khmer(c))
    return out


def cluster_floor(s: str, i: int) -> int:
    """Largest index <= i that does not split a Khmer grapheme cluster."""
    while 0 < i < len(s) and (KHMER_SIGN_RE.match(s[i]) or s[i - 1] == COENG):
        i -= 1
    return i


def safe_cut(s: str, limit: int) -> int:
    """Where to cut an over-long Khmer sentence: last phrase space in the 2nd half, else a cluster boundary."""
    space = s.rfind(" ", limit // 2, limit + 1)
    return space if space > 0 else max(1, cluster_floor(s, limit))


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    """Sentence-aware chunking for Khmer (no spaces between words; sentences end with ។)."""
    sents = []
    for s in CHUNK_SPLIT_RE.split(normalize_khmer(text)):
        s = SPACE_RE.sub(" ", s).strip()
        while len(s) > chunk_chars:            # split very long sentences at phrase / cluster boundaries
            cut = safe_cut(s, chunk_chars)
            sents.append(s[:cut].rstrip())
            nxt = cluster_floor(s, max(1, cut - overlap_chars))
            s = s[nxt if 0 < nxt < cut else cut:].lstrip()
        if s:
            sents.append(s)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur and cur_len + len(s) + 1 > chunk_chars:
            chunks.append(" ".join(cur))
            keep, kept = [], 0                 # carry trailing sentences as overlap
            for prev in reversed(cur):
                if kept + len(prev) > overlap_chars:
                    break
                keep.insert(0, prev)
                kept += len(prev) + 1
            cur, cur_len = keep, kept
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

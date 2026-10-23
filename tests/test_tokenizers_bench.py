"""Khmer split regex, SentencePiece BPE / byte-level BPE tokenizers, and the tokenizer benchmark."""
import gzip
import json
import os
import random

import pytest

from khmer_llm.config import Config
from khmer_llm.data.bpe import GPT4_SPLIT_PATTERN, KHMER_SPLIT_PATTERN, SPECIALS, load_tokenizer
from khmer_llm.text import COENG, KHMER_SIGN_RE

TEXT = "សួស្តី! ខ្ញុំរស់នៅក្នុងភ្នំពេញ។ តម្លៃ ១២៥០០៛ ផ្សេងៗ ok, it's GPT-4 2024!\nបន្ទាត់ថ្មី (ក្រុមហ៊ុន)"


def test_khmer_regex_keeps_clusters_whole_in_both_engines():
    regex = pytest.importorskip("regex")
    tokenizers = pytest.importorskip("tokenizers")
    chunks = regex.findall(KHMER_SPLIT_PATTERN, TEXT)
    assert "".join(chunks) == TEXT and "\n" not in KHMER_SPLIT_PATTERN
    assert " ខ្ញុំរស់នៅក្នុងភ្នំពេញ" in chunks and "។" in chunks and " ផ្សេងៗ" in chunks
    assert " ១២៥" in chunks and "០០" in chunks          # numerals capped at 3 digits
    assert all(not KHMER_SIGN_RE.match(c.lstrip()) and not c.endswith(COENG) for c in chunks)
    onig = tokenizers.pre_tokenizers.Split(tokenizers.Regex(KHMER_SPLIT_PATTERN), behavior="isolated")
    assert [p for p, _ in onig.pre_tokenize_str(TEXT)] == chunks   # HF (Oniguruma) splits identically
    gpt4 = regex.findall(GPT4_SPLIT_PATTERN, "ខ្ញុំរស់នៅ")
    assert any(KHMER_SIGN_RE.match(c) for c in gpt4)            # GPT-4's pattern breaks Khmer clusters


def _khmer_lines(n=4000, seed=0):
    rnd = random.Random(seed)
    cons, vow = "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ", "ាិីឹឺុូួើឿៀេែៃោៅ"
    words = ["".join(rnd.choice(cons) + rnd.choice(vow) + rnd.choice(["", "្" + rnd.choice(cons)])
                     for _ in range(rnd.randint(1, 3))) for _ in range(500)]
    return [" ".join(rnd.choice(words) for _ in range(rnd.randint(4, 20))) + "។" for _ in range(n)]


def _write_sample(work, lines):
    os.makedirs(work, exist_ok=True)
    for name in ("khmer_corpus.txt.gz", "khmer_tokenizer_sample.txt.gz"):
        with gzip.open(os.path.join(work, name), "wt", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(work, "khmer_corpus.txt.gz.done"), "w") as f:
        f.write("{}")


def test_byte_bpe_tokenizer_roundtrip_specials_and_retrieval(tmp_path):
    pytest.importorskip("tokenizers")
    cfg = Config(work_dir=str(tmp_path), tokenizer_type="byte_bpe", vocab_size=600)
    _write_sample(cfg.work_dir, _khmer_lines())
    tok = load_tokenizer(cfg, cfg.tokenizer_sample_path)
    assert tok.model_file.endswith("khmer_byte_bpe_0k.json") and os.path.exists(tok.model_file)
    for text in [TEXT, "ok", "  spaces  and\ttabs", "😀 emoji ✓"]:
        assert tok.decode(tok.encode(text)) == text                # byte-level: lossless for any text
    ids = tok.encode("</s> <|user|> ក")
    assert all(i < cfg.vocab_size - len(SPECIALS) for i in ids)   # specials never come from plain text
    assert tok.eos_id == cfg.vocab_size - len(SPECIALS) + 1 and tok.marker("<|assistant|>") == [cfg.vocab_size - 1]
    assert tok.decode(ids + [tok.eos_id]) == tok.decode(ids)
    a, b = tok.encode("ការ"), tok.encode(" ការ")
    assert a != b and tok.retrieval_terms(b) == tok.retrieval_terms(a)   # " word" folded to "word"
    assert load_tokenizer(cfg).fingerprint == tok.fingerprint


def test_sentencepiece_bpe_option(tmp_path):
    cfg = Config(work_dir=str(tmp_path), tokenizer_type="sp_bpe", vocab_size=600)
    _write_sample(cfg.work_dir, _khmer_lines())
    tok = load_tokenizer(cfg, cfg.tokenizer_sample_path)
    assert tok.model_file.endswith("khmer_sp_bpe_0k.model") and tok.sp.get_piece_size() == 600
    assert tok.decode(tok.encode("ខ្ញុំរស់នៅ។")) == "ខ្ញុំរស់នៅ។"
    assert cfg.train_bin.endswith("train_sp_bpe_600.bin") and Config(vocab_size=600).train_bin.endswith("train_600.bin")
    with pytest.raises(ValueError):
        load_tokenizer(Config(work_dir=str(tmp_path), tokenizer_type="nope"))


def test_benchmark_runs_all_tokenizers_and_ranks_by_bpb(tmp_path):
    pytest.importorskip("tokenizers")
    from khmer_llm.pipeline import main
    cfg = Config(work_dir=str(tmp_path / "w"), vocab_size=600, context_length=64, d_model=64, n_layers=2, n_heads=4,
                 n_kv_heads=1, n_routed_experts=4, top_k=2, d_ff_shared=64, d_ff_expert=16, batch_size=4,
                 grad_accum_steps=1, bench_steps=6, warmup_iters=2, log_interval=3, eval_iters=2,
                 val_every_n_lines=20, run_self_test=False, tokenizer_sample_lines=10_000)
    _write_sample(cfg.work_dir, _khmer_lines())
    results = main(cfg, stages="bench")["bench"]
    assert set(results) == {"unigram", "sp_bpe", "byte_bpe"}
    for r in results.values():
        assert r["steps"] == 6 and 0 < r["bpb"] < 20 and r["chars_per_token"] > 0.3
    saved = json.load(open(os.path.join(cfg.work_dir, "bench_results.json"), encoding="utf-8"))
    assert set(saved) == set(results)
    with open(os.path.join(cfg.work_dir, "bench_byte_bpe", "train_byte_bpe_600.bin.json")) as f:
        meta = json.load(f)
    with open(os.path.join(cfg.work_dir, "bench_unigram", "train_600.bin.json")) as f:
        assert json.load(f)["val_bytes"] == meta["val_bytes"] > 0   # identical held-out text for every tokenizer

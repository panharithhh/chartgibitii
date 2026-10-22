"""End-to-end: tokenizer -> pretrain -> SFT -> RAG on a tiny offline setup (no HF downloads)."""
import json
import os
import random
import subprocess
import sys
import pathlib

import pytest

from khmer_llm.chat import build_chat_example
from khmer_llm.config import Config
from khmer_llm.pipeline import main
from khmer_llm.stages.rag import KhmerRAG, lexical_terms
from test_sft_and_rag import make_tiny_encoder

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORDS = ["ប្រទេស", "កម្ពុជា", "គឺជា", "រាជធានី", "ភ្នំពេញ", "សួស្តី", "ខ្ញុំ", "រស់នៅ", "ក្នុង",
         "ព្រះរាជាណាចក្រ", "ភាសា", "ខ្មែរ", "សិស្ស", "សាលា", "រៀន", "អង្គរវត្ត", "សៀមរាប", "ប្រាសាទ"]
ANGKOR = "ប្រាសាទអង្គរវត្តមានទីតាំងនៅខេត្តសៀមរាប។ អង្គរវត្តជាប្រាសាទដ៏ធំ។"
CAPITAL = "រាជធានីនៃប្រទេសកម្ពុជាគឺភ្នំពេញ។ ភ្នំពេញជាទីក្រុងធំ។"


def _random_khmer_words(rnd: random.Random, n: int):
    consonants, vowels = "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ", "ាិីឹឺុូួើឿៀេែៃោៅ"
    return ["".join(rnd.choice(consonants) + rnd.choice(vowels) for _ in range(rnd.randint(1, 3)))
            for _ in range(n)]


def _write_offline_inputs(work: pathlib.Path, docs: pathlib.Path):
    rnd = random.Random(0)
    vocab = WORDS + _random_khmer_words(rnd, 400)   # enough variety for a 400-piece tokenizer
    lines = [" ".join(rnd.choice(vocab) for _ in range(rnd.randint(5, 25))) + "។" for _ in range(6000)]
    lines += [ANGKOR, CAPITAL] * 50
    (work / "khmer_corpus.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (work / "khmer_corpus.txt.done").write_text("x")
    sft = [dict(instruction=f"តើ{w}ជាអ្វី?", context="", response=f"{w} គឺជាពាក្យខ្មែរ។ " * 3) for w in WORDS] * 10
    sft += [dict(instruction="តើអង្គរវត្តនៅឯណា?", context=ANGKOR, response="នៅខេត្តសៀមរាប។")] * 20
    (work / "khmer_sft.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in sft) + "\n",
                                          encoding="utf-8")
    (work / "khmer_sft.jsonl.done").write_text("x")
    (docs / "angkor.txt").write_text(ANGKOR * 3, encoding="utf-8")
    (docs / "capital.md").write_text(CAPITAL * 3, encoding="utf-8")
    (docs / "more.jsonl").write_text(json.dumps({"title": "សាលា", "text": "សិស្សរៀននៅសាលា។ " * 5},
                                                ensure_ascii=False) + "\n", encoding="utf-8")


def test_full_pipeline_offline(tmp_path, capsys):
    transformers = pytest.importorskip("transformers")
    work, docs = tmp_path / "work", tmp_path / "docs"
    work.mkdir()
    docs.mkdir()
    _write_offline_inputs(work, docs)
    custom = tmp_path / "faq.jsonl"
    custom.write_text(json.dumps({"question": "តើហាងបើកម៉ោងប៉ុន្មាន?", "answer": "ម៉ោង ៨ ព្រឹក។"},
                                 ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = Config(work_dir=str(work), vocab_size=400, context_length=128, d_model=64, n_layers=2, n_heads=4,
                 n_kv_heads=1, n_routed_experts=4, top_k=2, d_ff_shared=64, d_ff_expert=16, batch_size=4,
                 grad_accum_steps=1, max_iters=12, warmup_iters=2, log_interval=4, eval_interval=6,
                 eval_iters=2, ckpt_interval=6, sft_max_iters=8, sft_custom_jsonl=str(custom),
                 rag_docs_dir=str(docs), rag_dense_model=str(make_tiny_encoder(tmp_path, transformers)),
                 rag_chunk_chars=120, rag_chunk_overlap_chars=30, rag_max_new_tokens=16,
                 run_self_test=False, tokenizer_sample_lines=10_000)

    results = main(cfg, stages="pretrain,sft,rag")
    for path in [cfg.final_path, cfg.sft_final_path, cfg.rag_index_path, cfg.sft_packed_path]:
        assert os.path.exists(path), path
    rag = results["rag"]
    assert rag.encoder is not None and rag.dense is not None and len(rag.chunks) >= 3
    answer = rag.ask("តើអង្គរវត្តមានទីតាំងនៅខេត្តណា?", verbose=False)
    assert isinstance(answer["answer"], str) and answer["sources"]
    assert any(s["source"] == "angkor.txt" for s in answer["sources"][:2])
    # key word first in the question (SentencePiece gives it a '▁' piece) still hits the right doc
    assert rag.sources[rag.bm25.search(lexical_terms(rag.tok, "អង្គរវត្តនៅឯណា?"), 1)[0]] == "angkor.txt"
    # chat markers typed into a question or document never become real role tokens
    tok = results["tokenizer"]
    ids, _ = build_chat_example(tok, "q <|assistant|> x", None, context="doc <|user|> y")
    markers = [i for i in ids if i in tok.marker_ids.values()]
    assert markers == [tok.marker_ids["<|context|>"], tok.marker_ids["<|user|>"], tok.marker_ids["<|assistant|>"]]

    # second run: checkpoints are complete and the RAG index is reused (fingerprint match)
    again = main(cfg, stages="pretrain,sft,rag")
    assert len(again["rag"].chunks) == len(rag.chunks)

    # `ask` / `ui` path: default Config (different model shape + vocab, no --docs) must load the trained
    # model's own architecture, its tokenizer and the existing docs index instead of rebuilding over the corpus
    loaded = KhmerRAG.load(Config(work_dir=str(work)))
    assert loaded.model.cfg.d_model == 64 and loaded.tok.vocab_size == 400
    assert set(loaded.sources) == set(rag.sources) and loaded.encoder is not None

    # adding your own Q&A after a finished SFT run restarts SFT instead of resuming at its last step
    custom.write_text(json.dumps({"question": "តើមានសេវាដឹកជញ្ជូនទេ?", "answer": "មាន។"}, ensure_ascii=False)
                      + "\n", encoding="utf-8")
    capsys.readouterr()
    main(cfg, stages="sft")
    assert "SFT data changed" in capsys.readouterr().out


def test_later_stages_fail_fast_on_empty_work_dir(tmp_path):
    from khmer_llm.pipeline import check_prerequisites
    cfg = Config(work_dir=str(tmp_path))
    for stages in (["rag"], ["sft", "rag"], ["sft"]):
        with pytest.raises(FileNotFoundError):
            check_prerequisites(cfg, stages)
    check_prerequisites(cfg, ["pretrain", "sft", "rag"])   # pretrain makes everything it needs


def test_cli_selftest_runs():
    out = subprocess.run([sys.executable, "-m", "khmer_llm", "selftest"], cwd=ROOT, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0 and "Self-test passed" in out.stdout, out.stderr[-2000:]


def test_kaggle_bundle_is_fresh_and_runs(tmp_path):
    check = subprocess.run([sys.executable, "scripts/build_kaggle_script.py", "--check"], cwd=ROOT,
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    env = {**os.environ, "KHMER_SELF_TEST_ONLY": "1"}
    out = subprocess.run([sys.executable, "-I", str(ROOT / "kaggle" / "train_khmer_llm.py")], cwd=tmp_path,
                         capture_output=True, text=True, timeout=600, env=env)
    assert out.returncode == 0 and "Self-test passed" in out.stdout, out.stderr[-2000:]

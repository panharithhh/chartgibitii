import pytest

from khmer_llm.config import Config, apply_overrides
from khmer_llm.text import chunk_text, is_valid_khmer, khmer_ratio, normalize_khmer, text_to_lines

KH = "ប្រទេសកម្ពុជាមានរាជធានីភ្នំពេញ"


def test_normalize_drops_zero_width_and_controls():
    assert normalize_khmer("ក​ខ\x07គ") == "កខ គ"


def test_text_to_lines_keeps_khmer_full_stop():
    lines = text_to_lines(f"{KH}។\n\n hello world \n{KH}។")
    assert lines == [f"{KH}។", f"{KH}។"]   # Latin-only line filtered, ។ kept


def test_text_to_lines_splits_long_paragraphs():
    para = "។".join([KH] * 200) + "។"
    lines = text_to_lines(para)
    assert len(lines) > 1 and all(len(l) <= 2000 for l in lines)


def test_khmer_filters():
    assert is_valid_khmer(KH) and not is_valid_khmer("hello world, this is English")
    assert khmer_ratio(KH) == 1.0


def test_chunk_text_respects_size_and_overlaps():
    text = "។ ".join([KH + str(i) for i in range(40)]) + "។"
    chunks = chunk_text(text, chunk_chars=150, overlap_chars=50)
    assert len(chunks) > 3 and all(len(c) <= 150 for c in chunks)
    assert all(set(a.split(" ")) & set(b.split(" ")) for a, b in zip(chunks, chunks[1:]))


def test_config_overrides_coerce_types():
    cfg = apply_overrides(Config(), ["max_iters=10", "lr=1e-3", "compile=true", "rag_dense_model=none",
                                     "adam_betas=0.8,0.9", "sft_max_iters=5", "rag_docs_dir=docs/"])
    assert cfg.max_iters == 10 and cfg.lr == 1e-3 and cfg.compile is True
    assert cfg.rag_dense_model is None and cfg.adam_betas == (0.8, 0.9)
    assert cfg.sft_max_iters == 5 and cfg.rag_docs_dir == "docs/"
    with pytest.raises(ValueError):
        apply_overrides(Config(), ["not_a_field=1"])


def test_chunking_never_splits_khmer_clusters():
    from khmer_llm.text import COENG, KHMER_SIGN_RE
    word = "ស្ត្រី" + "ក្សត្រ" + "ព្រះ"          # coeng stacks + dependent vowels, no spaces
    chunks = chunk_text(word * 60, chunk_chars=97, overlap_chars=23)
    assert len(chunks) > 3
    for c in chunks:
        assert not KHMER_SIGN_RE.match(c[0]) and c[-1] != COENG and len(c) <= 97


def test_harvest_writes_gz_corpus_with_budget_dedup_and_sample(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    import gzip
    import json
    import khmer_llm.data.corpus as corpus
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps({"text": f"{KH}{i}។\n{KH}។"}) for i in range(300)), encoding="utf-8")
    b.write_text("\n".join(json.dumps({"body": f"{KH}ខ{i}។"}) for i in range(300)), encoding="utf-8")
    monkeypatch.setattr(corpus, "HARVEST_SOURCES", [
        dict(name="A", loads=[dict(path="json", data_files=str(a))], fields=["text"]),
        dict(name="B", loads=[dict(path="json", data_files=str(b))], fields=None),   # unknown schema
        dict(name="C", loads=[dict(path="json", data_files=str(a))], fields=["text"]),
    ])
    cfg = Config(work_dir=str(tmp_path), max_corpus_chars=12_000, tokenizer_sample_lines=50)
    corpus.harvest_khmer_corpus(cfg)
    lines = list(corpus.iter_lines(cfg.corpus_path))
    assert len(lines) == len(set(lines)) and f"{KH}។" in lines            # exact dedup across rows
    assert 12_000 <= sum(map(len, lines)) < 12_000 + 2 * 40               # stops at the char budget
    stats = json.load(open(cfg.corpus_path + ".done"))
    assert stats["sources"]["A"]["lines"] > 0 and "C" not in stats["sources"]   # budget reached before C
    with gzip.open(cfg.tokenizer_sample_path, "rt", encoding="utf-8") as f:
        sample = f.read().split("\n")[:-1]
    assert len(sample) == 50 and set(sample) <= set(lines)
    corpus.harvest_khmer_corpus(cfg)   # second call is a no-op (done flag)


def test_flatten_with_eos():
    from khmer_llm.data.corpus import _flatten_with_eos
    assert _flatten_with_eos([[5, 6], [7], [8, 9, 10]], 2).tolist() == [5, 6, 2, 7, 2, 8, 9, 10, 2]
    assert _flatten_with_eos([], 2).tolist() == []


def test_parse_json_spec_and_madlad_unescape(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    import json
    import khmer_llm.data.corpus as corpus
    from khmer_llm.data.hf import open_hf_stream
    nested = tmp_path / "hplt.jsonl"   # nested / varying fields, like HPLT 3.0
    nested.write_text("\n".join([json.dumps({"text": f"{KH}០។", "pii": [[1, 2]], "doc_scores": [0.5]}),
                                 json.dumps({"text": f"{KH}១។", "web-register": {"a": 1}})]), encoding="utf-8")
    label, rows = open_hf_stream([dict(path="text", data_files=[str(nested)], parse_json=True)], require=["text"])
    assert "1 files" in label and [r["text"] for r in rows] == [f"{KH}០។", f"{KH}១។"]
    madlad = tmp_path / "madlad.jsonl"
    madlad.write_text(json.dumps({"text": f"{KH}ក។\\n{KH}ខ។ &amp; ok"}), encoding="utf-8")   # literal backslash-n
    monkeypatch.setattr(corpus, "HARVEST_SOURCES", [
        dict(name="M", loads=[dict(path="json", data_files=str(madlad))], fields=["text"], unescape=True)])
    cfg = Config(work_dir=str(tmp_path / "w"))
    import os
    os.makedirs(cfg.work_dir)
    corpus.harvest_khmer_corpus(cfg)
    assert list(corpus.iter_lines(cfg.corpus_path)) == [f"{KH}ក។", f"{KH}ខ។ & ok"]


def test_import_dir_accepts_several_folders_newest_first(tmp_path):
    import os
    from khmer_llm.runtime import link_previous_outputs
    new, old = tmp_path / "gpu-run", tmp_path / "data-run"
    new.mkdir()
    old.mkdir()
    (new / "ckpt.pt").write_text("new")
    (old / "ckpt.pt").write_text("old")
    (old / "train_16000.bin").write_text("bins")
    (old / "x.pt.tmp").write_text("partial")
    cfg = Config(work_dir=str(tmp_path / "work"), import_dir=f"{new}, {old}, {tmp_path / 'missing'}")
    link_previous_outputs(cfg)
    assert open(cfg.path("ckpt.pt")).read() == "new" and open(cfg.path("train_16000.bin")).read() == "bins"
    assert not os.path.exists(cfg.path("x.pt.tmp"))

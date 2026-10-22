import json

import numpy as np
import pytest

from khmer_llm.selftest import ToyTokenizer, check_chat_template, check_rag_components, check_sft_steps
from khmer_llm.stages.rag import BM25Index, rrf_fuse
from khmer_llm.stages.sft import pack_examples, read_custom_examples, sft_config


def test_chat_template_and_truncation(cfg):
    check_chat_template(cfg)


def test_sft_packing_masking_and_learning(cfg, rt):
    check_sft_steps(cfg, rt)


def test_packing_never_splits_an_example():
    tok = ToyTokenizer(128)
    examples = [([10 + i] * n, [0] * (n - 2) + [1, 1]) for i, n in enumerate([30, 30, 30, 5, 60])]
    toks, masks = pack_examples(examples, 65, tok.pad_id, tok.eos_id)
    assert toks.shape == (3, 65) and int(masks.sum()) == 10
    for row in toks:
        ids = [t for t in row[1:] if t != tok.pad_id]
        assert all(ids.count(v) in (5, 30, 60) for v in set(ids))


def test_custom_examples_accept_aliases(tmp_path):
    p = tmp_path / "faq.jsonl"
    rows = [{"question": "តើហាងបើកម៉ោងប៉ុន្មាន?", "answer": "ម៉ោង ៨ ព្រឹក។"},
            {"instruction": "q", "response": "a", "context": "c"}, {"instruction": "", "response": "x"}]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n\n", encoding="utf-8")
    ex = read_custom_examples(str(p))
    assert len(ex) == 2 and ex[0]["response"] == "ម៉ោង ៨ ព្រឹក។" and ex[1]["context"] == "c"


def test_sft_iters_follow_epochs(cfg):
    scfg = sft_config(cfg, n_train_rows=160, world_size=1)   # 8 x 2 rows per step
    assert scfg.max_iters == 20 and scfg.lr == cfg.sft_lr and scfg.lr_schedule == "cosine"
    assert sft_config(cfg.__class__(**{**cfg.__dict__, "sft_max_iters": 7}), 160, 1).max_iters == 7


def test_rag_components():
    check_rag_components()


def test_bm25_prefers_rare_terms_and_handles_empty():
    docs = [[1, 2, 3], [1, 1, 1, 4], [5]]
    idx = BM25Index().build(docs, vocab_size=8)
    assert idx.search([4], 3) == [1]
    assert idx.search([7], 3) == [] and idx.search([], 3) == []
    assert idx.search([1, 5], 3)[0] == 2   # rare term 5 beats common term 1


def test_rrf_handles_disjoint_lists():
    assert set(rrf_fuse([[1], [2]], top_k=5)) == {1, 2}


def test_dense_encoder_with_local_model(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from khmer_llm.stages.rag import DenseEncoder
    model_dir = make_tiny_encoder(tmp_path, transformers)
    enc = DenseEncoder(str(model_dir), "cpu")
    texts = ["ក ខ គ", "ឃ", "ក ខ"]
    emb = enc.encode(texts, is_query=False)
    assert emb.shape == (3, 32)
    assert torch.allclose(emb.float().norm(dim=-1), torch.ones(3), atol=1e-2)
    single = enc.encode([texts[1]], is_query=False)
    assert torch.allclose(emb[1].float(), single[0].float(), atol=1e-2)   # order restored after length sort


def make_tiny_encoder(tmp_path, transformers):
    """A random 1-layer BERT + char vocab saved locally, so dense retrieval runs offline."""
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [chr(c) for c in range(0x1780, 0x17FF)] \
        + list("abcdefghijklmnopqrstuvwxyz0123456789")
    d = tmp_path / "tiny-encoder"
    d.mkdir()
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    transformers.BertTokenizer(str(d / "vocab.txt"), do_lower_case=False).save_pretrained(str(d))
    config = transformers.BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=512)
    transformers.BertModel(config).save_pretrained(str(d))
    return d


def test_extract_pair_handles_common_schemas():
    from khmer_llm.stages.sft import extract_pair
    auto = dict(instruction=None, context=None, response=None)
    assert extract_pair({"instruction": "q", "input": "c", "output": "a"}, auto) == dict(instruction="q", context="c", response="a")
    assert extract_pair({"prompt": "q", "completion": "a"}, auto)["response"] == "a"
    msgs = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                         {"role": "assistant", "content": "a"}]}
    assert extract_pair(msgs, auto) == dict(instruction="q", context="", response="a")
    sharegpt = {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}
    assert extract_pair(sharegpt, auto)["response"] == "a"
    assert extract_pair({"text": "only text"}, auto) is None
    explicit = dict(instruction="instruction", context="input", response="output")
    assert extract_pair({"instruction": "q", "output": "a"}, explicit)["context"] == ""


def test_extract_qa_crops_context_around_answer():
    from khmer_llm.stages.sft import extract_qa
    filler = "នេះជាប្រយោគមិនពាក់ព័ន្ធ។" * 80
    context = filler + "រាជធានីគឺភ្នំពេញ។" + filler
    ex = extract_qa({"question": "តើរាជធានីគឺអ្វី?", "context": context, "answers": {"text": ["ភ្នំពេញ"]}}, window=300)
    assert ex is not None and "ភ្នំពេញ" in ex["context"] and len(ex["context"]) <= 300
    assert ex["response"] == "ភ្នំពេញ"
    assert extract_qa({"question": "q", "context": "abc", "answers": {"text": ["zzz"]}}) is None
    assert extract_qa({"question": "q", "context": "abc", "answers": {"text": []}}) is None


def test_open_hf_stream_checks_schema_and_falls_back(tmp_path):
    pytest.importorskip("datasets")
    from khmer_llm.data.hf import open_hf_stream
    wrong, right = tmp_path / "wrong.jsonl", tmp_path / "right.jsonl"
    wrong.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
    right.write_text("\n".join(json.dumps({"instruction": f"q{i}", "output": "a"}) for i in range(3)) + "\n",
                     encoding="utf-8")
    loads = [dict(path="definitely/not-a-real-dataset-xyz"), dict(path="json", data_files=str(wrong)),
             dict(path="json", data_files=str(right))]
    label, rows = open_hf_stream(loads, require=["instruction", "output"])
    assert "right.jsonl" in label and [r["instruction"] for r in rows] == ["q0", "q1", "q2"]
    with pytest.raises(RuntimeError):
        open_hf_stream(loads[:2], require=["instruction"])


def test_iter_json_file_reads_gz_arrays_and_jsonl(tmp_path):
    import gzip
    from khmer_llm.data.hf import iter_json_file
    arr = tmp_path / "km.json.gz"
    with gzip.open(arr, "wt", encoding="utf-8") as f:
        json.dump([{"instruction": "q", "output": "a"}] * 2, f)
    lines = tmp_path / "rows.jsonl"
    lines.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert len(list(iter_json_file(str(arr)))) == 2 and [r["a"] for r in iter_json_file(str(lines))] == [1, 2]

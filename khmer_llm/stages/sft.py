"""Stage 2 — post-training (SFT): Khmer instruction data, chat template, answer-only loss."""
import json
import math
import os
import time
import zlib
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ..chat import SFT_SYSTEM_PROMPT, build_chat_example, chat
from ..config import Config
from ..data.hf import maybe_hf_login, open_hf_stream
from ..data.sources import SFT_SOURCES
from ..data.tokenizer import KhmerTokenizer
from ..model.lm import build_model
from ..runtime import Runtime, barrier, free_memory, time_left
from ..text import clean_multiline, khmer_ratio
from ..train.checkpoint import first_existing, load_model_weights, save_inference_weights
from ..train.loop import any_rank, run_training, wrap_model

SFT_SAMPLE_QUESTIONS = [
    "តើរាជធានីនៃប្រទេសកម្ពុជាគឺជាអ្វី?",          # What is the capital of Cambodia?
    "សូមពន្យល់ពីសារៈសំខាន់នៃការអប់រំ។",           # Explain the importance of education.
]


QA_CONTEXT_CHARS = 700   # passage window kept around the answer for QA-with-context data
PAIR_SCHEMAS = [("instruction", "output", "input"), ("instruction", "response", "input"),
                ("prompt", "response", None), ("prompt", "completion", None), ("question", "answer", "context"),
                ("inputs", "targets", None), ("input", "output", None)]
CHAT_COLUMNS = ("messages", "conversations", "conversation")


def _make_example(instruction, response, context="") -> Optional[dict]:
    ex = dict(instruction=clean_multiline(instruction), context=clean_multiline(context),
              response=clean_multiline(response))
    return ex if ex["instruction"] and ex["response"] else None


def extract_pair(row: dict, src: dict) -> Optional[dict]:
    """Instruction/response example from explicit columns, a chat `messages` list, or a known schema."""
    if src.get("instruction"):
        return _make_example(row.get(src["instruction"]), row.get(src["response"]),
                             row.get(src["context"]) if src.get("context") else "")
    for col in CHAT_COLUMNS:
        turns = row.get(col)
        if isinstance(turns, list):
            role = lambda t: (t.get("role") or t.get("from") or "").lower() if isinstance(t, dict) else ""
            user = next((t for t in turns if role(t) in ("user", "human")), None)
            asst = next((t for t in turns if role(t) in ("assistant", "gpt", "model", "bot")), None)
            if user and asst:
                return _make_example(user.get("content") or user.get("value"), asst.get("content") or asst.get("value"))
    for q, a, c in PAIR_SCHEMAS:
        if isinstance(row.get(q), str) and isinstance(row.get(a), str):
            return _make_example(row[q], row[a], (row.get(c) or "") if c else "")
    return None


def crop_around(context: str, answer: str, window: int) -> Optional[str]:
    """~window chars of context containing the answer, snapped to sentence ends; None if absent."""
    i = context.find(answer)
    if i < 0:
        return None
    if len(context) <= window:
        return context
    start = max(0, min(i - (window - len(answer)) // 2, len(context) - window))
    end = min(len(context), start + window)
    b = context.find("។", start, i)                       # begin after a sentence end if one is close
    if b != -1 and b - start < window // 4:
        start = b + 1
    e = context.rfind("។", i + len(answer), end)          # stop at a sentence end if one is close
    if e != -1 and end - e < window // 4:
        end = e + 1
    return context[start:end].strip()


def extract_qa(row: dict, window: int = QA_CONTEXT_CHARS) -> Optional[dict]:
    """SQuAD-style row -> (question, cropped passage as context, answer)."""
    answers = row.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text") or []
    elif isinstance(answers, list):
        texts = [a.get("text") if isinstance(a, dict) else a for a in answers]
    else:
        texts = [answers]
    answer = next((t for t in texts if isinstance(t, str) and t.strip()), None)
    context = row.get("context")
    if not answer or not isinstance(context, str):
        return None
    passage = crop_around(context, answer, window)
    return None if passage is None else _make_example(row.get("question"), answer, passage)


def harvest_sft_examples(cfg: Config) -> str:
    path, done_flag = cfg.sft_jsonl_path, cfg.sft_jsonl_path + ".done"
    if os.path.exists(path) and os.path.exists(done_flag) and os.path.getsize(path) > 0:
        print(f"[✓] SFT examples found at {path}. Skipping download.")
        return path
    maybe_hf_login()
    print("\n[📥] Downloading Khmer instruction (SFT) datasets...")
    seen, total = set(), 0
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for src in SFT_SOURCES:
            budget = cfg.sft_max_examples - total
            if budget <= 0:
                break
            if src.get("cap"):
                budget = min(budget, src["cap"])
            n_src, t0 = 0, time.time()
            try:
                label, ds = open_hf_stream(src["loads"], src.get("require"))
                print(f"  -> {src['name']} [{label}] (license: {src.get('license', 'see dataset card')})")
                for row in ds:
                    ex = extract_qa(row) if src.get("kind") == "qa" else extract_pair(row, src)
                    if ex is None or khmer_ratio(ex["instruction"] + ex["response"]) < 0.2:
                        continue   # keep Khmer examples only
                    key = hash((ex["instruction"], ex["context"], ex["response"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_src += 1
                    if n_src >= budget:
                        break
            except Exception as e:
                print(f"  -> {src['name']}: [x] stopped: {type(e).__name__}: {str(e)[:300]}")
            total += n_src
            print(f"     [✓] +{n_src:,} examples | total {total:,} | {time.time() - t0:.0f}s")
    if total == 0 and not cfg.sft_custom_jsonl:
        raise RuntimeError("No SFT examples downloaded. Turn on Internet, edit SFT_SOURCES "
                           "or set Config.sft_custom_jsonl.")
    os.replace(path + ".tmp", path)
    if total > 0:   # an empty (offline) download must not block fetching next time
        with open(done_flag + ".tmp", "w") as fl:
            fl.write(str(total))
        os.replace(done_flag + ".tmp", done_flag)
    return path


def read_custom_examples(path: str) -> List[dict]:
    """Your own Q&A, one JSON object per line: instruction/question, response/answer/output, context/input."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ex = _make_example(row.get("instruction") or row.get("question"),
                               row.get("response") or row.get("answer") or row.get("output"),
                               row.get("context") or row.get("input") or "")
            if ex is not None:
                out.append(ex)
    return out


def pack_examples(examples, row_len: int, pad_id: int, eos_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy packing into fixed rows of row_len tokens; each row starts with EOS (document start)."""
    rows_t, rows_m = [], []
    cur_t, cur_m = [eos_id], [0]
    for ids, mask in examples:
        if len(cur_t) + len(ids) > row_len:
            pad = row_len - len(cur_t)
            rows_t.append(cur_t + [pad_id] * pad)
            rows_m.append(cur_m + [0] * pad)
            cur_t, cur_m = [eos_id], [0]
        cur_t += ids
        cur_m += mask
    if len(cur_t) > 1:
        pad = row_len - len(cur_t)
        rows_t.append(cur_t + [pad_id] * pad)
        rows_m.append(cur_m + [0] * pad)
    return (np.asarray(rows_t, dtype=np.uint16).reshape(-1, row_len),
            np.asarray(rows_m, dtype=np.uint8).reshape(-1, row_len))


def _sft_fingerprint(cfg: Config, tok: KhmerTokenizer) -> str:
    custom = cfg.sft_custom_jsonl
    stat = os.stat(custom) if custom and os.path.exists(custom) else None
    return json.dumps(dict(custom=custom, custom_mtime=stat.st_mtime if stat else None,
                           custom_size=stat.st_size if stat else None, repeat=cfg.sft_custom_repeat,
                           ctx=cfg.context_length, tokenizer=tok.fingerprint, system=SFT_SYSTEM_PROMPT,
                           hf_size=os.path.getsize(cfg.sft_jsonl_path)))


def prepare_sft_data(cfg: Config, tok: KhmerTokenizer):
    fingerprint = _sft_fingerprint(cfg, tok)
    if os.path.exists(cfg.sft_packed_path):
        with np.load(cfg.sft_packed_path) as old:
            if "fingerprint" in old.files and str(old["fingerprint"]) == fingerprint:
                print(f"[✓] Packed SFT data found: {cfg.sft_packed_path}")
                return
    print("\n[🔄] Tokenizing + packing SFT examples...")
    train_ex, val_ex, dropped = [], [], 0

    def build(ex):
        nonlocal dropped
        built = build_chat_example(tok, ex["instruction"], ex["response"], ex.get("context", ""),
                                   max_len=cfg.context_length, system=SFT_SYSTEM_PROMPT)
        dropped += built is None
        return built

    with open(cfg.sft_jsonl_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="SFT examples", unit=" ex", mininterval=10):
            ex = json.loads(line)
            built = build(ex)
            if built is not None:
                is_val = zlib.crc32(ex["instruction"].encode("utf-8")) % cfg.sft_val_every_n == 0
                (val_ex if is_val else train_ex).append(built)
    rng = np.random.default_rng(cfg.seed)
    train_ex = [train_ex[i] for i in rng.permutation(len(train_ex))]
    if len(val_ex) < 8 and train_ex:   # small download: move a few distinct examples to validation
        n_val = min(max(8, len(train_ex) // 50), len(train_ex) // 2)
        val_ex, train_ex = val_ex + train_ex[:n_val], train_ex[n_val:]
    custom = []
    if cfg.sft_custom_jsonl:
        custom = [b for b in map(build, read_custom_examples(cfg.sft_custom_jsonl)) if b is not None]
        train_ex += [b for b in custom for _ in range(cfg.sft_custom_repeat)]   # all custom data is trained on
        print(f"     + {len(custom):,} custom examples x{cfg.sft_custom_repeat} from {cfg.sft_custom_jsonl}")
        if not val_ex:   # custom-only: monitor on (copies of) a few custom examples
            val_ex = custom[:8]
    if not train_ex:
        raise RuntimeError("No usable SFT examples (all empty or longer than context_length).")
    train_ex = [train_ex[i] for i in rng.permutation(len(train_ex))]
    tr_t, tr_m = pack_examples(train_ex, cfg.context_length + 1, tok.pad_id, tok.eos_id)
    va_t, va_m = pack_examples(val_ex or train_ex[:8], cfg.context_length + 1, tok.pad_id, tok.eos_id)
    tmp = cfg.sft_packed_path + ".tmp"
    with open(tmp, "wb") as f:   # write-then-rename (see atomic_torch_save)
        np.savez(f, train_tokens=tr_t, train_mask=tr_m, val_tokens=va_t, val_mask=va_m,
                 fingerprint=np.array(fingerprint))
    os.replace(tmp, cfg.sft_packed_path)
    print(f"[✓] SFT: {len(train_ex):,} train / {len(val_ex):,} val examples -> {len(tr_t):,} / {len(va_t):,} "
          f"rows of {cfg.context_length} | dropped {dropped:,} too-long | answer tokens {int(tr_m.sum()):,}")


class SFTLoader:
    """Shuffled epochs over packed rows; targets outside the answer mask are -100."""

    def __init__(self, tokens: np.ndarray, mask: np.ndarray, batch_size: int, device: str, seed: int):
        assert len(tokens) > 0, "no packed SFT rows"
        self.tokens, self.mask, self.B, self.device = tokens, mask.astype(bool), batch_size, device
        self.pin = device.startswith("cuda")
        self.rng = np.random.default_rng(seed)
        self.order = np.empty(0, dtype=np.int64)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.order) < self.B:
            reps = max(1, -(-self.B // len(self.tokens)))
            self.order = np.concatenate([self.order] + [self.rng.permutation(len(self.tokens)) for _ in range(reps)])
        idx, self.order = self.order[:self.B], self.order[self.B:]
        t = torch.from_numpy(self.tokens[idx].astype(np.int64))
        m = torch.from_numpy(self.mask[idx])
        x, y = t[:, :-1], t[:, 1:].masked_fill(~m[:, 1:], -100)
        if self.pin:
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def sft_config(cfg: Config, n_train_rows: int, world_size: int) -> Config:
    """SFT hyperparameters as a Config copy, so the shared training loop can be reused."""
    rows_per_step = cfg.batch_size * cfg.grad_accum_steps * world_size
    iters = cfg.sft_max_iters or max(1, math.ceil(cfg.sft_epochs * n_train_rows / rows_per_step))
    return replace(cfg, lr=cfg.sft_lr, max_iters=iters, warmup_iters=min(cfg.sft_warmup_iters, iters // 4 + 1),
                   lr_schedule="cosine", eval_interval=max(1, min(cfg.eval_interval, iters // 4)),
                   ckpt_interval=max(1, min(cfg.ckpt_interval, iters // 2)), eval_iters=min(cfg.eval_iters, 10))


def run_sft(cfg: Config, rt: Runtime, tok: KhmerTokenizer):
    if any_rank(not time_left(cfg), rt):
        if rt.master:
            print("[⏱] No time left for SFT; skipping (rerun with stages 'sft,rag' to continue).")
        return
    if rt.master:
        harvest_sft_examples(cfg)
        prepare_sft_data(cfg, tok)
        licenses = sorted({s.get("license") for s in SFT_SOURCES if s.get("license")})
        if licenses:
            print(f"[!] SFT datasets licensed {', '.join(licenses)}: check terms before commercial use.")
    barrier(rt)
    with np.load(cfg.sft_packed_path) as data:
        tr_t, tr_m, va_t, va_m = data["train_tokens"], data["train_mask"], data["val_tokens"], data["val_mask"]
        data_fingerprint = str(data["fingerprint"])

    # An SFT checkpoint only resumes if it was trained on this exact data (e.g. you added your own Q&A
    # after a finished run): otherwise SFT restarts from the base model so the new data gets a full run.
    data_marker = cfg.sft_ckpt_path + ".data"
    resume = cfg.resume and os.path.exists(cfg.sft_ckpt_path)
    previous = None
    if os.path.exists(data_marker):
        with open(data_marker) as f:
            previous = f.read()
    if resume and previous != data_fingerprint:
        resume = False
        if rt.master:
            print("[↻] SFT data changed since the last SFT run; restarting SFT from the base model.")
    barrier(rt)   # every rank decided before the marker is rewritten
    if rt.master:
        with open(data_marker + ".tmp", "w") as f:
            f.write(data_fingerprint)
        os.replace(data_marker + ".tmp", data_marker)
    scfg = replace(sft_config(cfg, len(tr_t), rt.world_size), resume=resume)

    raw_model = build_model(cfg, rt)
    if not resume:
        # prefer the fp32 training checkpoint over the half-precision inference copy
        base = first_existing([cfg.pretrained_ckpt, cfg.ckpt_path, cfg.final_path])
        if base is None:
            raise FileNotFoundError("SFT needs pretrained weights: run the 'pretrain' stage first, "
                                    "or set Config.pretrained_ckpt / Config.import_dir.")
        load_model_weights(raw_model, base)
        if rt.master:
            print(f"[↪] SFT starts from base weights {base}")
    model = wrap_model(raw_model, scfg, rt)

    train_loader = SFTLoader(tr_t, tr_m, cfg.batch_size, rt.device, cfg.seed + rt.rank)
    val_loader = SFTLoader(va_t, va_m, cfg.batch_size, rt.device, cfg.seed + 30_000)
    sample_fn = lambda: chat(raw_model, tok, SFT_SAMPLE_QUESTIONS[0], rt, max_new_tokens=64)
    step = run_training(scfg, rt, model, raw_model, train_loader, {"val": val_loader}, cfg.sft_ckpt_path,
                        sample_fn, "POST-TRAIN (SFT)")

    if rt.master and step == 0:
        print("[!] No SFT steps ran; chat model weights not written.")
    elif rt.master:
        save_inference_weights(cfg.sft_final_path, raw_model, scfg, tok.model_file, rt)
        print(f"[✓] Chat model weights: {cfg.sft_final_path}")
        for q in SFT_SAMPLE_QUESTIONS:
            print(f"  Q: {q}\n  A: {chat(raw_model, tok, q, rt)}")
    del model, raw_model
    free_memory()

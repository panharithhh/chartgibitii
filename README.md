# Khmer LLM — pretrain → SFT → RAG

Train a small Khmer language model from scratch, teach it to answer questions, and let customers ask
Khmer questions that it answers from **your** documents.

```
Khmer corpora ─► SentencePiece tokenizer ─► PRETRAIN ─► base model
    ─► POST-TRAIN (SFT: question → answer) ─► chat model
    ─► RAG: Khmer question ─► find passages in your docs ─► answer + sources
```

**Model:** Kimi-style sparse MoE transformer, ~106M parameters with ~67M active per token by default.
It uses GQA (8 query / 2 KV heads), QK-Norm, RoPE (θ = 50k), and 1 shared + 8 routed SwiGLU experts
with top-2 routing and aux-loss-free load balancing. Training uses Muon (Newton-Schulz-5) + AdamW.
On T4 / P100 it runs in fp16 with GradScaler; on A100 / H100 it runs in bf16.

## Folder layout

```
khmer_llm/                 the package
  config.py                every setting (one dataclass)
  runtime.py               GPU detection, mixed precision, multi-GPU helpers
  text.py                  Khmer cleaning, filtering, chunking
  chat.py                  chat template + answer generation
  data/                    dataset registry, HF streaming, tokenizer, token bins
  model/                   layers (attention, MoE) + the language model
  optim/                   Muon + AdamW, LR schedule
  train/                   shared training loop, checkpoints
  stages/                  pretrain.py, sft.py, rag.py
  ui.py                    customer web page (Gradio)
  pipeline.py, cli.py      run stages / command line
kaggle/train_khmer_llm.py  the whole package as ONE file for Kaggle (generated)
scripts/                   build_kaggle_script.py (regenerates the Kaggle file)
examples/                  sample RAG documents + custom Q&A format
tests/                     pytest suite (runs offline on CPU)
```

## Quick start

### Kaggle (one cell)
1. Notebook settings: **GPU on, Internet on**.
2. Paste `kaggle/train_khmer_llm.py` into one cell. Put your settings in its **last line**, for example:
   ```python
   RESULTS = main(Config(rag_docs_dir="/kaggle/input/my-docs", launch_ui=True))
   ```
   Then run it. It runs all three stages and saves everything to `/kaggle/working`.
3. Ask questions afterwards:
   ```python
   RESULTS["rag"].ask("តើរាជធានីនៃប្រទេសកម្ពុជាគឺជាអ្វី?")
   ```
   `launch_ui=True` prints a public web link for customers.

#### No-paste setup (recommended)
Upload `kaggle/train_khmer_llm.py` into the same Kaggle dataset as your `shop_replies.jsonl`. Then in a notebook:
```python
!cp /kaggle/input/<your-dataset>/train_khmer_llm.py train.py
!KHMER_SET="stages=data;sft_custom_jsonl=/kaggle/input/<your-dataset>/shop_replies.jsonl" python train.py
```
`KHMER_SET` sets any `Config` field (`key=value;key=value`) without editing the file. Use
`torchrun --nproc_per_node=2 train.py` instead of `python train.py` to train on both T4s.

#### Plan for the free tier (12 h sessions, ~30 GPU h per week)
| Session | Hardware | Last line |
|---|---|---|
| 0: data | **CPU** (free, no GPU quota) | `main(Config(stages="data", sft_custom_jsonl=...))` downloads ~6B chars of Khmer text, trains the tokenizer, and builds token files and SFT data |
| 1, 2, …: training | GPU T4 x2 | `main(Config(import_dir="/kaggle/input/<session-0-output>", sft_custom_jsonl=...))` |

Each later session lists the newest output first: `import_dir="/kaggle/input/<gpu-run-1>,/kaggle/input/<data-run>"`.
Every stage resumes from its checkpoint and saves before the 12 h cut. The log prints how many epochs of
the corpus a run covers, plus an `[eta]` line.

**Optional, more data:** CulturaX Khmer is gated. Accept its terms at huggingface.co/datasets/uonlp/CulturaX,
then add your HF token as a Kaggle secret named `HF_TOKEN` (Add-ons → Secrets). Without it, CulturaX is
skipped automatically.

### Local / server
```bash
pip install -e ".[rag,ui]"                          # needs torch already installed
python -m khmer_llm run                             # all stages
python -m khmer_llm run --stages sft,rag --docs examples/docs
python -m khmer_llm ask "តើអង្គរវត្តនៅខេត្តណា?"     # one question (reuses the docs index from the last run)
python -m khmer_llm ask                             # interactive
python -m khmer_llm ui                              # web page for customers
python -m khmer_llm run --set max_iters=100 --set vocab_size=32000   # override any setting
```
To use 2 GPUs: `torchrun --nproc_per_node=2 -m khmer_llm run`.

## Tokenizer benchmark (pick the best tokenizer)
Trains the **same model** for `bench_steps` (default 2000) with each tokenizer on the same text, then ranks
them by **bits per byte** on held-out text. Loss per token can't be compared across tokenizers; BPB can.

| `tokenizer_type` | What | Used by |
|---|---|---|
| `unigram` (default) | SentencePiece Unigram + byte fallback | T5, XLNet |
| `sp_bpe` | SentencePiece BPE + byte fallback, works on raw unsegmented text | Llama 2, Gemma, SeaLLM |
| `byte_bpe` | byte-level BPE (HF `tokenizers`) + **Khmer-aware split regex** | Llama 3 / Qwen / GPT-4 style |

GPT-4's split regex uses `\p{L}+`, which leaves out Khmer vowel signs and the coeng subscript marker, so it
cuts every syllable (`ខ្ញុំ` → `ខ` `្ញ` `ុំ`). `KHMER_SPLIT_PATTERN` (in `khmer_llm/data/bpe.py`) keeps
each Khmer run whole, caps numerals at 3 digits and gives `។ ៕ ៖ ៛` their own chunks.

Kaggle, one GPU notebook (downloads a smaller corpus itself):
```python
!KHMER_SET="stages=bench;max_corpus_chars=600000000;max_harvest_hours=1.5" torchrun --nproc_per_node=2 train.py
```
Output: a results table, `bench_results.json`, and `🏆 Best: <type>`. Then train for real with `tokenizer_type=<best>`.

## Use your own data

| What | How |
|---|---|
| Documents to answer from (RAG) | Put `.txt`, `.md` or `.jsonl` (`{"title", "text", "source"}`) files in a folder and pass `--docs <folder>`. See `examples/docs/`. |
| Your own Q&A pairs (SFT) | One JSON object per line with `question` and `answer`, plus an optional `context`. Pass `--set sft_custom_jsonl=<file>`. These are upsampled 3×. If you add or change them after a finished run, SFT restarts from the base model. See `examples/sft_custom.jsonl`. |
| Shop replies in 3 styles (SFT) | `{"input": <customer message>, "output": {"ok": ..., "neutral": ..., "no": ...}}` per line, passed the same way (`sft_custom_jsonl`). Each line trains 3 examples tagged with their style. Afterwards: `rag.suggest_replies(msg)` gives all 3, `rag.ask(msg, stance="no")` gives one, `python -m khmer_llm ask --replies "..."` from the CLI, and the web page shows the 3 side by side. See `examples/shop_replies.jsonl`. |

With no docs folder, RAG falls back to a slice of the harvested Khmer corpus.

## Outputs (`work_dir`)

| File | What |
|---|---|
| `khmer_sp_unigram_16k.model` | tokenizer |
| `khmer_kimi_moe_final.pt` | base model (after pretraining) |
| `khmer_kimi_moe_sft_final.pt` | chat model (after SFT), used for answering |
| `khmer_rag_index.pt` | search index over your documents |
| `*_latest.pt` | resumable training checkpoints |

## ⚠️ Data licenses
The default SFT datasets (Bactrian-X Khmer, SAIL Alpaca Khmer) are **CC BY-NC 4.0, non-commercial**.
Before using this with paying customers, replace them in `khmer_llm/data/sources.py` with data you are
allowed to use, and/or rely on your own Q&A (`sft_custom_jsonl`). Check every dataset card, including
the pretraining corpora.

## Development
```bash
pip install -e ".[dev]"
pytest -q                                        # ~35 s on CPU, no internet needed
python scripts/build_kaggle_script.py            # after editing the package: regenerate the Kaggle file
```
CI checks that the Kaggle file is in sync with the package.

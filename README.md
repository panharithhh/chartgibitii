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
2. Paste `kaggle/train_khmer_llm.py` into one cell and run it. It runs all three stages and saves
   everything to `/kaggle/working`.
3. Ask questions afterwards:
   ```python
   RESULTS["rag"].ask("តើរាជធានីនៃប្រទេសកម្ពុជាគឺជាអ្វី?")
   ```
   For a public web page for customers, set `Config.launch_ui = True` before running.

Kaggle stops after 12 hours, so long runs are split across sessions. Save the notebook version, add
its output as an input, and set `Config.import_dir = "/kaggle/input/<that-output>"`. Every stage then
resumes from its checkpoint.

### Local / server
```bash
pip install -e ".[rag,ui]"                          # needs torch already installed
python -m khmer_llm run                             # all stages
python -m khmer_llm run --stages sft,rag --docs examples/docs
python -m khmer_llm ask "តើអង្គរវត្តនៅខេត្តណា?"     # one question
python -m khmer_llm ask                             # interactive
python -m khmer_llm ui                              # web page for customers
python -m khmer_llm run --set max_iters=100 --set vocab_size=32000   # override any setting
```
To use 2 GPUs: `torchrun --nproc_per_node=2 -m khmer_llm run`.

## Use your own data

| What | How |
|---|---|
| Documents to answer from (RAG) | Put `.txt`, `.md` or `.jsonl` (`{"title", "text", "source"}`) files in a folder and pass `--docs <folder>`. See `examples/docs/`. |
| Your own Q&A pairs (SFT) | One JSON object per line with `question` and `answer`, plus an optional `context`. Pass `--set sft_custom_jsonl=<file>`. These are upsampled 3×. See `examples/sft_custom.jsonl`. |

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

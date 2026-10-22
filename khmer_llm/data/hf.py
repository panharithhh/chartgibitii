"""Hugging Face streaming helpers shared by the corpus and SFT harvesters."""
import gzip
import itertools
import json
import os
from typing import Iterator, List, Optional

from ..runtime import ensure_package


def maybe_hf_login():
    """Picks up an HF_TOKEN Kaggle secret if present (needed only for gated datasets)."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass


def row_text(row: dict, fields: Optional[List[str]], sep: str) -> str:
    keys = [k for k in (fields or []) if k in row]
    if not keys:   # unknown schema -> use every string column
        keys = [k for k, v in row.items() if isinstance(v, str)]
    return sep.join(row[k] for k in keys if isinstance(row.get(k), str) and row[k].strip())


def iter_json_file(path: str) -> Iterator[dict]:
    """Rows of a .json (array) or .jsonl file, optionally gzipped."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        yield from json.loads(text)
    else:
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)


def _iter_spec(spec: dict) -> Iterator[dict]:
    if spec.get("hub_file"):   # raw file in a dataset repo (works for script-based datasets on datasets>=4)
        from huggingface_hub import hf_hub_download
        return iter_json_file(hf_hub_download(spec["path"], spec["hub_file"], repo_type="dataset"))
    from datasets import load_dataset
    files = spec.get("data_files")
    if any(str(f).endswith(".zst") for f in (files if isinstance(files, list) else [files or ""])):
        ensure_package("zstandard")
    kwargs = dict(split=spec.get("split", "train"), streaming=True)
    if files:
        kwargs["data_files"] = files
    rows = iter(load_dataset(spec["path"], spec.get("config"), **kwargs))
    if spec.get("parse_json"):   # "text" builder + json.loads: no schema inference over nested fields
        return (json.loads(r["text"]) for r in rows if r.get("text", "").strip())
    return rows


def open_hf_stream(loads: List[dict], require: Optional[List[str]] = None):
    """
    Streams the first alternative that loads AND has the `require`d columns. Reading one row
    forces lazy errors (missing repo, script-based dataset, wrong config) to surface here.
    Returns (label, row iterator).
    """
    errors = []
    for spec in loads:
        try:
            it = _iter_spec(spec)
            first = next(it)
            missing = [c for c in (require or []) if c not in first]
            if missing:
                raise KeyError(f"missing columns {missing}; has {sorted(first)}")
            files = spec.get("data_files") or spec.get("hub_file")
            files = f"{len(files)} files" if isinstance(files, list) else files
            label = spec["path"] + (f" [{spec['config']}]" if spec.get("config") else "") + (f" ({files})" if files else "")
            return label, itertools.chain([first], it)
        except Exception as e:
            errors.append(f"{spec['path']}: {type(e).__name__}: {str(e)[:160]}")
    raise RuntimeError(" | ".join(errors))

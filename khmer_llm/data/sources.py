"""Dataset registry. "loads" = alternatives tried in order; the first one that yields a row with the
required columns wins. A source that fails everywhere is skipped with a message, never a crash.

Column names, configs and file layouts come from dataset cards, HPLT/MADLAD stats files and
third-party loaders (HF was not reachable when this list was assembled), which is why every source
has fallbacks and a first-row schema check. Load spec keys:
  path / config / split / data_files  -> datasets.load_dataset(path, config, data_files=..., streaming=True)
  hub_file                            -> one raw .json/.jsonl(.gz) file from a dataset repo
  parse_json=True                     -> read data_files with the "text" builder and json.loads each line
                                         (robust to nested fields that break schema inference)
Source keys: fields (text columns, None = every string column), sep, cap_chars (per-source share),
  unescape=True (literal "\\n" -> newline + HTML entities, for MADLAD-400).
"""

# MBZUAI/Bactrian-X ships a loading script that datasets>=4 refuses to run, so read its raw file first.
_BACTRIAN_KM = [dict(path="MBZUAI/Bactrian-X", hub_file="data/km.json.gz"),
                dict(path="MBZUAI/Bactrian-X", config="km"),
                dict(path="rinabuoy/Bactrian-X-km")]

_HPLT3_KM = [f"https://data.hplt-project.org/three/sorted/khm_Khmr/{b}_1.jsonl.zst" for b in (10, 9, 8, 7, 6, 5)]

# Pretraining corpus, in priority order: curated + Khmer-specific first, then web crawls. Exact line
# dedup charges overlap to the LATER source, so the big overlapping crawls come after FineWeb-2.
# cap_chars = max NEW characters taken from one source (None = no per-source limit).
HARVEST_SOURCES = [
    # ---- curated / clean ----
    dict(name="Chuon Nath Dictionary 44k", loads=[dict(path="seanghay/khmer-dictionary-44k")],
         fields=["word", "definition", "example"], sep=" "),
    dict(name="Khmer Wikipedia", loads=[dict(path="wikimedia/wikipedia", config="20231101.km"),
                                        dict(path="navinnseab/khmer-wikipedia-article")],
         fields=["text", "content"]),
    dict(name="CADT Khmer news", loads=[dict(path="CADT-IDRI/Khmer_News_classification_unsegmented"),
                                        dict(path="nattkorat/raw_khmer_news_dataset")],
         fields=["title", "content"]),
    dict(name="Khmer articles (Vanda10)", loads=[dict(path="Vanda10/khmer_text_articles_data")],
         fields=["full_article"]),
    dict(name="Khmer news corpus (Thareah)", loads=[dict(path="Thareah/khmer_news_corpus")], fields=["text"]),
    dict(name="KHEED Khmer news", loads=[dict(path="Cheaming/KHEED")], fields=["title", "content"]),
    dict(name="Bactrian-X Khmer 67k", loads=_BACTRIAN_KM, fields=["instruction", "input", "output"]),
    dict(name="Alpaca Khmer 52k", loads=[dict(path="saillab/alpaca-khmer-cleaned")],
         fields=["instruction", "input", "output"]),
    dict(name="Khmer text pairs", loads=[dict(path="rinabuoy/text-pairs-khmer-part-2")],
         fields=["text"], cap_chars=100_000_000),
    # ---- Khmer-specific corpora ----
    dict(name="khmer-lm-corpus 1.25M", loads=[dict(path="phonsobon/khmer-lm-corpus")], fields=["text"]),
    dict(name="khmer-text-corpus 4.9M", loads=[dict(path="Panhapich/khmer-text-corpus", config="raw"),
                                               dict(path="Panhapich/khmer-text-corpus", data_files="all_text.txt"),
                                               dict(path="Panhapich/khmer-text-corpus")],
         fields=["text"]),
    dict(name="khmer-raw-text v2", loads=[dict(path="nphearum/khmer-raw-text-3M-v2")],
         fields=["text"], cap_chars=600_000_000),
    # ---- web crawls (largest; overlap with each other) ----
    dict(name="FineWeb-2 khm_Khmr", loads=[dict(path="HuggingFaceFW/fineweb-2", config="khm_Khmr")],
         fields=["text"], cap_chars=1_600_000_000),
    dict(name="HPLT 3.0 khm_Khmr", loads=[dict(path="text", data_files=_HPLT3_KM, parse_json=True),
                                          dict(path="HPLT/HPLT2.0_cleaned", config="khm_Khmr")],
         fields=["text"], cap_chars=2_000_000_000),
    dict(name="FinePDFs khm_Khmr", loads=[dict(path="HuggingFaceFW/finepdfs", config="khm_Khmr")],
         fields=["text"], cap_chars=500_000_000),
    dict(name="CulturaX km (needs HF_TOKEN + accepted terms)",
         loads=[dict(path="parquet", data_files="hf://datasets/uonlp/CulturaX/km/*.parquet"),
                dict(path="allenai/c4", config="km")],
         fields=["text"], cap_chars=1_000_000_000),
    dict(name="khmer-pile", loads=[dict(path="sovannpanhaseng/khmer-pile"),
                                   dict(path="sovannpanhaseng/khmer-pile-raw")],
         fields=["text"], cap_chars=800_000_000),
    dict(name="MADLAD-400 km clean",
         loads=[dict(path="json", data_files="hf://datasets/allenai/MADLAD-400/data/km/km_clean_*.jsonl.gz")],
         fields=["text"], unescape=True, cap_chars=700_000_000),
    dict(name="CC-100 km", loads=[dict(path="text", data_files="https://data.statmt.org/cc-100/km.txt.xz")],
         fields=["text"], cap_chars=400_000_000),
    dict(name="Sailor2 synthetic Khmer",
         loads=[dict(path="json", data_files="hf://datasets/sailor2/sea-synthetic/khmer/chunk_*.jsonl")],
         fields=["text"], cap_chars=300_000_000),
]

# Post-training (SFT).
#   kind="pair": instruction / optional context / response columns (None -> auto-detect common schemas)
#   kind="qa":   SQuAD-style {question, context, answers}; the context is cropped around the answer.
# "context" is the slot RAG fills with retrieved passages, so pairs with an input field and the QA
# data teach the model to answer from text placed there.
# LICENSES: check each card before commercial use. Bactrian-X and the SAIL Alpaca are CC BY-NC 4.0
# (non-commercial). For a product, swap in data you may use and/or add Config.sft_custom_jsonl.
SFT_SOURCES = [
    dict(name="Bactrian-X Khmer 67k", loads=_BACTRIAN_KM, kind="pair", require=["instruction", "output"],
         instruction="instruction", context="input", response="output", cap=None, license="CC BY-NC 4.0"),
    dict(name="Alpaca Khmer 52k", loads=[dict(path="saillab/alpaca-khmer-cleaned")], kind="pair",
         require=["instruction", "output"], instruction="instruction", context="input", response="output",
         cap=None, license="CC BY-NC 4.0"),
    dict(name="MultiWikiQA Khmer (answer from passage)", loads=[dict(path="alexandrainst/multi-wiki-qa", config="km")],
         kind="qa", require=["question", "context", "answers"], cap=40_000, license="see dataset card"),
    dict(name="Bayon SFT (Khmer)", loads=[dict(path="attentionlab/bayon-sft")], kind="pair",
         instruction=None, context=None, response=None, cap=20_000, license="see dataset card"),
]

"""Dataset registry. "loads" = alternatives tried in order; the first one that yields a row with the
required columns wins. A source that fails everywhere is skipped with a message, never a crash.

Column names come from dataset cards and third-party loaders (HF was not reachable when this list
was assembled), which is why every source has fallbacks and a first-row schema check.
"""

# MBZUAI/Bactrian-X ships a loading script that datasets>=4 refuses to run, so read its raw file first.
_BACTRIAN_KM = [dict(path="MBZUAI/Bactrian-X", hub_file="data/km.json.gz"),
                dict(path="MBZUAI/Bactrian-X", config="km"),
                dict(path="rinabuoy/Bactrian-X-km")]

# Pretraining corpus. Order matters: curated sources first, large web crawls fill the budget.
HARVEST_SOURCES = [
    dict(name="Chuon Nath Dictionary 44k", loads=[dict(path="seanghay/khmer-dictionary-44k")],
         fields=["word", "definition", "example"], sep=" ", cap=None),
    dict(name="Bactrian-X Khmer 67k", loads=_BACTRIAN_KM, fields=["instruction", "input", "output"], cap=None),
    dict(name="Alpaca Khmer 52k", loads=[dict(path="saillab/alpaca-khmer-cleaned")],
         fields=["instruction", "input", "output"], cap=None),
    dict(name="Khmer text pairs", loads=[dict(path="rinabuoy/text-pairs-khmer-part-2")],
         fields=["text"], cap=200_000),
    dict(name="FineWeb-2 khm_Khmr", loads=[dict(path="HuggingFaceFW/fineweb-2", config="khm_Khmr")],
         fields=["text"], cap=800_000),
    dict(name="khmer-lm-corpus 1.25M", loads=[dict(path="phonsobon/khmer-lm-corpus")],
         fields=["text"], cap=1_000_000),
    dict(name="khmer-pile", loads=[dict(path="sovannpanhaseng/khmer-pile"),
                                   dict(path="sovannpanhaseng/khmer-pile-raw")],
         fields=["text"], cap=700_000),
    dict(name="khmer-text-corpus 4.9M", loads=[dict(path="Panhapich/khmer-text-corpus", config="raw"),
                                               dict(path="Panhapich/khmer-text-corpus", data_files="all_text.txt"),
                                               dict(path="Panhapich/khmer-text-corpus")],
         fields=["text"], cap=None),
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

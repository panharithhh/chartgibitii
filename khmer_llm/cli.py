"""Command line:  python -m khmer_llm {run,ask,ui,selftest} [--work-dir DIR] [--docs DIR] [--set key=value ...]"""
import argparse
from typing import List, Optional

from .config import Config, apply_overrides


def _config(args) -> Config:
    cfg = Config()
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if getattr(args, "docs", None):
        cfg.rag_docs_dir = args.docs
    return apply_overrides(cfg, args.set)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(prog="khmer-llm", description="Khmer LLM: pretrain -> SFT -> RAG")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in [("run", "run pipeline stages"), ("ask", "answer Khmer questions with RAG"),
                            ("ui", "launch the Gradio web page"), ("selftest", "quick correctness check")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--work-dir", help="where data, tokenizer and checkpoints live")
        p.add_argument("--docs", help="folder of .txt/.md/.jsonl documents for RAG")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override any Config field, e.g. --set max_iters=100")
        if name == "run":
            p.add_argument("--stages", help="comma list of pretrain,sft,rag (default: Config.stages)")
        if name == "ask":
            p.add_argument("question", nargs="*", help="question; omit for an interactive prompt")
        if name == "ui":
            p.add_argument("--no-share", action="store_true", help="local URL only, no public link")
    args = parser.parse_args(argv)
    cfg = _config(args)

    if args.cmd == "run":
        from .pipeline import main as run_pipeline
        run_pipeline(cfg, stages=args.stages)
    elif args.cmd == "ask":
        from .stages.rag import KhmerRAG
        rag = KhmerRAG.load(cfg)
        if args.question:
            rag.ask(" ".join(args.question))
        else:
            print("Type a Khmer question (empty line to quit).")
            while True:
                q = input("❓ ").strip()
                if not q:
                    break
                rag.ask(q)
    elif args.cmd == "ui":
        from .stages.rag import KhmerRAG
        from .ui import launch_ui
        launch_ui(KhmerRAG.load(cfg), share=not args.no_share)
    elif args.cmd == "selftest":
        from .runtime import setup_runtime
        from .selftest import self_test
        self_test(setup_runtime(allow_ddp=False))


if __name__ == "__main__":
    main()

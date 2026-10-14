"""Command-line helpers.

    python -m app.cli ingest sample_docs
    python -m app.cli ask "How many days of annual leave do employees get?"
    python -m app.cli --workspace hr ingest hr_docs
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.config import get_settings
from app.service import DEFAULT_WORKSPACE, RAGService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Document library to use (created if missing)")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Ingest every PDF/Word/Excel/CSV/TXT/MD file in a directory")
    ingest.add_argument("directory", type=Path)
    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    service = RAGService(get_settings())
    workspace = args.workspace
    if workspace not in {ws.name for ws in service.list_workspaces()}:
        service.create_workspace(workspace)

    if args.command == "ingest":
        for result in service.ingest_directory(args.directory, workspace):
            print(f"{result.status:>9}  {result.document.name}  ({result.document.num_chunks} chunks)")
        return 0

    answer = service.ask(args.question, args.top_k, workspace=workspace)
    print(answer.answer)
    print(f"\nstatus={answer.status} reason={answer.reason} max_similarity={answer.max_similarity}")
    for source in answer.sources:
        print(json.dumps(asdict(source), ensure_ascii=False, indent=2))
    for warning in answer.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

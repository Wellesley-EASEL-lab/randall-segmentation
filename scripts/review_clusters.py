#!/usr/bin/env python3
"""Initialize, run, or export the clustering-assisted review application."""

from __future__ import annotations

import argparse
from pathlib import Path

from randall_pipeline.cluster_app import create_cluster_app
from randall_pipeline import review_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-db")
    init.add_argument("--db", type=Path, required=True)
    init.add_argument("--inventory", type=Path, required=True)
    init.add_argument("--assignments-csv", type=Path, required=True)
    init.add_argument("--summaries-csv", type=Path)
    init.add_argument("--reset", action="store_true")
    serve = commands.add_parser("serve")
    serve.add_argument("--db", type=Path, required=True)
    serve.add_argument("--asset-root", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5000)
    serve.add_argument("--debug", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--db", type=Path, required=True)
    export.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init-db":
        if args.db.exists() and not args.reset:
            raise FileExistsError(f"database already exists; choose a new path or pass --reset: {args.db}")
        args.db.parent.mkdir(parents=True, exist_ok=True)
        with review_db.connect(args.db) as conn:
            review_db.reset(conn) if args.reset else review_db.init_schema(conn)
            count = review_db.import_inventory(conn, args.inventory)
            review_db.import_clusters(conn, args.assignments_csv, args.summaries_csv)
        print(f"Initialized {args.db} with {count} assets and cluster suggestions")
    elif args.command == "export":
        with review_db.connect(args.db) as conn:
            count = review_db.export_reviews(conn, args.output_csv)
        print(f"Exported {count} reviews to {args.output_csv}")
    else:
        if not args.db.is_file():
            raise FileNotFoundError("initialize the cluster review database before serving")
        app = create_cluster_app(args.db, args.asset_root)
        app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

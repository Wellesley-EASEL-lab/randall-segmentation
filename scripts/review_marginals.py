#!/usr/bin/env python3
"""Initialize, run, or export the local unassisted marginal-review application."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from randall_pipeline.io import safe_join
from randall_pipeline import review_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-db")
    init.add_argument("--db", type=Path, required=True)
    init.add_argument("--inventory", type=Path, required=True)
    init.add_argument("--reset", action="store_true")
    serve = commands.add_parser("serve")
    serve.add_argument("--db", type=Path, required=True)
    serve.add_argument("--asset-root", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    export = commands.add_parser("export")
    export.add_argument("--db", type=Path, required=True)
    export.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def create_app(db_path: Path, asset_root: Path) -> gr.Blocks:
    asset_root = asset_root.resolve()

    def rows():
        with review_db.connect(db_path) as conn:
            return [dict(row) for row in review_db.list_assets(conn)]

    def display(index: int):
        assets = rows()
        if not assets:
            return 0, None, None, None, "", False, True, False, "", "No assets in database"
        index = max(0, min(int(index), len(assets) - 1))
        row = assets[index]
        crop = safe_join(asset_root, row["location"])
        source = safe_join(asset_root, row["source_path"]) if row["source_path"] else None
        segmentation = safe_join(asset_root, row["segmentation_path"]) if row["segmentation_path"] else None
        status = f"{index + 1}/{len(assets)} · {row['manuscript']} · {row['page']} · {row['asset_id']}"
        return (
            index, str(crop), str(source) if source and source.is_file() else None,
            str(segmentation) if segmentation and segmentation.is_file() else None,
            row["asset_id"], bool(row["checked"]), bool(row["include"]),
            bool(row["query"]), row["notes"], status,
        )

    def save(index, asset_id, checked, include, query, notes):
        with review_db.connect(db_path) as conn:
            review_db.update_asset(
                conn, str(asset_id), checked=bool(checked), include=bool(include),
                query=bool(query), notes=str(notes or ""),
            )
        return display(index)

    def move(delta, index, asset_id, checked, include, query, notes):
        saved = save(index, asset_id, checked, include, query, notes)
        return display(saved[0] + delta)

    with gr.Blocks(title="Randall marginal review") as app:
        index = gr.State(0)
        asset_id = gr.Textbox(visible=False)
        status = gr.Markdown()
        with gr.Row():
            source = gr.Image(label="Source page", type="filepath", height=500)
            segmentation = gr.Image(label="Segmentation", type="filepath", height=500)
            crop = gr.Image(label="Marginal crop", type="filepath", height=500)
        with gr.Row():
            checked = gr.Checkbox(label="Checked")
            include = gr.Checkbox(label="Include", value=True)
            query = gr.Checkbox(label="Query")
        notes = gr.Textbox(label="Notes", lines=3)
        with gr.Row():
            previous = gr.Button("Previous")
            save_button = gr.Button("Save")
            next_button = gr.Button("Next")
        outputs = [index, crop, source, segmentation, asset_id, checked, include, query, notes, status]
        app.load(display, inputs=[index], outputs=outputs)
        save_button.click(save, inputs=[index, asset_id, checked, include, query, notes], outputs=outputs)
        previous.click(lambda *values: move(-1, *values), inputs=[index, asset_id, checked, include, query, notes], outputs=outputs)
        next_button.click(lambda *values: move(1, *values), inputs=[index, asset_id, checked, include, query, notes], outputs=outputs)
    return app


def main() -> None:
    args = parse_args()
    if args.command == "init-db":
        if args.db.exists() and not args.reset:
            raise FileExistsError(f"database already exists; choose a new path or pass --reset: {args.db}")
        args.db.parent.mkdir(parents=True, exist_ok=True)
        with review_db.connect(args.db) as conn:
            review_db.reset(conn) if args.reset else review_db.init_schema(conn)
            count = review_db.import_inventory(conn, args.inventory)
        print(f"Initialized {args.db} with {count} assets")
    elif args.command == "export":
        with review_db.connect(args.db) as conn:
            count = review_db.export_reviews(conn, args.output_csv)
        print(f"Exported {count} reviews to {args.output_csv}")
    else:
        if not args.db.is_file():
            raise FileNotFoundError("initialize the review database before serving")
        app = create_app(args.db, args.asset_root)
        app.launch(
            server_name=args.host, server_port=args.port, share=False,
            allowed_paths=[str(args.asset_root.resolve())],
        )


if __name__ == "__main__":
    main()

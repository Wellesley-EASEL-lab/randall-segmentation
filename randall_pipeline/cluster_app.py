"""Flask application factory for clustering-assisted marginal review."""

from __future__ import annotations

import math
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_file, url_for

from .io import parse_bool, safe_join
from . import review_db


def create_cluster_app(db_path: Path, asset_root: Path) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.update(DB_PATH=str(db_path), ASSET_ROOT=str(asset_root.resolve()))
    app.secret_key = "local-only-randall-review"

    @app.get("/")
    def index():
        show_checked = parse_bool(request.args.get("show_checked"))
        requested = request.args.get("cluster", type=int)
        page = max(1, request.args.get("page", default=1, type=int))
        with review_db.connect(Path(app.config["DB_PATH"])) as conn:
            clusters = [dict(row) for row in review_db.cluster_labels(conn, include_checked=show_checked)]
            labels = [int(row["cluster_label"]) for row in clusters]
            cluster_label = requested if requested in labels else (labels[0] if labels else None)
            assets = []
            if cluster_label is not None:
                assets = [dict(row) for row in review_db.cluster_assets(
                    conn, cluster_label, include_checked=show_checked, limit=50, offset=(page - 1) * 50
                )]
        return render_template(
            "cluster_review.html", clusters=clusters, assets=assets,
            cluster_label=cluster_label, page=page, show_checked=show_checked,
        )

    @app.post("/update/<asset_id>")
    def update(asset_id: str):
        with review_db.connect(Path(app.config["DB_PATH"])) as conn:
            review_db.update_asset(
                conn, asset_id, checked=parse_bool(request.form.get("checked")),
                include=parse_bool(request.form.get("include")),
                query=parse_bool(request.form.get("query")),
                notes=str(request.form.get("notes", "")),
            )
        return redirect(url_for(
            "index", cluster=request.form.get("cluster"), page=request.form.get("page", 1),
            show_checked=request.form.get("show_checked", ""),
        ))

    @app.get("/asset/<asset_id>/<kind>")
    def asset(asset_id: str, kind: str):
        columns = {"crop": "location", "source": "source_path", "segmentation": "segmentation_path"}
        if kind not in columns:
            abort(404)
        with review_db.connect(Path(app.config["DB_PATH"])) as conn:
            row = conn.execute(f"SELECT {columns[kind]} AS path FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        if row is None or not row["path"]:
            abort(404)
        path = safe_join(Path(app.config["ASSET_ROOT"]), row["path"])
        if not path.is_file():
            abort(404)
        return send_file(path)

    return app


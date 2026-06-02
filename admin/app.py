"""Simple Flask admin UI for the PII pipeline audit database.

Manages:
  - pii_table_targets     — per-run source/target table pairs
  - pii_pipeline_config   — runtime-tunable key/value config
  - pii_column_exclusions — columns excluded from anonymization
  - pii_pipeline_runs     — read-only audit log
"""

from __future__ import annotations

import os

from flask import Flask, flash, redirect, render_template, request, url_for
from sqlalchemy import BigInteger, Boolean, Column, Integer, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-prod")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://pipeline:pipeline@localhost:5432/pii_audit"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# ── Models ─────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class TableTarget(Base):
    __tablename__ = "pii_table_targets"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_uri = Column(Text, nullable=False)
    target_uri = Column(Text, nullable=False)
    table_name = Column(Text)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP(timezone=True))


class PipelineConfig(Base):
    __tablename__ = "pii_pipeline_config"

    key = Column(Text, primary_key=True)
    value = Column(Text, nullable=False)
    description = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True))


class ColumnExclusion(Base):
    __tablename__ = "pii_column_exclusions"

    table_name = Column(Text, primary_key=True)
    column_name = Column(Text, primary_key=True)
    reason = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True))


class PipelineRun(Base):
    __tablename__ = "pii_pipeline_runs"

    run_id = Column(UUID, primary_key=True)
    pipeline_version = Column(Text)
    started_at = Column(TIMESTAMP(timezone=True))
    finished_at = Column(TIMESTAMP(timezone=True))
    table_name = Column(Text)
    source_uri = Column(Text)
    target_uri = Column(Text)
    total_rows = Column(Integer)
    total_columns = Column(Integer)
    columns_scanned = Column(Integer)
    suppressed_rows = Column(Integer)
    residual_pii = Column(Integer)
    status = Column(Text)
    error_msg = Column(Text)
    output_type = Column(Text)
    stage_seconds = Column(JSONB)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_session() -> Session:
    return Session(engine)


# ── Routes: dashboard ──────────────────────────────────────────────────────────

@app.get("/")
def index():
    with get_session() as db:
        targets_count = db.query(TableTarget).count()
        enabled_count = db.query(TableTarget).filter_by(enabled=True).count()
        config_count = db.query(PipelineConfig).count()
        exclusions_count = db.query(ColumnExclusion).count()
        runs_count = db.query(PipelineRun).count()
        recent_runs = (
            db.query(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .limit(5)
            .all()
        )
    return render_template(
        "index.html",
        targets_count=targets_count,
        enabled_count=enabled_count,
        config_count=config_count,
        exclusions_count=exclusions_count,
        runs_count=runs_count,
        recent_runs=recent_runs,
    )


# ── Routes: pii_table_targets ──────────────────────────────────────────────────

@app.get("/targets")
def targets_list():
    with get_session() as db:
        targets = db.query(TableTarget).order_by(TableTarget.id).all()
    return render_template("targets.html", targets=targets)


@app.post("/targets/add")
def targets_add():
    source = request.form.get("source_uri", "").strip()
    target = request.form.get("target_uri", "").strip()
    name = request.form.get("table_name", "").strip() or None
    if not source or not target:
        flash("Source URI and Target URI are required.", "danger")
        return redirect(url_for("targets_list"))
    if source == target:
        flash("Source and Target URIs must differ.", "danger")
        return redirect(url_for("targets_list"))
    with get_session() as db:
        db.add(TableTarget(source_uri=source, target_uri=target, table_name=name, enabled=True))
        db.commit()
    flash(f"Table target added{(' — ' + name) if name else ''}.", "success")
    return redirect(url_for("targets_list"))


@app.post("/targets/<int:target_id>/toggle")
def targets_toggle(target_id: int):
    with get_session() as db:
        row = db.get(TableTarget, target_id)
        if row:
            row.enabled = not row.enabled
            db.commit()
            flash(
                f"Target {'enabled' if row.enabled else 'disabled'}: {row.table_name or row.source_uri}",
                "success",
            )
    return redirect(url_for("targets_list"))


@app.post("/targets/<int:target_id>/delete")
def targets_delete(target_id: int):
    with get_session() as db:
        row = db.get(TableTarget, target_id)
        if row:
            db.delete(row)
            db.commit()
            flash(f"Target deleted: {row.table_name or row.source_uri}", "warning")
    return redirect(url_for("targets_list"))


# ── Routes: pii_pipeline_config ────────────────────────────────────────────────

@app.get("/config")
def config_list():
    with get_session() as db:
        rows = db.query(PipelineConfig).order_by(PipelineConfig.key).all()
    return render_template("config.html", rows=rows)


@app.post("/config/save")
def config_save():
    key = request.form.get("key", "").strip().upper()
    value = request.form.get("value", "").strip()
    description = request.form.get("description", "").strip() or None
    if not key or not value:
        flash("Key and Value are required.", "danger")
        return redirect(url_for("config_list"))
    with get_session() as db:
        existing = db.get(PipelineConfig, key)
        if existing:
            existing.value = value
            existing.description = description
            flash(f"Updated config key: {key}", "success")
        else:
            db.add(PipelineConfig(key=key, value=value, description=description))
            flash(f"Added config key: {key}", "success")
        db.commit()
    return redirect(url_for("config_list"))


@app.post("/config/<key>/delete")
def config_delete(key: str):
    with get_session() as db:
        row = db.get(PipelineConfig, key)
        if row:
            db.delete(row)
            db.commit()
            flash(f"Deleted config key: {key}", "warning")
    return redirect(url_for("config_list"))


# ── Routes: pii_column_exclusions ──────────────────────────────────────────────

@app.get("/exclusions")
def exclusions_list():
    with get_session() as db:
        rows = (
            db.query(ColumnExclusion)
            .order_by(ColumnExclusion.table_name, ColumnExclusion.column_name)
            .all()
        )
    return render_template("exclusions.html", rows=rows)


@app.post("/exclusions/add")
def exclusions_add():
    table = request.form.get("table_name", "").strip().lower()
    column = request.form.get("column_name", "").strip().lower()
    reason = request.form.get("reason", "").strip() or None
    if not table or not column:
        flash("Table name and Column name are required.", "danger")
        return redirect(url_for("exclusions_list"))
    with get_session() as db:
        existing = db.get(ColumnExclusion, (table, column))
        if existing:
            flash(f"Exclusion already exists: {table}.{column}", "warning")
        else:
            db.add(ColumnExclusion(table_name=table, column_name=column, reason=reason))
            db.commit()
            flash(f"Exclusion added: {table}.{column}", "success")
    return redirect(url_for("exclusions_list"))


@app.post("/exclusions/delete")
def exclusions_delete():
    table = request.form.get("table_name", "").strip()
    column = request.form.get("column_name", "").strip()
    with get_session() as db:
        row = db.get(ColumnExclusion, (table, column))
        if row:
            db.delete(row)
            db.commit()
            flash(f"Exclusion removed: {table}.{column}", "warning")
    return redirect(url_for("exclusions_list"))


# ── Routes: pii_pipeline_runs (read-only) ─────────────────────────────────────

@app.get("/runs")
def runs_list():
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 25
    with get_session() as db:
        total = db.query(PipelineRun).count()
        runs = (
            db.query(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
    pages = (total + per_page - 1) // per_page
    return render_template("runs.html", runs=runs, page=page, pages=pages, total=total)


@app.get("/runs/<run_id>")
def runs_detail(run_id: str):
    with get_session() as db:
        run = db.get(PipelineRun, run_id)
    if run is None:
        flash("Run not found.", "danger")
        return redirect(url_for("runs_list"))
    return render_template("run_detail.html", run=run)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=os.environ.get("FLASK_DEBUG", "0") == "1")

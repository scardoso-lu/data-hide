"""Python Flask admin UI for the PII pipeline audit database.

All routes live under /admin so the Next.js proxy can forward them without
any path rewriting.  Every request is verified against ADMIN_TOKEN — a
shared secret injected by the Next.js proxy.  Flask is not exposed on any
external port; only the proxy can reach it on the internal Docker network.

Manages:
  - pii_table_targets     — per-run source/target table pairs
  - pii_pipeline_config   — runtime-tunable key/value config
  - pii_column_exclusions — columns excluded from anonymization
  - pii_pipeline_runs     — read-only audit log
"""

from __future__ import annotations

import os

from flask import (
    Blueprint,
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)
from sqlalchemy import BigInteger, Boolean, Column, Integer, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-prod")
app.config["SESSION_COOKIE_NAME"] = "pii_admin_session"
app.config["SESSION_COOKIE_PATH"] = "/admin"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://pipeline:pipeline@localhost:5432/pii_audit"
)
engine = create_engine(_DATABASE_URL, pool_pre_ping=True)


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


# ── Blueprint & auth guard ─────────────────────────────────────────────────────

bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.before_request
def require_admin_token() -> None:
    """Reject any request that didn't come through the Next.js proxy."""
    if _ADMIN_TOKEN and request.headers.get("X-Admin-Token") != _ADMIN_TOKEN:
        abort(403)
    g.user_email = request.headers.get("X-Forwarded-User", "")
    g.user_name = request.headers.get("X-Forwarded-Name", "") or g.user_email


def db() -> Session:
    return Session(engine)


# ── Dashboard ──────────────────────────────────────────────────────────────────

@bp.get("/")
def index():
    with db() as s:
        targets_count = s.query(TableTarget).count()
        enabled_count = s.query(TableTarget).filter_by(enabled=True).count()
        config_count = s.query(PipelineConfig).count()
        exclusions_count = s.query(ColumnExclusion).count()
        runs_count = s.query(PipelineRun).count()
        recent_runs = (
            s.query(PipelineRun)
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


# ── pii_table_targets ──────────────────────────────────────────────────────────

@bp.get("/targets")
def targets_list():
    with db() as s:
        targets = s.query(TableTarget).order_by(TableTarget.id).all()
    return render_template("targets.html", targets=targets)


@bp.post("/targets/add")
def targets_add():
    source = request.form.get("source_uri", "").strip()
    target = request.form.get("target_uri", "").strip()
    name = request.form.get("table_name", "").strip() or None
    if not source or not target:
        flash("Source URI and Target URI are required.", "danger")
        return redirect(url_for("admin.targets_list"))
    if source == target:
        flash("Source and Target URIs must differ.", "danger")
        return redirect(url_for("admin.targets_list"))
    with db() as s:
        s.add(TableTarget(source_uri=source, target_uri=target, table_name=name, enabled=True))
        s.commit()
    flash(f"Target added{(' — ' + name) if name else ''}.", "success")
    return redirect(url_for("admin.targets_list"))


@bp.post("/targets/<int:target_id>/toggle")
def targets_toggle(target_id: int):
    with db() as s:
        row = s.get(TableTarget, target_id)
        if row:
            row.enabled = not row.enabled
            s.commit()
            flash(
                f"Target {'enabled' if row.enabled else 'disabled'}: {row.table_name or row.source_uri}",
                "success",
            )
    return redirect(url_for("admin.targets_list"))


@bp.post("/targets/<int:target_id>/delete")
def targets_delete(target_id: int):
    with db() as s:
        row = s.get(TableTarget, target_id)
        if row:
            label = row.table_name or row.source_uri
            s.delete(row)
            s.commit()
            flash(f"Target deleted: {label}", "warning")
    return redirect(url_for("admin.targets_list"))


# ── pii_pipeline_config ────────────────────────────────────────────────────────

@bp.get("/config")
def config_list():
    with db() as s:
        rows = s.query(PipelineConfig).order_by(PipelineConfig.key).all()
    return render_template("config.html", rows=rows)


@bp.post("/config/save")
def config_save():
    key = request.form.get("key", "").strip().upper()
    value = request.form.get("value", "").strip()
    description = request.form.get("description", "").strip() or None
    if not key or not value:
        flash("Key and Value are required.", "danger")
        return redirect(url_for("admin.config_list"))
    with db() as s:
        existing = s.get(PipelineConfig, key)
        if existing:
            existing.value = value
            existing.description = description
            flash(f"Updated: {key}", "success")
        else:
            s.add(PipelineConfig(key=key, value=value, description=description))
            flash(f"Added: {key}", "success")
        s.commit()
    return redirect(url_for("admin.config_list"))


@bp.post("/config/<key>/delete")
def config_delete(key: str):
    with db() as s:
        row = s.get(PipelineConfig, key)
        if row:
            s.delete(row)
            s.commit()
            flash(f"Deleted config key: {key}", "warning")
    return redirect(url_for("admin.config_list"))


# ── pii_column_exclusions ──────────────────────────────────────────────────────

@bp.get("/exclusions")
def exclusions_list():
    with db() as s:
        rows = (
            s.query(ColumnExclusion)
            .order_by(ColumnExclusion.table_name, ColumnExclusion.column_name)
            .all()
        )
    return render_template("exclusions.html", rows=rows)


@bp.post("/exclusions/add")
def exclusions_add():
    table = request.form.get("table_name", "").strip().lower()
    column = request.form.get("column_name", "").strip().lower()
    reason = request.form.get("reason", "").strip() or None
    if not table or not column:
        flash("Table name and Column name are required.", "danger")
        return redirect(url_for("admin.exclusions_list"))
    with db() as s:
        existing = s.get(ColumnExclusion, (table, column))
        if existing:
            flash(f"Already excluded: {table}.{column}", "warning")
        else:
            s.add(ColumnExclusion(table_name=table, column_name=column, reason=reason))
            s.commit()
            flash(f"Excluded: {table}.{column}", "success")
    return redirect(url_for("admin.exclusions_list"))


@bp.post("/exclusions/delete")
def exclusions_delete():
    table = request.form.get("table_name", "").strip()
    column = request.form.get("column_name", "").strip()
    with db() as s:
        row = s.get(ColumnExclusion, (table, column))
        if row:
            s.delete(row)
            s.commit()
            flash(f"Removed exclusion: {table}.{column}", "warning")
    return redirect(url_for("admin.exclusions_list"))


# ── pii_pipeline_runs (read-only) ─────────────────────────────────────────────

@bp.get("/runs")
def runs_list():
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 25
    with db() as s:
        total = s.query(PipelineRun).count()
        runs = (
            s.query(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
    pages = (total + per_page - 1) // per_page
    return render_template("runs.html", runs=runs, page=page, pages=pages, total=total)


@bp.get("/runs/<run_id>")
def runs_detail(run_id: str):
    with db() as s:
        run = s.get(PipelineRun, run_id)
    if run is None:
        flash("Run not found.", "danger")
        return redirect(url_for("admin.runs_list"))
    return render_template("run_detail.html", run=run)


# ── Register & run ─────────────────────────────────────────────────────────────

app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )

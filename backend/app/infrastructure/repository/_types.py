"""Shared data-transfer objects for the repository layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableMapping:
    source_uri: str
    target_uri: str
    table_name: str | None = None
    read_mode: str = "delta"  # "delta" | "sql"
    # SQL schema of the source for `read_mode="sql"` mappings; preserved
    # in the destination path so the closest source-to-target layout is
    # maintained across environments.  Ignored for `read_mode="delta"`.
    schema: str | None = None

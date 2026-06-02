"""URI parsing and shared utility helpers for the repository layer.

Pure utilities — no Azure SDK or I/O dependencies.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone


TEMPORAL_NAME_TOKENS = {
    "date", "time", "timestamp", "datetime",
    "created", "updated", "recorded", "captured", "occurred", "ts", "dt",
}
DEFAULT_READ_LOOKBACK_DAYS = 365
DELTA_DISCOVERY_RECURSIVE_THRESHOLD = 20


def _identifier_tokens(name: str) -> set[str]:
    return {part.lower() for part in re.split(r"[^A-Za-z0-9]+|(?<=[a-z])(?=[A-Z])", name) if part}


def _looks_temporal_by_name(name: str) -> bool:
    return bool(_identifier_tokens(name) & TEMPORAL_NAME_TOKENS)


def _parse_abfss_uri(uri: str) -> tuple[str, str, str]:
    abfss_match = re.match(r"^abfss://([^@/]+)@([^/]+)(?:/(.*))?$", uri)
    if abfss_match:
        return abfss_match.group(1), abfss_match.group(2), abfss_match.group(3) or ""

    https_match = re.match(r"^https://([^/]+)/([^/]+)/(.+)$", uri)
    if https_match:
        return https_match.group(2), https_match.group(1), https_match.group(3)

    if not abfss_match:
        raise ValueError(
            f"Cannot parse storage URI: '{uri}'. "
            "Expected: abfss://filesystem@host/path or https://host/filesystem/path"
        )


def _format_abfss_uri(filesystem: str, host: str, path: str) -> str:
    return f"abfss://{filesystem}@{host}/{path.strip('/')}"


def _account_name(uri: str) -> str:
    try:
        _, host, _ = _parse_abfss_uri(uri)
    except ValueError:
        raise ValueError(
            f"Cannot parse account name from URI: '{uri}'. "
            "Expected: abfss://container@account.dfs.fabric.microsoft.com/... "
            "or https://account.dfs.fabric.microsoft.com/container/..."
        ) from None
    return host.split(".", 1)[0]


def _looks_like_uuid(value: str) -> bool:
    return bool(re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        value,
    ))


def _ensure_lakehouse_tables_target_base(path: str) -> None:
    normalized = path.strip("/")
    if "/Files/" in f"/{normalized}/":
        raise ValueError(
            "TARGET_BASE_ABFSS_URI points to Lakehouse Files. Delta table output must target "
            "the Lakehouse Tables root, for example "
            "abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables."
        )
    if not normalized.endswith("/Tables"):
        raise ValueError(
            "TARGET_BASE_ABFSS_URI must end with '<lakehouse>.Lakehouse/Tables' so Fabric "
            "registers each output as a Lakehouse table."
        )


def _ensure_lakehouse_delta_table_uri(path: str) -> None:
    normalized = path.strip("/")
    if "/Files/" in f"/{normalized}/":
        raise ValueError(
            "Delta table output cannot be written under Lakehouse Files. Set "
            "TARGET_BASE_ABFSS_URI to '<lakehouse>.Lakehouse/Tables'."
        )
    if "/Tables/" not in f"/{normalized}/":
        raise ValueError("Delta table output URI must point under '<lakehouse>.Lakehouse/Tables/<table>'.")


def _env_int_at_least(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be {minimum} or greater")
    return value


def _max_upload_workers() -> int:
    return _env_int_at_least("MAX_UPLOAD_WORKERS", 4, 1)


def _read_lookback_days() -> int:
    return _env_int_at_least("READ_LOOKBACK_DAYS", DEFAULT_READ_LOOKBACK_DAYS, 0)


def read_cutoff_ts(now: datetime | None = None) -> datetime:
    """Return the UTC lower bound for source reads."""
    base = now or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base.astimezone(timezone.utc) - timedelta(days=_read_lookback_days())

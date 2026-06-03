"""Fabric REST API helpers â€” workspace/lakehouse name resolution and table discovery.

Cache dicts (``_fabric_*_cache``) and ``requests`` live in the *package*
namespace so tests can clear caches and patch ``app.repository.requests.get``
without modifying imports in this module.
"""

from __future__ import annotations

import logging

from ._types import TableMapping
from ._utils import (
    _parse_abfss_uri,
    _format_abfss_uri,
    _looks_like_uuid,
    _ensure_lakehouse_tables_target_base,
)
from .delta import _discover_delta_mappings, _data_lake_service_client

logger = logging.getLogger(__name__)


def _fabric_workspace_guid_for_name(workspace_name: str) -> str | None:
    """Resolve a workspace friendly name to its GUID via the Fabric REST API.

    ``GET /v1/workspaces`` lists all workspaces the credential can see.  The
    response is paginated; all pages are consumed so the cache is warm after the
    first call.  Returns ``None`` when the workspace is not found or the API
    call fails â€” callers must handle the ``None`` case.
    """
    import app.infrastructure.repository as _r

    if workspace_name in _r._fabric_workspace_id_cache:
        return _r._fabric_workspace_id_cache[workspace_name]

    token = _r.acquire_cached_token(_r.FABRIC_TOKEN_SCOPE)
    url: str | None = "https://api.fabric.microsoft.com/v1/workspaces"
    while url:
        response = _r.requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        for ws in body.get("value", []):
            name = ws.get("displayName", "")
            guid = ws.get("id")
            if name and guid:
                _r._fabric_workspace_id_cache[name] = guid
        url = body.get("continuationUri")

    # Cache a negative result to avoid repeated API calls on miss.
    if workspace_name not in _r._fabric_workspace_id_cache:
        _r._fabric_workspace_id_cache[workspace_name] = None

    return _r._fabric_workspace_id_cache.get(workspace_name)


def _fabric_item_display_name(workspace_guid: str, item_id: str) -> str | None:
    """Return the display name for a Fabric item given its workspace and item GUIDs.

    ``workspace_guid`` must be a GUID â€” the Fabric REST API rejects friendly
    names in the ``/v1/workspaces/{workspaceId}`` path segment.
    """
    import app.infrastructure.repository as _r

    cache_key = (workspace_guid, item_id)
    if cache_key in _r._fabric_item_name_cache:
        return _r._fabric_item_name_cache[cache_key]

    token = _r.acquire_cached_token(_r.FABRIC_TOKEN_SCOPE)
    response = _r.requests.get(
        f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_guid}/items/{item_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    display_name = response.json().get("displayName")
    result = display_name if isinstance(display_name, str) and display_name else None
    _r._fabric_item_name_cache[cache_key] = result
    return result


def _fabric_lakehouse_guid_for_name(workspace_guid: str, lakehouse_name: str) -> str | None:
    """Resolve a lakehouse friendly name to its item GUID.

    Lists all items in *workspace_guid* via ``GET /v1/workspaces/{id}/items``
    and returns the ``id`` of the first item whose ``displayName`` matches
    *lakehouse_name* (with or without a ``.Lakehouse`` suffix).  Returns
    ``None`` when no match is found or the API call fails.

    *workspace_guid* must be a GUID â€” the Fabric REST API rejects friendly
    workspace names in that path segment.
    """
    import app.infrastructure.repository as _r

    bare_name = lakehouse_name.removesuffix(".Lakehouse")
    cache_key = (workspace_guid, bare_name)
    if cache_key in _r._fabric_lakehouse_id_cache:
        return _r._fabric_lakehouse_id_cache[cache_key]

    token = _r.acquire_cached_token(_r.FABRIC_TOKEN_SCOPE)
    url: str | None = (
        f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_guid}/items"
    )
    while url:
        response = _r.requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        for item in body.get("value", []):
            item_name = item.get("displayName", "").removesuffix(".Lakehouse")
            item_id = item.get("id")
            if item_name and item_id:
                _r._fabric_lakehouse_id_cache[(workspace_guid, item_name)] = item_id
        url = body.get("continuationUri")

    if cache_key not in _r._fabric_lakehouse_id_cache:
        _r._fabric_lakehouse_id_cache[cache_key] = None

    return _r._fabric_lakehouse_id_cache.get(cache_key)


def _resolve_onelake_item_id_path(filesystem: str, host: str, path: str) -> str:
    """Resolve mixed-mode OneLake ABFSS paths to a consistent identifier form.

    OneLake enforces a strict consistency rule: the workspace identifier (the
    ``filesystem`` component of the ABFSS URI) and the lakehouse identifier
    (the first path segment) must both be GUIDs *or* both be friendly names.
    Mixing modes in either direction is rejected with ``FriendlyNameSupportDisabled``.

    Two mixed-mode shapes are handled:

    A. **Friendly workspace + GUID lakehouse** â€” resolve the GUID to its
       display name so all components use the friendly-name mode.
    B. **GUID workspace + friendly-name lakehouse** â€” resolve the friendly
       lakehouse name to its GUID so all components use GUID mode.

    When resolution fails the sentinel ``None`` is returned so
    ``discover_table_mappings`` can fall back to workspace-root scanning.
    """
    if host.lower() != "onelake.dfs.fabric.microsoft.com":
        return path

    parts = path.strip("/").split("/", 1)
    first_segment = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    first_is_guid = _looks_like_uuid(first_segment)
    fs_is_guid = _looks_like_uuid(filesystem)

    if not first_segment:
        # Empty path â€” workspace-root scan; no resolution needed.
        return path

    if fs_is_guid and first_is_guid:
        # GUID workspace + GUID lakehouse â€” OneLake accepts this pair as-is.
        return path

    if not fs_is_guid and not first_is_guid:
        # Friendly workspace + friendly lakehouse â€” consistent; no change needed.
        return path

    if not fs_is_guid and first_is_guid:
        # â”€â”€ Case A: friendly workspace + GUID lakehouse â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        item_id = first_segment
        display_name: str | None = None
        last_exc: Exception | None = None

        # Pass 1 â€” some tenants accept friendly workspace names in the items API.
        try:
            display_name = _fabric_item_display_name(filesystem, item_id)
            if display_name:
                logger.info(
                    "Resolved lakehouse GUID '%s' â†’ '%s' (workspace name used directly).",
                    item_id, display_name,
                )
        except Exception as exc:
            last_exc = exc
            logger.debug(
                "Pass 1 (workspace name as ID) failed for item '%s' in '%s': %s",
                item_id, filesystem, exc,
            )

        # Pass 2 â€” resolve workspace name â†’ GUID, then retry items API.
        if not display_name:
            try:
                workspace_guid = _fabric_workspace_guid_for_name(filesystem)
                if workspace_guid:
                    display_name = _fabric_item_display_name(workspace_guid, item_id)
                    if display_name:
                        logger.info(
                            "Resolved lakehouse GUID '%s' â†’ '%s' via workspace GUID '%s'.",
                            item_id, display_name, workspace_guid,
                        )
                else:
                    logger.debug(
                        "Workspace '%s' not found in GET /v1/workspaces response.", filesystem,
                    )
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "Pass 2 (workspace GUID resolution) failed for item '%s' in '%s': %s",
                    item_id, filesystem, exc,
                )

        if not display_name:
            logger.warning(
                "Could not resolve lakehouse GUID '%s' in workspace '%s' to a friendly "
                "name (last error: %s). Falling back to workspace-root scan. "
                "Fix: update SOURCE_BASE_ABFSS_URI to use the lakehouse friendly name: "
                "abfss://%s@%s/<LakehouseName>.Lakehouse/%s",
                item_id, filesystem, last_exc, filesystem, host, rest,
            )
            return None  # type: ignore[return-value]  # sentinel â†’ caller falls back

        item_path = (
            display_name
            if display_name.endswith(".Lakehouse")
            else f"{display_name}.Lakehouse"
        )
        return f"{item_path}/{rest}" if rest else item_path

    # â”€â”€ Case B: GUID workspace + friendly-name lakehouse â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lakehouse_name = first_segment
    last_exc = None

    try:
        lakehouse_guid = _fabric_lakehouse_guid_for_name(filesystem, lakehouse_name)
        if lakehouse_guid:
            logger.info(
                "Resolved lakehouse name '%s' â†’ GUID '%s' (workspace GUID '%s').",
                lakehouse_name, lakehouse_guid, filesystem,
            )
            return f"{lakehouse_guid}/{rest}" if rest else lakehouse_guid
        else:
            logger.debug(
                "Lakehouse '%s' not found in workspace '%s' item listing.",
                lakehouse_name, filesystem,
            )
    except Exception as exc:
        last_exc = exc
        logger.debug(
            "Case B (friendly lakehouse â†’ GUID) failed for '%s' in workspace '%s': %s",
            lakehouse_name, filesystem, exc,
        )

    logger.warning(
        "Could not resolve lakehouse name '%s' to a GUID in workspace '%s' "
        "(last error: %s). Falling back to workspace-root scan. "
        "Fix: update SOURCE_BASE_ABFSS_URI to use the lakehouse GUID: "
        "abfss://%s@%s/<LakehouseGUID>.Lakehouse/%s",
        lakehouse_name, filesystem, last_exc, filesystem, host, rest,
    )
    return None  # type: ignore[return-value]  # sentinel â†’ caller falls back


def discover_table_mappings(
    source_base_uri: str,
    target_base_uri: str,
    *,
    sql_endpoint: str | None = None,
    sql_database: str | None = None,
) -> list[TableMapping]:
    """Return TableMappings for every Delta table under source_base_uri plus any
    SQL-only shortcuts discovered via the Fabric SQL Analytics Endpoint.

    Delta tables discovered via ADLS always take precedence; a table that appears
    in both ADLS and SQL is included exactly once as read_mode="delta".
    """
    import app.infrastructure.repository as _r

    if _r.DataLakeServiceClient is None:
        from azure.storage.filedatalake import DataLakeServiceClient as _DLSC
        _r.DataLakeServiceClient = _DLSC

    filesystem, host, base_path = _parse_abfss_uri(source_base_uri)
    resolved = _resolve_onelake_item_id_path(filesystem, host, base_path)
    if resolved is None:
        # Fabric API resolution failed â€” scan the workspace root instead.
        base_path = ""
    else:
        base_path = resolved.rstrip("/")

    target_filesystem, target_host, target_path = _parse_abfss_uri(target_base_uri)
    resolved_target = _resolve_onelake_item_id_path(target_filesystem, target_host, target_path)
    if resolved_target is not None:
        # When the Fabric API can't resolve the lakehouse id to a friendly name
        # (e.g. SP has storage-only RBAC), keep the raw id path rather than
        # crashing — symmetric with the source-path fallback above. The target
        # must stay a concrete writable path, so we never blank it to "".
        target_path = resolved_target
    _ensure_lakehouse_tables_target_base(target_path)
    target_base = _format_abfss_uri(target_filesystem, target_host, target_path).rstrip("/")

    service_client = _data_lake_service_client(f"https://{host}")
    fs_client = service_client.get_file_system_client(file_system=filesystem)

    mappings = _discover_delta_mappings(fs_client, filesystem, host, base_path, target_base)
    logger.info("Discovered %d Delta table(s) under %s", len(mappings), source_base_uri)

    if sql_endpoint and sql_database:
        from .sql import _discover_sql_table_names
        try:
            sql_specs = _discover_sql_table_names(sql_endpoint, sql_database)
            # Compute the SQL-endpoint identity each Delta source already covers.
            delta_sql_keys: set[str] = set()
            for mapping in mappings:
                relative = mapping.target_uri[len(target_base):].lstrip("/")
                if "/" in relative:
                    delta_sql_keys.add(relative)
                else:
                    delta_sql_keys.add(f"dbo/{relative}")
            shortcuts = [
                (schema, name) for schema, name in sql_specs
                if f"{schema}/{name}" not in delta_sql_keys
            ]
            for schema, name in shortcuts:
                source_uri = f"sql://{sql_endpoint}/{sql_database}/{schema}/{name}"
                target_uri = f"{target_base}/{schema}/{name}"
                mappings.append(TableMapping(
                    source_uri=source_uri,
                    target_uri=target_uri,
                    table_name=name,
                    read_mode="sql",
                    schema=schema,
                ))
            logger.info(
                "Discovered %d SQL shortcut(s) via '%s' (%d already covered by Delta)",
                len(shortcuts), sql_endpoint, len(sql_specs) - len(shortcuts),
            )
        except Exception as exc:
            logger.warning("SQL shortcut discovery failed (non-fatal): %s", exc)

    return mappings

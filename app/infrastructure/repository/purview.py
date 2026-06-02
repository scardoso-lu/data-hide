"""Microsoft Purview catalog client (azure-purview-catalog SDK).

The SDK is imported lazily inside ``PurviewClient.__init__`` so environments
without the package (e.g. unit-test runs that mock this class) don't fail
at module import time.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PurviewClient:
    """Purview catalog client backed by the official ``azure-purview-catalog`` SDK.

    Uses ``PurviewCatalogClient`` with ``DefaultAzureCredential`` — no manual
    token acquisition needed.  The SDK is imported lazily so environments
    without the package (e.g. unit-test runs that mock this class) don't fail
    at module import time.
    """

    def __init__(self, account_name: str) -> None:
        try:
            from azure.purview.catalog import PurviewCatalogClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "azure-purview-catalog is required for Purview integration. "
                "Install it with: uv add 'azure-purview-catalog>=0.2.0,<1.0.0'"
            ) from exc
        from .auth import _credential_instance
        self._client = PurviewCatalogClient(
            endpoint=f"https://{account_name}.purview.azure.com",
            credential=_credential_instance(),
        )

    def column_classifications(self, qualified_name: str) -> dict[str, list[str]]:
        """Return ``{column_name: [classification_type, …]}`` for every column
        the Purview catalog has classified on the given entity path.

        Errors (HTTP, auth, or SDK) are logged and return an empty dict so the
        caller treats Purview as unavailable rather than aborting the pipeline.
        """
        try:
            data = self._client.entity.get_by_unique_attribute(
                "azure_datalake_gen2_path",
                attr_qualified_name=qualified_name,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is not None:
                logger.warning("Purview HTTP %s: %s", status, exc)
            else:
                logger.warning("Purview request failed: %s", exc)
            return {}

        result: dict[str, list[str]] = {}
        for entity in (data or {}).get("referredEntities", {}).values():
            if entity.get("typeName") != "azure_datalake_gen2_column":
                continue
            col = entity.get("attributes", {}).get("name", "")
            labels = [c["typeName"] for c in entity.get("classifications", [])]
            if col and labels:
                result[col] = labels
        return result

    @staticmethod
    def qualified_name(abfss_uri: str) -> str:
        """Convert an ABFSS URI to the qualified name format Purview uses."""
        without_scheme = abfss_uri.replace("abfss://", "")
        container, rest = without_scheme.split("@", 1)
        host, path = rest.split("/", 1)
        return f"https://{host}/{container}/{path}"


def run_purview_check(
    source_uri: str,
    df_columns: list[str],
    purview_account: str | None,
) -> dict:
    empty: dict = {
        "available": False,
        "flagged_columns": [],
        "column_labels": {},
        "discrepancies": [],
    }
    if not purview_account:
        return empty
    try:
        client = PurviewClient(purview_account)
        col_labels = client.column_classifications(PurviewClient.qualified_name(source_uri))
        flagged = list(col_labels.keys())
        return {
            "available": True,
            "flagged_columns": flagged,
            "column_labels": col_labels,
            "discrepancies": [c for c in flagged if c not in df_columns],
        }
    except Exception as exc:
        logger.warning("Purview check failed (non-fatal): %s", exc)
        return empty

"""
Unit tests for URI helpers, storage options, the optional Purview check,
free-text column flagging, quasi-identifier detection, k-anonymity enforcement,
and residual PII validation.
No external services or spaCy model required (except residual PII tests).
"""

from datetime import datetime, timezone

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from main import (
    PurviewClient,
    _account_name,
    _storage_opts,
    detect_identifier_columns,
    detect_quasi_identifiers,
    enforce_k_anonymity,
    flag_free_text_columns,
    pseudonymize_identifier_columns,
    run_purview_check,
    validate_residual_pii,
    write_delta,
)
from app.infrastructure.repository import (
    AuditDB,
    _coerce_null_columns_arrow,
    _fabric_workspace_guid_for_name,
    _parse_abfss_uri,
    _resolve_onelake_item_id_path,
    read_delta,
)


def _coerce_null_columns(df: pl.DataFrame):
    """Test helper bridging the old _coerce_null_columns(pandas) contract to the
    new _coerce_null_columns_arrow(table, null_col_names) API.

    Builds a PyArrow Table from the Polars frame and computes the all-null
    column names exactly as write_delta does.
    """
    import pyarrow as pa

    table = df.to_arrow()
    null_col_names = [f.name for f in table.schema if pa.types.is_null(f.type)]
    return _coerce_null_columns_arrow(table, null_col_names)


class _FakePseudonymizer:
    """Deterministic in-memory pseudonymizer for tests.

    Mimics KeyVaultPseudonymizer.pseudonymize: nulls pass through, every
    other value is hashed under a fixed secret derived from ``key``.
    """

    def __init__(self, key: bytes = b"test-key") -> None:
        self._key = key

    def __call__(self, value):
        import hmac
        import hashlib

        if value is None:
            return value
        raw = value if isinstance(value, str) else str(value)
        return hmac.new(self._key, raw.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


class TestAccountNameParser:

    @pytest.mark.parametrize("uri,expected", [
        (
            "abfss://workspace@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            "onelake",
        ),
        (
            "abfss://mycontainer@myaccount.dfs.core.windows.net/path/data",
            "myaccount",
        ),
        (
            "abfss://ws@storage123.dfs.core.windows.net/",
            "storage123",
        ),
        (
            "abfss://WorkspaceName@onelake.dfs.fabric.microsoft.com/Demo.Lakehouse/Tables/orders",
            "onelake",
        ),
    ])
    def test_extracts_correct_account(self, uri, expected):
        assert _account_name(uri) == expected

    def test_extracts_account_from_onelake_https_uri(self):
        uri = "https://onelake.dfs.fabric.microsoft.com/workspace-id/lakehouse-id/Tables/orders"
        assert _account_name(uri) == "onelake"

    @pytest.mark.parametrize("bad_uri", [
        "not-a-uri",
        "https://example.com/path",
        "",
        "abfss://no-at-sign-here",
        "abfss://workspace",
    ])
    def test_raises_on_invalid_uri(self, bad_uri):
        with pytest.raises(ValueError, match="Cannot parse"):
            _account_name(bad_uri)


class TestStorageOpts:

    def test_contains_account_name_and_token(self):
        uri = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t"
        opts = _storage_opts(uri, "my-bearer-token")
        assert opts["account_name"] == "onelake"
        assert opts["bearer_token"] == "my-bearer-token"

    def test_different_accounts_produce_different_opts(self):
        uri_a = "abfss://c@accountA.dfs.core.windows.net/p"
        uri_b = "abfss://c@accountB.dfs.core.windows.net/p"
        assert _storage_opts(uri_a, "tok")["account_name"] == "accountA"
        assert _storage_opts(uri_b, "tok")["account_name"] == "accountB"


class TestFabricWorkspaceGuidForName:
    """Tests for workspace friendly-name â†’ GUID resolution via the Fabric REST API."""

    def _mock_requests_get(self, mocker, pages):
        """Patch requests.get to return paginated workspace list responses."""
        import app.infrastructure.repository as repo

        responses = []
        for i, workspaces in enumerate(pages):
            body = {"value": workspaces}
            if i < len(pages) - 1:
                body["continuationUri"] = f"https://api.fabric.microsoft.com/v1/workspaces?cont={i+1}"
            mock_resp = mocker.MagicMock()
            mock_resp.json.return_value = body
            responses.append(mock_resp)

        mocker.patch("app.infrastructure.repository.requests.get", side_effect=responses)
        mocker.patch("app.infrastructure.repository.acquire_cached_token", return_value="fake-token")
        # Clear the cache before each test.
        repo._fabric_workspace_id_cache.clear()

    def test_returns_guid_for_known_workspace(self, mocker):
        self._mock_requests_get(mocker, [[
            {"displayName": "MyWorkspace", "id": "ffb5e061-3824-486b-ab7c-aaef61221403"},
            {"displayName": "OtherWorkspace", "id": "aaaabbbb-cccc-dddd-eeee-ffffffffffff"},
        ]])
        assert _fabric_workspace_guid_for_name("MyWorkspace") == "ffb5e061-3824-486b-ab7c-aaef61221403"

    def test_returns_none_for_unknown_workspace(self, mocker):
        self._mock_requests_get(mocker, [[
            {"displayName": "OtherWorkspace", "id": "aaaabbbb-cccc-dddd-eeee-ffffffffffff"},
        ]])
        assert _fabric_workspace_guid_for_name("MyWorkspace") is None

    def test_consumes_all_pages(self, mocker):
        self._mock_requests_get(mocker, [
            [{"displayName": "Page1WS", "id": "11111111-1111-1111-1111-111111111111"}],
            [{"displayName": "MyWorkspace", "id": "ffb5e061-3824-486b-ab7c-aaef61221403"}],
        ])
        assert _fabric_workspace_guid_for_name("MyWorkspace") == "ffb5e061-3824-486b-ab7c-aaef61221403"

    def test_result_is_cached(self, mocker):
        import app.infrastructure.repository as repo
        self._mock_requests_get(mocker, [[
            {"displayName": "MyWorkspace", "id": "ffb5e061-3824-486b-ab7c-aaef61221403"},
        ]])
        _fabric_workspace_guid_for_name("MyWorkspace")
        _fabric_workspace_guid_for_name("MyWorkspace")
        # requests.get should have been called only once (first call); second uses cache.
        assert repo.requests.get.call_count == 1


class TestResolveOnelakeItemIdPath:
    """Tests for the two-step workspace-name â†’ GUID â†’ lakehouse-name resolution."""

    def test_non_onelake_host_unchanged(self):
        path = "container/Tables"
        assert _resolve_onelake_item_id_path("ws", "storage.dfs.core.windows.net", path) == path

    def test_already_friendly_lakehouse_unchanged(self):
        path = "MyLakehouse.Lakehouse/Tables"
        assert _resolve_onelake_item_id_path("MyWorkspace", "onelake.dfs.fabric.microsoft.com", path) == path

    def test_guid_workspace_guid_lakehouse_unchanged(self):
        path = "f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables"
        result = _resolve_onelake_item_id_path(
            "ffb5e061-3824-486b-ab7c-aaef61221403",
            "onelake.dfs.fabric.microsoft.com",
            path,
        )
        assert result == path

    def test_friendly_workspace_guid_lakehouse_resolved(self, mocker):
        """The root cause of FriendlyNameSupportDisabled: workspace is a name but
        lakehouse is a GUID. Resolution must use the workspace GUID for the API call."""
        import app.infrastructure.repository as repo
        repo._fabric_workspace_id_cache.clear()
        repo._fabric_item_name_cache.clear()
        mocker.patch(
            "app.infrastructure.repository.fabric._fabric_workspace_guid_for_name",
            return_value="ffb5e061-3824-486b-ab7c-aaef61221403",
        )
        mocker.patch(
            "app.infrastructure.repository.fabric._fabric_item_display_name",
            return_value="SourceLakehouse",
        )
        result = _resolve_onelake_item_id_path(
            "MyWorkspace",
            "onelake.dfs.fabric.microsoft.com",
            "f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables",
        )
        assert result == "SourceLakehouse.Lakehouse/Tables"

    def test_workspace_resolution_failure_returns_none(self, mocker):
        """If both resolution passes fail, return None so discover_table_mappings
        can fall back to workspace-root scanning without any Fabric API access."""
        mocker.patch(
            "app.infrastructure.repository.fabric._fabric_item_display_name",
            side_effect=Exception("403 Forbidden"),
        )
        mocker.patch("app.infrastructure.repository.fabric._fabric_workspace_guid_for_name", return_value=None)
        result = _resolve_onelake_item_id_path(
            "MyWorkspace",
            "onelake.dfs.fabric.microsoft.com",
            "f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables",
        )
        assert result is None

    def test_pass1_success_skips_workspace_resolution(self, mocker):
        """If the items API accepts the workspace name directly, no workspace GUID lookup is needed."""
        import app.infrastructure.repository as repo
        repo._fabric_item_name_cache.clear()
        mocker.patch(
            "app.infrastructure.repository.fabric._fabric_item_display_name",
            return_value="SourceLakehouse",
        )
        ws_resolver = mocker.patch("app.infrastructure.repository.fabric._fabric_workspace_guid_for_name")
        result = _resolve_onelake_item_id_path(
            "MyWorkspace",
            "onelake.dfs.fabric.microsoft.com",
            "f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables",
        )
        assert result == "SourceLakehouse.Lakehouse/Tables"
        ws_resolver.assert_not_called()


class TestStorageUriParser:

    def test_parses_abfss_uri(self):
        assert _parse_abfss_uri("abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables") == (
            "ws",
            "onelake.dfs.fabric.microsoft.com",
            "lh/Tables",
        )

    def test_parses_onelake_https_uri(self):
        assert _parse_abfss_uri(
            "https://onelake.dfs.fabric.microsoft.com/ffb5e061-3824-486b-ab7c-aaef61221403/f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables"
        ) == (
            "ffb5e061-3824-486b-ab7c-aaef61221403",
            "onelake.dfs.fabric.microsoft.com",
            "f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables",
        )


class TestReadDeltaLookback:

    class _FakeDeltaTable:
        def __init__(self, uri, storage_options=None):
            self.uri = uri
            self.storage_options = storage_options

        def to_pyarrow_dataset(self):
            import pyarrow as pa

            return pa.table({
                "id": [1, 2, 3],
                "created_at": [
                    datetime(2023, 12, 31, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2025, 1, 2, tzinfo=timezone.utc),
                ],
                "name": ["old", "edge", "new"],
            })

    def test_delta_read_uses_duckdb_cutoff_before_dataframe_materialization(self, monkeypatch, mocker):
        import app.infrastructure.repository as repo

        monkeypatch.setattr(repo, "DeltaTable", self._FakeDeltaTable)
        mocker.patch("app.infrastructure.repository.delta.read_cutoff_ts", return_value=datetime(2024, 1, 1, tzinfo=timezone.utc))

        df = read_delta("abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t", {})

        assert list(df["id"]) == [2, 3]

    def test_delta_read_filters_string_temporal_columns_by_name(self, monkeypatch, mocker):
        import pyarrow as pa
        import app.infrastructure.repository as repo

        class FakeDeltaTable:
            def __init__(self, uri, storage_options=None):
                pass

            def to_pyarrow_dataset(self):
                return pa.table({
                    "id": [1, 2],
                    "event_date": ["2023-12-31", "2024-01-02"],
                })

        monkeypatch.setattr(repo, "DeltaTable", FakeDeltaTable)
        mocker.patch("app.infrastructure.repository.delta.read_cutoff_ts", return_value=datetime(2024, 1, 1, tzinfo=timezone.utc))

        df = read_delta("abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t", {})

        assert list(df["id"]) == [2]

    def test_delta_read_falls_back_to_all_rows_when_filter_returns_empty(self, monkeypatch, mocker):
        """A small or old table where every row pre-dates the cutoff must not
        be silently discarded â€” the pipeline should fall back to a full read."""
        import pyarrow as pa
        import app.infrastructure.repository as repo

        class FakeDeltaTable:
            def __init__(self, uri, storage_options=None):
                pass

            def to_pyarrow_dataset(self):
                return pa.table({
                    "id": [1, 2],
                    # both rows are from 2020 â€” far older than the 365-day cutoff
                    "created_at": [
                        datetime(2020, 1, 1, tzinfo=timezone.utc),
                        datetime(2020, 6, 1, tzinfo=timezone.utc),
                    ],
                })

        monkeypatch.setattr(repo, "DeltaTable", FakeDeltaTable)
        mocker.patch("app.infrastructure.repository.delta.read_cutoff_ts", return_value=datetime(2024, 1, 1, tzinfo=timezone.utc))

        df = read_delta("abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t", {})

        # fallback must return both rows, not an empty DataFrame
        assert len(df) == 2
        assert set(df["id"]) == {1, 2}


class TestDeltaTemporalPushdown:
    """The cutoff filter compares typed date/timestamp columns directly so
    DuckDB can prune Parquet row groups (huge-table memory fix); only string
    columns keep TRY_CAST, which cannot be pruned."""

    def test_split_classifies_typed_vs_string(self):
        import pyarrow as pa
        from app.infrastructure.repository.delta import _split_temporal_columns

        schema = pa.schema([
            ("id", pa.int64()),
            ("created_at", pa.timestamp("us", tz="UTC")),
            ("updated_on", pa.date32()),
            ("event_date", pa.string()),      # temporal by name, string-typed
            ("notes", pa.string()),           # not temporal
        ])
        typed, string = _split_temporal_columns(schema)
        assert set(typed) == {"created_at", "updated_on"}
        assert string == ["event_date"]

    def test_typed_columns_are_not_wrapped_in_try_cast(self):
        from app.infrastructure.repository.delta import _duckdb_temporal_filter_sql

        sql = _duckdb_temporal_filter_sql(["created_at", "updated_on"], ["event_date"])
        # Typed columns compared directly → DuckDB pushes the predicate into the
        # Parquet scan and prunes row groups.
        assert '"created_at" >= ?' in sql
        assert '"updated_on" >= ?' in sql
        assert 'TRY_CAST("created_at"' not in sql
        # String columns still need the cast.
        assert 'TRY_CAST("event_date" AS TIMESTAMP) >= ?' in sql


class TestReadSqlTableLookback:
    """read_sql_table must fall back to a full read when the 365-day filter
    returns no rows (table is too small or entirely pre-dates the cutoff)."""

    def _make_cursor(self, filtered_rows, all_rows, columns):
        """Return a fake cursor whose first execute() returns filtered_rows
        and whose second execute() (fallback) returns all_rows."""

        class FakeCursor:
            def __init__(self):
                self._calls = 0
                self._filtered = filtered_rows
                self._all = all_rows
                self.description = [(c,) for c in columns]

            def execute(self, sql, *args):
                self._calls += 1
                if self._calls == 1:
                    self._rows = self._filtered
                else:
                    self._rows = self._all

            def fetchall(self):
                return self._rows

        return FakeCursor()

    def _make_conn(self, cursor):
        class FakeConn:
            def __init__(self, cur):
                self._cursor = cur

            def cursor(self):
                return self._cursor

            def close(self):
                pass

        return FakeConn(cursor)

    def test_sql_read_falls_back_when_filter_returns_empty(self, monkeypatch, mocker):
        from app.infrastructure.repository import read_sql_table, _sql_temporal_columns
        import app.infrastructure.repository as repo

        columns = ["id", "event_date"]
        # Filtered result is empty; full table has two rows
        cur = self._make_cursor(
            filtered_rows=[],
            all_rows=[(1, "2020-01-01"), (2, "2020-06-01")],
            columns=columns,
        )

        mocker.patch("app.infrastructure.repository.sql.read_cutoff_ts", return_value=datetime(2024, 1, 1, tzinfo=timezone.utc))
        monkeypatch.setattr(repo.sql, "_sql_connection", lambda *a, **kw: self._make_conn(cur))
        monkeypatch.setattr(repo.sql, "_sql_temporal_columns", lambda *a, **kw: ["event_date"])

        df = read_sql_table("events", "fake-endpoint", "fake-db")

        assert len(df) == 2
        assert set(df["id"]) == {1, 2}
        assert cur._calls == 2   # filtered query + fallback query

    def test_sql_read_does_not_fall_back_when_filter_has_rows(self, monkeypatch, mocker):
        from app.infrastructure.repository import read_sql_table
        import app.infrastructure.repository as repo

        columns = ["id", "event_date"]
        cur = self._make_cursor(
            filtered_rows=[(3, "2024-03-01")],
            all_rows=[(1, "2020-01-01"), (2, "2020-06-01"), (3, "2024-03-01")],
            columns=columns,
        )

        mocker.patch("app.infrastructure.repository.sql.read_cutoff_ts", return_value=datetime(2024, 1, 1, tzinfo=timezone.utc))
        monkeypatch.setattr(repo.sql, "_sql_connection", lambda *a, **kw: self._make_conn(cur))
        monkeypatch.setattr(repo.sql, "_sql_temporal_columns", lambda *a, **kw: ["event_date"])

        df = read_sql_table("events", "fake-endpoint", "fake-db")

        assert len(df) == 1
        assert cur._calls == 1   # only the filtered query


class TestWriteDelta:

    class _FakeDuckDB:
        def connect(self):
            return self._Connection()

        class _Connection:
            def install_extension(self, name):
                pass

            def load_extension(self, name):
                pass

            def register(self, name, df):
                pass

            def execute(self, sql):
                import pathlib
                import re

                match = re.search(r"COPY _df TO '([^']+)'", sql)
                if match:
                    table_path = pathlib.Path(match.group(1))
                    (table_path / "_delta_log").mkdir(parents=True)
                    (table_path / "_delta_log" / "00000000000000000000.json").write_text("{}", encoding="utf-8")
                    (table_path / "part-00000.parquet").write_bytes(b"parquet")

            def close(self):
                pass

    def test_uploads_delta_directory_to_explicit_path(self, monkeypatch, mocker):
        import app.infrastructure.repository as repo

        fs_client = mocker.MagicMock()
        file_client = mocker.MagicMock()
        service = mocker.MagicMock()
        service.get_file_system_client.return_value = fs_client
        fs_client.get_file_client.return_value = file_client
        service_cls = mocker.MagicMock(return_value=service)

        monkeypatch.setattr(repo, "_duckdb", self._FakeDuckDB())
        monkeypatch.setattr(repo, "DataLakeServiceClient", service_cls)
        monkeypatch.setattr(repo, "_credential_instance", lambda: object())

        write_delta(
            pl.DataFrame({"x": [1]}),
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers",
            {},
        )

        service_cls.assert_called_once()
        service.get_file_system_client.assert_called_once_with(file_system="ws")
        uploaded_paths = {call.kwargs["file_path"] for call in fs_client.get_file_client.call_args_list}
        assert "lh.Lakehouse/Tables/customers/_delta_log/00000000000000000000.json" in uploaded_paths
        assert any(
            path.startswith("lh.Lakehouse/Tables/customers/part-") and path.endswith(".parquet")
            for path in uploaded_paths
        )
        assert file_client.upload_data.call_count == 2
        assert file_client.upload_data.call_args.kwargs["overwrite"] is True

    def test_replaces_existing_remote_delta_directory_before_upload(self, monkeypatch, mocker):
        import app.infrastructure.repository as repo

        fs_client = mocker.MagicMock()
        directory_client = mocker.MagicMock()
        directory_client.exists.return_value = True
        file_client = mocker.MagicMock()
        service = mocker.MagicMock()
        service.get_file_system_client.return_value = fs_client
        fs_client.get_directory_client.return_value = directory_client
        fs_client.get_file_client.return_value = file_client

        monkeypatch.setattr(repo, "_duckdb", self._FakeDuckDB())
        monkeypatch.setattr(repo, "DataLakeServiceClient", mocker.MagicMock(return_value=service))
        monkeypatch.setattr(repo, "_credential_instance", lambda: object())

        write_delta(
            pl.DataFrame({"x": [1]}),
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers",
            {},
        )

        fs_client.get_directory_client.assert_called_once_with("lh.Lakehouse/Tables/customers")
        directory_client.delete_directory.assert_called_once()
        method_names = [call[0] for call in fs_client.method_calls]
        assert method_names.index("get_directory_client") < method_names.index("get_file_client")

    def test_uses_table_folder_name_for_delta_output(self, monkeypatch, mocker):
        import app.infrastructure.repository as repo

        fs_client = mocker.MagicMock()
        file_client = mocker.MagicMock()
        service = mocker.MagicMock()
        service.get_file_system_client.return_value = fs_client
        fs_client.get_file_client.return_value = file_client

        monkeypatch.setattr(repo, "_duckdb", self._FakeDuckDB())
        monkeypatch.setattr(repo, "DataLakeServiceClient", mocker.MagicMock(return_value=service))
        monkeypatch.setattr(repo, "_credential_instance", lambda: object())

        write_delta(pl.DataFrame({"x": [1]}), "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers", {})

        uploaded_paths = {call.kwargs["file_path"] for call in fs_client.get_file_client.call_args_list}
        assert any(
            path.startswith("lh.Lakehouse/Tables/customers/part-") and path.endswith(".parquet")
            for path in uploaded_paths
        )
        assert "lh.Lakehouse/Tables/customers/_delta_log/00000000000000000000.json" in uploaded_paths

    def test_write_uses_delta_rs_not_duckdb_extension(self, monkeypatch, mocker):
        import app.infrastructure.repository as repo

        service = mocker.MagicMock()
        service.get_file_system_client.return_value = mocker.MagicMock()

        duckdb = mocker.MagicMock()
        monkeypatch.setattr(repo, "_duckdb", duckdb)
        monkeypatch.setattr(repo, "DataLakeServiceClient", mocker.MagicMock(return_value=service))
        monkeypatch.setattr(repo, "_credential_instance", lambda: object())

        write_delta(
            pl.DataFrame({"x": [1]}),
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers",
            {},
        )

        duckdb.connect.assert_not_called()

    def test_rejects_lakehouse_files_target(self):
        with pytest.raises(ValueError, match="Lakehouse Files"):
            write_delta(
                pl.DataFrame({"x": [1]}),
                "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Files/out/customers",
                {},
            )

    def test_rejects_empty_schema_before_storage_access(self, monkeypatch, mocker):
        import app.infrastructure.repository as repo

        service_cls = mocker.MagicMock()
        monkeypatch.setattr(repo, "DataLakeServiceClient", service_cls)

        with pytest.raises(ValueError, match="at least one column"):
            write_delta(
                pl.DataFrame(),
                "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers",
                {},
            )

        service_cls.assert_not_called()


class TestAuditSchemaMigration:

    class _Cursor:
        def __init__(self, executed):
            self.executed = executed

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            self.executed.append(sql)

    class _Connection:
        def __init__(self, executed):
            self.executed = executed

        def cursor(self):
            return TestAuditSchemaMigration._Cursor(self.executed)

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    def test_init_schema_adds_missing_audit_columns(self, monkeypatch):
        import app.infrastructure.repository as repo

        executed: list[str] = []
        monkeypatch.setattr(
            repo.psycopg2,
            "connect",
            lambda dsn: self._Connection(executed),
        )

        AuditDB("postgresql://example")

        joined = "\n".join(executed)
        assert "ADD COLUMN IF NOT EXISTS key_vault_key_version TEXT" in joined
        assert "ADD COLUMN IF NOT EXISTS stage_seconds JSONB" in joined


class TestPurviewQualifiedName:

    @pytest.mark.parametrize("uri,expected_qn", [
        (
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            "https://onelake.dfs.fabric.microsoft.com/ws/lh.Lakehouse/Tables/t",
        ),
        (
            "abfss://MyWorkspace@onelake.dfs.fabric.microsoft.com/Demo.Lakehouse/Tables/customers",
            "https://onelake.dfs.fabric.microsoft.com/MyWorkspace/Demo.Lakehouse/Tables/customers",
        ),
    ])
    def test_converts_abfss_to_qualified_name(self, uri, expected_qn):
        assert PurviewClient.qualified_name(uri) == expected_qn

    def test_scheme_stripped(self):
        qn = PurviewClient.qualified_name(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t"
        )
        assert not qn.startswith("abfss://")
        assert qn.startswith("https://")


class TestRunPurviewCheck:

    def test_skipped_when_account_not_set(self):
        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            ["col1", "col2"],
            purview_account=None,
        )
        assert result["available"] is False
        assert result["flagged_columns"] == []
        assert result["discrepancies"] == []

    def test_returns_flagged_columns(self, mocker):
        mock_cls = mocker.patch("main.PurviewClient")
        mock_cls.qualified_name.return_value = "https://onelake.dfs.fabric.microsoft.com/ws/lh/t"
        mock_cls.return_value.column_classifications.return_value = {
            "email":  ["MICROSOFT.PERSONAL.EMAIL"],
            "name":   ["MICROSOFT.PERSONAL.NAME"],
        }

        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            df_columns=["email", "name", "score"],
            purview_account="my-purview",
        )

        assert result["available"] is True
        assert set(result["flagged_columns"]) == {"email", "name"}
        assert result["discrepancies"] == []

    def test_discrepancy_when_flagged_column_absent_from_dataframe(self, mocker):
        mock_cls = mocker.patch("main.PurviewClient")
        mock_cls.qualified_name.return_value = "https://..."
        mock_cls.return_value.column_classifications.return_value = {
            "ssn":   ["MICROSOFT.PERSONAL.SSN"],
            "email": ["MICROSOFT.PERSONAL.EMAIL"],
        }

        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            df_columns=["email", "score"],
            purview_account="my-purview",
        )

        assert "ssn" in result["discrepancies"]
        assert "email" not in result["discrepancies"]

    def test_non_fatal_on_auth_failure(self, mocker):
        mock_cls = mocker.patch("main.PurviewClient")
        mock_cls.side_effect = Exception("auth failed")
        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            df_columns=[],
            purview_account="my-purview",
        )
        assert result["available"] is False

    def test_non_fatal_on_http_404(self, mocker):
        from azure.core.exceptions import HttpResponseError
        mock_cls = mocker.patch("main.PurviewClient")
        mock_cls.qualified_name.return_value = "https://..."
        mock_cls.return_value.column_classifications.side_effect = HttpResponseError(
            message="Not Found", response=mocker.MagicMock(status_code=404)
        )

        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            df_columns=[],
            purview_account="my-purview",
        )
        assert result["available"] is False


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Free-text column flagging
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestFlagFreeTextColumns:

    def test_notes_column_flagged(self):
        df = pl.DataFrame({"notes": ["This is a long free text value with multiple words."], "price": [9.99]})
        assert "notes" in flag_free_text_columns(df)

    def test_description_column_flagged(self):
        df = pl.DataFrame({"description": ["Widget A needs a longer narrative description for review."], "id": [1]})
        assert "description" in flag_free_text_columns(df)

    def test_feedback_column_flagged(self):
        df = pl.DataFrame({"customer_feedback": ["The delivery was late and the customer explained the problem in detail."], "qty": [1]})
        assert "customer_feedback" in flag_free_text_columns(df)

    def test_numeric_column_not_flagged(self):
        df = pl.DataFrame({"price": pl.Series([9.99], dtype=pl.Float64), "qty": pl.Series([1], dtype=pl.Int64)})
        assert flag_free_text_columns(df) == []

    def test_non_text_object_column_with_matching_name_still_flagged(self):
        df = pl.DataFrame({"notes": pl.Series("notes", [{"key": "val"}], dtype=pl.Object)})
        assert "notes" not in flag_free_text_columns(df)

    def test_unrelated_object_column_not_flagged(self):
        df = pl.DataFrame({"customer_id": ["CID-001", "CID-002"]})
        assert "customer_id" not in flag_free_text_columns(df)

    def test_empty_dataframe_returns_empty(self):
        df = pl.DataFrame()
        assert flag_free_text_columns(df) == []


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Quasi-identifier detection
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestDetectQuasiIdentifiers:

    def test_explicit_cols_used_when_provided(self):
        df = pl.DataFrame({"age": [25], "city": ["NYC"], "score": [10]})
        qi = detect_quasi_identifiers(df, explicit_cols=["age", "city"])
        assert qi == ["age", "city"]

    def test_explicit_cols_filtered_to_present(self):
        df = pl.DataFrame({"age": [25, 25, 25, 30, 30, 30], "score": [10, 11, 12, 13, 14, 15]})
        qi = detect_quasi_identifiers(df, explicit_cols=["age", "missing_col"])
        assert qi == ["age"]
        assert "missing_col" not in qi

    def test_keyword_detection_on_age(self):
        df = pl.DataFrame({"age": [25, 25, 25, 30, 30, 30], "score": [10, 11, 12, 13, 14, 15]})
        qi = detect_quasi_identifiers(df)
        assert "age" in qi

    def test_keyword_detection_on_gender(self):
        df = pl.DataFrame({"gender": ["M", "M", "M", "F", "F", "F"], "score": [10, 11, 12, 13, 14, 15]})
        qi = detect_quasi_identifiers(df)
        assert "gender" in qi

    def test_non_qi_column_not_detected(self):
        df = pl.DataFrame({"product_name": ["Widget"], "price": [9.99]})
        qi = detect_quasi_identifiers(df)
        assert qi == []

    def test_empty_explicit_cols_falls_back_to_keyword(self):
        df = pl.DataFrame({"age": [25, 25, 25, 30, 30, 30], "city": ["NYC", "NYC", "NYC", "LUX", "LUX", "LUX"]})
        qi = detect_quasi_identifiers(df, explicit_cols=[])
        assert "age" in qi


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# k-Anonymity enforcement
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestEnforceKAnonymity:

    def test_no_suppression_when_k_met(self):
        df = pl.DataFrame({"age": [25, 25, 30, 30, 30], "score": [1, 2, 3, 4, 5]})
        result_df, info = enforce_k_anonymity(df, ["age"], k=2)
        assert len(result_df) == 5
        assert info["suppressed_rows"] == 0

    def test_suppresses_rare_groups(self):
        df = pl.DataFrame({"age": [25, 25, 30, 30, 99]})
        result_df, info = enforce_k_anonymity(df, ["age"], k=2)
        assert 99 not in result_df["age"].to_list()
        assert info["suppressed_rows"] == 1

    def test_empty_quasi_cols_returns_unchanged(self):
        df = pl.DataFrame({"score": [1, 2, 3]})
        result_df, info = enforce_k_anonymity(df, [], k=5)
        assert_frame_equal(result_df, df)
        assert info["suppressed_rows"] == 0

    def test_all_rows_suppressed_when_none_meet_k(self):
        df = pl.DataFrame({"age": [10, 20, 30]})  # each age appears once
        result_df, info = enforce_k_anonymity(df, ["age"], k=2)
        assert len(result_df) == 0
        assert info["suppressed_rows"] == 3

    def test_multi_column_quasi_identifiers(self):
        df = pl.DataFrame({
            "age":    [25, 25, 25, 30],
            "gender": ["M", "M", "F", "M"],
        })
        result_df, info = enforce_k_anonymity(df, ["age", "gender"], k=2)
        assert len(result_df) == 2   # only (25, M) has count >= 2
        assert info["suppressed_rows"] == 2

    def test_missing_quasi_col_treated_gracefully(self):
        df = pl.DataFrame({"score": [1, 2, 3]})
        result_df, info = enforce_k_anonymity(df, ["nonexistent"], k=2)
        assert_frame_equal(result_df, df)
        assert info["suppressed_rows"] == 0

    def test_k_value_returned_in_info(self):
        df = pl.DataFrame({"age": [25, 25]})
        _, info = enforce_k_anonymity(df, ["age"], k=3)
        assert info["k"] == 3


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Residual PII validation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestValidateResidualPII:

    def test_non_identifier_ner_residuals_do_not_abort(self):
        df = pl.DataFrame({
            "indicator_label": ["Luxembourg", "France", "Kayl"],
            "commune": ["Luxembourg", "Kayl", "France"],
            "notes": ["Luxembourg", "France", "Kayl"],
        })

        assert validate_residual_pii(df) == 0

    def test_structured_phone_false_positive_does_not_abort(self):
        df = pl.DataFrame({
            "record_key": ["source_2024_001"],
            "source_file": ["bronze_communes_2024.csv"],
        })

        assert validate_residual_pii(df) == 0

    def test_direct_phone_residual_still_fails(self):
        df = pl.DataFrame({"phone": ["+352 621 123 456"]})

        with pytest.raises(RuntimeError, match="phone.PHONE_NUMBER=1"):
            validate_residual_pii(df)

    def test_direct_email_residual_still_fails(self):
        df = pl.DataFrame({"source_file": ["alice@example.com.csv"], "email": ["bob@example.com"]})

        with pytest.raises(RuntimeError, match="email.EMAIL_ADDRESS=1"):
            validate_residual_pii(df)

    def test_direct_url_residual_still_fails(self):
        df = pl.DataFrame({"url": ["https://example.com/private/customer"]})

        with pytest.raises(RuntimeError, match="url.URL=1"):
            validate_residual_pii(df)

    def test_direct_ip_residual_still_fails(self):
        df = pl.DataFrame({"ip_address": ["192.168.10.25"]})

        with pytest.raises(RuntimeError, match="ip_address.IP_ADDRESS=1"):
            validate_residual_pii(df)

    def test_generated_tokens_are_not_residual_pii(self):
        df = pl.DataFrame({"name": ["PERSON_0"], "email": ["EMAIL_ADDRESS_0"]})
        assert validate_residual_pii(df) == 0

    def test_residual_error_summarizes_column_without_value(self):
        df = pl.DataFrame({"email": ["alice@example.com"]})
        with pytest.raises(RuntimeError) as exc_info:
            validate_residual_pii(df)
        message = str(exc_info.value)
        assert "email.EMAIL_ADDRESS=1" in message
        assert "alice@example.com" not in message

    def test_passes_clean_dataframe(self):
        df = pl.DataFrame({"note": ["No issues found."], "qty": [5]})
        count = validate_residual_pii(df)
        assert count == 0

    def test_raises_on_residual_email(self):
        df = pl.DataFrame({"email": ["alice@example.com"]})
        with pytest.raises(RuntimeError, match="Residual PII"):
            validate_residual_pii(df)

    def test_raises_message_contains_count(self):
        df = pl.DataFrame({"email": ["alice@example.com", "bob@company.org"]})
        with pytest.raises(RuntimeError) as exc_info:
            validate_residual_pii(df)
        assert "finding" in str(exc_info.value)

    def test_skips_non_object_columns(self):
        df = pl.DataFrame({
            "id":    pl.Series([1, 2, 3], dtype=pl.Int64),
            "score": pl.Series([0.1, 0.2, 0.3], dtype=pl.Float64),
        })
        count = validate_residual_pii(df)
        assert count == 0

    def test_skips_non_string_values_in_object_column(self):
        df = pl.DataFrame({"mixed": pl.Series("mixed", [None, 42, {"key": "val"}], dtype=pl.Object)})
        count = validate_residual_pii(df)
        assert count == 0

    def test_empty_dataframe_passes(self):
        df = pl.DataFrame({"email": pl.Series("email", [], dtype=pl.String)})
        count = validate_residual_pii(df)
        assert count == 0

    def test_entity_token_passes(self):
        """ENTITY_TYPE_N pseudonym tokens must not be flagged as PII."""
        df = pl.DataFrame({
            "name":  ["PERSON_0", "PERSON_1"],
            "email": ["EMAIL_ADDRESS_0", "EMAIL_ADDRESS_1"],
        })
        count = validate_residual_pii(df)
        assert count == 0

    def test_raises_on_pii_inside_json_string(self):
        df = pl.DataFrame({"payload": ['{"email": "alice@example.com"}']})
        with pytest.raises(RuntimeError, match="Residual PII"):
            validate_residual_pii(df)

    def test_raises_on_pii_inside_native_dict(self):
        df = pl.DataFrame({"data": pl.Series("data", [{"email": "alice@example.com"}], dtype=pl.Object)})
        with pytest.raises(RuntimeError, match="Residual PII"):
            validate_residual_pii(df)

    def test_raises_on_pii_inside_json_key(self):
        df = pl.DataFrame({"data": pl.Series("data", [{"alice@example.com": "primary contact"}], dtype=pl.Object)})
        with pytest.raises(RuntimeError, match=r"data:\$\.<key>\.EMAIL_ADDRESS"):
            validate_residual_pii(df)

    def test_metadata_column_exemption_does_not_hide_json_value_pii(self):
        df = pl.DataFrame({"source_file": pl.Series("source_file", [{"email": "alice@example.com"}], dtype=pl.Object)})
        with pytest.raises(RuntimeError, match=r"source_file:\$\.email\.EMAIL_ADDRESS=1"):
            validate_residual_pii(df)

    def test_pseudonym_passes_validation(self):
        """24-hex pseudonym tokens must not be detected as PII."""
        token = _FakePseudonymizer()("EMP001")
        df = pl.DataFrame({"employee_id": [token]})
        count = validate_residual_pii(df)
        assert count == 0


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Identifier column detection
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestDetectIdentifierColumns:

    def test_employee_id_detected(self):
        df = pl.DataFrame({"employee_id": [1], "name": ["Alice"]})
        assert "employee_id" in detect_identifier_columns(df)

    def test_microsoft_id_detected(self):
        df = pl.DataFrame({"microsoft_id": ["abc123"], "score": [10]})
        assert "microsoft_id" in detect_identifier_columns(df)

    def test_matricule_detected(self):
        df = pl.DataFrame({"matricule": ["M001"], "score": [10]})
        assert "matricule" in detect_identifier_columns(df)

    def test_person_id_detected(self):
        df = pl.DataFrame({"person_id": ["P001"]})
        assert "person_id" in detect_identifier_columns(df)

    def test_user_id_detected(self):
        df = pl.DataFrame({"user_id": [42]})
        assert "user_id" in detect_identifier_columns(df)

    def test_column_with_space_normalised(self):
        df = pl.DataFrame({"employee id": ["E001"]})
        assert "employee id" in detect_identifier_columns(df)

    def test_non_identifier_column_not_detected(self):
        df = pl.DataFrame({"product_name": ["Widget"], "price": [9.99]})
        assert detect_identifier_columns(df) == []

    def test_explicit_cols_override(self):
        df = pl.DataFrame({"emp_id": [1], "score": [10]})
        assert detect_identifier_columns(df, explicit_cols=["emp_id"]) == ["emp_id"]

    def test_explicit_cols_filtered_to_present(self):
        df = pl.DataFrame({"emp_id": [1]})
        result = detect_identifier_columns(df, explicit_cols=["emp_id", "missing_col"])
        assert result == ["emp_id"]
        assert "missing_col" not in result

    def test_placeholder_identifier_column_detected(self):
        """Legacy placeholder identifier column names are still detected."""
        df = pl.DataFrame({"IDENTIFIER_0": ["x"]})
        assert "IDENTIFIER_0" in detect_identifier_columns(df)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Identifier column pseudonymization (Key Vault-bound)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPseudonymizeIdentifierColumns:

    def test_string_value_pseudonymized(self):
        df = pl.DataFrame({"employee_id": ["EMP001", "EMP002"]})
        result_df, pseudonymized = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert "EMP001" not in result_df["employee_id"].to_list()
        assert "EMP002" not in result_df["employee_id"].to_list()
        assert "employee_id" in pseudonymized

    def test_mapping_is_deterministic_for_same_key(self):
        df = pl.DataFrame({"employee_id": ["EMP001"]})
        r1, _ = pseudonymize_identifier_columns(df.clone(), ["employee_id"], _FakePseudonymizer())
        r2, _ = pseudonymize_identifier_columns(df.clone(), ["employee_id"], _FakePseudonymizer())
        assert r1["employee_id"][0] == r2["employee_id"][0]

    def test_same_value_same_pseudonym_across_rows(self):
        df = pl.DataFrame({"employee_id": ["EMP001", "EMP001"]})
        result_df, _ = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert result_df["employee_id"][0] == result_df["employee_id"][1]

    def test_different_values_different_pseudonyms(self):
        df = pl.DataFrame({"employee_id": ["EMP001", "EMP002"]})
        result_df, _ = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert result_df["employee_id"][0] != result_df["employee_id"][1]

    def test_null_preserved(self):
        df = pl.DataFrame({"employee_id": [None, "EMP001"]})
        result_df, _ = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert result_df["employee_id"][0] is None

    def test_integer_id_pseudonymized(self):
        df = pl.DataFrame({"employee_id": pl.Series([12345, 67890], dtype=pl.Int64)})
        result_df, pseudonymized = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert 12345 not in result_df["employee_id"].to_list()
        assert "employee_id" in pseudonymized

    def test_different_keys_produce_different_pseudonyms(self):
        df = pl.DataFrame({"employee_id": ["EMP001"]})
        r1, _ = pseudonymize_identifier_columns(df.clone(), ["employee_id"], _FakePseudonymizer(b"key_a"))
        r2, _ = pseudonymize_identifier_columns(df.clone(), ["employee_id"], _FakePseudonymizer(b"key_b"))
        assert r1["employee_id"][0] != r2["employee_id"][0]

    def test_empty_id_cols_returns_unchanged(self):
        df = pl.DataFrame({"employee_id": ["EMP001"]})
        result_df, pseudonymized = pseudonymize_identifier_columns(df, [], _FakePseudonymizer())
        assert result_df["employee_id"][0] == "EMP001"
        assert pseudonymized == []

    def test_pseudonym_has_fixed_length(self):
        df = pl.DataFrame({"employee_id": ["short", "a_much_longer_employee_id_string"]})
        result_df, _ = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert len(result_df["employee_id"][0]) == 24
        assert len(result_df["employee_id"][1]) == 24

    def test_original_dataframe_not_mutated(self):
        df = pl.DataFrame({"employee_id": ["EMP001"]})
        original = df["employee_id"][0]
        pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert df["employee_id"][0] == original

    def test_missing_column_silently_skipped(self):
        df = pl.DataFrame({"employee_id": ["EMP001"]})
        result_df, pseudonymized = pseudonymize_identifier_columns(
            df, ["employee_id", "nonexistent"], _FakePseudonymizer(),
        )
        assert "employee_id" in pseudonymized
        assert "nonexistent" not in pseudonymized

    def test_none_pseudonymizer_raises(self):
        df = pl.DataFrame({"employee_id": ["EMP001"]})
        with pytest.raises(ValueError, match="pseudonymizer is required"):
            pseudonymize_identifier_columns(df, ["employee_id"], None)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Null-column coercion for Delta write
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestCoerceNullColumns:
    """_coerce_null_columns_arrow must convert pa.null() columns to pa.string()
    so delta-rs can write them without SchemaMismatchError."""

    def test_all_null_column_becomes_string(self):
        import pyarrow as pa
        df = pl.DataFrame({"id": ["A", "B"], "notes": pl.Series("notes", [None, None], dtype=pl.Null)})
        table = _coerce_null_columns(df)
        assert pa.types.is_string(table.schema.field("notes").type) or \
               pa.types.is_large_string(table.schema.field("notes").type)

    def test_null_values_preserved_as_null(self):
        import pyarrow as pa
        df = pl.DataFrame({"id": ["A"], "notes": pl.Series("notes", [None], dtype=pl.Null)})
        table = _coerce_null_columns(df)
        assert table.column("notes")[0].as_py() is None

    def test_non_null_columns_unchanged(self):
        import pyarrow as pa
        df = pl.DataFrame({"id": ["A", "B"], "score": [1.0, 2.0]})
        table = _coerce_null_columns(df)
        assert pa.types.is_floating(table.schema.field("score").type)

    def test_empty_dataframe_null_column_coerced(self):
        import pyarrow as pa
        df = pl.DataFrame({
            "id": pl.Series("id", [], dtype=pl.Null),
            "notes": pl.Series("notes", [], dtype=pl.Null),
        })
        table = _coerce_null_columns(df)
        # At minimum, the table should be writeable â€” no pa.null() fields remain
        for field in table.schema:
            assert not pa.types.is_null(field.type), f"Column {field.name!r} still has null type"

    def test_no_null_columns_returns_table_unchanged(self):
        import pyarrow as pa
        df = pl.DataFrame({"name": ["Alice"], "age": [30]})
        table = _coerce_null_columns(df)
        assert table.schema.field("name").type == pa.string() or \
               pa.types.is_large_string(table.schema.field("name").type)
        assert pa.types.is_integer(table.schema.field("age").type)

    def test_column_order_preserved(self):
        df = pl.DataFrame({
            "z": pl.Series("z", [None], dtype=pl.Null),
            "a": ["x"],
            "m": pl.Series("m", [None], dtype=pl.Null),
        })
        table = _coerce_null_columns(df)
        assert table.schema.names == ["z", "a", "m"]

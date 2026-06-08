"""sqlite3-backed executor for the chinook SQL agent's tool surface.

Maps each of the 15 tool names to a Python function that runs against
a chinook database file and returns a JSON-serialisable result. The
generator calls into here whenever the model emits a tool call; the
result is then fed back to the model as a tool-response message.

Three layers of defense, in priority order:

1. **Read-only connection.** The sqlite3 connection is opened with
   `?mode=ro` URI flags so even a successful `run_query("DROP ...")`
   would error out at the SQLite layer rather than mutate the file.
   This is the load-bearing defense — every other layer assumes it.
2. **Table-name whitelist.** Tools that interpolate a table name into
   raw SQL (PRAGMA, SELECT FROM, ORDER BY) check `table_name` against
   `CHINOOK_TABLES` before constructing the statement. Defense in
   depth against a model that ignores the schema and dreams up names;
   without this, `[name]` quoting is the only barrier and that's not
   bulletproof against engineered inputs.
3. **Control tools always refuse.** `delete_record`, `drop_table`, and
   `truncate_table` return a structured error message regardless of
   the `operator_confirmed` flag. The flag's job is to teach the model
   the *protocol* during SFT — at corpus-gen time we want the
   protocol modelled in the assistant's reasoning, not actually
   executed on disk.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator


# Opaque marker passed back when a control tool is invoked. The
# generator includes this in the tool-response so the model sees the
# refusal pattern in-context.
CONTROL_REFUSAL_MESSAGE = (
    "REFUSED: this is a destructive control tool. The executor "
    "refuses to mutate the database in corpus-generation mode. "
    "If your prompt included operator_confirmed=true, the call shape "
    "is correct — the refusal here is environment policy, not your error."
)


# Canonical chinook tables (lerocha schema). The 15-tool agent surface
# only ever needs to operate on these. Any other `table_name` argument
# is rejected — defense against a hallucinated table name reaching raw
# SQL despite the read-only connection.
CHINOOK_TABLES: frozenset[str] = frozenset({
    "Album", "Artist", "Customer", "Employee", "Genre",
    "Invoice", "InvoiceLine", "MediaType",
    "Playlist", "PlaylistTrack", "Track",
})


def _check_table(table: str) -> dict | None:
    """Return an error dict if `table` isn't a canonical chinook table,
    else None. Tools call this before any SQL that interpolates the
    name; whitelist + read-only conn is the layered defense."""
    if table not in CHINOOK_TABLES:
        return {
            "error": f"unknown table: {table!r}. Valid tables: "
                     + ", ".join(sorted(CHINOOK_TABLES)),
        }
    return None


# Column-name pattern that's safe to interpolate. SQLite identifiers
# are letters, digits, and underscores; no quoting needed if the value
# matches this. Used as a guard before interpolating into PRAGMA / ORDER
# BY / GROUP BY positions where parameter binding isn't supported.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_ident(name: str, *, kind: str = "column") -> dict | None:
    if not _IDENT_RE.match(name):
        return {"error": f"invalid {kind} identifier: {name!r}"}
    return None


@contextmanager
def _ro_conn(db_path: str) -> Iterator[sqlite3.Connection]:
    """Open a read-only connection. URI mode lets us pass `mode=ro`,
    which is the SQLite-blessed way to refuse writes from any code path
    in the same process. Closes deterministically on context exit."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _rows_to_list(cursor: sqlite3.Cursor, limit: int = 100) -> list[dict]:
    """Materialise up to `limit` rows as plain dicts. We cap because
    a single tool response in the corpus shouldn't be many KB — the
    model only needs a representative sample to keep reasoning."""
    rows = cursor.fetchmany(limit)
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────
# Catalog cluster
# ─────────────────────────────────────────────────────────────────────


def list_tables(db_path: str, args: dict) -> dict:
    """Return all user tables in the database."""
    with _ro_conn(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        names = [row["name"] for row in cur.fetchall()]
    return {"tables": names}


def describe_table(db_path: str, args: dict) -> dict:
    """Return column metadata for a single table."""
    table = _require_str(args, "table_name")
    err = _check_table(table)
    if err:
        return err
    with _ro_conn(db_path) as conn:
        # PRAGMA doesn't accept parameter binding, but the table name
        # has been whitelisted above and the connection is read-only.
        cur = conn.execute(f"PRAGMA table_info([{table}])")
        cols = [dict(r) for r in cur.fetchall()]
    if not cols:
        return {"error": f"table not found: {table!r}"}
    return {"table": table, "columns": cols}


def sample_rows(db_path: str, args: dict) -> dict:
    """Return the first N rows of a table."""
    table = _require_str(args, "table_name")
    err = _check_table(table)
    if err:
        return err
    n = int(args.get("n", 5))
    with _ro_conn(db_path) as conn:
        try:
            cur = conn.execute(f"SELECT * FROM [{table}] LIMIT ?", (n,))
            rows = _rows_to_list(cur, limit=max(n, 1))
        except sqlite3.Error as e:
            return {"error": f"sqlite: {e}"}
    return {"table": table, "rows": rows}


# ─────────────────────────────────────────────────────────────────────
# Query cluster
# ─────────────────────────────────────────────────────────────────────


def run_query(db_path: str, args: dict) -> dict:
    """Execute a SELECT/WITH/EXPLAIN. The read-only connection rejects
    DML/DDL at the engine layer."""
    query = _require_str(args, "query")
    with _ro_conn(db_path) as conn:
        try:
            cur = conn.execute(query)
            rows = _rows_to_list(cur, limit=100)
        except sqlite3.Error as e:
            return {"error": f"sqlite: {e}"}
    return {"rows": rows, "row_count_capped_at": 100}


def count_rows(db_path: str, args: dict) -> dict:
    table = _require_str(args, "table_name")
    err = _check_table(table)
    if err:
        return err
    with _ro_conn(db_path) as conn:
        try:
            cur = conn.execute(f"SELECT COUNT(*) AS n FROM [{table}]")
            row = cur.fetchone()
        except sqlite3.Error as e:
            return {"error": f"sqlite: {e}"}
    return {"table": table, "count": row["n"]}


def top_n_by(db_path: str, args: dict) -> dict:
    table = _require_str(args, "table_name")
    order_by = _require_str(args, "order_by")
    err = _check_table(table) or _check_ident(order_by)
    if err:
        return err
    n = int(args.get("n", 10))
    ascending = bool(args.get("ascending", False))
    direction = "ASC" if ascending else "DESC"
    with _ro_conn(db_path) as conn:
        try:
            cur = conn.execute(
                f"SELECT * FROM [{table}] ORDER BY [{order_by}] {direction} LIMIT ?",
                (n,),
            )
            rows = _rows_to_list(cur, limit=max(n, 1))
        except sqlite3.Error as e:
            return {"error": f"sqlite: {e}"}
    return {"table": table, "order_by": order_by, "n": n, "rows": rows}


# ─────────────────────────────────────────────────────────────────────
# Search cluster (natural-language query → fuzzy LIKE match)
# ─────────────────────────────────────────────────────────────────────


def find_artist(db_path: str, args: dict) -> dict:
    return _fuzzy_lookup(db_path, args, table="Artist", column="Name")


def find_album(db_path: str, args: dict) -> dict:
    return _fuzzy_lookup(db_path, args, table="Album", column="Title")


def find_track(db_path: str, args: dict) -> dict:
    return _fuzzy_lookup(db_path, args, table="Track", column="Name")


def _fuzzy_lookup(db_path: str, args: dict, *, table: str, column: str) -> dict:
    # `table` and `column` are hardcoded by the caller (find_artist /
    # find_album / find_track); they aren't user input. Still cheap to
    # validate so the invariant is enforced one place.
    err = _check_table(table) or _check_ident(column)
    if err:
        return err
    query = _require_str(args, "query")
    limit = int(args.get("limit", 5))
    pattern = f"%{query}%"
    with _ro_conn(db_path) as conn:
        try:
            cur = conn.execute(
                f"SELECT * FROM [{table}] WHERE [{column}] LIKE ? LIMIT ?",
                (pattern, limit),
            )
            rows = _rows_to_list(cur, limit=max(limit, 1))
        except sqlite3.Error as e:
            return {"error": f"sqlite: {e}"}
    return {"table": table, "matches": rows}


# ─────────────────────────────────────────────────────────────────────
# Aggregate cluster
# ─────────────────────────────────────────────────────────────────────


def sum_by(db_path: str, args: dict) -> dict:
    return _aggregate(
        db_path, args, expr="SUM",
        agg_arg="sum_column", out_key="total",
    )


def avg_by(db_path: str, args: dict) -> dict:
    return _aggregate(
        db_path, args, expr="AVG",
        agg_arg="avg_column", out_key="avg",
    )


def group_summary(db_path: str, args: dict) -> dict:
    table = _require_str(args, "table_name")
    group_by = _require_str(args, "group_by")
    value_col = _require_str(args, "value_column")
    err = (
        _check_table(table)
        or _check_ident(group_by)
        or _check_ident(value_col)
    )
    if err:
        return err
    limit = int(args.get("limit", 25))
    sql = (
        f"SELECT [{group_by}] AS group_key, "
        f"COUNT(*) AS n, "
        f"SUM([{value_col}]) AS sum, AVG([{value_col}]) AS avg, "
        f"MIN([{value_col}]) AS min, MAX([{value_col}]) AS max "
        f"FROM [{table}] GROUP BY [{group_by}] LIMIT ?"
    )
    with _ro_conn(db_path) as conn:
        try:
            cur = conn.execute(sql, (limit,))
            rows = _rows_to_list(cur, limit=max(limit, 1))
        except sqlite3.Error as e:
            return {"error": f"sqlite: {e}"}
    return {"table": table, "group_by": group_by, "rows": rows}


def _aggregate(
    db_path: str, args: dict, *,
    expr: str, agg_arg: str, out_key: str,
) -> dict:
    table = _require_str(args, "table_name")
    group_by = _require_str(args, "group_by")
    agg_col = _require_str(args, agg_arg)
    err = (
        _check_table(table)
        or _check_ident(group_by)
        or _check_ident(agg_col)
    )
    if err:
        return err
    limit = int(args.get("limit", 25))
    sql = (
        f"SELECT [{group_by}] AS group_key, "
        f"{expr}([{agg_col}]) AS {out_key} "
        f"FROM [{table}] GROUP BY [{group_by}] LIMIT ?"
    )
    with _ro_conn(db_path) as conn:
        try:
            cur = conn.execute(sql, (limit,))
            rows = _rows_to_list(cur, limit=max(limit, 1))
        except sqlite3.Error as e:
            return {"error": f"sqlite: {e}"}
    return {"table": table, "group_by": group_by, "rows": rows}


# ─────────────────────────────────────────────────────────────────────
# Control cluster — always refuse during corpus-gen
# ─────────────────────────────────────────────────────────────────────


def delete_record(db_path: str, args: dict) -> dict:
    return {"error": CONTROL_REFUSAL_MESSAGE, "tool": "delete_record"}


def drop_table(db_path: str, args: dict) -> dict:
    return {"error": CONTROL_REFUSAL_MESSAGE, "tool": "drop_table"}


def truncate_table(db_path: str, args: dict) -> dict:
    return {"error": CONTROL_REFUSAL_MESSAGE, "tool": "truncate_table"}


# ─────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────


_DISPATCH = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "sample_rows": sample_rows,
    "run_query": run_query,
    "count_rows": count_rows,
    "top_n_by": top_n_by,
    "find_artist": find_artist,
    "find_album": find_album,
    "find_track": find_track,
    "sum_by": sum_by,
    "avg_by": avg_by,
    "group_summary": group_summary,
    "delete_record": delete_record,
    "drop_table": drop_table,
    "truncate_table": truncate_table,
}


def execute(tool_name: str, args: dict, *, db_path: str) -> str:
    """Dispatch a tool call by name. Returns the result as a JSON
    string (the shape the chat template expects on a `tool` role
    message's `content` field)."""
    handler = _DISPATCH.get(tool_name)
    if handler is None:
        result = {"error": f"unknown tool: {tool_name!r}"}
    else:
        try:
            result = handler(db_path, args or {})
        except Exception as e:  # noqa: BLE001 — broad on purpose
            # Tool execution must never abort the agent loop. A failed
            # tool returns a structured error so the model can react
            # to it the same way it would a real backend error.
            result = {"error": f"{type(e).__name__}: {e}", "tool": tool_name}
    return json.dumps(result, default=str)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _require_str(args: dict, key: str) -> str:
    """Return `args[key]` as a string or raise. Used at the top of
    every handler so a missing/malformed arg fails fast with a
    pinpointed message rather than a generic KeyError."""
    val = args.get(key)
    if not isinstance(val, str) or not val:
        raise ValueError(f"missing or empty arg: {key!r}")
    return val

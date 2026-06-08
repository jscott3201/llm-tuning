"""Tool manifest for the chinook SQL agent.

15 tools across 5 functional clusters (see `_common/eval_scoring.py`'s
`RELATED_GROUPS`). Schemas follow OpenAI's tool-format JSON, which
vLLM accepts directly when posted to `/v1/chat/completions`.

Cluster summary:

  catalog/    list_tables, describe_table, sample_rows
  query/      run_query, count_rows, top_n_by
  search/     find_artist, find_album, find_track       (natural-language)
  aggregate/  sum_by, avg_by, group_summary
  control/    delete_record, drop_table, truncate_table  (need confirmation)

Why 15 and not 5: the eval rubric's selection axis uses F1 *with*
partial credit for sibling tools. With only one tool per cluster the
partial-credit hop never fires; with 3-4 per cluster, scoring has
real signal at the boundary between "right tool" and "neighbour tool"
and "wrong cluster entirely."
"""

from __future__ import annotations

# Each entry follows OpenAI's `{"type": "function", "function": {...}}`
# wrapper. Build the wrapped form in `MANIFEST` at the bottom.
_RAW_TOOLS: list[dict] = [
    # ── catalog cluster ────────────────────────────────────────────
    {
        "name": "list_tables",
        "description": (
            "List all tables in the database. Returns an array of "
            "table names. Call when you need to see what data exists "
            "before deciding which table to query."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "describe_table",
        "description": (
            "Return the column names, types, and primary-key info "
            "for a single table. Call before constructing a SQL query "
            "if you don't already know the schema."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Exact table name (case-sensitive in chinook).",
                },
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "sample_rows",
        "description": (
            "Return the first N rows of a table. Useful when you need "
            "to see real values to know how the data is shaped (free-"
            "text fields, units, encoding) before writing a query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "n": {
                    "type": "integer",
                    "description": "Row count to return. Defaults to 5.",
                    "default": 5,
                },
            },
            "required": ["table_name"],
        },
    },
    # ── query cluster ──────────────────────────────────────────────
    {
        "name": "run_query",
        "description": (
            "Execute an arbitrary SELECT (or WITH/EXPLAIN) statement "
            "against the database. The query string MUST be SQL. Use "
            "this for any read that doesn't fit one of the higher-"
            "level helpers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A SQL statement. Must start with SELECT, WITH, or EXPLAIN.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "count_rows",
        "description": "Return SELECT COUNT(*) for a table.",
        "parameters": {
            "type": "object",
            "properties": {"table_name": {"type": "string"}},
            "required": ["table_name"],
        },
    },
    {
        "name": "top_n_by",
        "description": (
            "Return the top-N rows of a table sorted by a column. "
            "Use this instead of run_query when the user asks "
            "'top X by Y' — it's cheaper and harder to get wrong."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "order_by": {
                    "type": "string",
                    "description": "Column name to sort by.",
                },
                "n": {"type": "integer", "default": 10},
                "ascending": {"type": "boolean", "default": False},
            },
            "required": ["table_name", "order_by"],
        },
    },
    # ── search cluster (natural-language; skipped by sql_syntax) ───
    {
        "name": "find_artist",
        "description": (
            "Fuzzy-search for an artist by name. The query is "
            "natural-language, NOT SQL. Returns the closest matching "
            "ArtistId(s)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-form artist name or partial match.",
                },
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_album",
        "description": "Fuzzy-search for an album by title. NL query, NOT SQL.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_track",
        "description": "Fuzzy-search for a track by name. NL query, NOT SQL.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    # ── aggregate cluster ──────────────────────────────────────────
    {
        "name": "sum_by",
        "description": "Group a table by one column and sum another.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "group_by": {"type": "string"},
                "sum_column": {"type": "string"},
                "limit": {"type": "integer", "default": 25},
            },
            "required": ["table_name", "group_by", "sum_column"],
        },
    },
    {
        "name": "avg_by",
        "description": "Group a table by one column and average another.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "group_by": {"type": "string"},
                "avg_column": {"type": "string"},
                "limit": {"type": "integer", "default": 25},
            },
            "required": ["table_name", "group_by", "avg_column"],
        },
    },
    {
        "name": "group_summary",
        "description": (
            "Group a table by one column and return count, sum, avg, "
            "min, max for a numeric column. The 'one call to rule "
            "them all' for ad-hoc aggregation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "group_by": {"type": "string"},
                "value_column": {"type": "string"},
                "limit": {"type": "integer", "default": 25},
            },
            "required": ["table_name", "group_by", "value_column"],
        },
    },
    # ── control cluster (require operator_confirmed=true) ──────────
    {
        "name": "delete_record",
        "description": (
            "Delete a single row by primary key. DESTRUCTIVE. Refuse "
            "to call without operator_confirmed=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "record_id": {"type": "integer"},
                "operator_confirmed": {
                    "type": "boolean",
                    "default": False,
                    "description": "Must be true. Otherwise the call is refused.",
                },
            },
            "required": ["table_name", "record_id", "operator_confirmed"],
        },
    },
    {
        "name": "drop_table",
        "description": (
            "Drop a table. EXTREMELY DESTRUCTIVE. Refuse without "
            "operator_confirmed=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "operator_confirmed": {"type": "boolean", "default": False},
            },
            "required": ["table_name", "operator_confirmed"],
        },
    },
    {
        "name": "truncate_table",
        "description": (
            "Delete all rows in a table without dropping the table. "
            "DESTRUCTIVE. Refuse without operator_confirmed=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "operator_confirmed": {"type": "boolean", "default": False},
            },
            "required": ["table_name", "operator_confirmed"],
        },
    },
]


MANIFEST: list[dict] = [
    {"type": "function", "function": tool} for tool in _RAW_TOOLS
]
"""OpenAI-shaped tool list. Pass directly as the `tools` field on a
`/v1/chat/completions` request."""


TOOL_NAMES: list[str] = [t["name"] for t in _RAW_TOOLS]
"""Flat list of tool names — useful for sanity checks and the eval
scorer's `RELATED_GROUPS` cross-reference."""

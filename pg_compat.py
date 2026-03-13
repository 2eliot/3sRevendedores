"""
PostgreSQL compatibility layer for Revendedores51.
Wraps psycopg (v3) to expose a sqlite3-compatible interface so minimal
changes are needed in the rest of the codebase.

Handles automatically:
  - ? -> %s placeholder conversion
  - datetime('now') -> NOW()
  - PRAGMA -> no-op (silently ignored)
  - CREATE TABLE schema fixes: AUTOINCREMENT->SERIAL, DATETIME->TIMESTAMP
  - Row objects that support both dict-key and positional (row[0]) access
  - row_factory assignment (no-op, always uses dict_row)
"""

import os
import re
import logging

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL conversion helpers
# ---------------------------------------------------------------------------

_PRAGMA_RE   = re.compile(r'^\s*PRAGMA\s+', re.IGNORECASE)
_QM_RE       = re.compile(r'\?')
_DT_NOW_RE   = re.compile(r"datetime\('now'\)", re.IGNORECASE)
_AUTOINCR_RE = re.compile(r'\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b', re.IGNORECASE)
_DATETIME_RE = re.compile(r'\bDATETIME\b', re.IGNORECASE)
_TEXT_DT_RE  = re.compile(r"TEXT\s+DEFAULT\s+\(datetime\('now'\)\)", re.IGNORECASE)
_SQLITE_MASTER_RE = re.compile(r'\bsqlite_master\b', re.IGNORECASE)
_STRFTIME_RE = re.compile(r"strftime\(\s*'([^']+)'\s*,\s*([^)]+)\)", re.IGNORECASE)
# DATE(expr, '-N hours/minutes/days') → DATE(expr - INTERVAL 'N hours/minutes/days')
_DATE_MODIFIER_RE = re.compile(
    r"DATE\((.+?),\s*'([+-]?\s*\d+)\s+(hours?|minutes?|days?|seconds?)'\s*\)",
    re.IGNORECASE
)
# datetime('now', '-N hours/minutes') → (NOW() - INTERVAL 'N hours/minutes')
_DT_NOW_MOD_RE = re.compile(
    r"datetime\(\s*'now'\s*,\s*'([+-]?\s*\d+)\s+(hours?|minutes?|days?|seconds?)'\s*\)",
    re.IGNORECASE
)


def _replace_date_modifier(match):
    expr = match.group(1).strip()
    offset = match.group(2).strip()
    unit = match.group(3).strip()
    # Normalize sign: SQLite uses '-48 hours'; PG needs INTERVAL subtraction
    if offset.startswith('-'):
        return f"DATE({expr} - INTERVAL '{offset.lstrip('-').strip()} {unit}')"
    elif offset.startswith('+'):
        return f"DATE({expr} + INTERVAL '{offset.lstrip('+').strip()} {unit}')"
    else:
        return f"DATE({expr} + INTERVAL '{offset} {unit}')"


def _replace_dt_now_modifier(match):
    offset = match.group(1).strip()
    unit = match.group(2).strip()
    if offset.startswith('-'):
        return f"(NOW() - INTERVAL '{offset.lstrip('-').strip()} {unit}')"
    elif offset.startswith('+'):
        return f"(NOW() + INTERVAL '{offset.lstrip('+').strip()} {unit}')"
    else:
        return f"(NOW() + INTERVAL '{offset} {unit}')"


def _replace_strftime(match):
    fmt = (match.group(1) or '').strip()
    expr = (match.group(2) or '').strip()
    fmt_map = {
        '%Y-%m': 'YYYY-MM',
        '%Y-%m-%d': 'YYYY-MM-DD',
        '%d/%m/%Y': 'DD/MM/YYYY',
        '%H:%M:%S': 'HH24:MI:SS',
        '%Y': 'YYYY',
        '%m': 'MM',
        '%d': 'DD',
    }
    pg_fmt = fmt_map.get(fmt)
    if not pg_fmt:
        # fallback best-effort: drop % markers to avoid psycopg placeholder conflicts
        pg_fmt = fmt.replace('%', '')
    return f"to_char({expr}, '{pg_fmt}')"


def _convert_sql(sql: str):
    """
    Convert SQLite SQL to PostgreSQL SQL.
    Returns None for PRAGMA (no-op).
    Returns (converted_sql, is_returning) tuple otherwise.
    """
    stripped = sql.strip()

    # PRAGMA → no-op
    if _PRAGMA_RE.match(stripped):
        return None

    # ? → %s  (only outside string literals — simple replacement is fine
    # because we never use ? inside string values)
    sql = sql.replace('?', '%s')

    # SQLite strftime(...) -> PostgreSQL to_char(...)
    sql = _STRFTIME_RE.sub(_replace_strftime, sql)

    # DATE(expr, '-N hours') → DATE(expr - INTERVAL 'N hours')
    sql = _DATE_MODIFIER_RE.sub(_replace_date_modifier, sql)

    # datetime('now', '-N hours') → (NOW() - INTERVAL 'N hours')  — must come BEFORE plain datetime('now')
    sql = _DT_NOW_MOD_RE.sub(_replace_dt_now_modifier, sql)

    # datetime('now') → NOW()
    sql = _DT_NOW_RE.sub("NOW()", sql)

    # Remaining datetime(expr) → (expr)::timestamp  (SQLite cast; PG columns are already timestamps)
    sql = re.sub(r"datetime\(([^)]+)\)", r"(\1)::timestamp", sql, flags=re.IGNORECASE)

    # Escape literal % so psycopg doesn't parse e.g. %Y as placeholders.
    # Preserve valid placeholders: %s, %b, %t and %%.
    sql = re.sub(r'%(?![%sbt])', '%%', sql)

    # sqlite_master → information_schema.tables  (used in table_exists checks)
    sql = _SQLITE_MASTER_RE.sub("sqlite_master", sql)  # handled at call site

    upper = sql.upper()

    # Schema-level fixes (only needed in CREATE TABLE statements)
    if 'CREATE TABLE' in upper:
        sql = _AUTOINCR_RE.sub('SERIAL PRIMARY KEY', sql)
        sql = _TEXT_DT_RE.sub("TIMESTAMP DEFAULT NOW()", sql)
        sql = _DATETIME_RE.sub('TIMESTAMP', sql)

    return sql


# ---------------------------------------------------------------------------
# PgRow: dict + positional access (mimics sqlite3.Row)
# ---------------------------------------------------------------------------

class PgRow:
    """
    Wraps a psycopg dict row.
    Supports row['key']  (dict access)
    AND      row[0]      (positional access like sqlite3.Row)
    """
    __slots__ = ('_data', '_keys')

    def __init__(self, source):
        if source is None:
            self._data = {}
            self._keys = []
        else:
            self._data = dict(source)
            self._keys = list(self._data.keys())

    # --- dict-like access ---
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[self._keys[key]]
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self):
        return f'PgRow({self._data!r})'


# ---------------------------------------------------------------------------
# PgCursor wrapper
# ---------------------------------------------------------------------------

class PgCursor:
    """Wraps a psycopg v3 dict-row cursor to return PgRow objects."""

    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql: str, params=None):
        sql_pg = _convert_sql(sql)
        if sql_pg is None:
            return self  # PRAGMA no-op
        try:
            self._cur.execute(sql_pg, params)
        except Exception:
            raise
        return self

    def executemany(self, sql: str, params_list):
        sql_pg = _convert_sql(sql)
        if sql_pg is None:
            return self
        self._cur.executemany(sql_pg, params_list)
        return self

    def fetchone(self):
        row = self._cur.fetchone()
        return PgRow(row) if row is not None else None

    def fetchall(self):
        return [PgRow(r) for r in (self._cur.fetchall() or [])]

    def __iter__(self):
        for row in self._cur:
            yield PgRow(row)

    @property
    def rowcount(self):
        return self._cur.rowcount

    @property
    def lastrowid(self):
        return getattr(self._cur, 'lastrowid', None)

    def close(self):
        self._cur.close()


# ---------------------------------------------------------------------------
# _NoOpCursor — returned for PRAGMA and other ignored statements
# ---------------------------------------------------------------------------

class _NoOpCursor:
    rowcount = 0
    lastrowid = None

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def __iter__(self):
        return iter([])

    def close(self):
        pass


# ---------------------------------------------------------------------------
# PgConnection wrapper
# ---------------------------------------------------------------------------

class PgConnection:
    """
    Wraps a psycopg (v3) connection to expose sqlite3-compatible interface.

    Usage is identical to sqlite3:
        conn = get_db_connection()
        row  = conn.execute('SELECT * FROM usuarios WHERE id = ?', (uid,)).fetchone()
        conn.commit()
        conn.close()
    """

    def __init__(self, dsn: str):
        self._conn = psycopg.connect(dsn, row_factory=dict_row)
        # Importante: el código legacy usa muchos try/except para DDL (ALTER TABLE ...)
        # asumiendo comportamiento SQLite. En PostgreSQL, un error deja abortada la
        # transacción completa. Usamos autocommit para emular el flujo SQLite y evitar
        # InFailedSqlTransaction cuando esos errores esperados se capturan y se ignoran.
        self._conn.autocommit = True

    # row_factory is set in many places — make it a no-op
    @property
    def row_factory(self):
        return None

    @row_factory.setter
    def row_factory(self, value):
        pass  # always use RealDictCursor

    # ------------------------------------------------------------------
    def _raw_cursor(self) -> PgCursor:
        return PgCursor(self._conn.cursor())

    def execute(self, sql: str, params=None):
        sql_pg = _convert_sql(sql)
        if sql_pg is None:
            return _NoOpCursor()
        cur = self._raw_cursor()
        try:
            cur._cur.execute(sql_pg, params)
        except Exception:
            raise
        return cur

    def executemany(self, sql: str, params_list):
        sql_pg = _convert_sql(sql)
        if sql_pg is None:
            return _NoOpCursor()
        cur = self._raw_cursor()
        cur._cur.executemany(sql_pg, params_list)
        return cur

    def cursor(self) -> PgCursor:
        return self._raw_cursor()

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            try:
                self.rollback()
            except Exception:
                pass
        else:
            self.commit()
        self.close()


# ---------------------------------------------------------------------------
# table_exists helper (replaces sqlite_master checks)
# ---------------------------------------------------------------------------

def table_exists(conn: PgConnection, table_name: str) -> bool:
    """Check if a table exists in the current PostgreSQL database."""
    try:
        cur = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s",
            (table_name,)
        )
        return cur.fetchone() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

_DATABASE_URL: str = None  # type: ignore[assignment]


def _get_database_url() -> str:
    global _DATABASE_URL
    if _DATABASE_URL is None:
        url = os.environ.get('DATABASE_URL', '').strip()
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set. "
                "Example: postgresql://user:password@localhost:5432/revendedores"
            )
        # Heroku-style postgres:// → postgresql://
        if url.startswith('postgres://'):
            url = 'postgresql://' + url[len('postgres://'):]
        _DATABASE_URL = url
    return _DATABASE_URL


def get_db_connection() -> PgConnection:
    """Return a new PostgreSQL connection (sqlite3-compatible interface)."""
    return PgConnection(_get_database_url())


def get_db_connection_optimized() -> PgConnection:
    """Same as get_db_connection() — PostgreSQL doesn't need PRAGMA tuning."""
    return get_db_connection()

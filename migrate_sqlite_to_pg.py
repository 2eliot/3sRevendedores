"""
migrate_sqlite_to_pg.py
=======================
Migrates Revendedores51 data from SQLite (usuarios.db) to PostgreSQL.

Usage (on VPS):
    python3 migrate_sqlite_to_pg.py --sqlite /home/apps/web-b-revendedores/data/usuarios.db

Requires DATABASE_URL set in environment (or .env file).
"""

import os
import sys
import sqlite3
import argparse
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    sys.exit("psycopg not installed. Run: pip install 'psycopg[binary]'")


# ---------------------------------------------------------------------------
# Tables to migrate in dependency order (parent tables first)
# ---------------------------------------------------------------------------
TABLES = [
    'usuarios',
    'transacciones',
    'historial_compras',
    'pines_freefire',
    'pines_freefire_global',
    'precios_paquetes',
    'precios_freefire_global',
    'precios_freefire_id',
    'precios_bloodstriker',
    'transacciones_bloodstriker',
    'transacciones_freefire_id',
    'configuracion_redeemer',
    'configuracion_fuentes_pines',
    'precios_compra',
    'profit_ledger',
    'profit_daily_aggregate',
    'monthly_user_spending',
    'ventas_semanales',
    'creditos_billetera',
    'noticias',
    'noticias_vistas',
    'notificaciones_personalizadas',
    'admin_imported_files',
    'juegos_dinamicos',
    'paquetes_dinamicos',
    'transacciones_dinamicas',
    'api_recharges_log',
    'recargas_binance',
]


def get_pg_conn(url: str):
    if url.startswith('postgres://'):
        url = 'postgresql://' + url[len('postgres://'):]
    conn = psycopg.connect(url, row_factory=dict_row)
    conn.autocommit = False
    return conn


def get_sqlite_conn(path: str):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists_pg(pg_cur, table_name: str) -> bool:
    pg_cur.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name=%s",
        (table_name,)
    )
    return pg_cur.fetchone() is not None


def table_exists_sqlite(sq_cur, table_name: str) -> bool:
    sq_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return sq_cur.fetchone() is not None


def get_columns(sq_cur, table_name: str):
    sq_cur.execute(f'PRAGMA table_info("{table_name}")')
    return [row[1] for row in sq_cur.fetchall()]


def get_pg_columns(pg_cur, table_name: str):
    pg_cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table_name,)
    )
    return [r['column_name'] for r in pg_cur.fetchall()]


def migrate_table(sq_conn, pg_conn, table_name: str, dry_run=False):
    sq_cur = sq_conn.cursor()
    pg_cur = pg_conn.cursor()

    if not table_exists_sqlite(sq_cur, table_name):
        log.info(f"  SKIP {table_name} (not in SQLite)")
        return 0

    if not table_exists_pg(pg_cur, table_name):
        log.warning(f"  SKIP {table_name} (not in PostgreSQL — run app first to create schema)")
        return 0

    # Get column names and migrate only common columns (schema drift-safe)
    sq_cols = get_columns(sq_cur, table_name)
    pg_cols = set(get_pg_columns(pg_cur, table_name))
    cols = [c for c in sq_cols if c in pg_cols]
    if not cols:
        log.warning(f"  SKIP {table_name} (no common columns between SQLite and PostgreSQL)")
        return 0

    # Read all rows
    sq_col_list = ', '.join(f'"{c}"' for c in cols)
    sq_cur.execute(f'SELECT {sq_col_list} FROM "{table_name}"')
    rows = sq_cur.fetchall()
    if not rows:
        log.info(f"  {table_name}: 0 rows (empty)")
        return 0

    col_list = ', '.join(f'"{c}"' for c in cols)
    placeholders = ', '.join(['%s'] * len(cols))
    insert_sql = (
        f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})'
        ' ON CONFLICT DO NOTHING'
    )

    inserted = 0
    skipped = 0
    for row in rows:
        values = tuple(row[c] for c in cols)
        if dry_run:
            inserted += 1
            continue
        try:
            pg_cur.execute(insert_sql, values)
            if pg_cur.rowcount and pg_cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            log.debug(f"  {table_name} row skip: {e}")
            skipped += 1

    if not dry_run:
        pg_conn.commit()

    log.info(f"  {table_name}: {inserted} inserted, {skipped} skipped")
    return inserted


def truncate_tables(pg_conn, tables):
    """Truncate selected tables in reverse dependency order."""
    pg_cur = pg_conn.cursor()
    for table in reversed(tables):
        try:
            pg_cur.execute(f'TRUNCATE TABLE "{table}" RESTART IDENTITY CASCADE')
            log.info(f"  Truncated {table}")
        except Exception as e:
            log.warning(f"  Could not truncate {table}: {e}")
    pg_conn.commit()


def reset_sequences(pg_conn):
    """Reset all SERIAL sequences to max(id)+1 after bulk import."""
    pg_cur = pg_conn.cursor()
    pg_cur.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND column_default LIKE 'nextval%'
          AND column_name = 'id'
    """)
    tables = pg_cur.fetchall()
    for row in tables:
        tbl = row['table_name']
        try:
            pg_cur.execute(
                f"SELECT setval(pg_get_serial_sequence('{tbl}', 'id'), "
                f"COALESCE(MAX(id), 0) + 1, false) FROM \"{tbl}\""
            )
            log.info(f"  Reset sequence for {tbl}.id")
        except Exception as e:
            log.warning(f"  Could not reset sequence for {tbl}: {e}")
            pg_conn.rollback()
    pg_conn.commit()


def main():
    parser = argparse.ArgumentParser(description='Migrate SQLite → PostgreSQL for Revendedores51')
    parser.add_argument('--sqlite', required=True, help='Path to usuarios.db SQLite file')
    parser.add_argument('--dry-run', action='store_true', help='Count rows without inserting')
    parser.add_argument('--tables', nargs='*', help='Specific tables to migrate (default: all)')
    parser.add_argument('--truncate', action='store_true', help='Truncate target tables before import')
    args = parser.parse_args()

    db_url = os.environ.get('DATABASE_URL', '').strip()
    if not db_url:
        sys.exit("DATABASE_URL not set. Add it to .env or export it.")

    if not os.path.isfile(args.sqlite):
        sys.exit(f"SQLite file not found: {args.sqlite}")

    log.info(f"SQLite source: {args.sqlite}")
    log.info(f"PostgreSQL target: {db_url.split('@')[-1]}")
    log.info(f"Dry run: {args.dry_run}")
    log.info("")

    sq_conn = get_sqlite_conn(args.sqlite)
    pg_conn = get_pg_conn(db_url)

    tables = args.tables if args.tables else TABLES

    if args.truncate and not args.dry_run:
        log.info("Truncating target tables before import...")
        truncate_tables(pg_conn, tables)

    total = 0
    for table in tables:
        log.info(f"Migrating: {table}")
        n = migrate_table(sq_conn, pg_conn, table, dry_run=args.dry_run)
        total += n

    if not args.dry_run:
        log.info("")
        log.info("Resetting SERIAL sequences...")
        reset_sequences(pg_conn)

    sq_conn.close()
    pg_conn.close()

    log.info("")
    log.info(f"Done. Total rows processed: {total}")
    if args.dry_run:
        log.info("(dry run — nothing was written)")


if __name__ == '__main__':
    main()

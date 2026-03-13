from flask import Blueprint, jsonify, request
import os
from datetime import datetime, timedelta
import pytz
from pg_compat import get_db_connection, table_exists as pg_table_exists

bp = Blueprint('admin_stats', __name__)


def get_conn():
    return get_db_connection()


def table_exists(conn, table_name: str) -> bool:
    return pg_table_exists(conn, table_name)


def get_admin_exclusions():
    ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
    emails_env = os.environ.get('ADMIN_EMAILS', '').strip()
    single_email = os.environ.get('ADMIN_EMAIL', '').strip()
    ids = []
    emails = []
    if ids_env:
        for part in ids_env.split(','):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
    if emails_env:
        for part in emails_env.split(','):
            part = part.strip()
            if part:
                emails.append(part)
    # Fallback/alias: ADMIN_EMAIL
    if single_email and single_email not in emails:
        emails.append(single_email)
    return ids, emails


def compute_legacy_profit_by_day(conn, start_utc: str, end_utc: str):
    admin_ids, admin_emails = get_admin_exclusions()
    params = [start_utc, end_utc]
    filters = []
    if admin_ids:
        placeholders = ','.join('?' for _ in admin_ids)
        filters.append(f"u.id NOT IN ({placeholders})")
        params.extend(admin_ids)
    if admin_emails:
        placeholders = ','.join('?' for _ in admin_emails)
        filters.append(f"u.correo NOT IN ({placeholders})")
        params.extend(admin_emails)
    # Excluir cuentas sin_ganancia
    filters.append("COALESCE(u.sin_ganancia,FALSE)=FALSE")
    where_ex = (" AND " + " AND ".join(filters)) if filters else ""

    # Traer transacciones en rango, excluyendo admin y sin_ganancia
    tx_rows = conn.execute(
        f"""
        SELECT t.usuario_id, t.monto, t.fecha
        FROM transacciones t
        JOIN usuarios u ON u.id = t.usuario_id
        WHERE t.fecha >= ? AND t.fecha < ? {where_ex}
        """,
        params
    ).fetchall()

    # Mapear precios por juego
    def load_price_map(table_name, juego):
        lst = conn.execute(f"SELECT id, precio FROM {table_name}").fetchall()
        return [(row['id'], float(row['precio']), juego) for row in lst]

    prices = []
    try:
        prices.extend(load_price_map('precios_paquetes', 'freefire_latam'))
    except Exception:
        pass
    try:
        prices.extend(load_price_map('precios_freefire_global', 'freefire_global'))
    except Exception:
        pass
    try:
        prices.extend(load_price_map('precios_bloodstriker', 'bloodstriker'))
    except Exception:
        pass
    try:
        prices.extend(load_price_map('precios_freefire_id', 'freefire_id'))
    except Exception:
        pass

    # Costos por juego/paquete
    cost_map = {}
    try:
        for row in conn.execute("SELECT juego, paquete_id, precio_compra FROM precios_compra WHERE activo = 1"):
            cost_map[(row['juego'], row['paquete_id'])] = float(row['precio_compra'])
    except Exception:
        pass

    # Helper para encontrar paquete por monto (tolerancia 0.01)
    def match_package_by_amount(amount):
        best = None
        for pid, precio, juego in prices:
            if abs(amount - precio) < 0.011:  # ligera tolerancia
                best = (pid, precio, juego)
                break
        return best

    # Agrupar por día
    profit_by_day = {}
    for tx in tx_rows:
        try:
            # Considerar monto de venta como valor absoluto
            sale = abs(float(tx['monto'] or 0.0))
            if sale <= 0:
                continue
            match = match_package_by_amount(sale)
            if not match:
                continue
            pid, precio_vta, juego = match
            costo = cost_map.get((juego, pid), 0.0)
            profit = max(precio_vta - float(costo), 0.0)
            day = str(tx['fecha'])[:10]
            profit_by_day[day] = profit_by_day.get(day, 0.0) + profit
        except Exception:
            continue

    # Convertir a lista ordenada
    out = [{'day': day, 'profit': round(amount, 6)} for day, amount in sorted(profit_by_day.items())]
    return out


def tz_ranges(tz_name: str):
    tz = pytz.timezone(tz_name)
    now_local = datetime.now(tz)
    today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = today_start.replace(day=1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)
    prev_month_start = (month_start - timedelta(days=1)).replace(day=1)
    return {
        'today_start': today_start,
        'today_end': today_start + timedelta(days=1),
        'yesterday_start': today_start - timedelta(days=1),
        'yesterday_end': today_start,
        'dby_start': today_start - timedelta(days=2),
        'dby_end': today_start - timedelta(days=1),
        'week_start': today_start - timedelta(days=6),
        'week_end': today_start + timedelta(days=1),
        'month_start': month_start,
        'month_end': next_month,
        'prev_month_start': prev_month_start,
        'prev_month_end': month_start,
        'tz': tz
    }


def to_utc_iso(dt_local):
    if dt_local.tzinfo is None:
        raise ValueError('Expected timezone-aware datetime')
    return dt_local.astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')


@bp.route('/top-clients')
def top_clients():
    tz_name = request.args.get('tz', 'America/Caracas')
    rng = tz_ranges(tz_name)
    ms = to_utc_iso(rng['month_start'])
    me = to_utc_iso(rng['month_end'])
    try:
        conn = get_conn()
        if table_exists(conn, 'users') and table_exists(conn, 'spent_ledger') and table_exists(conn, 'package_purchases_ledger'):
            sql = """
            WITH spent AS (
              SELECT sl.user_id, COALESCE(SUM(sl.amount), 0) AS total_spent
              FROM spent_ledger sl
              WHERE sl.created_at >= ? AND sl.created_at < ?
              GROUP BY sl.user_id
            ),
            purchases AS (
              SELECT ppl.user_id, SUM(CASE WHEN ppl.quantity > 0 THEN 1 ELSE 0 END) AS purchases_count
              FROM package_purchases_ledger ppl
              WHERE ppl.created_at >= ? AND ppl.created_at < ?
              GROUP BY ppl.user_id
            ),
            combined AS (
              SELECT u.id AS user_id,
                     COALESCE(u.name, u.email) AS display_name,
                     u.email AS email,
                     COALESCE(spent.total_spent, 0) AS total_spent,
                     COALESCE(purchases.purchases_count, 0) AS purchases_count
              FROM users u
              LEFT JOIN spent ON spent.user_id = u.id
              LEFT JOIN purchases ON purchases.user_id = u.id
              WHERE (COALESCE(spent.total_spent,0) > 0 OR COALESCE(purchases.purchases_count,0) > 0)
            ),
            max_total AS (
              SELECT MAX(total_spent) AS max_total FROM combined
            )
            SELECT c.user_id, c.display_name, c.email, c.total_spent, c.purchases_count,
                   CASE WHEN mt.max_total > 0 THEN (c.total_spent * 100.0 / mt.max_total) ELSE 0 END AS pct
            FROM combined c, max_total mt
            ORDER BY c.total_spent DESC, c.purchases_count DESC
            LIMIT 5
            """
            rows = conn.execute(sql, (ms, me, ms, me)).fetchall()
        else:
            # Legacy schema: usar monthly_user_spending si existe, sino transacciones
            if table_exists(conn, 'monthly_user_spending'):
                # Calcular year_month del rango
                from datetime import datetime
                import pytz
                tz = pytz.timezone('America/Caracas')
                start_local = datetime.fromisoformat(ms.replace(' ', 'T')).replace(tzinfo=pytz.utc).astimezone(tz)
                year_month = start_local.strftime('%Y-%m')
                
                sql = """
                WITH combined AS (
                  SELECT u.id AS user_id,
                         COALESCE(u.nombre || ' ' || u.apellido, u.correo) AS display_name,
                         u.correo AS email,
                         COALESCE(mus.total_spent, 0) AS total_spent,
                         COALESCE(mus.purchases_count, 0) AS purchases_count
                  FROM monthly_user_spending mus
                  JOIN usuarios u ON u.id = mus.usuario_id
                  WHERE mus.year_month = ? AND COALESCE(u.sin_ganancia,FALSE)=FALSE
                ),
                max_total AS (
                  SELECT MAX(total_spent) AS max_total FROM combined
                )
                SELECT c.user_id,
                       c.display_name,
                       c.email,
                       c.total_spent,
                       c.purchases_count,
                       CASE WHEN mt.max_total > 0 THEN (c.total_spent * 100.0 / mt.max_total) ELSE 0 END AS pct
                FROM combined c, max_total mt
                ORDER BY c.total_spent DESC, c.purchases_count DESC
                LIMIT 5
                """
                rows = conn.execute(sql, (year_month,)).fetchall()
            else:
                # Fallback a transacciones (menos confiable por la limpieza)
                sql = """
                WITH base AS (
                  SELECT t.usuario_id AS user_id,
                         SUM(CASE WHEN t.monto < 0 THEN -t.monto ELSE t.monto END) AS total_spent,
                         COUNT(*) AS purchases_count
                  FROM transacciones t
                  WHERE t.fecha >= ? AND t.fecha < ?
                  GROUP BY t.usuario_id
                ),
                combined AS (
                  SELECT u.id AS user_id,
                         COALESCE(u.nombre || ' ' || u.apellido, u.correo) AS display_name,
                         u.correo AS email,
                         COALESCE(b.total_spent,0) AS total_spent,
                         COALESCE(b.purchases_count,0) AS purchases_count
                  FROM base b
                  JOIN usuarios u ON u.id = b.user_id
                  WHERE COALESCE(u.sin_ganancia,FALSE)=FALSE
                ),
                max_total AS (
                  SELECT MAX(total_spent) AS max_total FROM combined
                )
                SELECT c.user_id,
                       c.display_name,
                       c.email,
                       c.total_spent,
                       c.purchases_count,
                       CASE WHEN mt.max_total > 0 THEN (COALESCE(c.total_spent,0) * 100.0 / mt.max_total) ELSE 0 END AS pct
                FROM combined c, max_total mt
                ORDER BY c.total_spent DESC, c.purchases_count DESC
                LIMIT 5
                """
                rows = conn.execute(sql, (ms, me)).fetchall()
        conn.close()
        data = [dict(r) for r in rows]
        # Rename admin accounts
        admin_ids, admin_emails = get_admin_exclusions()
        if admin_ids or admin_emails:
            for r in data:
                try:
                    if (r.get('user_id') in admin_ids) or (r.get('email') in admin_emails):
                        r['display_name'] = 'Admin'
                except Exception:
                    pass
        # Clean email from output (not needed on front)
        for r in data:
            if 'email' in r:
                del r['email']
        return jsonify({
            'top_clients_month': data,
            'period': {
                'from': ms,
                'to': me,
                'tz': tz_name
            }
        })
    except Exception as e:
        return jsonify({'error': 'query_failed', 'message': str(e)}), 500


@bp.route('/summary')
def summary():
    parts = {}
    tz_name = request.args.get('tz', 'America/Caracas')
    rng = tz_ranges(tz_name)
    ms = to_utc_iso(rng['month_start'])
    me = to_utc_iso(rng['month_end'])
    # Fechas locales para consultas en tablas legacy agregadas por día
    month_start_day = rng['month_start'].date().isoformat()
    month_end_day = rng['month_end'].date().isoformat()
    try:
        conn = get_conn()
        if table_exists(conn, 'spent_ledger'):
            row = conn.execute("SELECT COALESCE(SUM(amount),0) AS total_spent_all_time FROM spent_ledger").fetchone()
        else:
            row = conn.execute("SELECT COALESCE(SUM(CASE WHEN monto < 0 THEN -monto ELSE monto END),0) AS total_spent_all_time FROM transacciones").fetchone()
        parts['total_spent_all_time'] = row['total_spent_all_time'] if row else 0
    except Exception as e:
        parts['total_spent_all_time_error'] = str(e)
    try:
        conn = get_conn()
        if table_exists(conn, 'user_balances') and table_exists(conn, 'users'):
            row = conn.execute(
                "SELECT COALESCE(SUM(ub.balance),0) AS active_balance_total FROM user_balances ub JOIN users u ON u.id = ub.user_id WHERE COALESCE(u.is_admin,0)=0"
            ).fetchone()
        else:
            row = conn.execute("SELECT COALESCE(SUM(saldo),0) AS active_balance_total FROM usuarios WHERE COALESCE(sin_ganancia,FALSE)=FALSE").fetchone()
        parts['active_balance_total'] = row['active_balance_total'] if row else 0
    except Exception as e:
        parts['active_balance_total_error'] = str(e)
    try:
        conn = get_conn()
        if table_exists(conn, 'menu_item_pins') and table_exists(conn, 'menu_item_packages'):
            row = conn.execute(
                """
                WITH valid_pins AS (
                  SELECT mip.menu_item_package_id FROM menu_item_pins mip
                  JOIN menu_item_packages p ON p.id = mip.menu_item_package_id
                  WHERE COALESCE(mip.is_used,0)=0
                )
                SELECT COUNT(*) AS pins_available_total,
                       COALESCE(SUM(p.price),0) AS pins_balance_total
                FROM valid_pins vp
                JOIN menu_item_packages p ON p.id = vp.menu_item_package_id
                """
            ).fetchone()
            parts['pins_available_total'] = row['pins_available_total'] if row else 0
            parts['pins_balance_total'] = row['pins_balance_total'] if row else 0
        else:
            # Legacy: pines_freefire (+ _global) and price tables
            # available count
            row1 = conn.execute("SELECT COUNT(*) c FROM pines_freefire WHERE usado = FALSE").fetchone()
            row2 = None
            if table_exists(conn, 'pines_freefire_global'):
                row2 = conn.execute("SELECT COUNT(*) c FROM pines_freefire_global WHERE usado = FALSE").fetchone()
            pins_available_total = (row1['c'] if row1 else 0) + ((row2['c'] if row2 else 0) if row2 else 0)
            # balance total: sum per monto_id * precio
            sum_latam = conn.execute(
                """
                SELECT COALESCE(SUM(pp.precio * cnt.c),0) AS s
                FROM (
                  SELECT monto_id, COUNT(*) AS c FROM pines_freefire WHERE usado = FALSE GROUP BY monto_id
                ) cnt
                JOIN precios_paquetes pp ON pp.id = cnt.monto_id
                """
            ).fetchone()
            sum_global = {'s': 0}
            if table_exists(conn, 'pines_freefire_global') and table_exists(conn, 'precios_freefire_global'):
                sum_global = conn.execute(
                    """
                    SELECT COALESCE(SUM(pfg.precio * cnt.c),0) AS s
                    FROM (
                      SELECT monto_id, COUNT(*) AS c FROM pines_freefire_global WHERE usado = FALSE GROUP BY monto_id
                    ) cnt
                    JOIN precios_freefire_global pfg ON pfg.id = cnt.monto_id
                    """
                ).fetchone()
            parts['pins_available_total'] = pins_available_total
            parts['pins_balance_total'] = (sum_latam['s'] if sum_latam else 0) + (sum_global['s'] if sum_global else 0)
    except Exception as e:
        parts['pins_error'] = str(e)
    # Total de usuarios (canónico o legacy)
    try:
        conn = get_conn()
        if table_exists(conn, 'users'):
            r = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
            parts['users_total'] = r['c'] if r else 0
        else:
            r = conn.execute("SELECT COUNT(*) AS c FROM usuarios").fetchone()
            parts['users_total'] = r['c'] if r else 0
        conn.close()
    except Exception as e:
        parts['users_total_error'] = str(e)
    # Profit total del mes (excluyendo admin)
    try:
        conn = get_conn()
        if table_exists(conn, 'package_purchases_ledger') and table_exists(conn, 'users'):
            r = conn.execute(
                """
                SELECT COALESCE(SUM(ppl.profit),0) AS profit
                FROM package_purchases_ledger ppl
                JOIN users u ON u.id = ppl.user_id
                WHERE COALESCE(u.is_admin,0)=0 AND ppl.created_at >= ? AND ppl.created_at < ?
                """,
                (ms, me)
            ).fetchone()
            parts['profit_month_total'] = r['profit'] if r else 0
        else:
            # Legacy persistente: usar agregados diarios
            if table_exists(conn, 'profit_daily_aggregate'):
                row = conn.execute(
                    "SELECT COALESCE(SUM(profit_total),0) AS s FROM profit_daily_aggregate WHERE day >= ? AND day < ?",
                    (month_start_day, month_end_day)
                ).fetchone()
                parts['profit_month_total'] = row['s'] if row else 0
            else:
                # Fallback: calcular al vuelo
                lst = compute_legacy_profit_by_day(conn, ms, me)
                if isinstance(lst, list):
                    parts['profit_month_total'] = round(sum(item.get('profit', 0) for item in lst), 6)
                else:
                    parts['profit_month_total'] = 0
    except Exception as e:
        parts['profit_month_total_error'] = str(e)
    return jsonify({'summary': parts})


@bp.route('/pins-daily')
def pins_daily():
    tz_name = request.args.get('tz', 'America/Caracas')
    rng = tz_ranges(tz_name)
    periods = [
        ('today', rng['today_start'], rng['today_end']),
        ('yesterday', rng['yesterday_start'], rng['yesterday_end']),
        ('dby', rng['dby_start'], rng['dby_end'])
    ]
    out = []
    for key, start, end in periods:
        s = to_utc_iso(start)
        e = to_utc_iso(end)
        try:
            conn = get_conn()
            if table_exists(conn, 'menu_item_pins') and table_exists(conn, 'menu_item_pins_used'):
                a1 = conn.execute("SELECT COUNT(*) c FROM menu_item_pins WHERE created_at >= ? AND created_at < ?", (s, e)).fetchone()
                a2 = conn.execute("SELECT COUNT(*) c FROM menu_item_pins_used WHERE created_at >= ? AND created_at < ?", (s, e)).fetchone()
                u1 = conn.execute("SELECT COUNT(*) c FROM menu_item_pins_used WHERE used_at >= ? AND used_at < ?", (s, e)).fetchone()
                pins_added = (a1['c'] if a1 else 0) + (a2['c'] if a2 else 0)
                pins_used = u1['c'] if u1 else 0
            else:
                # Legacy: count aggregated per day using fecha_agregado and fecha_usado
                a_latam = conn.execute("SELECT COUNT(*) c FROM pines_freefire WHERE fecha_agregado >= ? AND fecha_agregado < ?", (s, e)).fetchone()
                a_global = None
                if table_exists(conn, 'pines_freefire_global'):
                    a_global = conn.execute("SELECT COUNT(*) c FROM pines_freefire_global WHERE fecha_agregado >= ? AND fecha_agregado < ?", (s, e)).fetchone()
                u_latam = conn.execute("SELECT COUNT(*) c FROM pines_freefire WHERE fecha_usado IS NOT NULL AND fecha_usado >= ? AND fecha_usado < ?", (s, e)).fetchone()
                u_global = None
                if table_exists(conn, 'pines_freefire_global'):
                    u_global = conn.execute("SELECT COUNT(*) c FROM pines_freefire_global WHERE fecha_usado IS NOT NULL AND fecha_usado >= ? AND fecha_usado < ?", (s, e)).fetchone()
                pins_added = (a_latam['c'] if a_latam else 0) + ((a_global['c'] if a_global else 0) if a_global else 0)
                pins_used = (u_latam['c'] if u_latam else 0) + ((u_global['c'] if u_global else 0) if u_global else 0)
            conn.close()
            out.append({
                'date': start.date().isoformat(),
                'pins_added': pins_added,
                'pins_used': pins_used
            })
        except Exception as e2:
            out.append({'date': start.date().isoformat(), 'error': str(e2)})
    return jsonify({'pins_daily': out, 'tz': tz_name})


@bp.route('/timeseries')
def timeseries():
    tz_name = request.args.get('tz', 'America/Caracas')
    rng = tz_ranges(tz_name)
    ms, me = to_utc_iso(rng['month_start']), to_utc_iso(rng['month_end'])
    ws, we = to_utc_iso(rng['week_start']), to_utc_iso(rng['week_end'])
    pms, pme = to_utc_iso(rng['prev_month_start']), to_utc_iso(rng['prev_month_end'])
    ms_day, me_day = rng['month_start'].date().isoformat(), rng['month_end'].date().isoformat()
    pms_day, pme_day = rng['prev_month_start'].date().isoformat(), rng['prev_month_end'].date().isoformat()
    res = {
        'daily_spent_month': [],
        'daily_spent_week': [],
        'profit_daily_month': [],
        'profit_daily_prev_month': []
    }
    try:
        conn = get_conn()
        if table_exists(conn, 'spent_ledger'):
            q1 = conn.execute(
                "SELECT date(created_at) AS day, COALESCE(SUM(amount),0) total_spent FROM spent_ledger WHERE created_at >= ? AND created_at < ? GROUP BY date(created_at) ORDER BY day",
                (ms, me)
            ).fetchall()
        else:
            q1 = conn.execute(
                "SELECT date(fecha) AS day, COALESCE(SUM(CASE WHEN monto < 0 THEN -monto ELSE monto END),0) total_spent FROM transacciones WHERE fecha >= ? AND fecha < ? GROUP BY date(fecha) ORDER BY day",
                (ms, me)
            ).fetchall()
        res['daily_spent_month'] = [dict(r) for r in q1]
    except Exception as e:
        res['daily_spent_month_error'] = str(e)
    try:
        conn = get_conn()
        if table_exists(conn, 'spent_ledger'):
            q2 = conn.execute(
                "SELECT date(created_at) AS day, COALESCE(SUM(amount),0) total_spent FROM spent_ledger WHERE created_at >= ? AND created_at < ? GROUP BY date(created_at) ORDER BY day",
                (ws, we)
            ).fetchall()
        else:
            q2 = conn.execute(
                "SELECT date(fecha) AS day, COALESCE(SUM(CASE WHEN monto < 0 THEN -monto ELSE monto END),0) total_spent FROM transacciones WHERE fecha >= ? AND fecha < ? GROUP BY date(fecha) ORDER BY day",
                (ws, we)
            ).fetchall()
        res['daily_spent_week'] = [dict(r) for r in q2]
    except Exception as e:
        res['daily_spent_week_error'] = str(e)
    try:
        conn = get_conn()
        if table_exists(conn, 'package_purchases_ledger') and table_exists(conn, 'users'):
            q3 = conn.execute(
                """
                SELECT date(ppl.created_at) AS day, COALESCE(SUM(ppl.profit),0) AS profit
                FROM package_purchases_ledger ppl
                JOIN users u ON u.id = ppl.user_id
                WHERE COALESCE(u.is_admin,0)=0 AND ppl.created_at >= ? AND ppl.created_at < ?
                GROUP BY date(ppl.created_at)
                ORDER BY day
                """,
                (ms, me)
            ).fetchall()
        elif table_exists(conn, 'profit_daily_aggregate'):
            q3 = conn.execute(
                "SELECT day, profit_total AS profit FROM profit_daily_aggregate WHERE day >= ? AND day < ? ORDER BY day",
                (ms_day, me_day)
            ).fetchall()
        elif table_exists(conn, 'transacciones'):
            q3 = compute_legacy_profit_by_day(conn, ms, me)
        else:
            q3 = []
        res['profit_daily_month'] = [dict(r) for r in q3] if isinstance(q3, list) else [dict(r) for r in q3]
    except Exception as e:
        res['profit_daily_month_error'] = str(e)
    try:
        conn = get_conn()
        if table_exists(conn, 'package_purchases_ledger') and table_exists(conn, 'users'):
            q4 = conn.execute(
                """
                SELECT date(ppl.created_at) AS day, COALESCE(SUM(ppl.profit),0) AS profit
                FROM package_purchases_ledger ppl
                JOIN users u ON u.id = ppl.user_id
                WHERE COALESCE(u.is_admin,0)=0 AND ppl.created_at >= ? AND ppl.created_at < ?
                GROUP BY date(ppl.created_at)
                ORDER BY day
                """,
                (pms, pme)
            ).fetchall()
        elif table_exists(conn, 'profit_daily_aggregate'):
            q4 = conn.execute(
                "SELECT day, profit_total AS profit FROM profit_daily_aggregate WHERE day >= ? AND day < ? ORDER BY day",
                (pms_day, pme_day)
            ).fetchall()
        elif table_exists(conn, 'transacciones'):
            q4 = compute_legacy_profit_by_day(conn, pms, pme)
        else:
            q4 = []
        res['profit_daily_prev_month'] = [dict(r) for r in q4] if isinstance(q4, list) else [dict(r) for r in q4]
    except Exception as e:
        res['profit_daily_prev_month_error'] = str(e)
    return jsonify({'timeseries': res, 'tz': tz_name})


@bp.route('/backfill-legacy-profit', methods=['POST'])
def backfill_legacy_profit():
    """Backfill de agregados diarios de profit para esquema legacy.
    Protegido por token: enviar ?token=... que debe coincidir con BACKFILL_TOKEN en env.
    Parámetros:
      - days: int (default 60)
      - tz: zona horaria (default America/Caracas)
    """
    token = request.args.get('token') or (request.get_json(silent=True) or {}).get('token')
    expected = os.environ.get('BACKFILL_TOKEN')
    if not expected or token != expected:
        return jsonify({'error': 'unauthorized'}), 401
    days = request.args.get('days', type=int) or (request.get_json(silent=True) or {}).get('days') or 60
    tz_name = request.args.get('tz', 'America/Caracas')
    rng = tz_ranges(tz_name)
    tz = rng['tz']
    try:
        conn = get_conn()
        cur = conn.cursor()
        today_local = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        count = 0
        for i in range(days):
            day_start = today_local - timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            s, e = to_utc_iso(day_start), to_utc_iso(day_end)
            # Calcular usando fallback actual
            lst = compute_legacy_profit_by_day(conn, s, e)
            total = 0.0
            if isinstance(lst, list):
                total = round(sum(item.get('profit', 0) for item in lst), 6)
            # Upsert en profit_daily_aggregate
            day = day_start.date().isoformat()
            ex = cur.execute("SELECT profit_total FROM profit_daily_aggregate WHERE day=?", (day,)).fetchone()
            if ex:
                cur.execute("UPDATE profit_daily_aggregate SET profit_total=?, updated_at=datetime('now') WHERE day=?", (total, day))
            else:
                cur.execute("INSERT INTO profit_daily_aggregate(day, profit_total) VALUES(?, ?)", (day, total))
            count += 1
        conn.commit()
        conn.close()
        return jsonify({'ok': True, 'days_processed': count})
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return jsonify({'error': 'backfill_failed', 'message': str(e)}), 500


@bp.route('/packages-history')
def packages_history():
    tz_name = request.args.get('tz', 'America/Caracas')
    period = request.args.get('period', 'month')
    rng = tz_ranges(tz_name)
    if period == 'day':
        rs, re = rng['today_start'], rng['today_end']
    elif period == 'week':
        rs, re = rng['week_start'], rng['week_end']
    else:
        rs, re = rng['month_start'], rng['month_end']
    s, e = to_utc_iso(rs), to_utc_iso(re)
    try:
        conn = get_conn()
        if table_exists(conn, 'package_purchases_ledger') and table_exists(conn, 'menu_item_packages'):
            sql = (
                "SELECT mip.menu_item_id AS category_id, mip.name AS package_name, COALESCE(SUM(ppl.quantity),0) AS total_quantity "
                "FROM package_purchases_ledger ppl "
                "JOIN menu_item_packages mip ON mip.id = ppl.menu_item_package_id "
                "JOIN users u ON u.id = ppl.user_id AND COALESCE(u.is_admin,0)=0 "
                "WHERE ppl.created_at >= ? AND ppl.created_at < ? "
                "GROUP BY mip.menu_item_id, mip.name ORDER BY total_quantity DESC"
            )
            rows = conn.execute(sql, (s, e)).fetchall()
        elif table_exists(conn, 'ventas_semanales'):
            rows = conn.execute(
                "SELECT paquete_id AS category_id, paquete_nombre AS package_name, COALESCE(SUM(cantidad_vendida),0) AS total_quantity FROM ventas_semanales WHERE fecha_venta >= ? AND fecha_venta < ? GROUP BY paquete_id, paquete_nombre ORDER BY total_quantity DESC",
                (s, e)
            ).fetchall()
        else:
            rows = []
        conn.close()
        return jsonify({f'packages_history_{period}': [dict(r) for r in rows], 'period': {'from': s, 'to': e, 'tz': tz_name}})
    except Exception as e:
        return jsonify({'error': 'query_failed', 'message': str(e)}), 500


@bp.route('/profit-packages-config', methods=['GET', 'POST'])
def profit_packages_config():
    if request.method == 'GET':
        try:
            conn = get_conn()
            if table_exists(conn, 'menu_item_packages') and table_exists(conn, 'package_cost_config'):
                sql = (
                    "SELECT mip.id AS package_id, 'canonical' AS source, mip.menu_item_id AS category_id, mip.name AS package_name, mip.price AS base_price, pcc.cost "
                    "FROM menu_item_packages mip LEFT JOIN package_cost_config pcc ON pcc.menu_item_package_id = mip.id"
                )
                rows = conn.execute(sql).fetchall()
            else:
                # Build union from legacy price tables + precios_compra
                rows = []
                # Freefire LATAM
                latam = conn.execute(
                    """
                    SELECT id AS package_id, 'freefire_latam' AS source, id AS category_id, nombre AS package_name, precio AS base_price, (
                       SELECT precio_compra FROM precios_compra pc WHERE pc.juego='freefire_latam' AND pc.paquete_id = precios_paquetes.id AND pc.activo = 1 LIMIT 1
                     ) AS cost FROM precios_paquetes
                    """
                ).fetchall()
                rows.extend(latam)
                if table_exists(conn, 'precios_freefire_global'):
                    g = conn.execute(
                        """
                        SELECT id AS package_id, 'freefire_global' AS source, id AS category_id, nombre AS package_name, precio AS base_price, (
                           SELECT precio_compra FROM precios_compra pc WHERE pc.juego='freefire_global' AND pc.paquete_id = precios_freefire_global.id AND pc.activo = 1 LIMIT 1
                         ) AS cost FROM precios_freefire_global
                        """
                    ).fetchall()
                    rows.extend(g)
                if table_exists(conn, 'precios_bloodstriker'):
                    b = conn.execute(
                        """
                        SELECT id AS package_id, 'bloodstriker' AS source, id AS category_id, nombre AS package_name, precio AS base_price, (
                           SELECT precio_compra FROM precios_compra pc WHERE pc.juego='bloodstriker' AND pc.paquete_id = precios_bloodstriker.id AND pc.activo = 1 LIMIT 1
                         ) AS cost FROM precios_bloodstriker
                        """
                    ).fetchall()
                    rows.extend(b)
                if table_exists(conn, 'precios_freefire_id'):
                    fi = conn.execute(
                        """
                        SELECT id AS package_id, 'freefire_id' AS source, id AS category_id, nombre AS package_name, precio AS base_price, (
                           SELECT precio_compra FROM precios_compra pc WHERE pc.juego='freefire_id' AND pc.paquete_id = precios_freefire_id.id AND pc.activo = 1 LIMIT 1
                         ) AS cost FROM precios_freefire_id
                        """
                    ).fetchall()
                    rows.extend(fi)
                if table_exists(conn, 'paquetes_dinamicos') and table_exists(conn, 'juegos_dinamicos'):
                    dyn = conn.execute(
                        """
                        SELECT pd.id AS package_id, 'dyn_' || jd.slug AS source, pd.id AS category_id, pd.nombre AS package_name, pd.precio AS base_price, (
                           SELECT precio_compra FROM precios_compra pc WHERE pc.juego = 'dyn_' || jd.slug AND pc.paquete_id = pd.id AND pc.activo = 1 LIMIT 1
                         ) AS cost FROM paquetes_dinamicos pd JOIN juegos_dinamicos jd ON jd.id = pd.juego_id
                        """
                    ).fetchall()
                    rows.extend(dyn)
            conn.close()
            return jsonify({'profit_packages_config': [dict(r) for r in rows]})
        except Exception as e:
            return jsonify({'error': 'query_failed', 'message': str(e)}), 500
    else:
        # POST: upsert costs
        try:
            payload = request.get_json(silent=True) or {}
            items = payload.get('items', [])
            if not isinstance(items, list):
                return jsonify({'error': 'invalid_payload'}), 400
            conn = get_conn()
            cur = conn.cursor()
            if table_exists(conn, 'menu_item_packages') and table_exists(conn, 'package_cost_config'):
                # canonical
                for it in items:
                    pkg_id = it.get('package_id')
                    cost = it.get('cost')
                    if pkg_id is None or cost is None:
                        continue
                    cur.execute("INSERT INTO package_cost_config (menu_item_package_id, cost) VALUES (?, ?) ON CONFLICT(menu_item_package_id) DO UPDATE SET cost=excluded.cost", (pkg_id, float(cost)))
            else:
                # legacy precios_compra
                for it in items:
                    juego = it.get('source')
                    paquete_id = it.get('package_id')
                    cost = it.get('cost')
                    if not juego or paquete_id is None or cost is None:
                        continue
                    # Upsert precios_compra (unique juego, paquete_id). Ensure activo=1
                    cur.execute(
                        """
                        INSERT INTO precios_compra (juego, paquete_id, precio_compra, activo)
                        VALUES (?, ?, ?, 1)
                        ON CONFLICT(juego, paquete_id) DO UPDATE SET precio_compra=excluded.precio_compra, activo=1
                        """,
                        (juego, int(paquete_id), float(cost))
                    )
            conn.commit()
            conn.close()
            return jsonify({'ok': True})
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            return jsonify({'error': 'update_failed', 'message': str(e)}), 500

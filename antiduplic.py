"""
Antiduplic (Referencias) — pagos de Bancamiga y verificación anti-duplicados (Blueprint Flask).

Flujo:
  1. Un bot en la PC del dueño descarga los movimientos de Bancamiga y los envía a
     POST /api/pagos-banco (token). Cada envío trae todos los movimientos del día:
     los ya existentes se ignoran gracias al índice único.
  2. Revendedores (sesión web) o el CRM (token) verifican una referencia + monto en
     POST /api/verificar-pago. Cada pago del banco solo puede usarse una vez.
  3. El admin ve los pagos, quién usó cada uno y resuelve empates en /admin/antiduplic.

Variables de entorno:
  PAGOS_BANCO_TOKEN   token Bearer para el bot del banco y el CRM (obligatorio para la API)
  DEFAULT_TZ          zona horaria para "hoy" (por defecto America/Caracas)
"""
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

from flask import Blueprint, flash, jsonify, redirect, render_template, request, session

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('antiduplic', __name__)

MAX_PAGOS_POR_ENVIO = 2000
REPORTES_MAX, REPORTES_VENTANA = 10, 60  # intentos por revendedor por minuto
CONFIG_REPORTAR = 'antiduplic_reportar_activo'
CONFIG_ULTIMO_ENVIO = 'antiduplic_ultimo_envio'
ORIGENES = ('revendedor', 'crm', 'manual')
TOL = 0.005  # tolerancia al comparar montos con 2 decimales


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(os.environ.get('DEFAULT_TZ', 'America/Caracas'))
    except Exception:
        return timezone(timedelta(hours=-4))


def ahora_local():
    return datetime.now(_tz()).strftime('%Y-%m-%d %H:%M:%S')


def hoy_local():
    return datetime.now(_tz()).strftime('%Y-%m-%d')


def solo_digitos(ref):
    """Referencia sin espacios; None si contiene algo que no sea dígito."""
    ref = re.sub(r'\s+', '', str(ref or ''))
    return ref if ref.isdigit() else None


def sin_ceros(ref):
    return ref.lstrip('0') or '0'


def parse_monto(v):
    """Acepta 7350, 7350.00, 7350,00 y 7.350,00. Devuelve float redondeado o None."""
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    s = re.sub(r'\s+', '', str(v or ''))
    if not s:
        return None
    if ',' in s and '.' in s:
        s = s.replace('.', '').replace(',', '.') if s.rfind(',') > s.rfind('.') else s.replace(',', '')
    elif ',' in s:
        s = s.replace(',', '.')
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def parse_fecha(v):
    s = str(v or '').strip()
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    return None


def parse_hora(v):
    s = str(v or '').strip()
    m = re.match(r'^(\d{1,2}):(\d{2})(?::(\d{2}))?$', s)
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        return None
    return f"{int(m.group(1)):02d}:{m.group(2)}" + (f":{m.group(3)}" if m.group(3) else '')


def _pago_dict(row):
    if not row:
        return None
    return {
        'id': row['id'], 'referencia': row['referencia'], 'fecha': str(row['fecha']),
        'hora': str(row['hora']), 'monto': float(row['monto']), 'tipo': row['tipo'],
        'estado': row['estado'],
    }


def _token_ok():
    esperado = os.environ.get('PAGOS_BANCO_TOKEN', '').strip()
    auth = request.headers.get('Authorization', '')
    if not esperado or not auth.startswith('Bearer '):
        return False
    return secrets.compare_digest(auth[7:].strip().encode(), esperado.encode())


def _config_get(conn, clave, default=None):
    row = conn.execute('SELECT valor FROM configuracion_redeemer WHERE clave = ?', (clave,)).fetchone()
    return row['valor'] if row and row['valor'] is not None else default


def _config_set(conn, clave, valor):
    conn.execute(
        "INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion) VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion",
        (clave, valor))


# ---------------------------------------------------------------------------
# Tablas (PASO 1)
# ---------------------------------------------------------------------------

_tablas_listas = False
_tablas_lock = threading.Lock()


def init_tablas():
    global _tablas_listas
    if _tablas_listas:
        return
    with _tablas_lock:
        if _tablas_listas:
            return
        conn = get_db_connection()
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pagos_banco (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    referencia TEXT NOT NULL,
                    ref_ultimos6 TEXT NOT NULL,
                    fecha TEXT NOT NULL,
                    hora TEXT NOT NULL,
                    monto NUMERIC(15,2) NOT NULL,
                    tipo TEXT,
                    estado TEXT NOT NULL DEFAULT 'disponible',
                    creado_en TEXT NOT NULL
                )''')
            conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS ux_pagos_banco_mov ON pagos_banco (referencia, fecha, hora, monto)')
            conn.execute('CREATE INDEX IF NOT EXISTS ix_pagos_banco_ult6_monto ON pagos_banco (ref_ultimos6, monto)')
            conn.execute('CREATE INDEX IF NOT EXISTS ix_pagos_banco_fecha ON pagos_banco (fecha)')
            conn.execute('CREATE INDEX IF NOT EXISTS ix_pagos_banco_estado ON pagos_banco (estado)')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS usos_referencia (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pago_id INTEGER NOT NULL UNIQUE REFERENCES pagos_banco (id),
                    origen TEXT NOT NULL,
                    revendedor_id INTEGER,
                    orden_id TEXT UNIQUE,
                    referencia_reportada TEXT,
                    monto_reportado NUMERIC(15,2),
                    usado_en TEXT NOT NULL
                )''')
            # Empates sin resolver: el admin asigna el pago correcto a mano
            conn.execute('''
                CREATE TABLE IF NOT EXISTS revisiones_pago (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    origen TEXT NOT NULL,
                    revendedor_id INTEGER,
                    orden_id TEXT,
                    referencia_reportada TEXT,
                    monto_reportado NUMERIC(15,2),
                    fecha_reportada TEXT,
                    hora_reportada TEXT,
                    candidatos TEXT NOT NULL,
                    estado TEXT NOT NULL DEFAULT 'pendiente',
                    creado_en TEXT NOT NULL,
                    resuelto_en TEXT
                )''')
            conn.execute('CREATE INDEX IF NOT EXISTS ix_revisiones_estado ON revisiones_pago (estado)')
            conn.commit()
            _tablas_listas = True
        except Exception as e:
            conn.rollback()
            logger.error(f'[Antiduplic] No se pudieron crear las tablas: {e}')
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Recepción de pagos del banco (PASO 2)
# ---------------------------------------------------------------------------

def guardar_pagos(pagos):
    """Inserta los pagos nuevos e ignora los repetidos. Devuelve (recibidos, insertados, errores)."""
    init_tablas()
    filas, errores = [], []
    for i, p in enumerate(pagos):
        if not isinstance(p, dict):
            errores.append(f'#{i}: no es un objeto')
            continue
        ref = solo_digitos(p.get('referencia'))
        fecha, hora, monto = parse_fecha(p.get('fecha')), parse_hora(p.get('hora')), parse_monto(p.get('monto'))
        if not ref or not fecha or not hora or monto is None or monto <= 0:
            errores.append(f'#{i}: datos inválidos')
            continue
        filas.append((ref, ref[-6:], fecha, hora, monto, str(p.get('tipo') or '')[:120], ahora_local()))
    conn = get_db_connection()
    try:
        insertados = 0
        for f in filas:
            cur = conn.execute(
                'INSERT INTO pagos_banco (referencia, ref_ultimos6, fecha, hora, monto, tipo, estado, creado_en) '
                "VALUES (?, ?, ?, ?, ?, ?, 'disponible', ?) "
                'ON CONFLICT (referencia, fecha, hora, monto) DO NOTHING', f)
            insertados += max(cur.rowcount, 0)
        _config_set(conn, CONFIG_ULTIMO_ENVIO, ahora_local())
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return len(pagos), insertados, errores


@bp.route('/api/pagos-banco', methods=['POST'])
def api_pagos_banco():
    if not _token_ok():
        return jsonify(error='No autorizado'), 401
    if (request.content_length or 0) > 2 * 1024 * 1024:
        return jsonify(error='Cuerpo demasiado grande'), 413
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        data = data.get('pagos')
    if not isinstance(data, list):
        return jsonify(error='Se espera una lista JSON de pagos'), 400
    if len(data) > MAX_PAGOS_POR_ENVIO:
        return jsonify(error=f'Máximo {MAX_PAGOS_POR_ENVIO} pagos por envío'), 413
    recibidos, insertados, errores = guardar_pagos(data)
    if insertados:
        logger.info(f'[Antiduplic] Pagos del banco: {insertados} nuevos de {recibidos}')
    resp = {'recibidos': recibidos, 'insertados': insertados}
    if errores:
        resp['rechazados'] = len(errores)
        resp['errores'] = errores[:20]
    return jsonify(resp)


# ---------------------------------------------------------------------------
# Verificación anti-duplicados (PASO 3)
# ---------------------------------------------------------------------------

def _r(ok, codigo, mensaje, pago=None, **extra):
    d = {'ok': ok, 'codigo': codigo, 'mensaje': mensaje}
    if pago:
        d['pago'] = pago
    d.update(extra)
    return d


def _coincide_ref(banco, rep, corta):
    """¿La referencia del banco corresponde a la reportada?"""
    b, r = sin_ceros(banco), sin_ceros(rep)
    if corta:  # 4-5 dígitos: los últimos dígitos de la del banco
        return b.endswith(r)
    return b.endswith(r) or r.endswith(b)


def _buscar(conn, rep, monto, fecha, corta, con_monto=True):
    """Pagos (cualquier estado) cuya referencia coincide; filtrados por monto si con_monto."""
    if corta:
        rows = conn.execute('SELECT * FROM pagos_banco WHERE fecha = ?', (fecha,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM pagos_banco WHERE ref_ultimos6 = ?', (rep[-6:],)).fetchall()
    out = [r for r in rows if _coincide_ref(r['referencia'], rep, corta)]
    if con_monto:
        out = [r for r in out if abs(float(r['monto']) - monto) < TOL]
    return out


def _desempatar(cands, fecha, hora):
    if fecha and len(cands) > 1:
        f = [c for c in cands if str(c['fecha']) == fecha]
        cands = f or cands
    if hora and len(cands) > 1:
        h = [c for c in cands if str(c['hora']).startswith(hora)]
        cands = h or cands  # si la hora no coincide con ninguno, sigue el empate (→ revisión)
    return cands


def _uso_de_orden(conn, orden_id):
    return conn.execute(
        'SELECT u.*, p.referencia, p.fecha, p.hora, p.monto, p.tipo, p.estado, p.id AS pid '
        'FROM usos_referencia u JOIN pagos_banco p ON p.id = u.pago_id WHERE u.orden_id = ?',
        (orden_id,)).fetchone()


def _pago_de_uso(row):
    return {'id': row['pid'], 'referencia': row['referencia'], 'fecha': str(row['fecha']),
            'hora': str(row['hora']), 'monto': float(row['monto']), 'tipo': row['tipo'], 'estado': row['estado']}


def usar_pago(conn, pago_id, origen, revendedor_id, orden_id, ref_rep, monto_rep):
    """Marca el pago como usado de forma atómica. True si lo tomó esta llamada."""
    cur = conn.execute("UPDATE pagos_banco SET estado = 'usado' WHERE id = ? AND estado IN ('disponible', 'revision')"
                       if origen == 'manual' else
                       "UPDATE pagos_banco SET estado = 'usado' WHERE id = ? AND estado = 'disponible'", (pago_id,))
    if cur.rowcount != 1:
        return False
    conn.execute(
        'INSERT INTO usos_referencia (pago_id, origen, revendedor_id, orden_id, referencia_reportada, monto_reportado, usado_en) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        (pago_id, origen, revendedor_id, orden_id, ref_rep, monto_rep, ahora_local()))
    return True


def verificar_y_usar_pago(referencia, monto, fecha=None, hora=None, origen='revendedor',
                          revendedor_id=None, orden_id=None):
    """Verifica una referencia contra los pagos del banco y, si hay uno solo, lo marca como usado."""
    init_tablas()
    if origen not in ORIGENES:
        return _r(False, 'revision', 'Origen no válido')
    orden_id = (str(orden_id).strip()[:80] or None) if orden_id is not None else None

    conn = get_db_connection()
    try:
        # Idempotencia para el CRM: la misma orden nunca consume un segundo pago
        if orden_id:
            previo = _uso_de_orden(conn, orden_id)
            if previo:
                return _r(True, 'ya_aprobado', 'Esta orden ya tiene un pago aprobado', _pago_de_uso(previo))

        ref = solo_digitos(referencia)
        if ref is None:
            return _r(False, 'ref_corta', 'Referencia inválida: escribe solo los números')
        ref = sin_ceros(ref)
        monto = parse_monto(monto)
        if monto is None or monto <= 0:
            return _r(False, 'monto_no_coincide', 'Monto inválido')
        fecha = parse_fecha(fecha) if fecha else None
        hora = parse_hora(hora) if hora else None
        if len(ref) < 4:
            return _r(False, 'ref_corta', 'Referencia muy corta: escribe al menos 4 dígitos')
        corta = len(ref) < 6
        if corta and not fecha:
            return _r(False, 'falta_fecha', 'Con 4 o 5 dígitos indica también la fecha del pago')

        todos = _buscar(conn, ref, monto, fecha, corta)
        disponibles = _desempatar([c for c in todos if c['estado'] == 'disponible'], fecha, hora)

        if not disponibles:
            usados = [c for c in todos if c['estado'] == 'usado']
            if usados:
                uso = conn.execute('SELECT usado_en FROM usos_referencia WHERE pago_id = ?', (usados[0]['id'],)).fetchone()
                cuando = uso['usado_en'] if uso else ''
                return _r(False, 'usado', f'Referencia ya utilizada{(" el " + str(cuando)) if cuando else ""}', usado_en=cuando)
            if any(c['estado'] == 'revision' for c in todos):
                return _r(False, 'revision', 'Este pago está en revisión manual')
            if _buscar(conn, ref, monto, fecha, corta, con_monto=False):
                return _r(False, 'monto_no_coincide', 'La referencia existe pero el monto no coincide')
            return _r(False, 'no_encontrado', 'Pago no encontrado. Puede que el banco aún no lo haya enviado: '
                                              'inténtalo de nuevo en 1 minuto.')

        if len(disponibles) > 1:
            if not hora:
                return _r(False, 'revision', 'Hay varios pagos con esa referencia y monto: indica la hora del pago',
                          pedir_hora=True)
            # Con hora y aún empatado: a revisión manual, sin aprobar
            ids = [c['id'] for c in disponibles]
            for pid in ids:
                conn.execute("UPDATE pagos_banco SET estado = 'revision' WHERE id = ? AND estado = 'disponible'", (pid,))
            conn.execute(
                'INSERT INTO revisiones_pago (origen, revendedor_id, orden_id, referencia_reportada, monto_reportado, '
                'fecha_reportada, hora_reportada, candidatos, estado, creado_en) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (origen, revendedor_id, orden_id, referencia, monto, fecha, hora, json.dumps(ids), 'pendiente', ahora_local()))
            conn.commit()
            return _r(False, 'revision', 'Requiere revisión manual: hay varios pagos iguales. Un administrador lo revisará.')

        pago = disponibles[0]
        try:
            tomado = usar_pago(conn, pago['id'], origen, revendedor_id, orden_id, str(referencia), monto)
            if not tomado:
                conn.rollback()
                return _r(False, 'usado', 'Referencia ya utilizada')
            conn.commit()
        except Exception:
            # p. ej. la misma orden_id aprobada a la vez por otra petición
            conn.rollback()
            if orden_id:
                previo = _uso_de_orden(conn, orden_id)
                if previo:
                    return _r(True, 'ya_aprobado', 'Esta orden ya tiene un pago aprobado', _pago_de_uso(previo))
            return _r(False, 'usado', 'Referencia ya utilizada')
        d = _pago_dict(pago)
        d['estado'] = 'usado'
        return _r(True, 'aprobado', 'Pago verificado', d)
    finally:
        conn.close()


_rate = {}
_rate_lock = threading.Lock()


def _rate_ok(clave):
    ahora = time.time()
    with _rate_lock:
        hits = [t for t in _rate.get(clave, []) if ahora - t < REPORTES_VENTANA]
        ok = len(hits) < REPORTES_MAX
        if ok:
            hits.append(ahora)
        _rate[clave] = hits
        return ok


@bp.route('/api/verificar-pago', methods=['POST'])
def api_verificar_pago():
    data = request.get_json(silent=True) or {}
    if _token_ok():
        origen = data.get('origen') or 'crm'
        if origen not in ('crm', 'manual'):
            return jsonify(_r(False, 'revision', "Con token el origen debe ser 'crm' o 'manual'")), 400
        if origen == 'crm' and not str(data.get('orden_id') or '').strip():
            return jsonify(_r(False, 'revision', 'Falta orden_id')), 400
        res = verificar_y_usar_pago(data.get('referencia'), data.get('monto'), data.get('fecha'), data.get('hora'),
                                    origen, data.get('revendedor_id'), data.get('orden_id'))
        return jsonify(res)
    if 'usuario' not in session or not session.get('user_db_id'):
        return jsonify(error='No autorizado'), 401
    if not request.is_json:
        return jsonify(error='Se espera JSON'), 415
    if not session.get('is_admin') and not reportar_activo():
        return jsonify(_r(False, 'revision', 'El reporte de pagos no está disponible')), 403
    if not _rate_ok(f"rev:{session['user_db_id']}"):
        return jsonify(_r(False, 'revision', 'Demasiados intentos. Espera un minuto.')), 429
    res = verificar_y_usar_pago(data.get('referencia'), data.get('monto'), data.get('fecha'), data.get('hora'),
                                'revendedor', session['user_db_id'], None)
    return jsonify(res)


# ---------------------------------------------------------------------------
# Pantalla del revendedor (PASO 4)
# ---------------------------------------------------------------------------

def reportar_activo():
    try:
        init_tablas()
        conn = get_db_connection()
        try:
            return _config_get(conn, CONFIG_REPORTAR, '0') == '1'
        finally:
            conn.close()
    except Exception:
        return False


@bp.route('/reportar-pago')
def reportar_pago():
    if 'usuario' not in session:
        return redirect('/auth')
    if not session.get('is_admin') and not reportar_activo():
        flash('El reporte de pagos no está disponible por ahora.', 'error')
        return redirect('/billetera')
    return render_template('reportar_pago.html', hoy=hoy_local())


# ---------------------------------------------------------------------------
# Administración (PASO 5)
# ---------------------------------------------------------------------------

def _solo_admin_json():
    if not session.get('is_admin'):
        return jsonify(error='Acceso denegado'), 403
    return None


@bp.route('/admin/antiduplic')
def admin_antiduplic():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    init_tablas()
    return render_template('admin_antiduplic.html', hoy=hoy_local(),
                           token_configurado=bool(os.environ.get('PAGOS_BANCO_TOKEN', '').strip()),
                           reportar=reportar_activo(), api_url=request.host_url.rstrip('/'))


@bp.route('/admin/antiduplic/datos')
def admin_antiduplic_datos():
    err = _solo_admin_json()
    if err:
        return err
    init_tablas()
    fecha = parse_fecha(request.args.get('fecha')) if request.args.get('fecha') else None
    estado = request.args.get('estado') or ''
    q = re.sub(r'\s+', '', request.args.get('q') or '')[:40]
    hoy = hoy_local()

    sql = ('SELECT p.*, u.origen, u.revendedor_id, u.orden_id, u.referencia_reportada, u.monto_reportado, u.usado_en, '
           'us.nombre AS rev_nombre, us.apellido AS rev_apellido, us.correo AS rev_correo '
           'FROM pagos_banco p LEFT JOIN usos_referencia u ON u.pago_id = p.id '
           'LEFT JOIN usuarios us ON us.id = u.revendedor_id WHERE 1=1')
    params = []
    if fecha:
        sql += ' AND p.fecha = ?'
        params.append(fecha)
    if estado in ('disponible', 'usado', 'revision'):
        sql += ' AND p.estado = ?'
        params.append(estado)
    if q:
        monto_q = parse_monto(q)
        if q.isdigit():
            sql += ' AND (p.referencia LIKE ?' + (' OR ABS(p.monto - ?) < 0.005)' if monto_q is not None else ')')
            params.append('%' + q)
            if monto_q is not None:
                params.append(monto_q)
        elif monto_q is not None:
            sql += ' AND ABS(p.monto - ?) < 0.005'
            params.append(monto_q)
    sql += ' ORDER BY p.fecha DESC, p.hora DESC, p.id DESC LIMIT 500'

    conn = get_db_connection()
    try:
        rows = conn.execute(sql, tuple(params)).fetchall()
        pagos = []
        for r in rows:
            d = _pago_dict(r)
            d['creado_en'] = r['creado_en']
            if r['usado_en']:
                d['uso'] = {
                    'origen': r['origen'], 'revendedor_id': r['revendedor_id'], 'orden_id': r['orden_id'],
                    'revendedor': ' '.join(x for x in (r['rev_nombre'], r['rev_apellido']) if x) or None,
                    'correo': r['rev_correo'], 'referencia_reportada': r['referencia_reportada'],
                    'monto_reportado': float(r['monto_reportado']) if r['monto_reportado'] is not None else None,
                    'usado_en': r['usado_en'],
                }
            pagos.append(d)

        res = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(monto), 0) AS total, "
            "SUM(CASE WHEN estado = 'disponible' THEN 1 ELSE 0 END) AS disp, "
            "SUM(CASE WHEN estado = 'usado' THEN 1 ELSE 0 END) AS usados, "
            "SUM(CASE WHEN estado = 'revision' THEN 1 ELSE 0 END) AS rev "
            "FROM pagos_banco WHERE fecha = ?", (hoy,)).fetchone()
        ultimo_pago = conn.execute('SELECT MAX(creado_en) AS t FROM pagos_banco').fetchone()['t']
        ultimo_mov = conn.execute('SELECT fecha, hora FROM pagos_banco ORDER BY fecha DESC, hora DESC LIMIT 1').fetchone()

        revs = []
        for v in conn.execute("SELECT r.*, us.nombre, us.apellido FROM revisiones_pago r "
                              "LEFT JOIN usuarios us ON us.id = r.revendedor_id "
                              "WHERE r.estado = 'pendiente' ORDER BY r.id").fetchall():
            ids = json.loads(v['candidatos'] or '[]')
            cands = [_pago_dict(conn.execute('SELECT * FROM pagos_banco WHERE id = ?', (i,)).fetchone()) for i in ids]
            revs.append({
                'id': v['id'], 'origen': v['origen'], 'revendedor_id': v['revendedor_id'], 'orden_id': v['orden_id'],
                'revendedor': ' '.join(x for x in (v['nombre'], v['apellido']) if x) or None,
                'referencia_reportada': v['referencia_reportada'],
                'monto_reportado': float(v['monto_reportado']) if v['monto_reportado'] is not None else None,
                'fecha_reportada': v['fecha_reportada'], 'hora_reportada': v['hora_reportada'],
                'creado_en': v['creado_en'], 'candidatos': [c for c in cands if c],
            })
        resumen = {
            'hoy': hoy, 'pagos_hoy': int(res['n'] or 0), 'total_hoy': float(res['total'] or 0),
            'disponibles_hoy': int(res['disp'] or 0), 'usados_hoy': int(res['usados'] or 0),
            'revision_hoy': int(res['rev'] or 0),
            'ultimo_envio_bot': _config_get(conn, CONFIG_ULTIMO_ENVIO),
            'ultimo_pago_nuevo': ultimo_pago,
            'ultimo_movimiento': f"{ultimo_mov['fecha']} {ultimo_mov['hora']}" if ultimo_mov else None,
        }
        return jsonify(pagos=pagos, resumen=resumen, revisiones=revs, ahora=ahora_local())
    finally:
        conn.close()


@bp.route('/admin/antiduplic/asignar', methods=['POST'])
def admin_antiduplic_asignar():
    """Asigna a mano un pago (disponible o en revisión) a un revendedor u orden."""
    err = _solo_admin_json()
    if err:
        return err
    if not request.is_json:
        return jsonify(ok=False, mensaje='Se espera JSON'), 415
    init_tablas()
    data = request.get_json(silent=True) or {}
    try:
        pago_id = int(data.get('pago_id'))
    except (TypeError, ValueError):
        return jsonify(ok=False, mensaje='Pago no válido'), 400
    revision_id = data.get('revision_id')
    rev_id = data.get('revendedor_id')
    orden_id = str(data.get('orden_id') or '').strip()[:80] or None

    conn = get_db_connection()
    try:
        revision = None
        if revision_id:
            revision = conn.execute("SELECT * FROM revisiones_pago WHERE id = ? AND estado = 'pendiente'",
                                    (int(revision_id),)).fetchone()
            if not revision:
                return jsonify(ok=False, mensaje='La revisión ya fue resuelta'), 409
            if pago_id not in json.loads(revision['candidatos'] or '[]'):
                return jsonify(ok=False, mensaje='Ese pago no es candidato de la revisión'), 400
            rev_id = revision['revendedor_id']
            orden_id = revision['orden_id']
        if rev_id not in (None, ''):
            try:
                rev_id = int(rev_id)
            except (TypeError, ValueError):
                return jsonify(ok=False, mensaje='ID de revendedor no válido'), 400
            if not conn.execute('SELECT 1 FROM usuarios WHERE id = ?', (rev_id,)).fetchone():
                return jsonify(ok=False, mensaje='No existe un usuario con ese ID'), 404
        else:
            rev_id = None
        if not rev_id and not orden_id:
            return jsonify(ok=False, mensaje='Indica el ID del revendedor o el ID de la orden'), 400
        if orden_id and _uso_de_orden(conn, orden_id):
            return jsonify(ok=False, mensaje='Esa orden ya tiene un pago asignado'), 409

        ref_rep = revision['referencia_reportada'] if revision else 'asignación manual'
        monto_rep = revision['monto_reportado'] if revision else None
        if not usar_pago(conn, pago_id, 'manual', rev_id, orden_id, ref_rep, monto_rep):
            conn.rollback()
            return jsonify(ok=False, mensaje='Ese pago ya fue usado'), 409
        if revision:
            for otro in json.loads(revision['candidatos'] or '[]'):
                if otro != pago_id:
                    conn.execute("UPDATE pagos_banco SET estado = 'disponible' WHERE id = ? AND estado = 'revision'", (otro,))
            conn.execute("UPDATE revisiones_pago SET estado = 'resuelta', resuelto_en = ? WHERE id = ?",
                         (ahora_local(), revision['id']))
        conn.commit()
        return jsonify(ok=True, mensaje='Pago asignado')
    except Exception as e:
        conn.rollback()
        logger.error(f'[Antiduplic] Error asignando pago {pago_id}: {e}')
        return jsonify(ok=False, mensaje='No se pudo asignar el pago'), 500
    finally:
        conn.close()


@bp.route('/admin/antiduplic/descartar', methods=['POST'])
def admin_antiduplic_descartar():
    """Cierra una revisión sin asignar: los candidatos vuelven a estar disponibles."""
    err = _solo_admin_json()
    if err:
        return err
    if not request.is_json:
        return jsonify(ok=False, mensaje='Se espera JSON'), 415
    init_tablas()
    try:
        revision_id = int((request.get_json(silent=True) or {}).get('revision_id'))
    except (TypeError, ValueError):
        return jsonify(ok=False, mensaje='Revisión no válida'), 400
    conn = get_db_connection()
    try:
        rev = conn.execute("SELECT * FROM revisiones_pago WHERE id = ? AND estado = 'pendiente'", (revision_id,)).fetchone()
        if not rev:
            return jsonify(ok=False, mensaje='La revisión ya fue resuelta'), 409
        for pid in json.loads(rev['candidatos'] or '[]'):
            conn.execute("UPDATE pagos_banco SET estado = 'disponible' WHERE id = ? AND estado = 'revision'", (pid,))
        conn.execute("UPDATE revisiones_pago SET estado = 'descartada', resuelto_en = ? WHERE id = ?", (ahora_local(), revision_id))
        conn.commit()
        return jsonify(ok=True, mensaje='Revisión descartada')
    finally:
        conn.close()


@bp.route('/admin/antiduplic/config', methods=['POST'])
def admin_antiduplic_config():
    err = _solo_admin_json()
    if err:
        return err
    if not request.is_json:
        return jsonify(ok=False), 415
    init_tablas()
    activo = bool((request.get_json(silent=True) or {}).get('reportar'))
    conn = get_db_connection()
    try:
        _config_set(conn, CONFIG_REPORTAR, '1' if activo else '0')
        conn.commit()
    finally:
        conn.close()
    return jsonify(ok=True, reportar=activo)

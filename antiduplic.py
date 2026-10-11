"""
Antiduplic (Referencias) — pagos bancarios y verificación anti-duplicados, POR USUARIO (Blueprint Flask).

Cada usuario (revendedor) tiene sus cuentas bancarias, sus pagos, sus revisiones y sus claves.
Ningún usuario ve ni toca datos de otro: toda consulta filtra por usuario_id.

Flujo:
  1. Un bot envía los movimientos de una cuenta bancaria a POST /api/pagos-banco con una
     clave de lector (Bearer lec_…) que decide el usuario y la cuenta. PAGOS_BANCO_TOKEN
     sigue valiendo y entra como lector del usuario dueño (cuenta principal).
  2. Un sistema externo verifica referencia + monto en POST /api/verificar-pago con una
     clave de API de Referencias (X-API-Key) del usuario. Cada pago se usa una sola vez.
  3. Empates → revisión manual (el usuario en /referencias, el admin en /admin/antiduplic).

No toca el saldo de nadie: solo verifica y registra el uso de pagos.

Variables de entorno:
  PAGOS_BANCO_TOKEN          clave del bot antiguo → usuario dueño
  ANTIDUPLIC_USUARIO_DUENO   ID del usuario dueño de los datos anteriores (por defecto el admin principal)
  DEFAULT_TZ                 zona horaria para "hoy" (por defecto America/Caracas)
"""
import hashlib
import json
import logging
import os
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone

from flask import Blueprint, flash, jsonify, redirect, render_template, request, session

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('antiduplic', __name__)

MAX_PAGOS_POR_ENVIO = 2000
MAX_CUERPO = 2 * 1024 * 1024
CONFIG_ULTIMO_ENVIO = 'antiduplic_ultimo_envio'   # (antiguo, global) se migra a la cuenta principal
ORIGENES = ('revendedor', 'crm', 'manual')
TOL = 0.005          # tolerancia al comparar montos con 2 decimales
LECTOR_PREFIJO = 'lec_'
AVISO_BOT_MIN = 10   # minutos sin envíos del bot para mostrar aviso


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


def _f2(v):
    return round(float(v), 2) if v is not None else None


def _pago_dict(row):
    if not row:
        return None
    return {
        'id': row['id'], 'referencia': row['referencia'], 'fecha': str(row['fecha']),
        'hora': str(row['hora']), 'monto': _f2(row['monto']), 'tipo': row['tipo'],
        'estado': row['estado'],
    }


def _config_get(conn, clave, default=None):
    row = conn.execute('SELECT valor FROM configuracion_redeemer WHERE clave = ?', (clave,)).fetchone()
    return row['valor'] if row and row['valor'] is not None else default


def _config_set(conn, clave, valor):
    conn.execute(
        "INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion) VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion",
        (clave, valor))


def es_postgres():
    return bool(os.environ.get('DATABASE_URL', '').strip())


def usuario_dueno(conn=None):
    """Dueño de los datos anteriores y del PAGOS_BANCO_TOKEN: ANTIDUPLIC_USUARIO_DUENO o el admin principal."""
    env = os.environ.get('ANTIDUPLIC_USUARIO_DUENO', '').strip()
    if env.isdigit():
        return int(env)
    propia = conn is None
    conn = conn or get_db_connection()
    try:
        email = os.environ.get('ADMIN_EMAIL', 'admin@inefable.com').strip().lower()
        row = conn.execute('SELECT id FROM usuarios WHERE LOWER(correo) = ?', (email,)).fetchone()
        if row:
            return int(row['id'])
        row = conn.execute('SELECT MIN(id) AS id FROM usuarios').fetchone()
        return int(row['id']) if row and row['id'] else 1
    except Exception:
        return 1
    finally:
        if propia:
            conn.close()


# ---------------------------------------------------------------------------
# Tablas y migración
# ---------------------------------------------------------------------------

_tablas_listas = False
_tablas_lock = threading.Lock()


def _suelto(sql, params=()):
    """Ejecuta una sentencia en su propia conexión; devuelve False si falla (p. ej. columna ya existente)."""
    conn = get_db_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


_USOS_SQL = '''
    CREATE TABLE IF NOT EXISTS {nombre} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pago_id INTEGER NOT NULL UNIQUE REFERENCES pagos_banco (id),
        usuario_id INTEGER,
        origen TEXT NOT NULL,
        revendedor_id INTEGER,
        orden_id TEXT,
        referencia_reportada TEXT,
        monto_reportado NUMERIC(15,2),
        usado_en TEXT NOT NULL,
        hecho_por TEXT
    )'''


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
                    usuario_id INTEGER,
                    cuenta_banco_id INTEGER,
                    referencia TEXT NOT NULL,
                    ref_ultimos6 TEXT NOT NULL,
                    fecha TEXT NOT NULL,
                    hora TEXT NOT NULL,
                    monto NUMERIC(15,2) NOT NULL,
                    tipo TEXT,
                    estado TEXT NOT NULL DEFAULT 'disponible',
                    creado_en TEXT NOT NULL
                )''')
            conn.execute(_USOS_SQL.format(nombre='usos_referencia'))
            conn.execute('''
                CREATE TABLE IF NOT EXISTS revisiones_pago (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario_id INTEGER,
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
                    resuelto_en TEXT,
                    resuelto_por TEXT
                )''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cuentas_banco (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario_id INTEGER NOT NULL,
                    banco TEXT NOT NULL DEFAULT 'Bancamiga',
                    alias TEXT NOT NULL DEFAULT '',
                    ultimos_digitos TEXT DEFAULT '',
                    titular TEXT DEFAULT '',
                    activo BOOLEAN DEFAULT TRUE,
                    ultimo_envio TEXT,
                    creado_en TEXT NOT NULL
                )''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS claves_lector (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario_id INTEGER NOT NULL,
                    cuenta_banco_id INTEGER NOT NULL,
                    nombre TEXT NOT NULL DEFAULT '',
                    clave_hash TEXT NOT NULL UNIQUE,
                    prefijo_visible TEXT NOT NULL,
                    activo BOOLEAN DEFAULT TRUE,
                    ultimo_uso TEXT,
                    creado_en TEXT NOT NULL
                )''')
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f'[Antiduplic] No se pudieron crear las tablas: {e}')
            raise
        finally:
            conn.close()

        # Columnas nuevas en instalaciones anteriores (ya existentes → se ignora el error)
        for tabla, col, tipo in (('pagos_banco', 'usuario_id', 'INTEGER'), ('pagos_banco', 'cuenta_banco_id', 'INTEGER'),
                                 ('usos_referencia', 'usuario_id', 'INTEGER'), ('usos_referencia', 'hecho_por', 'TEXT'),
                                 ('revisiones_pago', 'usuario_id', 'INTEGER'), ('revisiones_pago', 'resuelto_por', 'TEXT')):
            _suelto(f'ALTER TABLE {tabla} ADD COLUMN {col} {tipo}')
        _quitar_unico_orden()

        # Índices por usuario (el único antiguo se reemplaza: dos usuarios pueden tener la misma referencia)
        _suelto('DROP INDEX IF EXISTS ux_pagos_banco_mov')
        _suelto('DROP INDEX IF EXISTS ix_pagos_banco_ult6_monto')
        _suelto('DROP INDEX IF EXISTS ix_pagos_banco_fecha')
        _suelto('DROP INDEX IF EXISTS ix_pagos_banco_estado')
        for sql in ('CREATE UNIQUE INDEX IF NOT EXISTS ux_pagos_banco_usuario_mov ON pagos_banco (usuario_id, referencia, fecha, hora, monto)',
                    'CREATE INDEX IF NOT EXISTS ix_pagos_banco_usuario_ult6 ON pagos_banco (usuario_id, ref_ultimos6, monto)',
                    'CREATE INDEX IF NOT EXISTS ix_pagos_banco_usuario_fecha ON pagos_banco (usuario_id, fecha)',
                    'CREATE INDEX IF NOT EXISTS ix_pagos_banco_usuario_estado ON pagos_banco (usuario_id, estado)',
                    'CREATE UNIQUE INDEX IF NOT EXISTS ux_usos_usuario_orden ON usos_referencia (usuario_id, orden_id)',
                    'CREATE INDEX IF NOT EXISTS ix_revisiones_usuario_estado ON revisiones_pago (usuario_id, estado)',
                    'CREATE INDEX IF NOT EXISTS ix_cuentas_banco_usuario ON cuentas_banco (usuario_id)',
                    'CREATE INDEX IF NOT EXISTS ix_claves_lector_usuario ON claves_lector (usuario_id)'):
            if not _suelto(sql):
                logger.error(f'[Antiduplic] No se pudo crear un índice: {sql}')
        _migrar_datos()
        _tablas_listas = True


def _quitar_unico_orden():
    """orden_id era único global; ahora es único por usuario (índice ux_usos_usuario_orden)."""
    if es_postgres():
        _suelto('ALTER TABLE usos_referencia DROP CONSTRAINT IF EXISTS usos_referencia_orden_id_key')
        return
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'usos_referencia'").fetchone()
        if not row or not re.search(r'orden_id\s+TEXT\s+UNIQUE', row['sql'] or '', re.IGNORECASE):
            return
        cols = 'id, pago_id, usuario_id, origen, revendedor_id, orden_id, referencia_reportada, monto_reportado, usado_en, hecho_por'
        conn.execute(_USOS_SQL.format(nombre='usos_referencia_nueva'))
        conn.execute(f'INSERT INTO usos_referencia_nueva ({cols}) SELECT {cols} FROM usos_referencia')
        conn.execute('DROP TABLE usos_referencia')
        conn.execute('ALTER TABLE usos_referencia_nueva RENAME TO usos_referencia')
        conn.commit()
        logger.info('[Antiduplic] usos_referencia: orden_id ahora es único por usuario')
    except Exception as e:
        conn.rollback()
        logger.error(f'[Antiduplic] No se pudo reconstruir usos_referencia: {e}')
    finally:
        conn.close()


def cuenta_principal(conn, usuario_id, crear=True):
    """Primera cuenta activa del usuario (la crea como «Bancamiga principal» si no tiene ninguna)."""
    row = conn.execute('SELECT id FROM cuentas_banco WHERE usuario_id = ? AND activo = TRUE ORDER BY id LIMIT 1',
                       (usuario_id,)).fetchone()
    if row:
        return int(row['id'])
    if not crear:
        return None
    cur = conn.execute("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) "
                       "VALUES (?, 'Bancamiga', 'Bancamiga principal', TRUE, ?) RETURNING id", (usuario_id, ahora_local()))
    return int(cur.fetchone()[0])


def _migrar_datos():
    """Los pagos, usos y revisiones sin dueño pasan al usuario dueño, en su cuenta principal."""
    conn = get_db_connection()
    try:
        pendiente = (conn.execute('SELECT 1 FROM pagos_banco WHERE usuario_id IS NULL LIMIT 1').fetchone()
                     or conn.execute('SELECT 1 FROM usos_referencia WHERE usuario_id IS NULL LIMIT 1').fetchone()
                     or conn.execute('SELECT 1 FROM revisiones_pago WHERE usuario_id IS NULL LIMIT 1').fetchone())
        if not pendiente:
            return
        dueno = usuario_dueno(conn)
        cuenta = cuenta_principal(conn, dueno)
        conn.execute('UPDATE pagos_banco SET usuario_id = ?, cuenta_banco_id = ? WHERE usuario_id IS NULL', (dueno, cuenta))
        conn.execute('UPDATE usos_referencia SET usuario_id = (SELECT p.usuario_id FROM pagos_banco p WHERE p.id = usos_referencia.pago_id) '
                     'WHERE usuario_id IS NULL')
        conn.execute('UPDATE revisiones_pago SET usuario_id = ? WHERE usuario_id IS NULL', (dueno,))
        antiguo = _config_get(conn, CONFIG_ULTIMO_ENVIO)
        if antiguo:
            conn.execute('UPDATE cuentas_banco SET ultimo_envio = ? WHERE id = ? AND ultimo_envio IS NULL', (antiguo, cuenta))
        conn.commit()
        logger.info(f'[Antiduplic] Datos anteriores asignados al usuario dueño {dueno} (cuenta {cuenta})')
    except Exception as e:
        conn.rollback()
        logger.error(f'[Antiduplic] Error migrando datos al usuario dueño: {e}')
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Claves de lector (bot del banco)
# ---------------------------------------------------------------------------

def hash_clave(clave):
    return hashlib.sha256(str(clave).encode('utf-8')).hexdigest()


def nueva_clave_lector():
    clave = LECTOR_PREFIJO + secrets.token_hex(24)
    return clave, hash_clave(clave), clave[:12]


def autenticar_bot():
    """(usuario_id, cuenta_banco_id, clave_lector_id) según la clave Bearer, o None."""
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return None
    token = auth[7:].strip()
    if not token:
        return None
    conn = get_db_connection()
    try:
        if token.startswith(LECTOR_PREFIJO):
            row = conn.execute(
                'SELECT k.id, k.usuario_id, k.cuenta_banco_id, k.activo, c.activo AS cuenta_activa, c.usuario_id AS cuenta_usuario '
                'FROM claves_lector k JOIN cuentas_banco c ON c.id = k.cuenta_banco_id WHERE k.clave_hash = ?',
                (hash_clave(token),)).fetchone()
            if not row or not row['activo'] or not row['cuenta_activa'] or row['cuenta_usuario'] != row['usuario_id']:
                return None
            return int(row['usuario_id']), int(row['cuenta_banco_id']), int(row['id'])
        esperado = os.environ.get('PAGOS_BANCO_TOKEN', '').strip()
        if esperado and secrets.compare_digest(token.encode(), esperado.encode()):
            dueno = usuario_dueno(conn)
            cuenta = cuenta_principal(conn, dueno)
            conn.commit()
            return dueno, cuenta, None
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Recepción de pagos del banco
# ---------------------------------------------------------------------------

def guardar_pagos(pagos, usuario_id=None, cuenta_banco_id=None, clave_lector_id=None):
    """Inserta los pagos nuevos del usuario e ignora los repetidos. Devuelve (recibidos, insertados, errores)."""
    init_tablas()
    conn = get_db_connection()
    try:
        if usuario_id is None:
            usuario_id = usuario_dueno(conn)
        if cuenta_banco_id is None:
            cuenta_banco_id = cuenta_principal(conn, usuario_id)
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
            filas.append((usuario_id, cuenta_banco_id, ref, ref[-6:], fecha, hora, monto, str(p.get('tipo') or '')[:120], ahora_local()))
        insertados = 0
        for f in filas:
            cur = conn.execute(
                'INSERT INTO pagos_banco (usuario_id, cuenta_banco_id, referencia, ref_ultimos6, fecha, hora, monto, tipo, estado, creado_en) '
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'disponible', ?) "
                'ON CONFLICT (usuario_id, referencia, fecha, hora, monto) DO NOTHING', f)
            insertados += max(cur.rowcount, 0)
        ahora = ahora_local()
        conn.execute('UPDATE cuentas_banco SET ultimo_envio = ? WHERE id = ?', (ahora, cuenta_banco_id))
        if clave_lector_id:
            conn.execute('UPDATE claves_lector SET ultimo_uso = ? WHERE id = ?', (ahora, clave_lector_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return len(pagos), insertados, errores


@bp.route('/api/pagos-banco', methods=['POST'])
def api_pagos_banco():
    init_tablas()
    quien = autenticar_bot()
    if not quien:
        return jsonify(error='No autorizado', mensaje='Clave de lector inválida o desactivada'), 401
    if (request.content_length or 0) > MAX_CUERPO:
        return jsonify(error='Cuerpo demasiado grande', mensaje='Máximo 2 MB por envío'), 413
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        data = data.get('pagos')
    if not isinstance(data, list):
        return jsonify(error='Se espera una lista JSON de pagos', mensaje='Envía una lista JSON de pagos'), 400
    if len(data) > MAX_PAGOS_POR_ENVIO:
        return jsonify(error=f'Máximo {MAX_PAGOS_POR_ENVIO} pagos por envío',
                       mensaje=f'Máximo {MAX_PAGOS_POR_ENVIO} pagos por envío'), 413
    uid, cuenta, clave_id = quien
    recibidos, insertados, errores = guardar_pagos(data, uid, cuenta, clave_id)
    if insertados:
        logger.info(f'[Antiduplic] Usuario {uid} cuenta {cuenta}: {insertados} pagos nuevos de {recibidos}')
    resp = {'recibidos': recibidos, 'insertados': insertados}
    if errores:
        resp['rechazados'] = len(errores)
        resp['errores'] = errores[:20]
    return jsonify(resp)


# ---------------------------------------------------------------------------
# Verificación anti-duplicados (la lógica no cambia; solo se filtra por usuario)
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


def _buscar(conn, uid, rep, monto, fecha, corta, con_monto=True):
    """Pagos del usuario (cualquier estado) cuya referencia coincide; filtrados por monto si con_monto."""
    if corta:
        rows = conn.execute('SELECT * FROM pagos_banco WHERE usuario_id = ? AND fecha = ?', (uid, fecha)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM pagos_banco WHERE usuario_id = ? AND ref_ultimos6 = ?', (uid, rep[-6:])).fetchall()
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


def _uso_de_orden(conn, uid, orden_id):
    return conn.execute(
        'SELECT u.*, p.referencia, p.fecha, p.hora, p.monto, p.tipo, p.estado, p.id AS pid '
        'FROM usos_referencia u JOIN pagos_banco p ON p.id = u.pago_id WHERE u.usuario_id = ? AND u.orden_id = ?',
        (uid, orden_id)).fetchone()


def _pago_de_uso(row):
    return {'id': row['pid'], 'referencia': row['referencia'], 'fecha': str(row['fecha']),
            'hora': str(row['hora']), 'monto': _f2(row['monto']), 'tipo': row['tipo'], 'estado': row['estado']}


def usar_pago(conn, uid, pago_id, origen, revendedor_id, orden_id, ref_rep, monto_rep, hecho_por='', desde_revision=False):
    """Marca el pago del usuario como usado de forma atómica. True si lo tomó esta llamada."""
    estados = "('disponible', 'revision')" if desde_revision else "('disponible')"
    cur = conn.execute(f"UPDATE pagos_banco SET estado = 'usado' WHERE id = ? AND usuario_id = ? AND estado IN {estados}",
                       (pago_id, uid))
    if cur.rowcount != 1:
        return False
    conn.execute(
        'INSERT INTO usos_referencia (pago_id, usuario_id, origen, revendedor_id, orden_id, referencia_reportada, '
        'monto_reportado, usado_en, hecho_por) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (pago_id, uid, origen, revendedor_id, orden_id, ref_rep, monto_rep, ahora_local(), hecho_por or None))
    return True


def verificar_y_usar_pago(referencia, monto, fecha=None, hora=None, origen='revendedor',
                          revendedor_id=None, orden_id=None, usuario_id=None, hecho_por=''):
    """Verifica una referencia contra los pagos DEL USUARIO y, si hay uno solo, lo marca como usado."""
    init_tablas()
    if origen not in ORIGENES:
        return _r(False, 'revision', 'Origen no válido')
    orden_id = (str(orden_id).strip()[:80] or None) if orden_id is not None else None

    conn = get_db_connection()
    try:
        uid = int(usuario_id) if usuario_id is not None else usuario_dueno(conn)
        # Idempotencia: la misma orden (de este usuario) nunca consume un segundo pago
        if orden_id:
            previo = _uso_de_orden(conn, uid, orden_id)
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

        todos = _buscar(conn, uid, ref, monto, fecha, corta)
        disponibles = _desempatar([c for c in todos if c['estado'] == 'disponible'], fecha, hora)

        if not disponibles:
            usados = [c for c in todos if c['estado'] == 'usado']
            if usados:
                uso = conn.execute('SELECT usado_en FROM usos_referencia WHERE pago_id = ?', (usados[0]['id'],)).fetchone()
                cuando = uso['usado_en'] if uso else ''
                return _r(False, 'usado', f'Referencia ya utilizada{(" el " + str(cuando)) if cuando else ""}', usado_en=cuando)
            if any(c['estado'] == 'revision' for c in todos):
                return _r(False, 'revision', 'Este pago está en revisión manual')
            if _buscar(conn, uid, ref, monto, fecha, corta, con_monto=False):
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
                conn.execute("UPDATE pagos_banco SET estado = 'revision' WHERE id = ? AND usuario_id = ? AND estado = 'disponible'",
                             (pid, uid))
            conn.execute(
                'INSERT INTO revisiones_pago (usuario_id, origen, revendedor_id, orden_id, referencia_reportada, monto_reportado, '
                'fecha_reportada, hora_reportada, candidatos, estado, creado_en) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (uid, origen, revendedor_id, orden_id, referencia, monto, fecha, hora, json.dumps(ids), 'pendiente', ahora_local()))
            conn.commit()
            return _r(False, 'revision', 'Requiere revisión manual: hay varios pagos iguales. Un administrador lo revisará.')

        pago = disponibles[0]
        try:
            tomado = usar_pago(conn, uid, pago['id'], origen, revendedor_id, orden_id, str(referencia), monto, hecho_por)
            if not tomado:
                conn.rollback()
                return _r(False, 'usado', 'Referencia ya utilizada')
            conn.commit()
        except Exception:
            # p. ej. la misma orden_id aprobada a la vez por otra petición
            conn.rollback()
            if orden_id:
                previo = _uso_de_orden(conn, uid, orden_id)
                if previo:
                    return _r(True, 'ya_aprobado', 'Esta orden ya tiene un pago aprobado', _pago_de_uso(previo))
            return _r(False, 'usado', 'Referencia ya utilizada')
        d = _pago_dict(pago)
        d['estado'] = 'usado'
        return _r(True, 'aprobado', 'Pago verificado', d)
    finally:
        conn.close()


@bp.route('/api/verificar-pago', methods=['POST'])
def api_verificar_pago():
    """Clave de API de Referencias con permiso «Verificar pagos». Verifica contra los pagos de SU usuario."""
    from api_panel import cuenta_por_clave, cuenta_puede, limite_ok
    cuenta = cuenta_por_clave(request.headers.get('X-API-Key'))
    if not cuenta:
        return jsonify(error='No autorizado', mensaje='Clave de API inválida o desactivada'), 401
    if not cuenta_puede(cuenta, 'verificar_pago'):
        return jsonify(error='Esta cuenta no tiene permiso para verificar pagos',
                       mensaje='Esta clave no tiene el permiso «Verificar pagos»'), 403
    if not limite_ok(cuenta['id'], 'escritura'):
        return jsonify(error='Demasiadas peticiones', mensaje='Demasiadas peticiones; espera un minuto'), 429
    data = request.get_json(silent=True) or {}
    origen = data.get('origen') or 'crm'
    if origen not in ('crm', 'manual'):
        return jsonify(_r(False, 'revision', "El origen debe ser 'crm' o 'manual'")), 400
    if origen == 'crm' and not str(data.get('orden_id') or '').strip():
        return jsonify(_r(False, 'revision', 'Falta orden_id')), 400
    res = verificar_y_usar_pago(data.get('referencia'), data.get('monto'), data.get('fecha'), data.get('hora'),
                                origen, None, data.get('orden_id'), usuario_id=cuenta['usuario_id'],
                                hecho_por=f"api:{cuenta['nombre']} (#{cuenta['id']})")
    return jsonify(res)


# ---------------------------------------------------------------------------
# Consultas y gestión compartidas (admin, panel del usuario y API)
#   uid = None → todos los usuarios (solo el admin)
# ---------------------------------------------------------------------------

def _uso_dict(r):
    if not r['usado_en']:
        return None
    return {
        'origen': r['origen'], 'revendedor_id': r['revendedor_id'], 'orden_id': r['orden_id'],
        'revendedor': ' '.join(x for x in (r['rev_nombre'], r['rev_apellido']) if x) or None,
        'correo': r['rev_correo'], 'referencia_reportada': r['referencia_reportada'],
        'monto_reportado': _f2(r['monto_reportado']), 'usado_en': r['usado_en'], 'hecho_por': r['hecho_por'],
    }


_SQL_PAGOS = ('SELECT p.*, u.origen, u.revendedor_id, u.orden_id, u.referencia_reportada, u.monto_reportado, u.usado_en, '
              'u.hecho_por, rv.nombre AS rev_nombre, rv.apellido AS rev_apellido, rv.correo AS rev_correo, '
              'du.nombre AS dueno_nombre, du.apellido AS dueno_apellido, du.correo AS dueno_correo, '
              'cb.alias AS cuenta_alias, cb.banco AS cuenta_banco_nombre '
              'FROM pagos_banco p LEFT JOIN usos_referencia u ON u.pago_id = p.id '
              'LEFT JOIN usuarios rv ON rv.id = u.revendedor_id '
              'LEFT JOIN usuarios du ON du.id = p.usuario_id '
              'LEFT JOIN cuentas_banco cb ON cb.id = p.cuenta_banco_id')


def _pago_completo(r):
    d = _pago_dict(r)
    d['creado_en'] = r['creado_en']
    d['usuario_id'] = r['usuario_id']
    d['usuario'] = ' '.join(x for x in (r['dueno_nombre'], r['dueno_apellido']) if x) or r['dueno_correo'] or None
    d['cuenta_id'] = r['cuenta_banco_id']
    d['cuenta'] = r['cuenta_alias'] or r['cuenta_banco_nombre'] or None
    d['uso'] = _uso_dict(r)
    return d


def listar_pagos(conn, uid=None, fecha=None, desde=None, hasta=None, estado='', q='', cuenta_id=None,
                 pagina=1, por_pagina=500):
    """(pagos, total) con filtros. uid=None → todos (admin)."""
    where, params = ['1=1'], []
    if uid is not None:
        where.append('p.usuario_id = ?')
        params.append(uid)
    if fecha:
        where.append('p.fecha = ?')
        params.append(fecha)
    if desde:
        where.append('p.fecha >= ?')
        params.append(desde)
    if hasta:
        where.append('p.fecha <= ?')
        params.append(hasta)
    if estado in ('disponible', 'usado', 'revision'):
        where.append('p.estado = ?')
        params.append(estado)
    if cuenta_id:
        where.append('p.cuenta_banco_id = ?')
        params.append(int(cuenta_id))
    q = re.sub(r'\s+', '', q or '')[:40]
    if q:
        monto_q = parse_monto(q)
        if q.isdigit():
            if monto_q is not None:
                where.append('(p.referencia LIKE ? OR ABS(p.monto - ?) < 0.005)')
                params += ['%' + q, monto_q]
            else:
                where.append('p.referencia LIKE ?')
                params.append('%' + q)
        elif monto_q is not None:
            where.append('ABS(p.monto - ?) < 0.005')
            params.append(monto_q)
    cond = ' AND '.join(where)
    total = conn.execute(f'SELECT COUNT(*) AS n FROM pagos_banco p WHERE {cond}', tuple(params)).fetchone()['n']
    por_pagina = max(1, min(int(por_pagina or 50), 500))
    pagina = max(1, int(pagina or 1))
    rows = conn.execute(f'{_SQL_PAGOS} WHERE {cond} ORDER BY p.fecha DESC, p.hora DESC, p.id DESC LIMIT ? OFFSET ?',
                        tuple(params) + (por_pagina, (pagina - 1) * por_pagina)).fetchall()
    return [_pago_completo(r) for r in rows], int(total or 0)


def detalle_pago(conn, uid, pago_id):
    """Un pago con su uso; None si no existe o no es del usuario (uid=None → cualquiera)."""
    sql, params = f'{_SQL_PAGOS} WHERE p.id = ?', [pago_id]
    if uid is not None:
        sql += ' AND p.usuario_id = ?'
        params.append(uid)
    row = conn.execute(sql, tuple(params)).fetchone()
    return _pago_completo(row) if row else None


def listar_cuentas(conn, uid, solo_activas=False):
    sql = 'SELECT * FROM cuentas_banco WHERE usuario_id = ?' + (' AND activo = TRUE' if solo_activas else '') + ' ORDER BY id'
    return [{'id': r['id'], 'banco': r['banco'], 'alias': r['alias'], 'ultimos_digitos': r['ultimos_digitos'] or '',
             'titular': r['titular'] or '', 'activo': bool(r['activo']), 'ultimo_envio': r['ultimo_envio'],
             'creado_en': r['creado_en']} for r in conn.execute(sql, (uid,)).fetchall()]


def resumen(conn, uid=None, fecha=None):
    """Resumen del día (por defecto hoy) y último envío del bot, por cuenta."""
    fecha = fecha or hoy_local()
    cond, params = ('usuario_id = ? AND ', (uid,)) if uid is not None else ('', ())
    res = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(monto), 0) AS total, "
        "SUM(CASE WHEN estado = 'disponible' THEN 1 ELSE 0 END) AS disp, "
        "SUM(CASE WHEN estado = 'usado' THEN 1 ELSE 0 END) AS usados, "
        "SUM(CASE WHEN estado = 'revision' THEN 1 ELSE 0 END) AS rev "
        f"FROM pagos_banco WHERE {cond}fecha = ?", params + (fecha,)).fetchone()
    ultimo_pago = conn.execute(f"SELECT MAX(creado_en) AS t FROM pagos_banco{' WHERE usuario_id = ?' if uid is not None else ''}",
                               params).fetchone()['t']
    ultimo_mov = conn.execute(f"SELECT fecha, hora FROM pagos_banco{' WHERE usuario_id = ?' if uid is not None else ''} "
                              'ORDER BY fecha DESC, hora DESC LIMIT 1', params).fetchone()
    if uid is not None:
        cuentas = [{'id': c['id'], 'alias': c['alias'], 'banco': c['banco'], 'ultimos_digitos': c['ultimos_digitos'],
                    'activo': c['activo'], 'ultimo_envio': c['ultimo_envio']} for c in listar_cuentas(conn, uid)]
    else:
        cuentas = [{'id': r['id'], 'alias': r['alias'], 'banco': r['banco'], 'ultimos_digitos': r['ultimos_digitos'] or '',
                    'activo': bool(r['activo']), 'ultimo_envio': r['ultimo_envio'], 'usuario_id': r['usuario_id']}
                   for r in conn.execute('SELECT * FROM cuentas_banco ORDER BY usuario_id, id').fetchall()]
    envios = [c['ultimo_envio'] for c in cuentas if c['activo'] and c['ultimo_envio']]
    return {
        'hoy': fecha, 'pagos_hoy': int(res['n'] or 0), 'total_hoy': _f2(res['total'] or 0),
        'disponibles_hoy': int(res['disp'] or 0), 'usados_hoy': int(res['usados'] or 0),
        'revision_hoy': int(res['rev'] or 0),
        'ultimo_envio_bot': max(envios) if envios else None,
        'ultimo_pago_nuevo': ultimo_pago,
        'ultimo_movimiento': f"{ultimo_mov['fecha']} {ultimo_mov['hora']}" if ultimo_mov else None,
        'aviso_bot_minutos': AVISO_BOT_MIN,
        'cuentas': cuentas,
    }


def listar_revisiones(conn, uid=None, estado='pendiente'):
    estado = estado if estado in ('pendiente', 'resuelta', 'descartada') else 'pendiente'
    sql = ('SELECT r.*, us.nombre, us.apellido, du.nombre AS dueno_nombre, du.apellido AS dueno_apellido, du.correo AS dueno_correo '
           'FROM revisiones_pago r LEFT JOIN usuarios us ON us.id = r.revendedor_id LEFT JOIN usuarios du ON du.id = r.usuario_id '
           'WHERE r.estado = ?')
    params = [estado]
    if uid is not None:
        sql += ' AND r.usuario_id = ?'
        params.append(uid)
    out = []
    for v in conn.execute(sql + ' ORDER BY r.id', tuple(params)).fetchall():
        ids = json.loads(v['candidatos'] or '[]')
        cands = []
        for i in ids:
            c = conn.execute('SELECT * FROM pagos_banco WHERE id = ? AND usuario_id = ?', (i, v['usuario_id'])).fetchone()
            if c:
                cands.append(_pago_dict(c))
        out.append({
            'id': v['id'], 'estado': v['estado'], 'usuario_id': v['usuario_id'],
            'usuario': ' '.join(x for x in (v['dueno_nombre'], v['dueno_apellido']) if x) or v['dueno_correo'] or None,
            'origen': v['origen'], 'revendedor_id': v['revendedor_id'], 'orden_id': v['orden_id'],
            'revendedor': ' '.join(x for x in (v['nombre'], v['apellido']) if x) or None,
            'referencia_reportada': v['referencia_reportada'], 'monto_reportado': _f2(v['monto_reportado']),
            'fecha_reportada': v['fecha_reportada'], 'hora_reportada': v['hora_reportada'],
            'creado_en': v['creado_en'], 'resuelto_en': v['resuelto_en'], 'resuelto_por': v['resuelto_por'],
            'candidatos': cands,
        })
    return out


def asignar(uid, pago_id, orden_id=None, revision_id=None, revendedor_id=None, hecho_por=''):
    """Asigna a mano un pago a una orden (o resuelve una revisión). uid=None → admin (cualquier usuario).

    Devuelve (http_status, dict con ok y mensaje). Un pago o revisión de otro usuario → 404.
    """
    init_tablas()
    try:
        pago_id = int(pago_id)
    except (TypeError, ValueError):
        return 400, {'ok': False, 'mensaje': 'Pago no válido'}
    orden_id = str(orden_id or '').strip()[:80] or None
    conn = get_db_connection()
    try:
        pago = conn.execute('SELECT id, usuario_id, estado FROM pagos_banco WHERE id = ?' + (' AND usuario_id = ?' if uid is not None else ''),
                            (pago_id, uid) if uid is not None else (pago_id,)).fetchone()
        if not pago:
            return 404, {'ok': False, 'mensaje': 'Pago no encontrado'}
        dueno = int(pago['usuario_id'])
        revision = None
        if revision_id:
            try:
                revision_id = int(revision_id)
            except (TypeError, ValueError):
                return 400, {'ok': False, 'mensaje': 'Revisión no válida'}
            revision = conn.execute('SELECT * FROM revisiones_pago WHERE id = ? AND usuario_id = ?', (revision_id, dueno)).fetchone()
            if not revision:
                return 404, {'ok': False, 'mensaje': 'Revisión no encontrada'}
            if revision['estado'] != 'pendiente':
                return 409, {'ok': False, 'mensaje': 'La revisión ya fue resuelta'}
            if pago_id not in json.loads(revision['candidatos'] or '[]'):
                return 400, {'ok': False, 'mensaje': 'Ese pago no es candidato de la revisión'}
            revendedor_id = revision['revendedor_id']
            orden_id = revision['orden_id']
        if revendedor_id not in (None, ''):
            try:
                revendedor_id = int(revendedor_id)
            except (TypeError, ValueError):
                return 400, {'ok': False, 'mensaje': 'ID de revendedor no válido'}
            if not conn.execute('SELECT 1 FROM usuarios WHERE id = ?', (revendedor_id,)).fetchone():
                return 404, {'ok': False, 'mensaje': 'No existe un usuario con ese ID'}
        else:
            revendedor_id = None
        if not revision and not orden_id and not revendedor_id:
            return 400, {'ok': False, 'mensaje': 'Indica el ID de la orden'}
        if orden_id and _uso_de_orden(conn, dueno, orden_id):
            return 409, {'ok': False, 'mensaje': 'Esa orden ya tiene un pago asignado'}

        ref_rep = revision['referencia_reportada'] if revision else 'asignación manual'
        monto_rep = revision['monto_reportado'] if revision else None
        if not usar_pago(conn, dueno, pago_id, 'manual', revendedor_id, orden_id, ref_rep, monto_rep, hecho_por,
                         desde_revision=True):
            conn.rollback()
            return 409, {'ok': False, 'mensaje': 'Ese pago ya fue usado'}
        if revision:
            for otro in json.loads(revision['candidatos'] or '[]'):
                if otro != pago_id:
                    conn.execute("UPDATE pagos_banco SET estado = 'disponible' WHERE id = ? AND usuario_id = ? AND estado = 'revision'",
                                 (otro, dueno))
            conn.execute("UPDATE revisiones_pago SET estado = 'resuelta', resuelto_en = ?, resuelto_por = ? WHERE id = ?",
                         (ahora_local(), hecho_por or None, revision['id']))
        conn.commit()
        return 200, {'ok': True, 'mensaje': 'Pago asignado'}
    except Exception as e:
        conn.rollback()
        logger.error(f'[Antiduplic] Error asignando pago {pago_id}: {e}')
        return 500, {'ok': False, 'mensaje': 'No se pudo asignar el pago'}
    finally:
        conn.close()


def descartar(uid, revision_id, hecho_por=''):
    """Cierra una revisión sin asignar; los candidatos vuelven a disponibles. uid=None → admin."""
    init_tablas()
    try:
        revision_id = int(revision_id)
    except (TypeError, ValueError):
        return 400, {'ok': False, 'mensaje': 'Revisión no válida'}
    conn = get_db_connection()
    try:
        sql = 'SELECT * FROM revisiones_pago WHERE id = ?' + (' AND usuario_id = ?' if uid is not None else '')
        rev = conn.execute(sql, (revision_id, uid) if uid is not None else (revision_id,)).fetchone()
        if not rev:
            return 404, {'ok': False, 'mensaje': 'Revisión no encontrada'}
        if rev['estado'] != 'pendiente':
            return 409, {'ok': False, 'mensaje': 'La revisión ya fue resuelta'}
        for pid in json.loads(rev['candidatos'] or '[]'):
            conn.execute("UPDATE pagos_banco SET estado = 'disponible' WHERE id = ? AND usuario_id = ? AND estado = 'revision'",
                         (pid, rev['usuario_id']))
        conn.execute("UPDATE revisiones_pago SET estado = 'descartada', resuelto_en = ?, resuelto_por = ? WHERE id = ?",
                     (ahora_local(), hecho_por or None, revision_id))
        conn.commit()
        return 200, {'ok': True, 'mensaje': 'Revisión descartada'}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Administración: vista general de todos los usuarios ("Referencias (todos)")
# ---------------------------------------------------------------------------

def _solo_admin_json():
    if not session.get('is_admin'):
        return jsonify(error='Acceso denegado', mensaje='Acceso denegado'), 403
    return None


def _admin_firma():
    return f"admin:{session.get('usuario') or 'admin'}"


@bp.route('/admin/antiduplic')
def admin_antiduplic():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    init_tablas()
    conn = get_db_connection()
    try:
        usuarios = [{'id': r['id'], 'nombre': ' '.join(x for x in (r['nombre'], r['apellido']) if x) or r['correo'],
                     'correo': r['correo']}
                    for r in conn.execute(
                        'SELECT DISTINCT us.id, us.nombre, us.apellido, us.correo FROM usuarios us WHERE us.id IN '
                        '(SELECT usuario_id FROM pagos_banco UNION SELECT usuario_id FROM cuentas_banco) ORDER BY us.id').fetchall()]
        dueno = usuario_dueno(conn)
    finally:
        conn.close()
    return render_template('admin_antiduplic.html', hoy=hoy_local(), usuarios=usuarios, dueno=dueno,
                           token_banco=bool(os.environ.get('PAGOS_BANCO_TOKEN', '').strip()),
                           api_url=request.host_url.rstrip('/'))


@bp.route('/admin/antiduplic/docs')
def admin_antiduplic_docs():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    return render_template('admin_antiduplic_docs.html', api_url=request.host_url.rstrip('/'),
                           token_banco=bool(os.environ.get('PAGOS_BANCO_TOKEN', '').strip()))


@bp.route('/admin/antiduplic/datos')
def admin_antiduplic_datos():
    err = _solo_admin_json()
    if err:
        return err
    init_tablas()
    uid = request.args.get('usuario_id')
    uid = int(uid) if uid and uid.isdigit() else None
    fecha = parse_fecha(request.args.get('fecha')) if request.args.get('fecha') else None
    conn = get_db_connection()
    try:
        pagos, total = listar_pagos(conn, uid, fecha=fecha, estado=request.args.get('estado') or '',
                                    q=request.args.get('q') or '', por_pagina=500)
        return jsonify(pagos=pagos, total=total, resumen=resumen(conn, uid), revisiones=listar_revisiones(conn, uid),
                       ahora=ahora_local())
    finally:
        conn.close()


@bp.route('/admin/antiduplic/asignar', methods=['POST'])
def admin_antiduplic_asignar():
    """Asigna a mano un pago (disponible o en revisión) de cualquier usuario a una orden."""
    err = _solo_admin_json()
    if err:
        return err
    if not request.is_json:
        return jsonify(ok=False, mensaje='Se espera JSON'), 415
    data = request.get_json(silent=True) or {}
    status, body = asignar(None, data.get('pago_id'), data.get('orden_id'), data.get('revision_id'),
                           data.get('revendedor_id'), hecho_por=_admin_firma())
    return jsonify(body), status


@bp.route('/admin/antiduplic/descartar', methods=['POST'])
def admin_antiduplic_descartar():
    err = _solo_admin_json()
    if err:
        return err
    if not request.is_json:
        return jsonify(ok=False, mensaje='Se espera JSON'), 415
    status, body = descartar(None, (request.get_json(silent=True) or {}).get('revision_id'), hecho_por=_admin_firma())
    return jsonify(body), status

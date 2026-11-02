"""
Campana de notificaciones de la barra (pedido del dueño, 2026-11-02): avisos del admin, noticias y saldo acreditado
(por el admin o por Binance), en vivo. static/campana.js pregunta cada 30 s; si hay algo nuevo la campana se sacude y el
número late hasta que se abre. Al abrirla se marca todo como visto.

El admin también la tiene: ve el saldo acreditado a todos los usuarios (lo visto se guarda en configuracion_redeemer).
No incluye las recargas de juegos (lo pidió así).
"""
import logging
import re
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request, session

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('campana', __name__)

HORAS_VE = 4   # las fechas se guardan en UTC; Venezuela = UTC-4


def _uid():
    uid = session.get('user_db_id')
    return int(uid) if uid and 'usuario' in session else None


def _fecha(v):
    """(texto para ordenar en UTC, 'dd/mm hh:mm' en hora de Venezuela)."""
    utc = str(v or '')[:19].replace('T', ' ')
    try:
        return utc, (datetime.strptime(utc, '%Y-%m-%d %H:%M:%S') - timedelta(hours=HORAS_VE)).strftime('%d/%m %H:%M')
    except ValueError:
        return utc, utc[:16]


def _sin_html(texto, largo=160):
    t = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', str(texto or ''))).strip()
    return t if len(t) <= largo else t[:largo - 1].rstrip() + '…'


def _filas(conn, sql, params):
    """Consulta que no rompe la campana si una tabla todavía no existe (se crean al usarse por primera vez)."""
    try:
        return conn.execute(sql, params).fetchall()
    except Exception:
        conn.rollback()
        return []


def _item(tipo, titulo, texto, fecha, nuevo):
    orden, visible = _fecha(fecha)
    return {'tipo': tipo, 'titulo': titulo, 'texto': texto, 'fecha': visible, 'nuevo': bool(nuevo), '_orden': orden}


def estado_usuario(conn, uid):
    items = []
    for c in _filas(conn, 'SELECT monto, fecha, visto FROM creditos_billetera WHERE usuario_id = ? ORDER BY fecha DESC, id DESC LIMIT 10', (uid,)):
        items.append(_item('saldo', 'Saldo acreditado', f'Se acreditaron ${float(c["monto"] or 0):.2f} a tu saldo', c['fecha'], not c['visto']))
    for n in _filas(conn, 'SELECT titulo, mensaje, fecha, visto FROM notificaciones_personalizadas WHERE usuario_id = ? '
                          'ORDER BY fecha DESC, id DESC LIMIT 10', (uid,)):
        items.append(_item('aviso', _sin_html(n['titulo'], 80), _sin_html(n['mensaje']), n['fecha'], not n['visto']))
    vistas = {int(r['noticia_id']) for r in _filas(conn, 'SELECT noticia_id FROM noticias_vistas WHERE usuario_id = ?', (uid,))}
    for n in _filas(conn, 'SELECT id, titulo, contenido, fecha FROM noticias ORDER BY fecha DESC, id DESC LIMIT 5', ()):
        items.append(_item('noticia', _sin_html(n['titulo'], 80), _sin_html(n['contenido']), n['fecha'], int(n['id']) not in vistas))
    items.sort(key=lambda i: i['_orden'], reverse=True)
    return items


def _clave_admin(uid):
    return f'campana_admin_visto_{uid}'


def _visto_admin(conn, uid):
    fila = _filas(conn, 'SELECT valor FROM configuracion_redeemer WHERE clave = ?', (_clave_admin(uid),))
    try:
        return int(fila[0]['valor']) if fila else 0
    except (TypeError, ValueError):
        return 0


def estado_admin(conn, uid):
    visto = _visto_admin(conn, uid)
    items = []
    for c in _filas(conn, 'SELECT c.id, c.monto, c.fecha, u.nombre, u.apellido, u.correo FROM creditos_billetera c '
                          'LEFT JOIN usuarios u ON u.id = c.usuario_id ORDER BY c.id DESC LIMIT 20', ()):
        quien = ' '.join(x for x in (c['nombre'], c['apellido']) if x) or c['correo'] or 'un usuario'
        items.append(_item('saldo', f'Saldo acreditado a {_sin_html(quien, 60)}', f'${float(c["monto"] or 0):.2f}', c['fecha'], int(c['id']) > visto))
    return items


@bp.route('/campana/estado')
def r_estado():
    uid = _uid()
    if uid is None:
        return jsonify(ok=False, error='Tu sesión expiró.'), 401
    admin = bool(session.get('is_admin'))
    conn = get_db_connection()
    try:
        items = estado_admin(conn, uid) if admin else estado_usuario(conn, uid)
    finally:
        conn.close()
    nuevos = sum(1 for i in items if i['nuevo'])
    for i in items:
        i.pop('_orden', None)
    return jsonify(ok=True, admin=admin, nuevos=nuevos, items=items[:20])


@bp.route('/campana/visto', methods=['POST'])
def r_visto():
    uid = _uid()
    if uid is None:
        return jsonify(ok=False, error='Tu sesión expiró.'), 401
    if not request.is_json:
        return jsonify(ok=False, error='Se espera JSON'), 415
    conn = get_db_connection()
    try:
        if session.get('is_admin'):
            fila = _filas(conn, 'SELECT MAX(id) AS m FROM creditos_billetera', ())
            ultimo = int(fila[0]['m'] or 0) if fila else 0
            conn.execute('INSERT INTO configuracion_redeemer (clave, valor) VALUES (?, ?) '
                         'ON CONFLICT (clave) DO UPDATE SET valor = excluded.valor', (_clave_admin(uid), str(ultimo)))
            conn.commit()
        else:
            for sql in ('UPDATE creditos_billetera SET visto = TRUE WHERE usuario_id = ? AND visto = FALSE',
                        'UPDATE notificaciones_personalizadas SET visto = TRUE WHERE usuario_id = ? AND visto = FALSE',
                        'INSERT INTO noticias_vistas (usuario_id, noticia_id) SELECT CAST(? AS INTEGER), n.id FROM noticias n WHERE n.id NOT IN '
                        '(SELECT noticia_id FROM noticias_vistas WHERE usuario_id = ?)'):
                try:
                    conn.execute(sql, (uid, uid) if sql.count('?') == 2 else (uid,))
                    conn.commit()
                except Exception as e:   # (una tabla que aún no existe: nada que marcar)
                    conn.rollback()
                    logger.debug(f'[Campana] {e}')
    finally:
        conn.close()
    return jsonify(ok=True)

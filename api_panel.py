"""
Apartado "API" del panel admin (Blueprint Flask).

Gestiona las cuentas de la API externa (tabla webservice_accounts de api_whitelabel)
con permisos por cuenta, y publica la documentación con el catálogo en vivo.

Permisos (columnas de webservice_accounts):
    perm_recargas        POST /api/v1/recharge        (por defecto activo: compatibilidad)
    perm_verificar_id    POST /api/v1/verify-id
    perm_verificar_pago  POST /api/verificar-pago     (Antiduplic / Bancamiga)

Endpoints:
    POST /api/v1/verify-id                       → nombre del jugador para un ID
    GET  /admin/api                              → pantalla de cuentas y permisos
    GET  /admin/api/docs                         → documentación con el catálogo actual
    GET  /admin/api/cuentas                      → lista (JSON)
    POST /admin/api/cuentas                      → crear (JSON)
    POST /admin/api/cuentas/<id>                 → actualizar nombre/usuario/webhook/permisos/activo (JSON)
    POST /admin/api/cuentas/<id>/regenerar       → nueva clave (JSON)
    POST /admin/api/cuentas/<id>/eliminar        → eliminar (JSON)
"""
import logging
import re
import threading
import time

from flask import Blueprint, flash, jsonify, redirect, render_template, request, session

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('api_panel', __name__)

PERMISOS = {
    'recargas': ('perm_recargas', True, 'Hacer recargas'),
    'verificar_id': ('perm_verificar_id', False, 'Verificar IDs de jugador'),
    'verificar_pago': ('perm_verificar_pago', False, 'Verificar pagos de Bancamiga'),
}
VERIFY_MAX, VERIFY_VENTANA = 30, 60  # verificaciones de ID por cuenta por minuto (límite del proveedor)


# ---------------------------------------------------------------------------
# Migración: columnas de permisos
# ---------------------------------------------------------------------------

_cols_listas = False
_cols_lock = threading.Lock()


def init_permisos():
    """Añade las columnas de permisos a webservice_accounts si faltan (una sentencia por conexión)."""
    global _cols_listas
    if _cols_listas:
        return
    with _cols_lock:
        if _cols_listas:
            return
        for _, (col, default, _) in PERMISOS.items():
            conn = get_db_connection()
            try:
                conn.execute(f"ALTER TABLE webservice_accounts ADD COLUMN {col} BOOLEAN DEFAULT {'TRUE' if default else 'FALSE'}")
                conn.commit()
            except Exception:
                conn.rollback()  # ya existe
            finally:
                conn.close()
        _cols_listas = True


def cuenta_puede(account, permiso):
    """¿La cuenta (dict de webservice_accounts) tiene ese permiso?"""
    col, default, _ = PERMISOS[permiso]
    val = (account or {}).get(col)
    return default if val is None else bool(val)


def cuenta_por_clave(api_key):
    from api_whitelabel import _get_account_by_key
    init_permisos()
    return _get_account_by_key((api_key or '').strip()) if api_key else None


def _clave_de_peticion():
    return (request.headers.get('X-API-Key') or '').strip()


# ---------------------------------------------------------------------------
# Catálogo: qué se puede recargar y verificar por API
# ---------------------------------------------------------------------------

def _verify_cfg_para(product_id, game=None):
    """Configuración de verificación de ID para un juego del catálogo (o {})."""
    import id_verify
    if product_id == -1:
        return id_verify.get_verify_config('freefire_id')
    if product_id == -155:
        if id_verify.get_inefable_key():
            return {'enabled': True, 'provider': 'inefable', 'inefable_game': 'Inefable-bloodstriker'}
        return {}
    if game:
        return id_verify.get_verify_config(f"dyn_{game['slug']}")
    return {}


def _verifica_id(product_id, game=None):
    import id_verify
    cfg = _verify_cfg_para(product_id, game)
    return bool(cfg.get('enabled') and id_verify.cfg_ready(cfg))


def catalogo():
    """Juegos activos con sus paquetes, indicando si se pueden recargar y verificar por API."""
    from dynamic_games import get_all_dynamic_games, get_dynamic_packages, parse_campos_config
    juegos = []

    conn = get_db_connection()
    try:
        ff = conn.execute('SELECT id, nombre, precio FROM precios_freefire_id WHERE activo = TRUE ORDER BY id').fetchall()
        bs = conn.execute('SELECT id, nombre, precio, gamepoint_package_id FROM precios_bloodstriker '
                          'WHERE activo = TRUE ORDER BY id').fetchall()
    finally:
        conn.close()

    if ff:
        juegos.append({
            'product_id': -1, 'nombre': 'Free Fire ID', 'slug': 'freefire-id', 'modo': 'id',
            'player_id2': None, 'servidor': None, 'verifica_id': _verifica_id(-1),
            'paquetes': [{'package_id': r['id'], 'nombre': r['nombre'], 'precio': float(r['precio']), 'recargable': True}
                         for r in ff],
        })
    if bs:
        juegos.append({
            'product_id': -155, 'nombre': 'Blood Strike', 'slug': 'bloodstriker', 'modo': 'id',
            'player_id2': None, 'servidor': None, 'verifica_id': _verifica_id(-155),
            'paquetes': [{'package_id': r['id'], 'nombre': r['nombre'], 'precio': float(r['precio']),
                          'recargable': bool(r['gamepoint_package_id'])} for r in bs],
        })
    for g in get_all_dynamic_games(only_active=True):
        campos = parse_campos_config(g)
        id2 = campos.get('campo_id2') or {}
        srv = campos.get('servidor') or {}
        modo = g.get('modo') or 'id'
        paquetes = []
        for p in get_dynamic_packages(g['id'], only_active=True):
            paquetes.append({
                'package_id': p['id'], 'nombre': p['nombre'], 'precio': float(p['precio']),
                # La API recarga juegos dinámicos solo por GamePoint (sin servidor)
                'recargable': bool(modo == 'id' and p.get('gamepoint_package_id') and g.get('gamepoint_product_id')
                                   and not srv.get('enabled')),
            })
        juegos.append({
            'product_id': g['id'], 'nombre': g['nombre'], 'slug': g['slug'], 'modo': modo,
            'player_id2': (id2.get('label') or 'Zone ID') if id2.get('enabled') else None,
            'servidor': (srv.get('opciones') or []) if srv.get('enabled') else None,
            'verifica_id': modo == 'id' and _verifica_id(g['id'], g),
            'paquetes': paquetes,
        })
    return juegos


# ---------------------------------------------------------------------------
# POST /api/v1/verify-id
# ---------------------------------------------------------------------------

_rate, _rate_lock = {}, threading.Lock()


def _rate_ok(clave):
    ahora = time.time()
    with _rate_lock:
        hits = [t for t in _rate.get(clave, []) if ahora - t < VERIFY_VENTANA]
        ok = len(hits) < VERIFY_MAX
        if ok:
            hits.append(ahora)
        _rate[clave] = hits
        return ok


@bp.route('/api/v1/verify-id', methods=['POST'])
def api_v1_verify_id():
    """Devuelve el nombre del jugador de un ID. Body: product_id, player_id, player_id2 (opcional)."""
    import id_verify
    from dynamic_games import get_dynamic_game_by_id
    account = cuenta_por_clave(_clave_de_peticion())
    if not account:
        return jsonify(ok=False, error='API key inválida o cuenta desactivada'), 401
    if not cuenta_puede(account, 'verificar_id'):
        return jsonify(ok=False, error='Esta cuenta no tiene permiso para verificar IDs'), 403
    data = request.get_json(silent=True) or {}
    try:
        product_id = int(data.get('product_id'))
    except (TypeError, ValueError):
        return jsonify(ok=False, error='product_id es requerido y debe ser numérico'), 400
    player_id = str(data.get('player_id') or '').strip()
    player_id2 = str(data.get('player_id2') or '').strip()
    if not id_verify.PLAYER_RE.match(player_id) or (player_id2 and not id_verify.PLAYER_RE.match(player_id2)):
        return jsonify(ok=False, error='player_id inválido (solo letras, números y - _ .)'), 400

    game = None
    if product_id > 0:
        game = get_dynamic_game_by_id(product_id)
        if not game or not game.get('activo'):
            return jsonify(ok=False, error='Juego no encontrado o inactivo'), 404
    elif product_id not in (-1, -155):
        return jsonify(ok=False, error='Juego no encontrado'), 404
    cfg = _verify_cfg_para(product_id, game)
    if not (cfg.get('enabled') and id_verify.cfg_ready(cfg)):
        return jsonify(ok=False, error='La verificación de ID no está disponible para este juego'), 404
    if not _rate_ok(account['id']):
        return jsonify(ok=False, error='Demasiadas verificaciones. Espera un minuto.'), 429

    ok, result = id_verify.call_verify_api(cfg, player_id, player_id2, str(data.get('servidor') or '')[:60])
    logger.info(f"[API] verify-id cuenta={account['id']} producto={product_id} -> {'OK' if ok else 'FAIL'}")
    if ok:
        return jsonify(ok=True, player_id=player_id, player_name=result)
    return jsonify(ok=False, error=result)


# ---------------------------------------------------------------------------
# Admin: pantalla, documentación y cuentas
# ---------------------------------------------------------------------------

def _solo_admin_json():
    if not session.get('is_admin'):
        return jsonify(ok=False, error='Acceso denegado'), 403
    if request.method == 'POST' and not request.is_json:
        return jsonify(ok=False, error='Se espera JSON'), 415
    return None


@bp.route('/admin/api')
def admin_api():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    init_permisos()
    return render_template('admin_api.html', permisos=[(k, v[2]) for k, v in PERMISOS.items()],
                           api_url=request.host_url.rstrip('/'))


@bp.route('/admin/api/docs')
def admin_api_docs():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    try:
        juegos = catalogo()
    except Exception as e:
        logger.error(f'[API] Error armando el catálogo para la documentación: {e}')
        juegos = []
    return render_template('admin_api_docs.html', juegos=juegos, api_url=request.host_url.rstrip('/'))


def _cuenta_json(r):
    d = {
        'id': r['id'], 'nombre': r['nombre'], 'api_key': r['api_key'], 'usuario_id': r['usuario_id'],
        'usuario': ' '.join(x for x in (r['u_nombre'], r['u_apellido']) if x) or None,
        'usuario_correo': r['u_correo'], 'usuario_saldo': float(r['u_saldo'] or 0),
        'webhook_url': r['webhook_url'] or '', 'activo': bool(r['activo']),
        'permisos': {k: cuenta_puede(dict(r), k) for k in PERMISOS},
    }
    return d


def _cargar_cuenta(conn, account_id):
    return conn.execute(
        'SELECT ws.*, u.nombre AS u_nombre, u.apellido AS u_apellido, u.correo AS u_correo, u.saldo AS u_saldo '
        'FROM webservice_accounts ws LEFT JOIN usuarios u ON u.id = ws.usuario_id WHERE ws.id = ?',
        (account_id,)).fetchone()


@bp.route('/admin/api/cuentas', methods=['GET', 'POST'])
def admin_api_cuentas():
    err = _solo_admin_json()
    if err:
        return err
    init_permisos()
    conn = get_db_connection()
    try:
        if request.method == 'GET':
            rows = conn.execute(
                'SELECT ws.*, u.nombre AS u_nombre, u.apellido AS u_apellido, u.correo AS u_correo, u.saldo AS u_saldo '
                'FROM webservice_accounts ws LEFT JOIN usuarios u ON u.id = ws.usuario_id ORDER BY ws.id').fetchall()
            return jsonify(ok=True, cuentas=[_cuenta_json(r) for r in rows])

        from api_whitelabel import _generate_api_key
        data = request.get_json(silent=True) or {}
        nombre = str(data.get('nombre') or '').strip()[:60]
        try:
            usuario_id = int(data.get('usuario_id'))
        except (TypeError, ValueError):
            return jsonify(ok=False, error='Indica el ID del usuario cuyo saldo pagará las recargas'), 400
        if not nombre:
            return jsonify(ok=False, error='Ponle un nombre a la cuenta (ej: CRM)'), 400
        if not conn.execute('SELECT 1 FROM usuarios WHERE id = ?', (usuario_id,)).fetchone():
            return jsonify(ok=False, error='No existe un usuario con ese ID'), 404
        permisos = data.get('permisos') or {}
        cols = [PERMISOS[k][0] for k in PERMISOS]
        vals = [bool(permisos.get(k, PERMISOS[k][1])) for k in PERMISOS]
        cur = conn.execute(
            f"INSERT INTO webservice_accounts (nombre, api_key, usuario_id, webhook_url, activo, {', '.join(cols)}) "
            f"VALUES (?, ?, ?, ?, TRUE, {', '.join('?' for _ in cols)}) RETURNING id",
            (nombre, _generate_api_key(), usuario_id, str(data.get('webhook_url') or '').strip()[:300], *vals))
        nuevo = cur.fetchone()[0]
        conn.commit()
        return jsonify(ok=True, cuenta=_cuenta_json(_cargar_cuenta(conn, nuevo)))
    except Exception as e:
        conn.rollback()
        logger.error(f'[API] Error en cuentas: {e}')
        return jsonify(ok=False, error='No se pudo guardar la cuenta'), 500
    finally:
        conn.close()


@bp.route('/admin/api/cuentas/<int:account_id>', methods=['POST'])
def admin_api_cuenta_update(account_id):
    err = _solo_admin_json()
    if err:
        return err
    init_permisos()
    data = request.get_json(silent=True) or {}
    sets, params = [], []
    if 'nombre' in data:
        nombre = str(data.get('nombre') or '').strip()[:60]
        if not nombre:
            return jsonify(ok=False, error='El nombre no puede quedar vacío'), 400
        sets.append('nombre = ?')
        params.append(nombre)
    if 'webhook_url' in data:
        sets.append('webhook_url = ?')
        params.append(str(data.get('webhook_url') or '').strip()[:300])
    if 'activo' in data:
        sets.append('activo = ?')
        params.append(bool(data['activo']))
    for k, v in (data.get('permisos') or {}).items():
        if k in PERMISOS:
            sets.append(f'{PERMISOS[k][0]} = ?')
            params.append(bool(v))
    conn = get_db_connection()
    try:
        if 'usuario_id' in data:
            try:
                uid = int(data['usuario_id'])
            except (TypeError, ValueError):
                return jsonify(ok=False, error='ID de usuario no válido'), 400
            if not conn.execute('SELECT 1 FROM usuarios WHERE id = ?', (uid,)).fetchone():
                return jsonify(ok=False, error='No existe un usuario con ese ID'), 404
            sets.append('usuario_id = ?')
            params.append(uid)
        if not sets:
            return jsonify(ok=False, error='Nada que cambiar'), 400
        cur = conn.execute(f"UPDATE webservice_accounts SET {', '.join(sets)}, fecha_actualizacion = CURRENT_TIMESTAMP "
                           'WHERE id = ?', (*params, account_id))
        if cur.rowcount != 1:
            conn.rollback()
            return jsonify(ok=False, error='Cuenta no encontrada'), 404
        conn.commit()
        return jsonify(ok=True, cuenta=_cuenta_json(_cargar_cuenta(conn, account_id)))
    finally:
        conn.close()


@bp.route('/admin/api/cuentas/<int:account_id>/regenerar', methods=['POST'])
def admin_api_cuenta_regenerar(account_id):
    err = _solo_admin_json()
    if err:
        return err
    from api_whitelabel import _generate_api_key
    conn = get_db_connection()
    try:
        cur = conn.execute('UPDATE webservice_accounts SET api_key = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ?',
                           (_generate_api_key(), account_id))
        if cur.rowcount != 1:
            conn.rollback()
            return jsonify(ok=False, error='Cuenta no encontrada'), 404
        conn.commit()
        return jsonify(ok=True, cuenta=_cuenta_json(_cargar_cuenta(conn, account_id)))
    finally:
        conn.close()


@bp.route('/admin/api/cuentas/<int:account_id>/eliminar', methods=['POST'])
def admin_api_cuenta_eliminar(account_id):
    err = _solo_admin_json()
    if err:
        return err
    conn = get_db_connection()
    try:
        # Si ya hizo recargas se conserva el historial: solo se desactiva
        if conn.execute('SELECT 1 FROM api_orders WHERE account_id = ? LIMIT 1', (account_id,)).fetchone():
            conn.execute('UPDATE webservice_accounts SET activo = FALSE WHERE id = ?', (account_id,))
            conn.commit()
            return jsonify(ok=True, desactivada=True,
                           mensaje='La cuenta tiene recargas en su historial: se desactivó en lugar de borrarse.')
        conn.execute('DELETE FROM webservice_accounts WHERE id = ?', (account_id,))
        conn.commit()
        return jsonify(ok=True, mensaje='Cuenta eliminada')
    finally:
        conn.close()

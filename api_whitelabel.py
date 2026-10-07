"""
API REST de Marca Blanca v1
===========================
Blueprint Flask que permite a webs externas (Inefablestore, etc.) conectarse,
consultar productos y ejecutar recargas usando el saldo de un usuario asignado.

Endpoints:
  GET  /api/v1/products          → Catálogo: juegos creados por el admin con paquetes mapeados
  POST /api/v1/recharge          → Crear orden de recarga
  GET  /api/v1/orders/<order_id> → Consultar estado de una orden
  GET  /api/v1/balance           → Consultar saldo de la cuenta

Admin (sesión):
  GET  /admin/webservice-accounts          → Listar cuentas
  POST /admin/webservice-accounts/create   → Crear nueva cuenta
  POST /admin/webservice-accounts/<id>/toggle  → Activar/desactivar
  POST /admin/webservice-accounts/<id>/regenerate-key → Regenerar API key
  POST /admin/webservice-accounts/<id>/delete  → Eliminar cuenta
"""

import functools
import json
import logging
import os
import secrets
import threading
import time as time_module

import requests as req_lib
from flask import Blueprint, jsonify, request, session, flash, redirect

logger = logging.getLogger(__name__)

bp = Blueprint('api_whitelabel', __name__)

# ---------------------------------------------------------------------------
# Helpers – DB connection (importados de pg_compat igual que el resto del app)
# ---------------------------------------------------------------------------

def _get_conn():
    from pg_compat import get_db_connection
    return get_db_connection()


# ---------------------------------------------------------------------------
# DDL – llamar desde init_db() de app.py
# ---------------------------------------------------------------------------

def init_whitelabel_tables(cursor):
    """Crea las tablas necesarias para la API de marca blanca.
    Llamar desde init_db() en app.py."""

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS webservice_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            api_key TEXT NOT NULL UNIQUE,
            usuario_id INTEGER NOT NULL,
            webhook_url TEXT DEFAULT '',
            activo BOOLEAN DEFAULT TRUE,
            fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
            fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_api_key ON webservice_accounts(api_key)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            usuario_id INTEGER NOT NULL,
            game_type TEXT NOT NULL,
            game_name TEXT DEFAULT '',
            package_id INTEGER NOT NULL,
            package_name TEXT DEFAULT '',
            player_id TEXT NOT NULL,
            player_id2 TEXT DEFAULT '',
            precio REAL NOT NULL,
            estado TEXT DEFAULT 'pendiente',
            reference_no TEXT DEFAULT '',
            player_name TEXT DEFAULT '',
            error_msg TEXT DEFAULT '',
            duration_seconds REAL DEFAULT 0,
            webhook_sent BOOLEAN DEFAULT FALSE,
            external_order_id TEXT DEFAULT '',
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            fecha_completada DATETIME,
            FOREIGN KEY (account_id) REFERENCES webservice_accounts (id),
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_orders_account ON api_orders(account_id, fecha DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_orders_estado ON api_orders(estado)')


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------

def _get_account_by_key(api_key):
    """Busca una WebServiceAccount activa por su api_key."""
    conn = _get_conn()
    row = conn.execute(
        'SELECT * FROM webservice_accounts WHERE api_key = ? AND activo = TRUE',
        (api_key,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def require_api_key(f):
    """Decorator: valida X-API-Key header o ?api_key param."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = (
            request.headers.get('X-API-Key')
            or request.args.get('api_key')
            or (request.get_json(silent=True) or {}).get('api_key')
            or ''
        ).strip()
        if not api_key:
            return jsonify({'ok': False, 'error': 'API key requerida'}), 401
        account = _get_account_by_key(api_key)
        if not account:
            return jsonify({'ok': False, 'error': 'API key inválida o cuenta desactivada'}), 401
        # Inyectar la cuenta en el request context
        request._ws_account = account
        return f(*args, **kwargs)
    return decorated


def _generate_api_key():
    """Genera un api_key seguro de 48 caracteres."""
    return 'wsk_' + secrets.token_hex(24)


def _get_linked_user_info(usuario_id):
    """Retorna datos del usuario vinculado a la cuenta API."""
    conn = _get_conn()
    row = conn.execute(
        'SELECT id, nombre, apellido, correo, saldo FROM usuarios WHERE id = ?',
        (usuario_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        'user_id': row['id'],
        'name': f"{row['nombre']} {row['apellido']}",
        'email': row['correo'],
        'balance': float(row['saldo']),
    }


# ---------------------------------------------------------------------------
# GET /api/v1/account  — info de la cuenta vinculada
# ---------------------------------------------------------------------------

@bp.route('/api/v1/account', methods=['GET'])
@require_api_key
def api_v1_account():
    """Retorna info de la cuenta API y del usuario vinculado."""
    account = request._ws_account
    user_info = _get_linked_user_info(account['usuario_id'])
    if not user_info:
        return jsonify({'ok': False, 'error': 'Usuario vinculado no encontrado'}), 404

    return jsonify({
        'ok': True,
        'account': {
            'id': account['id'],
            'name': account['nombre'],
            'active': bool(account['activo']),
            'webhook_url': account.get('webhook_url', ''),
            'created_at': str(account.get('fecha_creacion', '')),
        },
        'user': user_info,
    })


# ---------------------------------------------------------------------------
# GET /api/v1/products
# ---------------------------------------------------------------------------

@bp.route('/api/v1/products', methods=['GET'])
@require_api_key
def api_v1_products():
    """Catálogo: solo juegos creados por el admin con paquetes mapeados y su precio."""
    from api_panel import catalogo
    games = []
    try:
        for j in catalogo():
            games.append({
                'game_type': 'dynamic', 'game_id': j['product_id'], 'name': j['nombre'], 'slug': j['slug'],
                'mode': 'id', 'icon': j['icono'], 'description': '',
                'player_id2_label': j['player_id2'], 'servers': j['servidor'], 'verify_id': j['verifica_id'],
                'packages': [{'package_id': p['package_id'], 'name': p['nombre'], 'price': p['precio'], 'description': ''}
                             for p in j['paquetes']],
            })
    except Exception as e:
        logger.warning(f'[WL API] Error armando el catálogo: {e}')

    account = request._ws_account
    user_info = _get_linked_user_info(account['usuario_id'])

    return jsonify({
        'ok': True,
        'user': user_info,
        'games': games,
        'total_games': len(games),
        'total_packages': sum(len(g['packages']) for g in games),
    })


# ---------------------------------------------------------------------------
# POST /api/v1/recharge
# ---------------------------------------------------------------------------

@bp.route('/api/v1/recharge', methods=['POST'])
@require_api_key
def api_v1_recharge():
    """Crea una orden de recarga de un paquete mapeado. Descuenta saldo del usuario vinculado.

    Solo acepta juegos creados por el admin con Mapeo activo (los del catálogo de /api/v1/products);
    se recargan igual que en la web: Bot de Free Fire primero si está activo y, si no, el proveedor.

    Body JSON:
        product_id   (int)  - ID del juego (game_id del catálogo)
        package_id   (int)  - ID del paquete
        player_id    (str)  - ID del jugador
        player_id2   (str)  - Opcional, segundo ID (ej: Zone ID de Mobile Legends)
        external_order_id (str) - Opcional, referencia de la web cliente para trazabilidad
    """
    account = request._ws_account
    usuario_id = account['usuario_id']

    from api_panel import cuenta_puede, init_permisos
    init_permisos()
    if not cuenta_puede(account, 'recargas'):
        return jsonify({'ok': False, 'error': 'Esta cuenta no tiene permiso para hacer recargas'}), 403

    data = request.get_json(silent=True) or {}
    product_id = data.get('product_id')
    package_id = data.get('package_id')
    player_id = str(data.get('player_id', '')).strip()
    player_id2 = str(data.get('player_id2', '')).strip()
    external_order_id = str(data.get('external_order_id', '')).strip()

    if not package_id or not player_id:
        return jsonify({'ok': False, 'error': 'package_id y player_id son requeridos'}), 400

    # Idempotencia: la misma external_order_id nunca recarga ni cobra dos veces
    if external_order_id:
        external_order_id = external_order_id[:80]
        conn_prev = _get_conn()
        prev = conn_prev.execute(
            "SELECT * FROM api_orders WHERE account_id = ? AND external_order_id = ? AND estado IN ('procesando', 'completada') "
            'ORDER BY id DESC LIMIT 1', (account['id'], external_order_id)).fetchone()
        conn_prev.close()
        if prev:
            return jsonify({
                'ok': prev['estado'] == 'completada', 'duplicada': True, 'order_id': prev['id'], 'status': prev['estado'],
                'player_name': prev['player_name'] or '', 'reference_no': prev['reference_no'] or '',
                'mensaje': 'Esta external_order_id ya tiene una recarga; no se hizo otra.',
            }), 200 if prev['estado'] == 'completada' else 409

    try:
        package_id = int(package_id)
        if product_id is not None:
            product_id = int(product_id)
    except (ValueError, TypeError):
        return jsonify({'ok': False, 'error': 'package_id y product_id deben ser numéricos'}), 400

    # --- Resolver juego y paquete: solo juegos creados por el admin con Mapeo activo ---
    from api_panel import paquete_mapeado
    game, pkg = paquete_mapeado(product_id, package_id)
    if not game:
        return jsonify({'ok': False, 'error': f'Paquete {package_id} no encontrado, inactivo o sin mapeo'}), 404
    return _recharge_mapped(account, game, pkg, player_id, player_id2, external_order_id)


def _recharge_mapped(account, game, pkg, player_id, player_id2, external_order_id):
    """Recarga por Mapeo (con el Bot de Free Fire primero si está activo), igual que la compra web.
    Cobra el precio del paquete al usuario vinculado y lo devuelve si falla."""
    from dynamic_games import ejecutar_recarga_mapeo, units_de_mapeo
    usuario_id = account['usuario_id']
    precio = float(pkg['precio'])
    _start = time_module.time()

    conn = _get_conn()
    try:
        cur = conn.execute('''
            INSERT INTO api_orders (account_id, usuario_id, game_type, game_name, package_id, package_name,
                                    player_id, player_id2, precio, estado, external_order_id)
            VALUES (?, ?, 'mapped', ?, ?, ?, ?, ?, ?, 'procesando', ?)
            RETURNING id
        ''', (account['id'], usuario_id, game['nombre'], pkg['id'], pkg['nombre'], player_id, player_id2,
              precio, external_order_id))
        order_id = cur.fetchone()[0]
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f'[WL API] Error creando orden: {e}')
        return jsonify({'ok': False, 'error': 'Error interno al crear orden'}), 500
    finally:
        conn.close()

    res = ejecutar_recarga_mapeo(game, pkg, units_de_mapeo(game['id'], pkg['id']), usuario_id, True,
                                 player_id, player_id2, '', sufijo_historial=f" [API: {account['nombre']}]",
                                 _start=_start)
    estado = res['estado']
    duration = round(time_module.time() - _start, 1)
    status_code = 200
    if estado == 'aprobado':
        _actualizar_orden(order_id, 'completada', player_name=res['player_name'],
                          reference_no=', '.join(res.get('refs') or []), precio=res.get('cobrado', precio),
                          duracion=duration)
    elif estado == 'procesando':
        _actualizar_orden(order_id, 'procesando', reference_no=res['merchant_code'], duracion=duration)
        status_code = 202
    else:
        _actualizar_orden(order_id, 'fallida', error=res.get('error') or 'La recarga falló', duracion=duration)
        status_code = 402 if estado == 'sin_saldo' else (503 if estado == 'no_config' else 422)

    if account.get('webhook_url') and estado != 'procesando':
        _send_webhook_async(order_id, account['webhook_url'])

    remaining = 0.0
    try:
        conn_bal = _get_conn()
        row = conn_bal.execute('SELECT saldo FROM usuarios WHERE id = ?', (usuario_id,)).fetchone()
        conn_bal.close()
        remaining = float(row['saldo']) if row else 0.0
    except Exception:
        pass

    status = {'aprobado': 'completada', 'procesando': 'procesando'}.get(estado, 'fallida')
    body = {'ok': estado in ('aprobado', 'procesando'), 'order_id': order_id, 'status': status,
            'player_name': res.get('player_name') or '', 'reference_no': ', '.join(res.get('refs') or []),
            'duration': duration, 'user_id': usuario_id, 'remaining_balance': remaining}
    if estado == 'aprobado' and res.get('refund'):
        body.update(parcial=True, cobrado=res.get('cobrado'), devuelto=res['refund'])
    if status == 'fallida':
        body['error'] = 'Saldo insuficiente' if estado == 'sin_saldo' else (res.get('error') or 'La recarga falló')
        if estado == 'sin_saldo':
            body.update(saldo_actual=remaining, precio=precio)
    return jsonify(body), status_code


def _actualizar_orden(order_id, estado, player_name='', reference_no='', error='', precio=None, duracion=0):
    conn = _get_conn()
    try:
        sets = ['estado = ?', 'player_name = ?', 'reference_no = ?', 'error_msg = ?', 'duration_seconds = ?']
        params = [estado, player_name or '', reference_no or '', error or '', duracion]
        if precio is not None:
            sets.append('precio = ?')
            params.append(precio)
        if estado in ('completada', 'fallida'):
            sets.append('fecha_completada = CURRENT_TIMESTAMP')
        conn.execute(f"UPDATE api_orders SET {', '.join(sets)} WHERE id = ?", (*params, order_id))
        conn.commit()
    finally:
        conn.close()


def _sincronizar_orden(row):
    """Si una orden mapeada seguía 'procesando', copia el resultado del reconciliador."""
    if not row or row['estado'] != 'procesando' or row['game_type'] != 'mapped' or not str(row['reference_no'] or '').startswith('DG'):
        return row
    conn = _get_conn()
    try:
        tx = conn.execute('SELECT estado, ingame_name, gamepoint_referenceno, monto, notas FROM transacciones_dinamicas '
                          'WHERE transaccion_id = ?', (row['reference_no'],)).fetchone()
    finally:
        conn.close()
    if not tx or tx['estado'] not in ('aprobado', 'rechazado'):
        return row
    if tx['estado'] == 'aprobado':
        _actualizar_orden(row['id'], 'completada', player_name=tx['ingame_name'] or '',
                          reference_no=tx['gamepoint_referenceno'] or '', precio=float(tx['monto'] or row['precio']))
    else:
        _actualizar_orden(row['id'], 'fallida', reference_no=row['reference_no'], error='La recarga falló en el proveedor')
    conn = _get_conn()
    try:
        return conn.execute('SELECT * FROM api_orders WHERE id = ?', (row['id'],)).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# GET /api/v1/orders/<order_id>
# ---------------------------------------------------------------------------

@bp.route('/api/v1/orders/<int:order_id>', methods=['GET'])
@require_api_key
def api_v1_order_status(order_id):
    """Consulta el estado de una orden."""
    account = request._ws_account
    conn = _get_conn()
    row = conn.execute(
        'SELECT * FROM api_orders WHERE id = ? AND account_id = ?',
        (order_id, account['id'])
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({'ok': False, 'error': 'Orden no encontrada'}), 404
    row = _sincronizar_orden(row)

    return jsonify({
        'ok': True,
        'order': {
            'id': row['id'],
            'status': row['estado'],
            'game_type': row['game_type'],
            'game_name': row['game_name'],
            'package_id': row['package_id'],
            'package_name': row['package_name'],
            'player_id': row['player_id'],
            'player_name': row['player_name'],
            'precio': float(row['precio']),
            'reference_no': row['reference_no'],
            'error': row['error_msg'],
            'duration': row['duration_seconds'],
            'external_order_id': row['external_order_id'],
            'created_at': str(row['fecha']),
            'completed_at': str(row['fecha_completada']) if row['fecha_completada'] else None,
        }
    })


# ---------------------------------------------------------------------------
# GET /api/v1/order-status?external_order_id=XXX
# ---------------------------------------------------------------------------

@bp.route('/api/v1/order-status', methods=['GET'])
@require_api_key
def api_v1_order_status_by_external():
    """Consulta el estado de una orden por external_order_id."""
    account = request._ws_account
    ext_id = (request.args.get('external_order_id') or '').strip()
    if not ext_id:
        return jsonify({'ok': False, 'error': 'external_order_id requerido'}), 400

    conn = _get_conn()
    row = conn.execute(
        'SELECT * FROM api_orders WHERE external_order_id = ? AND account_id = ? ORDER BY id DESC LIMIT 1',
        (ext_id, account['id'])
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({'ok': True, 'found': False, 'status': 'not_found'})
    row = _sincronizar_orden(row)
    return jsonify({
        'ok': True,
        'found': True,
        'status': row['estado'],
        'order': {
            'id': row['id'],
            'status': row['estado'],
            'player_name': row['player_name'] or '',
            'reference_no': row['reference_no'] or '',
            'error': row['error_msg'] or '',
            'external_order_id': row['external_order_id'],
            'created_at': str(row['fecha']),
            'completed_at': str(row['fecha_completada']) if row['fecha_completada'] else None,
        }
    })


# ---------------------------------------------------------------------------
# GET /api/v1/balance
# ---------------------------------------------------------------------------

@bp.route('/api/v1/balance', methods=['GET'])
@require_api_key
def api_v1_balance():
    """Consulta el saldo del usuario vinculado a la cuenta API."""
    account = request._ws_account
    user_info = _get_linked_user_info(account['usuario_id'])
    if not user_info:
        return jsonify({'ok': False, 'error': 'Usuario vinculado no encontrado'}), 404

    return jsonify({
        'ok': True,
        'user': user_info,
        'account_name': account['nombre'],
    })


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------

def _send_webhook_async(order_id, webhook_url):
    """Envía una notificación POST al webhook_url de la web cliente (async)."""
    def _do_send():
        try:
            conn = _get_conn()
            row = conn.execute('SELECT * FROM api_orders WHERE id = ?', (order_id,)).fetchone()
            conn.close()
            if not row:
                return

            payload = {
                'event': 'order.updated',
                'order': {
                    'id': row['id'],
                    'status': row['estado'],
                    'game_type': row['game_type'],
                    'game_name': row['game_name'],
                    'package_id': row['package_id'],
                    'package_name': row['package_name'],
                    'player_id': row['player_id'],
                    'player_name': row['player_name'],
                    'precio': float(row['precio']),
                    'reference_no': row['reference_no'],
                    'error': row['error_msg'],
                    'external_order_id': row['external_order_id'],
                    'completed_at': str(row['fecha_completada']) if row['fecha_completada'] else None,
                }
            }

            resp = req_lib.post(
                webhook_url,
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json', 'User-Agent': 'Revendedores-Webhook/1.0'}
            )
            logger.info(f'[WL Webhook] order={order_id} url={webhook_url} status={resp.status_code}')

            # Marcar webhook como enviado
            conn = _get_conn()
            conn.execute('UPDATE api_orders SET webhook_sent = TRUE WHERE id = ?', (order_id,))
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f'[WL Webhook] Error enviando webhook order={order_id}: {e}')

    t = threading.Thread(target=_do_send, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Admin: Gestión de WebServiceAccounts
# ---------------------------------------------------------------------------

@bp.route('/admin/webservice-accounts', methods=['GET'])
def admin_list_ws_accounts():
    """Lista todas las cuentas de web service (JSON)."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403

    conn = _get_conn()
    rows = conn.execute('''
        SELECT ws.*, u.nombre as usuario_nombre, u.apellido as usuario_apellido, u.correo as usuario_correo, u.saldo as usuario_saldo
        FROM webservice_accounts ws
        JOIN usuarios u ON ws.usuario_id = u.id
        ORDER BY ws.id
    ''').fetchall()
    conn.close()

    accounts = []
    for r in rows:
        accounts.append({
            'id': r['id'],
            'nombre': r['nombre'],
            'api_key': r['api_key'],
            'usuario_id': r['usuario_id'],
            'usuario_nombre': f"{r['usuario_nombre']} {r['usuario_apellido']}",
            'usuario_correo': r['usuario_correo'],
            'usuario_saldo': float(r['usuario_saldo']),
            'webhook_url': r['webhook_url'],
            'activo': r['activo'],
            'fecha_creacion': str(r['fecha_creacion']),
        })

    return jsonify({'ok': True, 'accounts': accounts})


@bp.route('/admin/webservice-accounts/create', methods=['POST'])
def admin_create_ws_account():
    """Crea una nueva cuenta de web service."""
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')

    data = request.form or request.get_json(silent=True) or {}
    nombre = (data.get('ws_nombre') or '').strip()
    usuario_id = data.get('ws_usuario_id')
    webhook_url = (data.get('ws_webhook_url') or '').strip()

    if not nombre or not usuario_id:
        flash('Nombre y usuario son requeridos para crear una cuenta API.', 'error')
        return redirect('/admin')

    try:
        usuario_id = int(usuario_id)
    except (ValueError, TypeError):
        flash('ID de usuario inválido.', 'error')
        return redirect('/admin')

    # Verificar que el usuario existe
    conn = _get_conn()
    user = conn.execute('SELECT id FROM usuarios WHERE id = ?', (usuario_id,)).fetchone()
    if not user:
        conn.close()
        flash('Usuario no encontrado.', 'error')
        return redirect('/admin')

    api_key = _generate_api_key()

    try:
        conn.execute('''
            INSERT INTO webservice_accounts (nombre, api_key, usuario_id, webhook_url)
            VALUES (?, ?, ?, ?)
        ''', (nombre, api_key, usuario_id, webhook_url))
        conn.commit()
        flash(f'Cuenta API "{nombre}" creada. Key: {api_key}', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error creando cuenta: {e}', 'error')
    finally:
        conn.close()

    return redirect('/admin')


@bp.route('/admin/webservice-accounts/<int:account_id>/toggle', methods=['POST'])
def admin_toggle_ws_account(account_id):
    """Activa o desactiva una cuenta de web service."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403

    conn = _get_conn()
    row = conn.execute('SELECT activo FROM webservice_accounts WHERE id = ?', (account_id,)).fetchone()
    if not row:
        conn.close()
        flash('Cuenta no encontrada.', 'error')
        return redirect('/admin')

    new_val = not row['activo']
    conn.execute('UPDATE webservice_accounts SET activo = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ?',
                 (new_val, account_id))
    conn.commit()
    conn.close()

    estado = 'activada' if new_val else 'desactivada'
    flash(f'Cuenta API #{account_id} {estado}.', 'success')
    return redirect('/admin')


@bp.route('/admin/webservice-accounts/<int:account_id>/regenerate-key', methods=['POST'])
def admin_regenerate_ws_key(account_id):
    """Regenera la API key de una cuenta."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403

    conn = _get_conn()
    row = conn.execute('SELECT id FROM webservice_accounts WHERE id = ?', (account_id,)).fetchone()
    if not row:
        conn.close()
        flash('Cuenta no encontrada.', 'error')
        return redirect('/admin')

    new_key = _generate_api_key()
    conn.execute('UPDATE webservice_accounts SET api_key = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ?',
                 (new_key, account_id))
    conn.commit()
    conn.close()

    flash(f'Nueva API Key para cuenta #{account_id}: {new_key}', 'success')
    return redirect('/admin')


@bp.route('/admin/webservice-accounts/<int:account_id>/update', methods=['POST'])
def admin_update_ws_account(account_id):
    """Actualiza usuario vinculado y/o webhook de una cuenta."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403

    data = request.form or request.get_json(silent=True) or {}
    nuevo_usuario_id = data.get('ws_usuario_id')
    nuevo_webhook = data.get('ws_webhook_url')

    conn = _get_conn()
    row = conn.execute('SELECT * FROM webservice_accounts WHERE id = ?', (account_id,)).fetchone()
    if not row:
        conn.close()
        flash('Cuenta no encontrada.', 'error')
        return redirect('/admin')

    updates = []
    params = []

    if nuevo_usuario_id:
        try:
            nuevo_usuario_id = int(nuevo_usuario_id)
            user_exists = conn.execute('SELECT id FROM usuarios WHERE id = ?', (nuevo_usuario_id,)).fetchone()
            if not user_exists:
                conn.close()
                flash(f'Usuario ID {nuevo_usuario_id} no encontrado.', 'error')
                return redirect('/admin')
            updates.append('usuario_id = ?')
            params.append(nuevo_usuario_id)
        except (ValueError, TypeError):
            conn.close()
            flash('ID de usuario inválido.', 'error')
            return redirect('/admin')

    if nuevo_webhook is not None:
        updates.append('webhook_url = ?')
        params.append(nuevo_webhook.strip())

    if updates:
        updates.append('fecha_actualizacion = CURRENT_TIMESTAMP')
        params.append(account_id)
        conn.execute(f'UPDATE webservice_accounts SET {", ".join(updates)} WHERE id = ?', params)
        conn.commit()
        flash(f'Cuenta API #{account_id} actualizada.', 'success')
    conn.close()
    return redirect('/admin')


@bp.route('/admin/webservice-accounts/<int:account_id>/delete', methods=['POST'])
def admin_delete_ws_account(account_id):
    """Elimina una cuenta de web service."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403

    conn = _get_conn()
    conn.execute('DELETE FROM webservice_accounts WHERE id = ?', (account_id,))
    conn.commit()
    conn.close()

    flash(f'Cuenta API #{account_id} eliminada.', 'success')
    return redirect('/admin')

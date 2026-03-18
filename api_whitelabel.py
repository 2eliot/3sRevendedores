"""
API REST de Marca Blanca v1
===========================
Blueprint Flask que permite a webs externas (Inefablestore, etc.) conectarse,
consultar productos y ejecutar recargas usando el saldo de un usuario asignado.

Endpoints:
  GET  /api/v1/products          → Catálogo de juegos + paquetes activos
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
    """Retorna catálogo completo de juegos activos con paquetes y precios base."""
    from dynamic_games import get_all_dynamic_games, get_dynamic_packages

    games = []

    # 1. Juegos Dinámicos (GamePoint)
    try:
        dyn_games = get_all_dynamic_games(only_active=True)
        for game in dyn_games:
            pkgs = get_dynamic_packages(game['id'], only_active=True)
            packages = []
            for pkg in pkgs:
                packages.append({
                    'package_id': pkg['id'],
                    'name': pkg['nombre'],
                    'price': float(pkg['precio']),
                    'description': pkg.get('descripcion', ''),
                    'gamepoint_package_id': pkg.get('gamepoint_package_id'),
                })
            games.append({
                'game_type': 'dynamic',
                'game_id': game['id'],
                'name': game['nombre'],
                'slug': game['slug'],
                'mode': game.get('modo', 'id'),
                'icon': game.get('icono', '🎮'),
                'description': game.get('descripcion', ''),
                'packages': packages,
            })
    except Exception as e:
        logger.warning(f'[WL API] Error leyendo juegos dinámicos: {e}')

    # 2. Blood Strike
    try:
        conn = _get_conn()
        bs_rows = conn.execute(
            'SELECT id, nombre, precio, descripcion, gamepoint_package_id FROM precios_bloodstriker WHERE activo = TRUE ORDER BY id'
        ).fetchall()
        conn.close()
        bs_packages = []
        for r in bs_rows:
            bs_packages.append({
                'package_id': r['id'],
                'name': r['nombre'],
                'price': float(r['precio']),
                'description': r['descripcion'],
                'gamepoint_package_id': r.get('gamepoint_package_id'),
            })
        if bs_packages:
            games.append({
                'game_type': 'bloodstriker',
                'game_id': -155,
                'name': 'Blood Strike',
                'slug': 'bloodstriker',
                'mode': 'id',
                'icon': '🔫',
                'description': 'Blood Strike - Recargas de Gold',
                'packages': bs_packages,
            })
    except Exception as e:
        logger.warning(f'[WL API] Error leyendo Blood Strike: {e}')

    # 3. Free Fire ID
    try:
        conn = _get_conn()
        ff_rows = conn.execute(
            'SELECT id, nombre, precio, descripcion FROM precios_freefire_id WHERE activo = TRUE ORDER BY id'
        ).fetchall()
        conn.close()
        ff_packages = []
        for r in ff_rows:
            ff_packages.append({
                'package_id': r['id'],
                'name': r['nombre'],
                'price': float(r['precio']),
                'description': r.get('descripcion', ''),
            })
        if ff_packages:
            games.append({
                'game_type': 'freefire_id',
                'game_id': -1,
                'name': 'Free Fire ID',
                'slug': 'freefire-id',
                'mode': 'id',
                'icon': '🔥',
                'description': 'Free Fire - Recargas por ID',
                'packages': ff_packages,
            })
    except Exception as e:
        logger.warning(f'[WL API] Error leyendo Free Fire ID: {e}')

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
    """Crea una orden de recarga. Descuenta saldo del usuario vinculado.

    Body JSON:
        product_id   (int)  - ID del juego (game_id del catálogo, -155 para BS, -1 para FF)
        package_id   (int)  - ID del paquete
        player_id    (str)  - ID del jugador
        player_id2   (str)  - Opcional, segundo ID (ej: Zone ID de Mobile Legends)
        external_order_id (str) - Opcional, referencia de la web cliente para trazabilidad
    """
    account = request._ws_account
    usuario_id = account['usuario_id']

    data = request.get_json(silent=True) or {}
    product_id = data.get('product_id')
    package_id = data.get('package_id')
    player_id = str(data.get('player_id', '')).strip()
    player_id2 = str(data.get('player_id2', '')).strip()
    external_order_id = str(data.get('external_order_id', '')).strip()

    if not package_id or not player_id:
        return jsonify({'ok': False, 'error': 'package_id y player_id son requeridos'}), 400

    try:
        package_id = int(package_id)
        if product_id is not None:
            product_id = int(product_id)
    except (ValueError, TypeError):
        return jsonify({'ok': False, 'error': 'package_id y product_id deben ser numéricos'}), 400

    # --- Resolver juego y paquete ---
    game_type, game_name, pkg_name, precio, gp_package_id, gp_product_id = _resolve_package(product_id, package_id)
    if not game_type:
        return jsonify({'ok': False, 'error': f'Paquete {package_id} no encontrado o inactivo'}), 404

    # --- Verificar y descontar saldo atómicamente ---
    conn = _get_conn()
    try:
        cursor = conn.execute(
            'UPDATE usuarios SET saldo = saldo - ? WHERE id = ? AND saldo >= ?',
            (precio, usuario_id, precio)
        )
        if cursor.rowcount == 0:
            conn.close()
            # Obtener saldo actual para info
            conn2 = _get_conn()
            row = conn2.execute('SELECT saldo FROM usuarios WHERE id = ?', (usuario_id,)).fetchone()
            conn2.close()
            saldo_actual = row['saldo'] if row else 0
            return jsonify({
                'ok': False,
                'error': 'Saldo insuficiente',
                'saldo_actual': float(saldo_actual),
                'precio': float(precio),
            }), 402

        # Crear orden en estado pendiente
        cur = conn.execute('''
            INSERT INTO api_orders
            (account_id, usuario_id, game_type, game_name, package_id, package_name,
             player_id, player_id2, precio, estado, external_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'procesando', ?)
            RETURNING id
        ''', (account['id'], usuario_id, game_type, game_name, package_id, pkg_name,
              player_id, player_id2, precio, external_order_id))
        order_id = cur.fetchone()[0]
        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()
        logger.error(f'[WL API] Error creando orden: {e}')
        return jsonify({'ok': False, 'error': 'Error interno al crear orden'}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # --- Ejecutar recarga en background para no bloquear al cliente ---
    # (se ejecuta síncrono porque el cliente espera el resultado)
    result = _execute_recharge(order_id, game_type, package_id, player_id, player_id2,
                               precio, gp_package_id, gp_product_id, usuario_id, account,
                               game_name=game_name, pkg_name=pkg_name)

    return result


def _resolve_package(product_id, package_id):
    """Resuelve el tipo de juego, nombre, precio y IDs de GamePoint para un package_id.
    Returns: (game_type, game_name, pkg_name, precio, gp_package_id, gp_product_id) or (None,...) si no existe.
    """
    from dynamic_games import get_dynamic_package_by_id, get_dynamic_game_by_id

    # 1. Si product_id indica juego dinámico (> 0) o no especificado, buscar en dinámicos
    if product_id is None or (product_id is not None and product_id > 0):
        dyn_pkg = get_dynamic_package_by_id(package_id)
        if dyn_pkg and dyn_pkg.get('activo') and dyn_pkg.get('gamepoint_package_id'):
            if product_id and dyn_pkg.get('juego_id') != product_id:
                pass  # No coincide, seguir buscando
            else:
                game = get_dynamic_game_by_id(dyn_pkg['juego_id'])
                if game and game.get('activo'):
                    return (
                        'dynamic',
                        game['nombre'],
                        dyn_pkg['nombre'],
                        float(dyn_pkg['precio']),
                        dyn_pkg['gamepoint_package_id'],
                        game['gamepoint_product_id'],
                    )

    # 2. Blood Strike (product_id == -155 o fallback)
    if product_id is None or product_id == -155:
        try:
            conn = _get_conn()
            bs = conn.execute(
                'SELECT id, nombre, precio, gamepoint_package_id FROM precios_bloodstriker WHERE id = ? AND activo = TRUE',
                (package_id,)
            ).fetchone()
            conn.close()
            if bs and bs['gamepoint_package_id']:
                bs_product_id = int(os.environ.get('BLOODSTRIKE_PRODUCT_ID', '155'))
                return (
                    'bloodstriker',
                    'Blood Strike',
                    bs['nombre'],
                    float(bs['precio']),
                    bs['gamepoint_package_id'],
                    bs_product_id,
                )
        except Exception:
            pass

    # 3. Free Fire ID (product_id == -1 o fallback)
    if product_id is None or product_id == -1:
        try:
            conn = _get_conn()
            ff = conn.execute(
                'SELECT id, nombre, precio FROM precios_freefire_id WHERE id = ? AND activo = TRUE',
                (package_id,)
            ).fetchone()
            conn.close()
            if ff:
                return (
                    'freefire_id',
                    'Free Fire ID',
                    ff['nombre'],
                    float(ff['precio']),
                    None,  # No usa GamePoint
                    None,
                )
        except Exception:
            pass

    return (None, None, None, None, None, None)


def _execute_recharge(order_id, game_type, package_id, player_id, player_id2,
                      precio, gp_package_id, gp_product_id, usuario_id, account,
                      game_name='', pkg_name=''):
    """Ejecuta la recarga según el tipo de juego y actualiza la orden."""
    _start = time_module.time()

    try:
        if game_type in ('dynamic', 'bloodstriker'):
            result = _execute_gamepoint_recharge(
                order_id, game_type, package_id, player_id, player_id2,
                gp_package_id, gp_product_id
            )
        elif game_type == 'freefire_id':
            result = _execute_freefire_id_recharge(order_id, package_id, player_id)
        else:
            result = {'ok': False, 'error': f'Tipo de juego no soportado: {game_type}'}

    except Exception as e:
        logger.error(f'[WL API] Error ejecutando recarga order={order_id}: {e}')
        result = {'ok': False, 'error': f'Error interno: {str(e)}'}

    _duration = round(time_module.time() - _start, 1)

    # Actualizar orden
    conn = _get_conn()
    try:
        if result.get('ok'):
            conn.execute('''
                UPDATE api_orders
                SET estado = 'completada', reference_no = ?, player_name = ?,
                    duration_seconds = ?, fecha_completada = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (result.get('reference_no', ''), result.get('player_name', ''),
                  _duration, order_id))

            # ── Registrar en historial general (transacciones + historial_compras) ──
            try:
                _nc = f"WL-{secrets.token_hex(4).upper()}"
                _tid = f"WL-API-{order_id}"
                _player_name = result.get('player_name', '')
                if _player_name:
                    _pin_info = f"ID: {player_id} - Jugador: {_player_name}"
                else:
                    _pin_info = f"ID: {player_id}"
                if player_id2:
                    _pin_info = f"ID: {player_id}/{player_id2} - " + _pin_info.split(' - ', 1)[-1]
                _pin_info += f" [API: {account['nombre']}]"

                _paquete_display = f"{game_name} - {pkg_name}" if game_name else (pkg_name or f"Paquete {package_id}")

                conn.execute('''
                    INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto, duracion_segundos)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (usuario_id, _nc, _pin_info, _tid, _paquete_display, -precio, _duration))

                _saldo_row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (usuario_id,)).fetchone()
                _saldo = float(_saldo_row['saldo']) if _saldo_row else 0.0
                conn.execute('''
                    INSERT INTO historial_compras (usuario_id, monto, paquete_nombre, pin, tipo_evento, duracion_segundos, saldo_antes, saldo_despues)
                    VALUES (?, ?, ?, ?, 'compra', ?, ?, ?)
                ''', (usuario_id, precio, _paquete_display, _pin_info, _duration, _saldo + precio, _saldo))

                try:
                    from app import update_monthly_spending
                    update_monthly_spending(conn, usuario_id, precio)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f'[WL API] Error registrando transacción general order={order_id}: {e}')

        else:
            # Reembolsar saldo
            conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, usuario_id))
            conn.execute('''
                UPDATE api_orders
                SET estado = 'fallida', error_msg = ?, duration_seconds = ?,
                    fecha_completada = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (result.get('error', 'Error desconocido'), _duration, order_id))
        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error(f'[WL API] Error actualizando orden {order_id}: {e}')
    finally:
        conn.close()

    # Disparar webhook si la cuenta tiene URL configurada
    if account.get('webhook_url'):
        _send_webhook_async(order_id, account['webhook_url'])

    # Obtener saldo restante del usuario
    remaining_balance = 0.0
    try:
        conn_bal = _get_conn()
        bal_row = conn_bal.execute('SELECT saldo FROM usuarios WHERE id = ?', (usuario_id,)).fetchone()
        conn_bal.close()
        remaining_balance = float(bal_row['saldo']) if bal_row else 0.0
    except Exception:
        pass

    # Construir respuesta
    status_code = 200 if result.get('ok') else 422
    response = {
        'ok': result.get('ok', False),
        'order_id': order_id,
        'status': 'completada' if result.get('ok') else 'fallida',
        'player_name': result.get('player_name', ''),
        'reference_no': result.get('reference_no', ''),
        'duration': _duration,
        'user_id': usuario_id,
        'remaining_balance': remaining_balance,
    }
    if not result.get('ok'):
        response['error'] = result.get('error', '')

    return jsonify(response), status_code


def _execute_gamepoint_recharge(order_id, game_type, package_id, player_id, player_id2,
                                gp_package_id, gp_product_id):
    """Ejecuta recarga vía GamePoint API (juegos dinámicos y Blood Strike)."""
    from app import _gameclub_get_token, _gameclub_order_validate, _gameclub_order_create, _gameclub_order_inquiry

    # 1. Token
    gc_token, gc_err = _gameclub_get_token()
    if not gc_token:
        err = (gc_err or {}).get('message', 'No se pudo obtener token de GamePoint')
        return {'ok': False, 'error': f'Error proveedor: {err}'}

    # 2. Validate
    input_fields = {'input1': str(player_id)}
    if player_id2:
        input_fields['input2'] = str(player_id2)

    validate_data = _gameclub_order_validate(gc_token, gp_product_id, input_fields)
    validate_code = (validate_data or {}).get('code')
    if validate_code != 200 or not (validate_data or {}).get('validation_token'):
        err_msg = (validate_data or {}).get('message', 'Error validando orden')
        return {'ok': False, 'error': f'Validación falló: {err_msg}'}

    validation_token = validate_data['validation_token']

    # 3. Create order
    prefix = 'WL-DG' if game_type == 'dynamic' else 'WL-BS'
    merchant_code = f"{prefix}-{order_id}-" + secrets.token_hex(4).upper()
    create_data = _gameclub_order_create(gc_token, validation_token, gp_package_id, merchant_code)
    create_code = (create_data or {}).get('code')
    reference_no = (create_data or {}).get('referenceno', '')

    if create_code not in (100, 101):
        err_msg = (create_data or {}).get('message', 'Error creando orden en GamePoint')
        return {'ok': False, 'error': err_msg}

    # 4. Inquiry para obtener ingamename
    ingame_name = ''
    try:
        if reference_no:
            for _attempt in range(3):
                if _attempt > 0:
                    time_module.sleep(1.5)
                inq_data = _gameclub_order_inquiry(gc_token, reference_no)
                ingame_name = (inq_data or {}).get('ingamename') or ''
                if ingame_name:
                    break
    except Exception as e:
        logger.warning(f'[WL API] Inquiry error order={order_id}: {e}')

    return {
        'ok': True,
        'player_name': ingame_name,
        'reference_no': reference_no,
    }


def _execute_freefire_id_recharge(order_id, package_id, player_id):
    """Ejecuta recarga de Free Fire ID vía redención de pin."""
    from app import get_available_pin_freefire_global, redeem_pin_vps, get_redeemer_config_from_db

    pin_disponible = get_available_pin_freefire_global(package_id)
    if not pin_disponible:
        return {'ok': False, 'error': f'Sin stock para paquete {package_id}'}

    pin_codigo = pin_disponible['pin_codigo']

    redeemer_config = get_redeemer_config_from_db(_get_conn)
    try:
        redeem_result = redeem_pin_vps(pin_codigo, player_id, redeemer_config)
    except Exception as e:
        # Devolver pin al stock
        try:
            conn = _get_conn()
            conn.execute('INSERT OR IGNORE INTO pines_freefire_global (monto_id, pin_codigo) VALUES (?,?)',
                         (package_id, pin_codigo))
            conn.commit()
            conn.close()
        except Exception:
            pass
        return {'ok': False, 'error': f'Error redención: {str(e)}'}

    if redeem_result and redeem_result.success:
        return {
            'ok': True,
            'player_name': redeem_result.player_name or '',
            'reference_no': '',
        }
    else:
        # Devolver pin al stock
        try:
            conn = _get_conn()
            conn.execute('INSERT OR IGNORE INTO pines_freefire_global (monto_id, pin_codigo) VALUES (?,?)',
                         (package_id, pin_codigo))
            conn.commit()
            conn.close()
        except Exception:
            pass
        err_msg = (redeem_result.message if redeem_result else None) or 'Redención fallida'
        return {'ok': False, 'error': err_msg}


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

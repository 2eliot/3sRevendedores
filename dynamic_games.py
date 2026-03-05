"""
Sistema de Juegos Dinámicos — Blueprint Flask
Permite crear juegos vinculados a GamePoint desde el panel admin,
con auto-generación de página, menú, historial y sincronización de precios.
"""
import json
import os
import re
import secrets
import sqlite3
import time as time_module
import logging

from flask import Blueprint, jsonify, request, render_template, session, flash, redirect

logger = logging.getLogger(__name__)

bp = Blueprint('dynamic_games', __name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn():
    db_path = os.environ.get('DATABASE_PATH', 'usuarios.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_all_dynamic_games(only_active=False):
    conn = _get_conn()
    if only_active:
        rows = conn.execute('SELECT * FROM juegos_dinamicos WHERE activo = 1 ORDER BY nombre').fetchall()
    else:
        rows = conn.execute('SELECT * FROM juegos_dinamicos ORDER BY nombre').fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dynamic_game_by_slug(slug):
    conn = _get_conn()
    row = conn.execute('SELECT * FROM juegos_dinamicos WHERE slug = ?', (slug,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_dynamic_game_by_id(game_id):
    conn = _get_conn()
    row = conn.execute('SELECT * FROM juegos_dinamicos WHERE id = ?', (game_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_dynamic_packages(game_id, only_active=False):
    conn = _get_conn()
    if only_active:
        rows = conn.execute(
            'SELECT * FROM paquetes_dinamicos WHERE juego_id = ? AND activo = 1 ORDER BY orden, id',
            (game_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM paquetes_dinamicos WHERE juego_id = ? ORDER BY orden, id',
            (game_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dynamic_package_by_id(pkg_id):
    conn = _get_conn()
    row = conn.execute('SELECT * FROM paquetes_dinamicos WHERE id = ?', (pkg_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def parse_campos_config(game):
    """Parse the JSON campos_config from a game dict."""
    raw = game.get('campos_config', '{}') or '{}'
    try:
        return json.loads(raw)
    except Exception:
        return {}


def slugify(text):
    """Generate a URL-safe slug from text."""
    text = text.lower().strip()
    text = re.sub(r'[áàäâ]', 'a', text)
    text = re.sub(r'[éèëê]', 'e', text)
    text = re.sub(r'[íìïî]', 'i', text)
    text = re.sub(r'[óòöô]', 'o', text)
    text = re.sub(r'[úùüû]', 'u', text)
    text = re.sub(r'[ñ]', 'n', text)
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text or 'juego'


# ---------------------------------------------------------------------------
# GamePoint helpers — import from app at runtime to avoid circular imports
# ---------------------------------------------------------------------------

def _gp_helpers():
    """Lazy import of GamePoint helper functions from app module."""
    import app as _app
    return (
        _app._gameclub_get_token,
        _app._gameclub_post,
        _app._gameclub_order_validate,
        _app._gameclub_order_create,
        _app._gameclub_order_inquiry,
    )


# ---------------------------------------------------------------------------
# ADMIN: CRUD juegos dinámicos
# ---------------------------------------------------------------------------

@bp.route('/admin/dynamic-games')
def admin_dynamic_games_page():
    if not session.get('is_admin'):
        flash('Acceso denegado.', 'error')
        return redirect('/auth')
    games = get_all_dynamic_games()
    for g in games:
        g['_campos'] = parse_campos_config(g)
        g['_packages'] = get_dynamic_packages(g['id'])
    return render_template('admin_dynamic_games.html', games=games)


@bp.route('/admin/dynamic-games/create', methods=['POST'])
def admin_create_game():
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    data = request.get_json() or request.form
    nombre = (data.get('nombre') or '').strip()
    product_id = data.get('gamepoint_product_id')
    modo = data.get('modo', 'id')
    color = data.get('color_tema', '#a78bfa')
    icono = data.get('icono', '🎮')
    descripcion = data.get('descripcion', '')
    ganancia = float(data.get('ganancia_default', 0.10))

    # Build campos_config JSON
    campos = {}
    # Primary ID field
    campos['campo_id'] = {
        'label': data.get('campo_id_label', 'ID de Jugador'),
        'placeholder': data.get('campo_id_placeholder', 'Ingresa tu ID'),
        'required': True,
        'pattern': data.get('campo_id_pattern', ''),
        'pattern_msg': data.get('campo_id_pattern_msg', ''),
    }
    # Dual ID (e.g. Mobile Legends Zone ID)
    if data.get('dual_id') in (True, 'true', '1', 'on') or data.get('campo_id2_label'):
        campos['campo_id2'] = {
            'enabled': True,
            'label': data.get('campo_id2_label', 'Zone ID'),
            'placeholder': data.get('campo_id2_placeholder', 'Ingresa tu Zone ID'),
            'required': True,
            'pattern': data.get('campo_id2_pattern', ''),
            'pattern_msg': data.get('campo_id2_pattern_msg', ''),
        }
    # Server selector
    if data.get('servidor_enabled') in (True, 'true', '1', 'on') or data.get('campo_servidor') in (True, 'true', '1', 'on'):
        opciones_raw = data.get('servidor_opciones', '')
        opciones = [o.strip() for o in opciones_raw.split(',') if o.strip()] if isinstance(opciones_raw, str) else opciones_raw
        campos['servidor'] = {
            'enabled': True,
            'label': data.get('servidor_label', 'Servidor'),
            'opciones': opciones,
        }

    if not nombre or not product_id:
        return jsonify({'error': 'Nombre y Product ID son obligatorios'}), 400

    slug = slugify(nombre)

    # Check slug uniqueness
    existing = get_dynamic_game_by_slug(slug)
    if existing:
        slug = slug + '-' + secrets.token_hex(2)

    conn = _get_conn()
    try:
        conn.execute('''
            INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, color_tema, icono, activo, campos_config, descripcion, ganancia_default)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (nombre, slug, int(product_id), modo, color, icono, False, json.dumps(campos), descripcion, ganancia))
        conn.commit()
        game_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500
    conn.close()

    if request.is_json:
        return jsonify({'success': True, 'game_id': game_id, 'slug': slug})
    flash(f'Juego "{nombre}" creado. Ahora añade paquetes en la pestaña Precios.', 'success')
    return redirect('/admin#tab=precios')


@bp.route('/admin/dynamic-games/<int:game_id>/update', methods=['POST'])
def admin_update_game(game_id):
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    data = request.get_json() or request.form
    game = get_dynamic_game_by_id(game_id)
    if not game:
        return jsonify({'error': 'Juego no encontrado'}), 404

    nombre = (data.get('nombre') or game['nombre']).strip()
    product_id = data.get('gamepoint_product_id', game['gamepoint_product_id'])
    modo = data.get('modo', game['modo'])
    color = data.get('color_tema', game['color_tema'])
    icono = data.get('icono', game['icono'])
    descripcion = data.get('descripcion', game['descripcion'])
    ganancia = float(data.get('ganancia_default', game['ganancia_default']))
    activo = data.get('activo')
    if activo is not None:
        activo = activo in (True, 'true', '1', 'on', 1)
    else:
        activo = game['activo']

    # Rebuild campos_config
    campos = {}
    campos['campo_id'] = {
        'label': data.get('campo_id_label', 'ID de Jugador'),
        'placeholder': data.get('campo_id_placeholder', 'Ingresa tu ID'),
        'required': True,
        'pattern': data.get('campo_id_pattern', ''),
        'pattern_msg': data.get('campo_id_pattern_msg', ''),
    }
    if data.get('dual_id') in (True, 'true', '1', 'on'):
        campos['campo_id2'] = {
            'enabled': True,
            'label': data.get('campo_id2_label', 'Zone ID'),
            'placeholder': data.get('campo_id2_placeholder', 'Ingresa tu Zone ID'),
            'required': True,
            'pattern': data.get('campo_id2_pattern', ''),
            'pattern_msg': data.get('campo_id2_pattern_msg', ''),
        }
    if data.get('servidor_enabled') in (True, 'true', '1', 'on'):
        opciones_raw = data.get('servidor_opciones', '')
        opciones = [o.strip() for o in opciones_raw.split(',') if o.strip()] if isinstance(opciones_raw, str) else opciones_raw
        campos['servidor'] = {
            'enabled': True,
            'label': data.get('servidor_label', 'Servidor'),
            'opciones': opciones,
        }

    conn = _get_conn()
    conn.execute('''
        UPDATE juegos_dinamicos SET nombre=?, gamepoint_product_id=?, modo=?, color_tema=?, icono=?, activo=?,
        campos_config=?, descripcion=?, ganancia_default=?, fecha_actualizacion=CURRENT_TIMESTAMP
        WHERE id=?
    ''', (nombre, int(product_id), modo, color, icono, activo, json.dumps(campos), descripcion, ganancia, game_id))
    conn.commit()
    conn.close()

    if request.is_json:
        return jsonify({'success': True})
    flash(f'Juego "{nombre}" actualizado.', 'success')
    return redirect('/admin/dynamic-games')


@bp.route('/admin/dynamic-games/<int:game_id>/toggle', methods=['POST'])
def admin_toggle_game(game_id):
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    game = get_dynamic_game_by_id(game_id)
    if not game:
        return jsonify({'error': 'Juego no encontrado'}), 404
    new_state = not game['activo']
    conn = _get_conn()
    conn.execute('UPDATE juegos_dinamicos SET activo=?, fecha_actualizacion=CURRENT_TIMESTAMP WHERE id=?', (new_state, game_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'activo': new_state})


@bp.route('/admin/dynamic-games/<int:game_id>/delete', methods=['POST'])
def admin_delete_game(game_id):
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    conn = _get_conn()
    conn.execute('DELETE FROM paquetes_dinamicos WHERE juego_id=?', (game_id,))
    conn.execute('DELETE FROM transacciones_dinamicas WHERE juego_id=?', (game_id,))
    conn.execute('DELETE FROM juegos_dinamicos WHERE id=?', (game_id,))
    conn.commit()
    conn.close()
    if request.is_json:
        return jsonify({'success': True})
    flash('Juego eliminado.', 'success')
    return redirect('/admin/dynamic-games')


# ---------------------------------------------------------------------------
# ADMIN: Paquetes de un juego dinámico
# ---------------------------------------------------------------------------

@bp.route('/admin/dynamic-games/<int:game_id>/packages', methods=['GET'])
def admin_get_packages(game_id):
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    pkgs = get_dynamic_packages(game_id)
    return jsonify({'packages': pkgs})


@bp.route('/admin/dynamic-games/<int:game_id>/packages/add', methods=['POST'])
def admin_add_package(game_id):
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    data = request.get_json() or request.form
    nombre = (data.get('nombre') or '').strip()
    precio = float(data.get('precio', 0))
    descripcion = (data.get('descripcion') or '').strip()
    gp_pkg_id = data.get('gamepoint_package_id')
    orden = int(data.get('orden', 0))

    if not nombre or precio <= 0:
        return jsonify({'error': 'Nombre y precio son obligatorios'}), 400

    conn = _get_conn()
    conn.execute('''
        INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, descripcion, gamepoint_package_id, activo, orden)
        VALUES (?, ?, ?, ?, ?, 1, ?)
    ''', (game_id, nombre, precio, descripcion, int(gp_pkg_id) if gp_pkg_id else None, orden))
    conn.commit()
    pkg_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]

    # Also register in precios_compra for profit tracking
    game = get_dynamic_game_by_id(game_id)
    if game and gp_pkg_id:
        juego_key = f'dyn_{game["slug"]}'
        myr_to_usd = float(os.environ.get('BLOODSTRIKE_MYR_TO_USD_RATE', '0.2357'))
        # Estimate cost from price - default profit
        costo_estimado = max(0, precio - game.get('ganancia_default', 0.10))
        try:
            conn.execute(
                'INSERT OR REPLACE INTO precios_compra (juego, paquete_id, precio_compra) VALUES (?, ?, ?)',
                (juego_key, pkg_id, costo_estimado)
            )
            conn.commit()
        except Exception:
            pass
    conn.close()
    return jsonify({'success': True, 'package_id': pkg_id})


@bp.route('/admin/dynamic-games/packages/<int:pkg_id>/update', methods=['POST'])
def admin_update_package(pkg_id):
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    data = request.get_json() or request.form
    pkg = get_dynamic_package_by_id(pkg_id)
    if not pkg:
        return jsonify({'error': 'Paquete no encontrado'}), 404

    nombre = (data.get('nombre') or pkg['nombre']).strip()
    precio = float(data.get('precio', pkg['precio']))
    descripcion = (data.get('descripcion') or pkg.get('descripcion', '')).strip()
    gp_pkg_id = data.get('gamepoint_package_id', pkg.get('gamepoint_package_id'))
    activo = data.get('activo')
    orden = int(data.get('orden', pkg.get('orden', 0)))
    if activo is not None:
        activo = activo in (True, 'true', '1', 'on', 1)
    else:
        activo = pkg['activo']

    conn = _get_conn()
    conn.execute('''
        UPDATE paquetes_dinamicos SET nombre=?, precio=?, descripcion=?, gamepoint_package_id=?, activo=?, orden=?,
        fecha_actualizacion=CURRENT_TIMESTAMP WHERE id=?
    ''', (nombre, precio, descripcion, int(gp_pkg_id) if gp_pkg_id else None, activo, orden, pkg_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@bp.route('/admin/dynamic-games/packages/<int:pkg_id>/delete', methods=['POST'])
def admin_delete_package(pkg_id):
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    conn = _get_conn()
    conn.execute('DELETE FROM paquetes_dinamicos WHERE id=?', (pkg_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ---------------------------------------------------------------------------
# ADMIN: Auto-import packages from GamePoint catalog
# ---------------------------------------------------------------------------

@bp.route('/admin/dynamic-games/<int:game_id>/auto-import-packages', methods=['POST'])
def admin_auto_import_packages(game_id):
    """Fetch GP catalog and auto-create local packages (name + gp_id mapped, price $0)."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    game = get_dynamic_game_by_id(game_id)
    if not game:
        return jsonify({'error': 'Juego no encontrado'}), 404

    get_token, gp_post, *_ = _gp_helpers()
    gc_token, gc_err = get_token()
    if not gc_token:
        return jsonify({'error': (gc_err or {}).get('message', 'No se pudo obtener token')}), 500

    _, detail = gp_post('product/detail', {'token': gc_token, 'productid': game['gamepoint_product_id']})
    if (detail or {}).get('code') != 200:
        return jsonify({'error': (detail or {}).get('message', 'Error obteniendo catálogo')}), 500

    gp_packages = (detail or {}).get('package', [])
    if not gp_packages:
        return jsonify({'error': 'No se encontraron paquetes en GamePoint'}), 404

    conn = _get_conn()
    # Get existing GP IDs already mapped locally
    existing_gp_ids = set()
    for row in conn.execute('SELECT gamepoint_package_id FROM paquetes_dinamicos WHERE juego_id = ? AND gamepoint_package_id IS NOT NULL', (game_id,)):
        existing_gp_ids.add(int(row['gamepoint_package_id']))

    created = 0
    skipped = 0
    for idx, gp_pkg in enumerate(gp_packages):
        gp_id = int(gp_pkg['id'])
        if gp_id in existing_gp_ids:
            skipped += 1
            continue
        gp_name = gp_pkg.get('name', f'Paquete {gp_id}')
        # Clean HTML from name
        clean_name = re.sub(r'<[^>]+>', ' ', gp_name).strip()
        clean_name = ' '.join(clean_name.split())
        conn.execute('''
            INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, descripcion, gamepoint_package_id, activo, orden)
            VALUES (?, ?, 0, '', ?, 0, ?)
        ''', (game_id, clean_name, gp_id, idx))
        created += 1

    conn.commit()
    conn.close()

    return jsonify({
        'success': True,
        'created': created,
        'skipped': skipped,
        'total_gp': len(gp_packages),
    })


# ---------------------------------------------------------------------------
# ADMIN: Fetch GamePoint catalog for a product (to help mapping)
# ---------------------------------------------------------------------------

@bp.route('/admin/dynamic-games/<int:game_id>/gp-catalog', methods=['GET'])
def admin_gp_catalog(game_id):
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    game = get_dynamic_game_by_id(game_id)
    if not game:
        return jsonify({'error': 'Juego no encontrado'}), 404

    get_token, gp_post, *_ = _gp_helpers()
    gc_token, gc_err = get_token()
    if not gc_token:
        return jsonify({'error': (gc_err or {}).get('message', 'No se pudo obtener token')}), 500

    _, detail = gp_post('product/detail', {'token': gc_token, 'productid': game['gamepoint_product_id']})
    if (detail or {}).get('code') != 200:
        return jsonify({'error': (detail or {}).get('message', 'Error obteniendo catálogo')}), 500

    packages = (detail or {}).get('package', [])
    return jsonify({'success': True, 'packages': packages})


# ---------------------------------------------------------------------------
# ADMIN: Mapping for GameClub tab (same pattern as Blood Strike)
# ---------------------------------------------------------------------------

@bp.route('/admin/dynamic-games/<int:game_id>/gamepoint_packages')
def admin_dyn_gamepoint_packages(game_id):
    """Returns GP catalog + local packages for mapping UI (like Blood Strike)."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    game = get_dynamic_game_by_id(game_id)
    if not game:
        return jsonify({'error': 'Juego no encontrado'}), 404

    get_token, gp_post, *_ = _gp_helpers()
    gc_token, gc_err = get_token()
    if not gc_token:
        return jsonify({'error': (gc_err or {}).get('message', 'No se pudo obtener token')}), 500

    _, detail = gp_post('product/detail', {'token': gc_token, 'productid': game['gamepoint_product_id']})
    if (detail or {}).get('code') != 200:
        return jsonify({'error': (detail or {}).get('message', 'Error')}), 500

    conn = _get_conn()
    local_packages = conn.execute(
        'SELECT id, nombre, precio, gamepoint_package_id FROM paquetes_dinamicos WHERE juego_id = ? ORDER BY orden, id',
        (game_id,)
    ).fetchall()
    conn.close()

    return jsonify({
        'gamepoint_packages': (detail or {}).get('package', []),
        'local_packages': [dict(lp) for lp in local_packages],
        'product_id': game['gamepoint_product_id'],
        'game_name': game['nombre'],
    })


@bp.route('/admin/dynamic-games/<int:game_id>/set_gamepoint_id', methods=['POST'])
def admin_dyn_set_gamepoint_id(game_id):
    """Assign a gamepoint_package_id to a local dynamic package."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    data = request.get_json() or request.form
    local_id = data.get('local_id')
    gp_id = data.get('gamepoint_package_id')
    if not local_id:
        return jsonify({'error': 'Falta local_id'}), 400

    conn = _get_conn()
    pkg = conn.execute('SELECT id, nombre FROM paquetes_dinamicos WHERE id = ? AND juego_id = ?', (local_id, game_id)).fetchone()
    if not pkg:
        conn.close()
        return jsonify({'error': 'Paquete no encontrado'}), 404

    gp_val = int(gp_id) if gp_id and str(gp_id).strip() else None
    conn.execute('UPDATE paquetes_dinamicos SET gamepoint_package_id = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ?',
                 (gp_val, local_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'local_id': int(local_id), 'nombre': pkg['nombre'], 'gamepoint_package_id': gp_val})


# ---------------------------------------------------------------------------
# ADMIN: Sync prices for a specific dynamic game
# ---------------------------------------------------------------------------

def sync_dynamic_game_prices(game_id):
    """Sync prices for one dynamic game. Returns dict result."""
    game = get_dynamic_game_by_id(game_id)
    if not game:
        return {'error': 'Juego no encontrado'}

    myr_to_usd = float(os.environ.get('BLOODSTRIKE_MYR_TO_USD_RATE', '0.2357'))
    default_profit = game.get('ganancia_default', 0.10)
    juego_key = f'dyn_{game["slug"]}'

    get_token, gp_post, *_ = _gp_helpers()
    gc_token, gc_err = get_token()
    if not gc_token:
        return {'error': (gc_err or {}).get('message', 'No se pudo obtener token')}

    _, detail = gp_post('product/detail', {'token': gc_token, 'productid': game['gamepoint_product_id']})
    if (detail or {}).get('code') != 200:
        return {'error': (detail or {}).get('message', 'Error obteniendo catálogo')}

    gp_packages = (detail or {}).get('package', [])
    if not gp_packages:
        return {'error': 'No se encontraron paquetes en GamePoint'}

    conn = _get_conn()
    local_pkgs = conn.execute('SELECT * FROM paquetes_dinamicos WHERE juego_id = ?', (game_id,)).fetchall()

    # Load current costs
    local_costs = {}
    for row in conn.execute("SELECT paquete_id, precio_compra FROM precios_compra WHERE juego = ?", (juego_key,)).fetchall():
        local_costs[int(row['paquete_id'])] = float(row['precio_compra'])

    # Map gp_package_id -> local package
    gp_to_local = {}
    for lp in local_pkgs:
        if lp['gamepoint_package_id']:
            gp_to_local[int(lp['gamepoint_package_id'])] = dict(lp)

    report = []
    updated = 0

    for gp_pkg in gp_packages:
        gp_id = int(gp_pkg['id'])
        gp_name = gp_pkg.get('name', '')
        gp_price_myr = float(gp_pkg.get('price', 0))
        nuevo_costo = round(gp_price_myr * myr_to_usd, 4)

        local = gp_to_local.get(gp_id)
        if local:
            costo_actual = local_costs.get(local['id'], 0)
            if costo_actual > 0:
                ganancia = round(local['precio'] - costo_actual, 4)
            else:
                ganancia = default_profit
            nuevo_precio = round(nuevo_costo + ganancia, 2)

            entry = {
                'gp_id': gp_id, 'gp_name': gp_name, 'gp_myr': gp_price_myr,
                'costo_anterior': round(costo_actual, 4), 'costo_nuevo': round(nuevo_costo, 4),
                'ganancia': round(ganancia, 4), 'precio_nuevo': nuevo_precio,
                'local_id': local['id'], 'local_nombre': local['nombre'],
                'precio_anterior': local['precio'],
                'cambio': round(nuevo_precio - local['precio'], 4),
            }
            conn.execute('UPDATE paquetes_dinamicos SET precio=?, fecha_actualizacion=CURRENT_TIMESTAMP WHERE id=?',
                         (nuevo_precio, local['id']))
            conn.execute('INSERT OR REPLACE INTO precios_compra (juego, paquete_id, precio_compra) VALUES (?, ?, ?)',
                         (juego_key, local['id'], nuevo_costo))
            updated += 1
        else:
            nuevo_precio = round(nuevo_costo + default_profit, 2)
            entry = {
                'gp_id': gp_id, 'gp_name': gp_name, 'gp_myr': gp_price_myr,
                'costo_nuevo': round(nuevo_costo, 4), 'ganancia': default_profit,
                'precio_nuevo': nuevo_precio, 'local_id': None,
                'nota': 'Sin mapeo local',
            }
        report.append(entry)

    conn.commit()
    conn.close()

    return {
        'success': True,
        'game': game['nombre'],
        'packages_updated': updated,
        'total_gp': len(gp_packages),
        'report': report,
    }


@bp.route('/admin/dynamic-games/<int:game_id>/sync-prices', methods=['POST'])
def admin_sync_game_prices(game_id):
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    result = sync_dynamic_game_prices(game_id)
    if result.get('error'):
        return jsonify(result), 500
    return jsonify(result)


def sync_all_dynamic_games_prices():
    """Sync prices for ALL active dynamic games. Called by background thread."""
    games = get_all_dynamic_games(only_active=True)
    results = []
    for g in games:
        try:
            r = sync_dynamic_game_prices(g['id'])
            results.append({'game': g['nombre'], 'result': r})
        except Exception as e:
            results.append({'game': g['nombre'], 'error': str(e)})
    return results


# ---------------------------------------------------------------------------
# USER: Dynamic game page
# ---------------------------------------------------------------------------

@bp.route('/juego/d/<slug>')
def dynamic_game_page(slug):
    if 'usuario' not in session:
        return redirect('/auth')

    game = get_dynamic_game_by_slug(slug)
    if not game:
        flash('Juego no encontrado.', 'error')
        return redirect('/')

    is_admin = session.get('is_admin', False)
    if not is_admin and not game['activo']:
        flash('Este juego está desactivado temporalmente.', 'error')
        return redirect('/')

    # Refresh balance from DB
    user_id = session.get('user_db_id')
    if user_id:
        conn = _get_conn()
        user = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            session['saldo'] = user['saldo']
        conn.close()

    packages = get_dynamic_packages(game['id'], only_active=not is_admin)
    campos = parse_campos_config(game)

    # Check for successful purchase
    compra_exitosa = False
    compra_data = {}
    if request.args.get('compra') == 'exitosa' and f'compra_dyn_{slug}_exitosa' in session:
        compra_exitosa = True
        compra_data = session.pop(f'compra_dyn_{slug}_exitosa')

    # Lazy import to avoid circular dependency
    from app import get_games_active
    games_active = get_games_active()
    dynamic_games_menu = get_all_dynamic_games(only_active=True)

    return render_template('juego_dinamico.html',
                           game=game,
                           packages=packages,
                           campos=campos,
                           user_id=session.get('id', '00000'),
                           balance=session.get('saldo', 0),
                           is_admin=is_admin,
                           compra_exitosa=compra_exitosa,
                           games_active=games_active,
                           dynamic_games_menu=dynamic_games_menu,
                           **compra_data)


# ---------------------------------------------------------------------------
# USER: Purchase flow
# ---------------------------------------------------------------------------

@bp.route('/validar/dinamico/<slug>', methods=['POST'])
def validar_dinamico(slug):
    if 'usuario' not in session:
        return redirect('/auth')

    game = get_dynamic_game_by_slug(slug)
    if not game:
        flash('Juego no encontrado.', 'error')
        return redirect('/')

    is_admin = session.get('is_admin', False)
    if not is_admin and not game['activo']:
        flash('Este juego está desactivado temporalmente.', 'error')
        return redirect('/')

    campos = parse_campos_config(game)
    redirect_url = f'/juego/d/{slug}'

    # Collect form fields
    package_id = request.form.get('monto')
    player_id = request.form.get('player_id', '').strip()
    player_id2 = request.form.get('player_id2', '').strip()
    servidor = request.form.get('servidor', '').strip()

    if not package_id:
        flash('Selecciona un paquete.', 'error')
        return redirect(redirect_url)

    # Validate required fields based on game mode
    if game['modo'] == 'id':
        if not player_id:
            flash('Ingresa tu ID de jugador.', 'error')
            return redirect(redirect_url)
        # Validate patterns
        campo_id_cfg = campos.get('campo_id', {})
        pattern = campo_id_cfg.get('pattern', '')
        if pattern:
            if not re.match(pattern, player_id):
                flash(campo_id_cfg.get('pattern_msg', 'Formato de ID inválido.'), 'error')
                return redirect(redirect_url)
        # Dual ID
        campo_id2_cfg = campos.get('campo_id2', {})
        if campo_id2_cfg.get('enabled') and not player_id2:
            flash(f'Ingresa tu {campo_id2_cfg.get("label", "Zone ID")}.', 'error')
            return redirect(redirect_url)
        if campo_id2_cfg.get('enabled') and campo_id2_cfg.get('pattern'):
            if not re.match(campo_id2_cfg['pattern'], player_id2):
                flash(campo_id2_cfg.get('pattern_msg', 'Formato inválido.'), 'error')
                return redirect(redirect_url)
        # Server
        srv_cfg = campos.get('servidor', {})
        if srv_cfg.get('enabled') and not servidor:
            flash(f'Selecciona un {srv_cfg.get("label", "servidor")}.', 'error')
            return redirect(redirect_url)

    package_id = int(package_id)
    user_id = session.get('user_db_id')
    pkg = get_dynamic_package_by_id(package_id)
    if not pkg or pkg['juego_id'] != game['id']:
        flash('Paquete no encontrado.', 'error')
        return redirect(redirect_url)

    if not is_admin and not pkg['activo']:
        flash('Paquete no disponible.', 'error')
        return redirect(redirect_url)

    precio = pkg['precio']
    gp_package_id = pkg.get('gamepoint_package_id')
    if not gp_package_id:
        flash('Este paquete no tiene configurado el ID de GamePoint.', 'error')
        return redirect(redirect_url)

    # Check balance
    if not is_admin:
        conn = _get_conn()
        row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        saldo_actual = row['saldo'] if row else 0
        session['saldo'] = saldo_actual
        if saldo_actual < precio:
            flash(f'Saldo insuficiente. Necesitas ${precio:.2f} pero tienes ${saldo_actual:.2f}', 'error')
            return redirect(redirect_url)

    # === PURCHASE VIA GAMEPOINT ===
    _start = time_module.time()
    get_token, gp_post, order_validate, order_create, order_inquiry = _gp_helpers()

    try:
        # 1. Deduct balance atomically
        if not is_admin:
            conn = _get_conn()
            cursor = conn.execute('UPDATE usuarios SET saldo = saldo - ? WHERE id = ? AND saldo >= ?', (precio, user_id, precio))
            if cursor.rowcount == 0:
                conn.close()
                flash('Saldo insuficiente al momento de procesar.', 'error')
                return redirect(redirect_url)
            conn.commit()
            new_saldo = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
            conn.close()
            session['saldo'] = new_saldo['saldo'] if new_saldo else 0

        # 2. Get GamePoint token
        gc_token, gc_err = get_token()
        if not gc_token:
            _refund(user_id, precio, is_admin)
            flash(f'Error de conexión con proveedor. Tu saldo ha sido devuelto.', 'error')
            return redirect(redirect_url)

        # 3. Validate order — build input fields
        input_fields = {'input1': str(player_id)}
        if player_id2:
            input_fields['input2'] = str(player_id2)
        if servidor:
            input_fields['input3'] = str(servidor)

        validate_data = order_validate(gc_token, game['gamepoint_product_id'], input_fields)
        validate_code = (validate_data or {}).get('code')

        if validate_code != 200 or not (validate_data or {}).get('validation_token'):
            _refund(user_id, precio, is_admin)
            err_msg = (validate_data or {}).get('message', 'Error validando orden')
            logger.error(f"[DynGame:{game['slug']}] validate failed: code={validate_code} msg={err_msg}")
            flash(f'Error al validar: {err_msg}. Tu saldo ha sido devuelto.', 'error')
            return redirect(redirect_url)

        validation_token = validate_data['validation_token']
        # validate NO devuelve ingamename según docs — solo code, message, validation_token
        ingame_name = ''
        logger.info(f"[DynGame:{game['slug']}] validate OK | player={player_id}")

        # 4. Create order
        merchant_code = f"DG{game['id']}-" + secrets.token_hex(6).upper()
        create_data = order_create(gc_token, validation_token, gp_package_id, merchant_code)
        create_code = (create_data or {}).get('code')
        reference_no = (create_data or {}).get('referenceno', '')

        # 4.1 order/inquiry — extraer ingamename (docs: campo opcional, aparece cuando el pedido está procesado)
        item_name = ''
        try:
            if reference_no:
                # Retry inquiry con delay para dar tiempo a que el pedido se procese
                for _attempt in range(3):
                    if _attempt > 0:
                        time_module.sleep(1.5)
                    inq_data = order_inquiry(gc_token, reference_no)
                    ingame_name = (inq_data or {}).get('ingamename') or ''
                    item_name = (inq_data or {}).get('item') or ''
                    logger.info(f"[DynGame:{game['slug']}] inquiry attempt {_attempt+1}: ingamename='{ingame_name}' | item='{item_name}'")
                    if ingame_name:
                        break
                # Limpiar HTML del item (ej: "Blood Strike<br />300 + 20 Gold")
                if item_name:
                    item_name = re.sub(r'<[^>]+>', ' ', item_name).strip()
                    item_name = ' '.join(item_name.split())
        except Exception as e:
            logger.error(f"[DynGame:{game['slug']}] inquiry FAILED: {e}")

        _duration = round(time_module.time() - _start, 1)

        if create_code in (100, 101):
            # === SUCCESS ===
            numero_control = f"DG-{secrets.token_hex(4).upper()}"
            transaccion_id = merchant_code

            conn = _get_conn()
            conn.execute('''
                INSERT INTO transacciones_dinamicas
                (juego_id, usuario_id, player_id, player_id2, servidor, paquete_id,
                 numero_control, transaccion_id, monto, estado, gamepoint_referenceno, ingame_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game['id'], user_id, player_id, player_id2 or None, servidor or None,
                  package_id, numero_control, transaccion_id, precio, 'aprobado', reference_no, ingame_name))

            # Also insert into general transacciones table for unified history
            if ingame_name:
                pin_info = f"ID: {player_id} - Jugador: {ingame_name} - Ref: {reference_no}"
            else:
                pin_info = f"ID: {player_id} - Ref: {reference_no}"
            if player_id2:
                pin_info = f"ID: {player_id} / {player_id2} - " + pin_info.split(' - ', 1)[1]

            paquete_display = f"{game['nombre']} - {pkg['nombre']}"
            conn.execute('''
                INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto, duracion_segundos)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, numero_control, pin_info, transaccion_id, paquete_display, -precio, _duration))

            # Record in historial_compras
            _saldo_row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
            _saldo = _saldo_row['saldo'] if _saldo_row else 0
            conn.execute('''
                INSERT INTO historial_compras (usuario_id, monto, paquete_nombre, pin, tipo_evento, duracion_segundos, saldo_antes, saldo_despues)
                VALUES (?, ?, ?, ?, 'compra', ?, ?, ?)
            ''', (user_id, precio, paquete_display, pin_info, _duration, _saldo + precio, _saldo))

            # Record profit
            try:
                juego_key = f'dyn_{game["slug"]}'
                admin_ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
                admin_ids = [int(x.strip()) for x in admin_ids_env.split(',') if x.strip().isdigit()]
                is_admin_target = user_id in admin_ids
                costo_row = conn.execute('SELECT precio_compra FROM precios_compra WHERE juego=? AND paquete_id=?',
                                         (juego_key, package_id)).fetchone()
                costo_unit = costo_row['precio_compra'] if costo_row else 0
                profit_unit = round(precio - costo_unit, 4)
                conn.execute('''
                    INSERT INTO profit_ledger (usuario_id, juego, paquete_id, cantidad, precio_venta_unit, costo_unit, profit_unit, profit_total, transaccion_id)
                    VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                ''', (user_id, juego_key, package_id, precio, costo_unit, profit_unit, profit_unit, transaccion_id))
            except Exception:
                pass

            # Monthly spending
            if not is_admin:
                try:
                    from update_monthly_spending import update_monthly_spending
                    update_monthly_spending(conn, user_id, precio)
                except Exception:
                    pass

            conn.commit()
            conn.close()

            # Weekly sale
            if not is_admin:
                try:
                    from app import register_weekly_sale
                    register_weekly_sale(f'dyn_{game["slug"]}', package_id, pkg['nombre'], precio, 1)
                except Exception:
                    pass

            estado_txt = 'completado' if create_code == 100 else 'procesando'
            logger.info(f"[DynGame:{game['slug']}] storing in session: player_name='{ingame_name}' | player_id={player_id} | ref={reference_no}")
            session[f'compra_dyn_{slug}_exitosa'] = {
                'paquete_nombre': pkg['nombre'],
                'monto_compra': precio,
                'numero_control': numero_control,
                'transaccion_id': transaccion_id,
                'player_id': player_id,
                'player_id2': player_id2,
                'servidor': servidor,
                'player_name': ingame_name,
                'estado': estado_txt,
                'gamepoint_ref': reference_no,
            }
            return redirect(f'/juego/d/{slug}?compra=exitosa')

        else:
            # === FAILURE ===
            err_msg = (create_data or {}).get('message', 'Error creando orden')
            logger.error(f"[DynGame:{game['slug']}] create failed: code={create_code} msg={err_msg} | user={user_id}")

            conn = _get_conn()
            conn.execute('''
                INSERT INTO transacciones_dinamicas
                (juego_id, usuario_id, player_id, player_id2, servidor, paquete_id,
                 numero_control, transaccion_id, monto, estado, gamepoint_referenceno, notas)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game['id'], user_id, player_id, player_id2 or None, servidor or None,
                  package_id, f"DG-{secrets.token_hex(4).upper()}", merchant_code, precio,
                  'rechazado', reference_no, err_msg))
            conn.commit()
            conn.close()

            _refund(user_id, precio, is_admin)
            flash(f'La recarga falló: {err_msg}. Tu saldo ha sido devuelto.', 'error')
            return redirect(redirect_url)

    except Exception as e:
        logger.error(f"[DynGame:{game['slug']}] Error general: {str(e)}")
        _refund(user_id, precio, is_admin)
        flash('Error al procesar la compra. Tu saldo ha sido devuelto.', 'error')
        return redirect(redirect_url)


def _refund(user_id, precio, is_admin):
    """Refund balance to user."""
    if is_admin:
        return
    try:
        conn = _get_conn()
        conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, user_id))
        conn.commit()
        conn.close()
        session['saldo'] = session.get('saldo', 0) + precio
        logger.info(f"[DynGame] Saldo ${precio} reembolsado al usuario {user_id}")
    except Exception as e:
        logger.error(f"[DynGame] Error reembolsando: {e}")

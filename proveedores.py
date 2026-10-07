"""
Apartado "Proveedores" del panel admin (Blueprint Flask).

Proveedores de recargas del Mapeo:
  - Inefable     (REVENDEDORES_BASE_URL / REVENDEDORES_API_KEY), formato /api/v1/products, /recharge, /order-status
  - Central One  (CENTRALONE_API_KEY, opcional CENTRALONE_BASE_URL), ver central_one.py

Las llaves viven solo en variables de entorno del servidor; el panel nunca las muestra.
Respaldo automático (configuracion_redeemer 'proveedores_respaldo' = '1'/'0'): si el proveedor principal de
un paquete no entrega nada, se intenta con las recargas de respaldo definidas en su Mapeo.
"""
import logging
import os

from flask import Blueprint, flash, jsonify, redirect, render_template, request, session

import central_one
from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('proveedores', __name__)

CLAVE_RESPALDO = 'proveedores_respaldo'


def _solo_admin_json():
    if not session.get('is_admin'):
        return jsonify(ok=False, error='Acceso denegado'), 403
    if request.method == 'POST' and not request.is_json:
        return jsonify(ok=False, error='Se espera JSON'), 415
    return None


def _contar(conn, central):
    cond = "remote_product_id LIKE 'co:%'" if central else "remote_product_id NOT LIKE 'co:%'"
    row = conn.execute(f'SELECT COUNT(*) AS n, MAX(updated_at) AS t FROM rev_catalog_items WHERE active = TRUE AND {cond}').fetchone()
    return int(row['n'] or 0), str(row['t'] or '') or None


def _mapeados(conn, central):
    cond = "remote_product_id LIKE 'co:%'" if central else "remote_product_id NOT LIKE 'co:%'"
    row = conn.execute(f'SELECT COUNT(DISTINCT paquete_id) AS n FROM rev_item_mapping_steps WHERE {cond}').fetchone()
    return int(row['n'] or 0)


def respaldo_activo():
    from dynamic_games import respaldo_activo as _r
    return _r()


@bp.route('/admin/proveedores')
def admin_proveedores():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    return render_template('admin_proveedores.html')


@bp.route('/admin/proveedores/datos')
def admin_proveedores_datos():
    err = _solo_admin_json()
    if err:
        return err
    conn = get_db_connection()
    try:
        ine_items, ine_t = _contar(conn, False)
        co_items, co_t = _contar(conn, True)
        ine_map, co_map = _mapeados(conn, False), _mapeados(conn, True)
    finally:
        conn.close()

    ine_conf = bool(os.environ.get('REVENDEDORES_BASE_URL', '').strip() and os.environ.get('REVENDEDORES_API_KEY', '').strip())
    ine = {'clave': 'inefable', 'nombre': 'Inefable', 'configurado': ine_conf, 'variable': 'REVENDEDORES_API_KEY',
           'productos': ine_items, 'actualizado': ine_t, 'paquetes_mapeados': ine_map, 'saldo': None}
    if ine_conf and request.args.get('saldos') == '1':
        try:
            from app import get_revendedores_balance
            ine['saldo'] = {'disponible': f'{float(get_revendedores_balance()):.2f}', 'moneda': 'USD'}
        except Exception as e:
            ine['saldo'] = {'error': str(e)[:120]}

    co = {'clave': 'centralone', 'nombre': central_one.NOMBRE, 'configurado': central_one.configurado(),
          'variable': 'CENTRALONE_API_KEY', 'productos': co_items, 'actualizado': co_t,
          'paquetes_mapeados': co_map, 'saldo': None}
    if co['configurado'] and request.args.get('saldos') == '1':
        co['saldo'] = central_one.saldo()

    return jsonify(ok=True, proveedores=[ine, co], respaldo=respaldo_activo())


@bp.route('/admin/proveedores/centralone/sync', methods=['POST'])
def admin_centralone_sync():
    err = _solo_admin_json()
    if err:
        return err
    if not central_one.configurado():
        return jsonify(ok=False, error='Falta la variable CENTRALONE_API_KEY en el servidor'), 400
    try:
        r = central_one.sincronizar_catalogo(get_db_connection)
    except Exception as e:
        logger.error(f'[Central One] Error sincronizando catálogo: {e}')
        return jsonify(ok=False, error=str(e)[:200]), 502
    return jsonify(ok=True, **r)


@bp.route('/admin/proveedores/config', methods=['POST'])
def admin_proveedores_config():
    err = _solo_admin_json()
    if err:
        return err
    activo = bool((request.get_json(silent=True) or {}).get('respaldo'))
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion) VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion",
            (CLAVE_RESPALDO, '1' if activo else '0'))
        conn.commit()
    finally:
        conn.close()
    return jsonify(ok=True, respaldo=activo)

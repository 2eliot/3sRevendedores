"""
Referencias — el Antiduplic de cada usuario (Blueprint Flask).

API (X-API-Key de una clave de tipo «Referencias» del usuario; solo sus propios datos):
  GET  /api/v1/referencias/resumen                      ver_pagos
  GET  /api/v1/referencias/pagos                        ver_pagos
  GET  /api/v1/referencias/pagos/<id>                   ver_pagos
  GET  /api/v1/referencias/revisiones                   ver_pagos
  GET  /api/v1/referencias/cuentas                      ver_pagos
  POST /api/v1/referencias/asignar                      gestionar_pagos
  POST /api/v1/referencias/revisiones/<id>/resolver     gestionar_pagos
  POST /api/v1/referencias/revisiones/<id>/descartar    gestionar_pagos
(POST /api/verificar-pago y POST /api/pagos-banco viven en antiduplic.py.)

Un ID que no es del usuario responde 404 (nunca 403), para no confirmar que existe.
"""
import logging

from flask import Blueprint, jsonify, request

import antiduplic as ad
from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('referencias', __name__)

POR_PAGINA_MAX = 200


def _err(status, mensaje):
    return jsonify(ok=False, mensaje=mensaje), status


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _cuenta_api(permiso, tipo_limite='lectura'):
    """(cuenta, None) o (None, respuesta de error)."""
    from api_panel import cuenta_por_clave, cuenta_puede, limite_ok
    cuenta = cuenta_por_clave(request.headers.get('X-API-Key'))
    if not cuenta:
        return None, _err(401, 'Falta la clave de API o no es válida (cabecera X-API-Key)')
    if not cuenta_puede(cuenta, permiso):
        return None, _err(403, 'Esta clave no tiene el permiso necesario')
    if not limite_ok(cuenta['id'], tipo_limite):
        return None, _err(429, 'Demasiadas peticiones; espera un minuto')
    ad.init_tablas()
    return cuenta, None


def _firma(cuenta):
    return f"api:{cuenta['nombre']} (#{cuenta['id']})"


@bp.route('/api/v1/referencias/resumen')
def api_resumen():
    cuenta, err = _cuenta_api('ver_pagos')
    if err:
        return err
    fecha = request.args.get('fecha')
    if fecha and not ad.parse_fecha(fecha):
        return _err(400, 'Fecha no válida (usa AAAA-MM-DD)')
    conn = get_db_connection()
    try:
        return jsonify(ok=True, resumen=ad.resumen(conn, cuenta['usuario_id'], ad.parse_fecha(fecha) if fecha else None))
    finally:
        conn.close()


@bp.route('/api/v1/referencias/pagos')
def api_pagos():
    cuenta, err = _cuenta_api('ver_pagos')
    if err:
        return err
    a = request.args
    fechas = {}
    for k in ('fecha', 'desde', 'hasta'):
        if a.get(k):
            fechas[k] = ad.parse_fecha(a.get(k))
            if not fechas[k]:
                return _err(400, f'«{k}» no es una fecha válida (usa AAAA-MM-DD)')
    try:
        pagina = int(a.get('pagina') or 1)
        por_pagina = int(a.get('por_pagina') or 50)
        cuenta_id = int(a['cuenta_id']) if a.get('cuenta_id') else None
    except ValueError:
        return _err(400, 'pagina, por_pagina y cuenta_id deben ser números')
    if pagina < 1 or por_pagina < 1 or por_pagina > POR_PAGINA_MAX:
        return _err(400, f'por_pagina debe estar entre 1 y {POR_PAGINA_MAX}')
    estado = a.get('estado') or ''
    if estado and estado not in ('disponible', 'usado', 'revision'):
        return _err(400, 'estado debe ser disponible, usado o revision')
    conn = get_db_connection()
    try:
        pagos, total = ad.listar_pagos(conn, cuenta['usuario_id'], estado=estado, q=a.get('q') or '',
                                       cuenta_id=cuenta_id, pagina=pagina, por_pagina=por_pagina, **fechas)
        return jsonify(ok=True, pagos=pagos, total=total, pagina=pagina, por_pagina=por_pagina)
    finally:
        conn.close()


@bp.route('/api/v1/referencias/pagos/<int:pago_id>')
def api_pago(pago_id):
    cuenta, err = _cuenta_api('ver_pagos')
    if err:
        return err
    conn = get_db_connection()
    try:
        pago = ad.detalle_pago(conn, cuenta['usuario_id'], pago_id)
    finally:
        conn.close()
    return jsonify(ok=True, pago=pago) if pago else _err(404, 'Pago no encontrado')


@bp.route('/api/v1/referencias/revisiones')
def api_revisiones():
    cuenta, err = _cuenta_api('ver_pagos')
    if err:
        return err
    estado = request.args.get('estado') or 'pendiente'
    if estado not in ('pendiente', 'resuelta', 'descartada'):
        return _err(400, 'estado debe ser pendiente, resuelta o descartada')
    conn = get_db_connection()
    try:
        return jsonify(ok=True, revisiones=ad.listar_revisiones(conn, cuenta['usuario_id'], estado))
    finally:
        conn.close()


@bp.route('/api/v1/referencias/cuentas')
def api_cuentas():
    cuenta, err = _cuenta_api('ver_pagos')
    if err:
        return err
    conn = get_db_connection()
    try:
        cuentas = ad.listar_cuentas(conn, cuenta['usuario_id'])
    finally:
        conn.close()
    return jsonify(ok=True, cuentas=[{k: c[k] for k in ('id', 'banco', 'alias', 'ultimos_digitos', 'titular', 'activo', 'ultimo_envio')}
                                     for c in cuentas])


@bp.route('/api/v1/referencias/asignar', methods=['POST'])
def api_asignar():
    cuenta, err = _cuenta_api('gestionar_pagos', 'escritura')
    if err:
        return err
    data = request.get_json(silent=True) or {}
    if not str(data.get('orden_id') or '').strip():
        return _err(400, 'Falta orden_id')
    status, body = ad.asignar(cuenta['usuario_id'], data.get('pago_id'), orden_id=data.get('orden_id'), hecho_por=_firma(cuenta))
    return jsonify(body), status


@bp.route('/api/v1/referencias/revisiones/<int:revision_id>/resolver', methods=['POST'])
def api_resolver(revision_id):
    cuenta, err = _cuenta_api('gestionar_pagos', 'escritura')
    if err:
        return err
    data = request.get_json(silent=True) or {}
    status, body = ad.asignar(cuenta['usuario_id'], data.get('pago_id'), revision_id=revision_id, hecho_por=_firma(cuenta))
    return jsonify(body), status


@bp.route('/api/v1/referencias/revisiones/<int:revision_id>/descartar', methods=['POST'])
def api_descartar(revision_id):
    cuenta, err = _cuenta_api('gestionar_pagos', 'escritura')
    if err:
        return err
    status, body = ad.descartar(cuenta['usuario_id'], revision_id, hecho_por=_firma(cuenta))
    return jsonify(body), status

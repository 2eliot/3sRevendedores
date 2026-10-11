"""
Referencias — el Antiduplic de cada usuario (Blueprint Flask).

API (X-API-Key de una clave de tipo «Referencias» del usuario; solo sus propios datos y SOLO DEL BANCO DE LA CLAVE:
una clave del BDV solo ve el BDV y una de Bancamiga solo Bancamiga; un pago de otro banco responde 404):
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

Bancos: la pantalla tiene una pestaña por banco (Bancamiga y BDV) y cada una ve solo lo suyo (parámetro «banco»).
Las rutas /referencias/bdv/* gestionan la conexión del usuario con su BDV (ver bdv.py): nunca devuelven sus datos.
"""
import logging
import re

from flask import Blueprint, jsonify, redirect, render_template, request, session

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


def _banco(cuenta):
    """El banco lo decide la clave."""
    from api_panel import banco_clave
    return banco_clave(cuenta.get('banco'))


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
        return jsonify(ok=True, banco=_banco(cuenta),
                       resumen=ad.resumen(conn, cuenta['usuario_id'], ad.parse_fecha(fecha) if fecha else None, banco=_banco(cuenta)))
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
                                       cuenta_id=cuenta_id, pagina=pagina, por_pagina=por_pagina, banco=_banco(cuenta), **fechas)
        return jsonify(ok=True, banco=_banco(cuenta), pagos=pagos, total=total, pagina=pagina, por_pagina=por_pagina)
    finally:
        conn.close()


@bp.route('/api/v1/referencias/pagos/<int:pago_id>')
def api_pago(pago_id):
    cuenta, err = _cuenta_api('ver_pagos')
    if err:
        return err
    conn = get_db_connection()
    try:
        pago = ad.detalle_pago(conn, cuenta['usuario_id'], pago_id, banco=_banco(cuenta))
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
        return jsonify(ok=True, banco=_banco(cuenta), revisiones=ad.listar_revisiones(conn, cuenta['usuario_id'], estado,
                                                                                     banco=_banco(cuenta)))
    finally:
        conn.close()


@bp.route('/api/v1/referencias/cuentas')
def api_cuentas():
    cuenta, err = _cuenta_api('ver_pagos')
    if err:
        return err
    conn = get_db_connection()
    try:
        cuentas = ad.listar_cuentas(conn, cuenta['usuario_id'], banco=_banco(cuenta))
    finally:
        conn.close()
    campos = ('id', 'banco', 'alias', 'ultimos_digitos', 'titular', 'activo', 'ultimo_envio')
    salida = []
    for c in cuentas:
        d = {k: c[k] for k in campos}
        if 'lector' in c:   # BDV: estado del lector y si está conectada; nunca usuario, clave ni número completo
            d['conectada'] = c['conectada']
            d['lector'] = {k: c['lector'][k] for k in ('estado', 'mensaje', 'ultima_lectura', 'ultimo_pago', 'encendido')}
        salida.append(d)
    return jsonify(ok=True, banco=_banco(cuenta), cuentas=salida)


@bp.route('/api/v1/referencias/asignar', methods=['POST'])
def api_asignar():
    cuenta, err = _cuenta_api('gestionar_pagos', 'escritura')
    if err:
        return err
    data = request.get_json(silent=True) or {}
    if not str(data.get('orden_id') or '').strip():
        return _err(400, 'Falta orden_id')
    status, body = ad.asignar(cuenta['usuario_id'], data.get('pago_id'), orden_id=data.get('orden_id'), hecho_por=_firma(cuenta),
                              banco=_banco(cuenta))
    return jsonify(body), status


@bp.route('/api/v1/referencias/revisiones/<int:revision_id>/resolver', methods=['POST'])
def api_resolver(revision_id):
    cuenta, err = _cuenta_api('gestionar_pagos', 'escritura')
    if err:
        return err
    data = request.get_json(silent=True) or {}
    status, body = ad.asignar(cuenta['usuario_id'], data.get('pago_id'), revision_id=revision_id, hecho_por=_firma(cuenta),
                              banco=_banco(cuenta))
    return jsonify(body), status


@bp.route('/api/v1/referencias/revisiones/<int:revision_id>/descartar', methods=['POST'])
def api_descartar(revision_id):
    cuenta, err = _cuenta_api('gestionar_pagos', 'escritura')
    if err:
        return err
    status, body = ad.descartar(cuenta['usuario_id'], revision_id, hecho_por=_firma(cuenta), banco=_banco(cuenta))
    return jsonify(body), status


# ---------------------------------------------------------------------------
# Pantalla del usuario: /referencias (sesión; solo sus datos)
# ---------------------------------------------------------------------------

MAX_CLAVES_LECTOR = 10  # las claves de API tienen su límite en api_panel.MAX_CLAVES_USUARIO


def _uid():
    if 'usuario' not in session or not session.get('user_db_id'):
        return None
    return int(session['user_db_id'])


def _sesion_json(post=False):
    """(uid, None) o (None, respuesta). Los POST exigen JSON (protección contra envíos de otros sitios)."""
    uid = _uid()
    if uid is None:
        return None, _err(401, 'Tu sesión expiró. Vuelve a iniciar sesión.')
    if post and not request.is_json:
        return None, _err(415, 'Se espera JSON')
    ad.init_tablas()
    return uid, None


def _firma_usuario():
    return f"usuario:{session.get('usuario') or session.get('user_db_id')}"


def _banco_peticion():
    """Banco elegido en la pantalla: 'BDV' o 'Bancamiga' (parámetro o campo JSON «banco»)."""
    valor = request.args.get('banco')
    if valor is None and request.is_json:
        valor = (request.get_json(silent=True) or {}).get('banco')
    return ad.banco_de(valor)


@bp.route('/referencias')
def pagina_referencias():
    uid = _uid()
    if uid is None:
        return redirect('/auth')
    ad.init_tablas()
    return render_template('referencias.html', hoy=ad.hoy_local(), api_url=request.host_url.rstrip('/'),
                           user_id=session.get('id'), aviso_min=ad.AVISO_BOT_MIN)


@bp.route('/referencias/docs')
def referencias_docs():
    if _uid() is None:
        return redirect('/auth')
    return render_template('referencias_docs.html', api_url=request.host_url.rstrip('/'), admin=False,
                           token_banco=False, volver='/referencias')


@bp.route('/referencias/datos')
def referencias_datos():
    uid, err = _sesion_json()
    if err:
        return err
    a = request.args
    fecha = ad.parse_fecha(a.get('fecha')) if a.get('fecha') else None
    try:
        cuenta_id = int(a['cuenta_id']) if a.get('cuenta_id') else None
    except ValueError:
        cuenta_id = None
    banco = _banco_peticion()
    conn = get_db_connection()
    try:
        pagos, total = ad.listar_pagos(conn, uid, fecha=fecha, estado=a.get('estado') or '', q=a.get('q') or '',
                                       cuenta_id=cuenta_id, por_pagina=500, banco=banco)
        return jsonify(ok=True, banco=banco, pagos=pagos, total=total, resumen=ad.resumen(conn, uid, banco=banco),
                       revisiones=ad.listar_revisiones(conn, uid, banco=banco), ahora=ad.ahora_local())
    finally:
        conn.close()


@bp.route('/referencias/asignar', methods=['POST'])
def referencias_asignar():
    uid, err = _sesion_json(post=True)
    if err:
        return err
    data = request.get_json(silent=True) or {}
    if not data.get('revision_id') and not str(data.get('orden_id') or '').strip():
        return _err(400, 'Indica el ID de la orden')
    status, body = ad.asignar(uid, data.get('pago_id'), orden_id=data.get('orden_id'),
                              revision_id=data.get('revision_id'), hecho_por=_firma_usuario(), banco=_banco_peticion())
    return jsonify(body), status


@bp.route('/referencias/descartar', methods=['POST'])
def referencias_descartar():
    uid, err = _sesion_json(post=True)
    if err:
        return err
    status, body = ad.descartar(uid, (request.get_json(silent=True) or {}).get('revision_id'), hecho_por=_firma_usuario(),
                                banco=_banco_peticion())
    return jsonify(body), status


# ---------- Cuentas bancarias ----------

def _datos_cuenta(data):
    """(valores, error) a partir del JSON: banco, alias, ultimos_digitos (solo 4), titular."""
    banco = str(data.get('banco') or 'Bancamiga').strip()[:60]
    alias = str(data.get('alias') or '').strip()[:60]
    digitos = re.sub(r'[^0-9]', '', str(data.get('ultimos_digitos') or ''))
    titular = str(data.get('titular') or '').strip()[:80]
    if ad.banco_de(banco) == ad.BDV:
        return None, 'Las cuentas del BDV se conectan en la pestaña «BDV»'
    if not alias:
        return None, 'Ponle un nombre a la cuenta (por ejemplo «Bancamiga principal»)'
    if digitos and len(digitos) != 4:
        return None, 'Escribe solo los últimos 4 dígitos de la cuenta, nunca el número completo'
    return {'banco': banco or 'Bancamiga', 'alias': alias, 'ultimos_digitos': digitos, 'titular': titular}, None


@bp.route('/referencias/cuentas', methods=['GET', 'POST'])
def referencias_cuentas():
    uid, err = _sesion_json(post=request.method == 'POST')
    if err:
        return err
    conn = get_db_connection()
    try:
        if request.method == 'GET':
            return jsonify(ok=True, cuentas=ad.listar_cuentas(conn, uid, banco=ad.BANCAMIGA))
        vals, error = _datos_cuenta(request.get_json(silent=True) or {})
        if error:
            return _err(400, error)
        conn.execute('INSERT INTO cuentas_banco (usuario_id, banco, alias, ultimos_digitos, titular, activo, creado_en) '
                     'VALUES (?, ?, ?, ?, ?, TRUE, ?)',
                     (uid, vals['banco'], vals['alias'], vals['ultimos_digitos'], vals['titular'], ad.ahora_local()))
        conn.commit()
        return jsonify(ok=True, mensaje='Cuenta creada', cuentas=ad.listar_cuentas(conn, uid, banco=ad.BANCAMIGA))
    finally:
        conn.close()


@bp.route('/referencias/cuentas/<int:cuenta_id>', methods=['POST'])
def referencias_cuenta_editar(cuenta_id):
    uid, err = _sesion_json(post=True)
    if err:
        return err
    data = request.get_json(silent=True) or {}
    conn = get_db_connection()
    try:
        if not conn.execute("SELECT 1 FROM cuentas_banco WHERE id = ? AND usuario_id = ? AND banco <> 'BDV'",
                            (cuenta_id, uid)).fetchone():
            return _err(404, 'Cuenta no encontrada')  # las del BDV se gestionan en su pestaña
        if set(data) - {'activo'}:
            vals, error = _datos_cuenta(data)
            if error:
                return _err(400, error)
            conn.execute('UPDATE cuentas_banco SET banco = ?, alias = ?, ultimos_digitos = ?, titular = ? WHERE id = ? AND usuario_id = ?',
                         (vals['banco'], vals['alias'], vals['ultimos_digitos'], vals['titular'], cuenta_id, uid))
        if 'activo' in data:
            conn.execute('UPDATE cuentas_banco SET activo = ? WHERE id = ? AND usuario_id = ?', (bool(data['activo']), cuenta_id, uid))
        conn.commit()
        return jsonify(ok=True, mensaje='Cuenta guardada', cuentas=ad.listar_cuentas(conn, uid, banco=ad.BANCAMIGA))
    finally:
        conn.close()


# ---------- Claves (lector y API) ----------

def _claves(conn, uid, banco=ad.BANCAMIGA):
    from api_panel import claves_de_usuario
    if ad.banco_de(banco) == ad.BDV:   # el BDV no usa claves de lector: lo lee la web
        return {'lector': [], **claves_de_usuario(conn, uid, 'referencias', ad.BDV)}
    lector = [{'id': r['id'], 'nombre': r['nombre'], 'cuenta_id': r['cuenta_banco_id'], 'cuenta': r['alias'],
               'prefijo': r['prefijo_visible'], 'activo': bool(r['activo']), 'ultimo_uso': r['ultimo_uso'],
               'creado_en': r['creado_en']}
              for r in conn.execute('SELECT k.*, c.alias FROM claves_lector k LEFT JOIN cuentas_banco c ON c.id = k.cuenta_banco_id '
                                    'WHERE k.usuario_id = ? ORDER BY k.id', (uid,)).fetchall()]
    return {'lector': lector, **claves_de_usuario(conn, uid, 'referencias', 'Bancamiga')}


@bp.route('/referencias/claves')
def referencias_claves():
    uid, err = _sesion_json()
    if err:
        return err
    conn = get_db_connection()
    try:
        return jsonify(ok=True, **_claves(conn, uid, _banco_peticion()))
    finally:
        conn.close()


@bp.route('/referencias/claves-lector', methods=['POST'])
def referencias_lector_crear():
    uid, err = _sesion_json(post=True)
    if err:
        return err
    data = request.get_json(silent=True) or {}
    nombre = str(data.get('nombre') or '').strip()[:60] or 'Bot del banco'
    try:
        cuenta_id = int(data.get('cuenta_id'))
    except (TypeError, ValueError):
        return _err(400, 'Elige la cuenta bancaria de esta clave')
    conn = get_db_connection()
    try:
        if not conn.execute("SELECT 1 FROM cuentas_banco WHERE id = ? AND usuario_id = ? AND activo = TRUE AND banco <> 'BDV'",
                            (cuenta_id, uid)).fetchone():
            return _err(404, 'Cuenta no encontrada o desactivada')  # el BDV no usa claves de lector: lo lee la web
        activas = conn.execute('SELECT COUNT(*) AS n FROM claves_lector WHERE usuario_id = ? AND activo = TRUE', (uid,)).fetchone()['n']
        if activas >= MAX_CLAVES_LECTOR:
            return _err(409, f'Máximo {MAX_CLAVES_LECTOR} claves de lector activas')
        clave, h, pref = ad.nueva_clave_lector()
        conn.execute('INSERT INTO claves_lector (usuario_id, cuenta_banco_id, nombre, clave_hash, prefijo_visible, activo, creado_en) '
                     'VALUES (?, ?, ?, ?, ?, TRUE, ?)', (uid, cuenta_id, nombre, h, pref, ad.ahora_local()))
        conn.commit()
        return jsonify(ok=True, clave=clave, mensaje='Copia la clave ahora: no se volverá a mostrar', **_claves(conn, uid))
    finally:
        conn.close()


@bp.route('/referencias/claves-lector/<int:clave_id>/<accion>', methods=['POST'])
def referencias_lector_accion(clave_id, accion):
    uid, err = _sesion_json(post=True)
    if err:
        return err
    if accion not in ('regenerar', 'desactivar', 'activar'):
        return _err(404, 'Acción no encontrada')
    conn = get_db_connection()
    try:
        if not conn.execute('SELECT 1 FROM claves_lector WHERE id = ? AND usuario_id = ?', (clave_id, uid)).fetchone():
            return _err(404, 'Clave no encontrada')
        if accion == 'regenerar':
            clave, h, pref = ad.nueva_clave_lector()
            conn.execute('UPDATE claves_lector SET clave_hash = ?, prefijo_visible = ?, activo = TRUE WHERE id = ? AND usuario_id = ?',
                         (h, pref, clave_id, uid))
            extra = {'clave': clave, 'mensaje': 'Copia la clave nueva ahora: la anterior ya no sirve'}
        else:
            conn.execute('UPDATE claves_lector SET activo = ? WHERE id = ? AND usuario_id = ?', (accion == 'activar', clave_id, uid))
            extra = {'mensaje': 'Clave activada' if accion == 'activar' else 'Clave desactivada'}
        conn.commit()
        return jsonify(ok=True, **extra, **_claves(conn, uid))
    finally:
        conn.close()


@bp.route('/referencias/claves-api', methods=['POST'])
def referencias_api_crear():
    """El usuario crea sus claves de API de Referencias (las de Recargas están en /mi-api). Se muestra una sola vez."""
    uid, err = _sesion_json(post=True)
    if err:
        return err
    from api_panel import crear_clave_usuario
    data = request.get_json(silent=True) or {}
    nombre = str(data.get('nombre') or '').strip()[:60] or 'Mi sistema'
    conn = get_db_connection()
    try:
        banco = _banco_peticion()
        status, datos = crear_clave_usuario(conn, uid, 'referencias', nombre, data.get('permisos') or {}, banco=banco)
        if status != 200:
            return _err(status, datos['error'])
        return jsonify(ok=True, **datos, **_claves(conn, uid, banco))
    finally:
        conn.close()


@bp.route('/referencias/claves-api/<int:cuenta_id>/<accion>', methods=['POST'])
def referencias_api_accion(cuenta_id, accion):
    uid, err = _sesion_json(post=True)
    if err:
        return err
    from api_panel import accion_clave_usuario
    conn = get_db_connection()
    try:
        banco = _banco_peticion()
        status, datos = accion_clave_usuario(conn, uid, 'referencias', cuenta_id, accion,
                                             (request.get_json(silent=True) or {}).get('permisos'), banco=banco)
        if status != 200:
            return _err(status, datos['error'])
        return jsonify(ok=True, **datos, **_claves(conn, uid, banco))
    finally:
        conn.close()


# ---------- BDV: la conexión del usuario con su banco (la web lee sola sus movimientos) ----------

def _cuentas_bdv(conn, uid):
    import bdv
    return {'cuentas': ad.listar_cuentas(conn, uid, banco=ad.BDV), 'disponible': bdv.cifrado_disponible()}


@bp.route('/referencias/bdv/cuentas', methods=['GET', 'POST'])
def referencias_bdv_cuentas():
    """GET: sus conexiones del BDV con su estado (nunca sus datos). POST: guardar y conectar."""
    import bdv
    uid, err = _sesion_json(post=request.method == 'POST')
    if err:
        return err
    conn = get_db_connection()
    try:
        if request.method == 'GET':
            return jsonify(ok=True, **_cuentas_bdv(conn, uid))
        data = request.get_json(silent=True) or {}
        status, datos = bdv.guardar_conexion(conn, uid, data.get('usuario'), data.get('clave'), data.get('numero_cuenta'),
                                             data.get('alias'), bool(data.get('consentimiento')))
        if status != 200:
            return _err(status, datos['mensaje'])
        return jsonify(ok=True, mensaje=datos['mensaje'], **_cuentas_bdv(conn, uid))
    finally:
        conn.close()


@bp.route('/referencias/bdv/cuentas/<int:cuenta_id>/<accion>', methods=['POST'])
def referencias_bdv_accion(cuenta_id, accion):
    """leer (encender) / pausar / reanudar / borrar."""
    import bdv
    uid, err = _sesion_json(post=True)
    if err:
        return err
    conn = get_db_connection()
    try:
        if accion in ('leer', 'pausar'):
            status, datos = bdv.cambiar_lectura(conn, uid, cuenta_id, accion == 'leer')
        elif accion == 'reanudar':
            status, datos = bdv.reanudar(conn, uid, cuenta_id)
        elif accion == 'borrar':
            status, datos = bdv.borrar_conexion(conn, uid, cuenta_id)
        else:
            return _err(404, 'Acción no encontrada')
        if status != 200:
            return _err(status, datos['mensaje'])
        return jsonify(ok=True, mensaje=datos['mensaje'], **_cuentas_bdv(conn, uid))
    finally:
        conn.close()

"""
API del lector externo del BDV (Blueprint Flask): /api/lector-bdv/*. No aparece en la documentación de los usuarios.

El lector (lector_externo/, en otro servidor) no tiene una clave que alguien deba copiar: se identifica con su propio
par de llaves (PyNaCl).
  1. POST /api/lector-bdv/registro  {publica}   → la primera vez queda «pendiente» con un código (AB12-CD34) hasta que
     el admin lo aprueba en Referencias (todos) confirmando ese mismo código. Después responde «aprobado».
  2. POST /api/lector-bdv/reto      {publica}   → solo para el lector aprobado: una clave temporal (1 h) cerrada con su
     llave pública; solo él puede abrirla. Se guarda con hash. Es la clave «de tipo lector_bdv» de las demás rutas.
  3. Con Authorization: Bearer <clave temporal>:
     GET  /api/lector-bdv/cuentas   cuentas a leer (con sus datos CIFRADOS: la web no puede abrirlos) y las que debe
                                    cerrar (pausadas, borradas, detenidas), cada una con su versión.
                                    ETag/If-None-Match → 304 si nada cambió.
     POST /api/lector-bdv/estado    estado de cada cuenta (validado; se ignoran los reportes de una versión vieja)
                                    y datos del lector (memoria, cuentas leyendo).
     POST /api/lector-bdv/pagos     pagos leídos de una cuenta BDV; el usuario sale de la cuenta, nunca del cuerpo.
     Cada petición cuenta como latido: más de 2 min sin latido → la pantalla dice «El lector no responde».

Si BDV_CLAVE_PUBLICA está en el .env, esa llave queda fijada y no hace falta aprobar nada en el panel.
Nunca se registran claves temporales ni datos cifrados.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time

from flask import Blueprint, Response, jsonify, request, session

import antiduplic as ad
import bdv
from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('bdv_lector_api', __name__)

CONFIG_PENDIENTE = 'bdv_lector_pendiente'
CONFIG_SESIONES = 'bdv_lector_sesiones'
CONFIG_LATIDO = 'bdv_lector_latido'
CONFIG_ESTADO = 'bdv_lector_estado'
SESION_SEG = 3600
VIVO_SEG = 120
ESTADOS_LECTOR = ('conectando', 'conectado', 'banco_no_responde', 'rechazado', 'detenido', 'pausado', 'en_espera')
MAX_PAGOS = 2000
MAX_CUERPO = 2 * 1024 * 1024
LIMITES = {'registro': 10, 'reto': 20, 'cuentas': 60, 'estado': 120, 'pagos': 120}   # por minuto
_FECHA_HORA = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
_golpes, _golpes_lock = {}, threading.Lock()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _ip():
    """IP de quien llama (la web está detrás de Cloudflare). Solo se muestra; no se usa para dar acceso."""
    return (request.headers.get('CF-Connecting-IP') or request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
            or request.remote_addr or '?')[:45]


def _limite_ok(tipo, quien):
    ahora = time.time()
    clave = (tipo, quien)
    with _golpes_lock:
        if len(_golpes) > 5000:
            _golpes.clear()
        lista = [t for t in _golpes.get(clave, []) if ahora - t < 60]
        ok = len(lista) < LIMITES[tipo]
        if ok:
            lista.append(ahora)
        _golpes[clave] = lista
        return ok


def _err(status, mensaje):
    return jsonify(ok=False, error=mensaje), status


def _db(fn):
    conn = get_db_connection()
    try:
        r = fn(conn)
        conn.commit()
        return r
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _leer_json(conn, clave, defecto):
    try:
        return json.loads(ad._config_get(conn, clave) or 'null') or defecto
    except ValueError:
        return defecto


def _hash(clave):
    return hashlib.sha256(clave.encode('utf-8')).hexdigest()


def publica_aprobada(conn=None):
    """(llave pública aprobada, fija_por_env). La del .env manda sobre la aprobada en el panel."""
    env = os.environ.get('BDV_CLAVE_PUBLICA', '').strip()
    if env:
        return (env if bdv.publica_valida(env) else None), True
    if conn is None:
        return _db(lambda c: publica_aprobada(c))
    valor = ad._config_get(conn, bdv.CONFIG_PUBLICA)
    return (valor if bdv.publica_valida(valor) else None), False


def info_lector(conn):
    """Lo que el panel muestra del lector: aprobado, pendiente, último latido (con IP) y su estado. Sin secretos."""
    aprobada, por_env = publica_aprobada(conn)
    pendiente = _leer_json(conn, CONFIG_PENDIENTE, None)
    latido = _leer_json(conn, CONFIG_LATIDO, None)
    hace = round(time.time() - float(latido['t'])) if latido and latido.get('t') else None
    return {
        'aprobado': {'codigo': bdv.huella(aprobada), 'fija_por_env': por_env} if aprobada else None,
        'pendiente': ({'codigo': pendiente.get('codigo'), 'ip': pendiente.get('ip'), 'cuando': pendiente.get('cuando'),
                       'reemplaza': bool(aprobada)} if pendiente and not por_env else None),
        'latido': {'hace_seg': hace, 'ip': latido.get('ip'), 'cuando': latido.get('cuando')} if latido else None,
        'vivo': hace is not None and hace <= VIVO_SEG,
        'estado': _leer_json(conn, CONFIG_ESTADO, None),
    }


def lector_vivo(conn):
    latido = _leer_json(conn, CONFIG_LATIDO, None)
    return bool(latido and latido.get('t') and time.time() - float(latido['t']) <= VIVO_SEG)


# ---------------------------------------------------------------------------
# 1 y 2: registro y reto (sin clave: el lector se identifica con su llave)
# ---------------------------------------------------------------------------

def _publica_del_cuerpo():
    publica = str((request.get_json(silent=True) or {}).get('publica') or '').strip()
    return publica if bdv.publica_valida(publica) else None


@bp.route('/api/lector-bdv/registro', methods=['POST'])
def registro():
    if not _limite_ok('registro', _ip()):
        return _err(429, 'Demasiadas peticiones; espera un minuto')
    publica = _publica_del_cuerpo()
    if not publica:
        return _err(400, 'Llave pública no válida')
    ad.init_tablas()
    codigo = bdv.huella(publica)

    def q(conn):
        aprobada, por_env = publica_aprobada(conn)
        if aprobada and hmac.compare_digest(aprobada, publica):
            return 200, {'ok': True, 'estado': 'aprobado', 'codigo': codigo}
        if por_env:
            return 403, {'ok': False, 'estado': 'rechazado', 'codigo': codigo,
                         'error': 'Este servidor tiene fijada otra llave del lector (BDV_CLAVE_PUBLICA)'}
        ad._config_set(conn, CONFIG_PENDIENTE, json.dumps({'publica': publica, 'codigo': codigo, 'ip': _ip(),
                                                            'cuando': ad.ahora_local(), 't': time.time()}))
        return 202, {'ok': True, 'estado': 'pendiente', 'codigo': codigo,
                     'mensaje': 'Pide al administrador que apruebe este lector en Referencias (todos) con este código'}
    status, datos = _db(q)
    if datos['estado'] == 'pendiente':
        logger.info(f'[BDV] Un lector pide aprobación: código {codigo} desde {_ip()}')
    return jsonify(datos), status


@bp.route('/api/lector-bdv/reto', methods=['POST'])
def reto():
    from nacl.public import PublicKey, SealedBox
    if not _limite_ok('reto', _ip()):
        return _err(429, 'Demasiadas peticiones; espera un minuto')
    publica = _publica_del_cuerpo()
    if not publica:
        return _err(400, 'Llave pública no válida')
    ad.init_tablas()
    clave = secrets.token_urlsafe(32)

    def q(conn):
        aprobada, _ = publica_aprobada(conn)
        if not aprobada or not hmac.compare_digest(aprobada, publica):
            return False
        ahora = time.time()
        vigentes = [s for s in _leer_json(conn, CONFIG_SESIONES, []) if s.get('vence', 0) > ahora]
        # La llave pública no es secreta: cualquiera puede pedir retos (aunque no abrirlos). Por eso un reto nuevo
        # solo desplaza a otros retos sin usar, nunca a las claves que el lector ya está usando.
        sesiones = [s for s in vigentes if s.get('usada')][-3:] + [s for s in vigentes if not s.get('usada')][-2:]
        sesiones.append({'hash': _hash(clave), 'vence': ahora + SESION_SEG, 'llave': bdv.huella(publica), 'usada': False})
        ad._config_set(conn, CONFIG_SESIONES, json.dumps(sesiones))
        return True
    if not _db(q):
        return _err(401, 'Lector no aprobado')
    cerrada = SealedBox(PublicKey(base64.b64decode(publica))).encrypt(clave.encode('ascii'))
    return jsonify(ok=True, reto=base64.b64encode(cerrada).decode('ascii'), vence_en=SESION_SEG)


# ---------------------------------------------------------------------------
# 3: rutas con la clave temporal
# ---------------------------------------------------------------------------

def _autenticar(tipo):
    """None si la clave temporal es válida (y anota el latido); si no, la respuesta de error."""
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer ') or not auth[7:].strip():
        return _err(401, 'Falta la clave del lector')
    h = _hash(auth[7:].strip())
    if not _limite_ok(tipo, h):
        return _err(429, 'Demasiadas peticiones; espera un minuto')
    ad.init_tablas()

    def q(conn):
        ahora = time.time()
        aprobada, _ = publica_aprobada(conn)
        if not aprobada:
            return False
        sesiones = _leer_json(conn, CONFIG_SESIONES, [])
        sesion = next((s for s in sesiones if hmac.compare_digest(s.get('hash', ''), h)), None)
        valida = bool(sesion and sesion.get('vence', 0) > ahora and sesion.get('llave') == bdv.huella(aprobada))
        if valida:
            if not sesion.get('usada'):   # desde ahora un reto nuevo ya no la desplaza
                sesion['usada'] = True
                ad._config_set(conn, CONFIG_SESIONES, json.dumps(sesiones))
            ad._config_set(conn, CONFIG_LATIDO, json.dumps({'t': ahora, 'cuando': ad.ahora_local(), 'ip': _ip()}))
        return valida
    return None if _db(q) else _err(401, 'Clave del lector no válida o vencida')


@bp.route('/api/lector-bdv/cuentas')
def cuentas():
    err = _autenticar('cuentas')
    if err:
        return err

    def q(conn):
        return conn.execute("SELECT * FROM cuentas_banco WHERE banco = 'BDV' ORDER BY id").fetchall()
    leer, cerrar = [], []
    for r in _db(q):
        encendido = r['lector_encendido'] is None or bool(r['lector_encendido'])
        if r['bdv_clave_cifrada'] and r['activo'] and encendido and r['lector_estado'] != 'detenido':
            leer.append({'id': r['id'], 'usuario_id': r['usuario_id'], 'ultimos_digitos': r['ultimos_digitos'] or '',
                         'bdv_usuario_cifrado': r['bdv_usuario_cifrado'], 'bdv_clave_cifrada': r['bdv_clave_cifrada'],
                         'numero_cuenta_cifrado': r['numero_cuenta_cifrado'], 'version': r['lector_cambio'] or '',
                         'estado': r['lector_estado'] or 'pendiente', 'fallos_login': int(r['lector_fallos_login'] or 0)})
        else:   # con su versión: el lector la usa al reportar «pausado» (pausar cambia la versión)
            pausada = bool(r['bdv_clave_cifrada'] and r['activo'] and not encendido)
            # falta_pausa: pausada pero aún sin confirmar (p. ej. se pausó con el lector apagado): el lector la confirma
            cerrar.append({'id': r['id'], 'version': r['lector_cambio'] or '',
                           'falta_pausa': pausada and r['lector_estado'] != 'pausado'})
    # La ETag solo cambia con lo que exige actuar (cuentas, versiones y cuáles cerrar), no con el estado que reporta
    # el propio lector: así casi siempre se responde 304 sin cuerpo.
    base = json.dumps([[c['id'], c['version'], c['bdv_clave_cifrada'][-12:]] for c in leer] + [cerrar])
    etag = '"' + hashlib.sha256(base.encode()).hexdigest()[:32] + '"'
    if _etag_coincide(request.headers.get('If-None-Match'), etag):
        resp = Response(status=304)
    else:
        resp = jsonify(ok=True, cuentas=leer, cerrar=cerrar)
    resp.headers['ETag'] = etag
    resp.headers['Cache-Control'] = 'no-store'
    return resp


def _etag_coincide(cabecera, etag):
    """If-None-Match puede traer la ETag débil (W/"…"): Cloudflare la debilita cuando comprime la respuesta."""
    for parte in (cabecera or '').split(','):
        parte = parte.strip()
        if parte.startswith('W/'):
            parte = parte[2:]
        if parte == etag:
            return True
    return False


def _texto(v, largo):
    v = re.sub(r'[\x00-\x1f\x7f]', ' ', str(v or '')).strip()
    return v[:largo]


@bp.route('/api/lector-bdv/estado', methods=['POST'])
def estado():
    err = _autenticar('estado')
    if err:
        return err
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err(400, 'Se espera un objeto JSON')
    items = data.get('cuentas') or []
    if not isinstance(items, list) or len(items) > 500:
        return _err(400, 'cuentas debe ser una lista')
    cambios, ignoradas = [], 0
    for it in items:
        try:
            cid = int(it.get('id'))
            est = str(it.get('estado') or '')
            version = str(it.get('version') or '')
        except (TypeError, ValueError, AttributeError):
            ignoradas += 1
            continue
        if est not in ESTADOS_LECTOR:
            ignoradas += 1
            continue
        campos = {'lector_estado': est, 'lector_mensaje': _texto(it.get('mensaje'), 200)}
        for k, col in (('ultima_lectura', 'lector_ultima_lectura'), ('ultimo_pago', 'lector_ultimo_pago')):
            v = it.get(k)
            if v and _FECHA_HORA.match(str(v)):
                campos[col] = str(v)
        if 'fallos_login' in it:   # si no viene, se deja como está
            try:
                campos['lector_fallos_login'] = max(0, min(10, int(it.get('fallos_login') or 0)))
            except (TypeError, ValueError):
                pass
        cambios.append((cid, version, campos))

    def q(conn):
        n = 0
        for cid, version, campos in cambios:
            sets = ', '.join(f'{k} = ?' for k in campos)
            # Solo cuentas del BDV con datos y de la versión que el lector está usando: un reporte viejo (por
            # ejemplo «detenido» justo cuando el usuario guardó datos nuevos) no pisa lo nuevo.
            cur = conn.execute(f"UPDATE cuentas_banco SET {sets} WHERE id = ? AND banco = 'BDV' AND bdv_clave_cifrada IS NOT NULL "
                               "AND COALESCE(lector_cambio, '') = ?", (*campos.values(), cid, version))
            n += max(cur.rowcount, 0)
        lector = data.get('lector')
        if isinstance(lector, dict):
            limpio = {'latido': ad.ahora_local()}
            for k in ('memoria_mb', 'por_cuenta_mb'):
                try:
                    limpio[k] = round(float(lector[k]), 1) if lector.get(k) is not None else None
                except (TypeError, ValueError):
                    limpio[k] = None
            for k in ('cuentas_leyendo', 'max_cuentas', 'en_espera'):
                try:
                    limpio[k] = max(0, min(10000, int(lector.get(k) or 0)))
                except (TypeError, ValueError):
                    limpio[k] = 0
            limpio['version'] = _texto(lector.get('version'), 20)
            ad._config_set(conn, CONFIG_ESTADO, json.dumps(limpio))
        return n
    n = _db(q)
    return jsonify(ok=True, actualizadas=n, ignoradas=ignoradas + len(cambios) - n)


@bp.route('/api/lector-bdv/pagos', methods=['POST'])
def pagos():
    err = _autenticar('pagos')
    if err:
        return err
    if (request.content_length or 0) > MAX_CUERPO:
        return _err(413, 'Máximo 2 MB por envío')
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err(400, 'Se espera un objeto JSON')
    try:
        cid = int(data.get('cuenta_id'))
    except (TypeError, ValueError):
        return _err(400, 'Falta cuenta_id')
    lista = data.get('pagos')
    if not isinstance(lista, list) or len(lista) > MAX_PAGOS:
        return _err(400, f'pagos debe ser una lista de máximo {MAX_PAGOS}')
    fila = _db(lambda conn: conn.execute("SELECT id, usuario_id FROM cuentas_banco WHERE id = ? AND banco = 'BDV'",
                                         (cid,)).fetchone())
    if not fila:
        return _err(404, 'Cuenta del BDV no encontrada')
    limpios = []
    for p in lista:
        if isinstance(p, dict):
            tipo = _texto(p.get('tipo'), 120)
            limpios.append({**p, 'tipo': tipo if tipo.startswith('BDV ') else ('BDV ' + tipo)[:120]})
        else:
            limpios.append(p)
    # El usuario sale de la cuenta, nunca del cuerpo
    recibidos, insertados, errores = ad.guardar_pagos(limpios, int(fila['usuario_id']), int(fila['id']))
    if insertados:
        _db(lambda conn: conn.execute('UPDATE cuentas_banco SET lector_ultimo_pago = ? WHERE id = ?', (ad.ahora_local(), cid)))
        logger.info(f'[BDV] Cuenta {cid}: {insertados} pagos nuevos del lector')
    return jsonify(ok=True, recibidos=recibidos, insertados=insertados, rechazados=len(errores))


# ---------------------------------------------------------------------------
# Admin: aprobar, rechazar o desconectar el lector
# ---------------------------------------------------------------------------

def _solo_admin():
    if not session.get('is_admin'):
        return _err(403, 'Acceso denegado')
    if not request.is_json:
        return _err(415, 'Se espera JSON')
    return None


@bp.route('/admin/antiduplic/lector/<accion>', methods=['POST'])
def admin_lector(accion):
    """aprobar {codigo} (debe ser el del lector pendiente) / rechazar / desconectar."""
    err = _solo_admin()
    if err:
        return err
    if accion not in ('aprobar', 'rechazar', 'desconectar'):
        return _err(404, 'Acción no encontrada')
    ad.init_tablas()
    codigo = str((request.get_json(silent=True) or {}).get('codigo') or '').strip().upper()

    def q(conn):
        _, por_env = publica_aprobada(conn)
        if por_env:
            return 409, 'La llave del lector está fijada en el .env (BDV_CLAVE_PUBLICA): no se cambia desde el panel'
        pendiente = _leer_json(conn, CONFIG_PENDIENTE, None)
        if accion == 'aprobar':
            # Se aprueba exactamente el lector que el admin vio: si el código no coincide, no se aprueba nada
            if not pendiente or not codigo or not hmac.compare_digest(pendiente.get('codigo', ''), codigo):
                return 409, 'El código no coincide con el del lector que pide conectarse. Vuelve a cargar la página.'
            ad._config_set(conn, bdv.CONFIG_PUBLICA, pendiente['publica'])
            ad._config_set(conn, CONFIG_PENDIENTE, 'null')
            ad._config_set(conn, CONFIG_SESIONES, '[]')
            return 200, f'Lector {codigo} aprobado'
        if accion == 'rechazar':
            ad._config_set(conn, CONFIG_PENDIENTE, 'null')
            return 200, 'Solicitud rechazada'
        ad._config_set(conn, bdv.CONFIG_PUBLICA, '')
        ad._config_set(conn, CONFIG_SESIONES, '[]')
        return 200, 'Lector desconectado: ya no puede pedir cuentas ni mandar pagos'
    status, mensaje = _db(q)
    if status == 200:
        logger.info(f'[BDV] Admin: {mensaje}')
    return jsonify(ok=status == 200, mensaje=mensaje), status

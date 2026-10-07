"""
Cliente del proveedor Central One (https://portal.centraloneglobal.com).

API distinta a la de Inefable:
  - Authorization: Bearer co_live_...  (variable de entorno CENTRALONE_API_KEY; nunca en el navegador)
  - GET  /api/v1/catalog           productos (product_id UUID), precios en texto con 4 decimales
  - GET  /api/v1/balance           saldo
  - POST /api/v1/orders            crea un pedido; exige Idempotency-Key (reintentar con la misma no compra dos veces)
  - GET  /api/v1/orders/{id}       estado; delivered_count > 0 → hay códigos
  - GET  /api/v1/orders/{id}/codes códigos/PIN (permiso codes:read)

En el catálogo local (rev_catalog_items) sus productos se guardan con remote_product_id = 'co:<familia>'
y remote_package_id = UUID del producto, así el Mapeo los distingue de los de Inefable.

No hay sandbox: cualquier pedido mueve saldo real.
"""
import json
import logging
import os
import re
import time

import requests

logger = logging.getLogger(__name__)

PREFIJO = 'co:'
NOMBRE = 'Central One'
BASE_DEFAULT = 'https://portal.centraloneglobal.com'
ESPERA_ENTREGA_S = 15     # tiempo que la compra espera la entrega antes de dejarla "en proceso"
INTERVALO_S = 2
TIMEOUT_S = 20

_ZONA = ('zone', 'zona')
_SERVIDOR = ('server', 'servidor', 'region')
_ID = ('user_id', 'player_id', 'uid', 'game_user_id', 'account', 'id')

ERRORES = {
    'insufficient_balance': 'Saldo insuficiente en Central One',
    'insufficient_stock': 'Sin stock en Central One',
    'invalid_request': 'Central One rechazó los datos del pedido',
    'invalid_api_key': 'La llave de Central One no es válida',
    'insufficient_scope': 'La llave de Central One no tiene permiso para esto',
    'not_found': 'Producto o pedido no encontrado en Central One',
    'rate_limited': 'Demasiadas peticiones a Central One',
}


def es_de_central_one(remote_product_id):
    return str(remote_product_id or '').startswith(PREFIJO)


def base_url():
    return (os.environ.get('CENTRALONE_BASE_URL', '').strip() or BASE_DEFAULT).rstrip('/')


def configurado():
    return bool(os.environ.get('CENTRALONE_API_KEY', '').strip())


def _pedir(metodo, path, body=None, idem=None, sin_llave=False):
    """(status_http, json). status 0 = sin respuesta (red caída o timeout). Nunca lanza."""
    headers = {'Accept': 'application/json', 'User-Agent': '3sRevendedores/1.0'}
    if not sin_llave:
        headers['Authorization'] = 'Bearer ' + os.environ.get('CENTRALONE_API_KEY', '').strip()
    if idem:
        headers['Idempotency-Key'] = idem
    try:
        r = requests.request(metodo, base_url() + path, headers=headers, json=body, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        logger.warning(f'[Central One] {metodo} {path} sin respuesta: {e}')
        return 0, {}
    try:
        data = r.json()
    except ValueError:
        data = {}
    return r.status_code, data if isinstance(data, dict) else {}


def _error(status, data):
    code = ((data or {}).get('error') or {}).get('code') or ''
    return ERRORES.get(code) or f'Central One respondió con error ({status})'


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------

def salud():
    status, data = _pedir('GET', '/api/v1/health', sin_llave=True)
    return status == 200 and data.get('status') == 'ok'


def ping():
    status, data = _pedir('GET', '/api/v1/ping')
    return status == 200 and bool(data.get('ok')), (None if status == 200 else _error(status, data))


def saldo():
    """{'disponible', 'retenido', 'total', 'moneda'} o {'error'}."""
    status, data = _pedir('GET', '/api/v1/balance')
    if status != 200:
        return {'error': _error(status, data) if status else 'Central One no responde'}
    return {'disponible': data.get('available_balance'), 'retenido': data.get('held_balance'),
            'total': data.get('total_balance'), 'moneda': data.get('currency') or 'USD'}


def catalogo():
    """Lista de productos o lanza RuntimeError con un mensaje claro."""
    status, data = _pedir('GET', '/api/v1/catalog')
    if status != 200:
        raise RuntimeError(_error(status, data) if status else 'Central One no responde')
    items = data.get('items')
    if not isinstance(items, list):
        raise RuntimeError('Respuesta inesperada del catálogo de Central One')
    return items


def precio(item):
    """Precio del catálogo como float (viene como texto con 4 decimales)."""
    try:
        return float(str(item.get('reseller_price') or '0'))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Datos del jugador que pide cada producto
# ---------------------------------------------------------------------------

def campos_de(item):
    """[{key, label, type, options}] que el producto exige en target_payload."""
    if not item or not item.get('requires_target'):
        return []
    schema = item.get('target_schema') or []
    if schema:
        return [s for s in schema if isinstance(s, dict) and s.get('key')]
    return [{'key': k, 'label': k, 'type': 'text', 'options': None} for k in (item.get('target_fields') or [])]


def _clase(key):
    k = key.lower()
    if any(z in k for z in _ZONA):
        return 'zona'
    if any(s in k for s in _SERVIDOR):
        return 'servidor'
    if any(i in k for i in _ID):
        return 'id'
    return 'otro'


def armar_target(item, player_id, player_id2='', servidor=''):
    """(target_payload, error). Asigna el ID, la zona y el servidor a los campos que pide el producto.
    Si falta algo o un valor no está entre las opciones, devuelve error ANTES de pedir (no se cobra nada)."""
    campos = campos_de(item)
    if not campos:
        return None, None
    payload, usado_id = {}, False
    for c in campos:
        clase = _clase(c['key'])
        if clase == 'id' and not usado_id:
            valor, usado_id = player_id, True
        elif clase == 'zona':
            valor = player_id2
        elif clase == 'servidor':
            valor = servidor or player_id2
        elif not usado_id:
            valor, usado_id = player_id, True
        else:
            valor = player_id2 or servidor
        valor = str(valor or '').strip()
        if not valor:
            return None, f"Falta el dato «{c.get('label') or c['key']}» que pide Central One"
        if c.get('type') == 'select' and c.get('options'):
            opciones = [o for o in c['options'] if isinstance(o, dict)]
            elegido = next((o['value'] for o in opciones if str(o.get('value', '')).lower() == valor.lower()
                            or str(o.get('label', '')).lower() == valor.lower()), None)
            if elegido is None:
                validas = ', '.join(str(o.get('label') or o.get('value')) for o in opciones)
                return None, f"«{valor}» no es válido para {c.get('label') or c['key']} (opciones: {validas})"
            valor = elegido
        payload[c['key']] = valor
    return payload, None


# ---------------------------------------------------------------------------
# Pedidos
# ---------------------------------------------------------------------------

def _idem(ext):
    """Idempotency-Key válida (8-255, A-Z a-z 0-9 _ -) a partir del código de la recarga."""
    k = re.sub(r'[^A-Za-z0-9_-]', '-', str(ext or ''))[:255]
    return k if len(k) >= 8 else (k + '-' * 8)[:8]


def _codigos(order_id):
    status, data = _pedir('GET', f'/api/v1/orders/{order_id}/codes')
    if status != 200:
        return [], ''
    pines, seriales = [], []
    for it in ((data.get('order') or {}).get('items') or []):
        for c in (it.get('codes') or []):
            pin, _, serial = str(c).partition(' - ')
            pines.append(pin.strip())
            if serial.strip():
                seriales.append(serial.strip())
    return pines, ', '.join(seriales)


def _resultado(order):
    """Traduce un pedido de Central One al formato de las recargas: ok / pending / fail."""
    order = order or {}
    est = str(order.get('status') or '').lower()
    items = order.get('items') or []
    ref = order.get('reference_code') or order.get('id') or ''
    lineas = [str(i.get('status') or '').lower() for i in items]
    if est == 'completed' or (lineas and all(s == 'completed' for s in lineas)):
        res = {'ok': True, 'reference_no': ref, 'player_name': '', 'pin': ''}
        if any(int(i.get('delivered_count') or 0) > 0 for i in items):
            pines, serial = _codigos(order.get('id'))
            res['pin'] = ' | '.join(pines)
            if serial:
                res['reference_no'] = f'{ref} ({serial})'
        return res
    if est in ('failed', 'cancelled') or (lineas and all(s in ('failed', 'cancelled', 'refunded') for s in lineas)):
        return {'ok': False, 'error': 'Central One no pudo entregar el pedido', 'reference_no': ref}
    return {'ok': False, 'pending': True, 'status': 'procesando', 'reference_no': ref}


def consultar_pedido(order_id):
    status, data = _pedir('GET', f'/api/v1/orders/{order_id}')
    if status != 200:
        return None
    return data.get('order')


def recargar(package_uuid, item, player_id, player_id2, servidor, ext, esperar=True):
    """Crea el pedido (1 unidad) y espera unos segundos a que se entregue.

    Devuelve un dict como el de Inefable: {'ok', 'pending', 'network_error', 'error', 'reference_no',
    'pin', 'player_name', 'co_id'}. 'network_error' significa que no se sabe si el pedido se creó:
    reintentar con la misma ext es seguro (misma Idempotency-Key).
    """
    if not configurado():
        return {'ok': False, 'error': 'Central One no está configurado (falta CENTRALONE_API_KEY)'}
    target, err = armar_target(item or {}, player_id, player_id2, servidor)
    if err:
        return {'ok': False, 'error': err}
    linea = {'catalog_item_id': package_uuid, 'quantity': 1}
    if target:
        linea['target_payload'] = target
    status, data = _pedir('POST', '/api/v1/orders', {'items': [linea], 'note': str(ext)[:500]}, idem=_idem(ext))
    if status in (200, 201):
        order = data.get('order') or {}
        co_id = order.get('id')
        res = _resultado(order)
        deadline = time.time() + (ESPERA_ENTREGA_S if esperar else 0)
        while res.get('pending') and co_id and time.time() < deadline:
            time.sleep(INTERVALO_S)
            o = consultar_pedido(co_id)
            if o:
                res = _resultado(o)
        res['co_id'] = co_id
        return res
    if status == 0 or status == 429 or status >= 500:
        # Sin respuesta clara: el pedido pudo crearse. El reconciliador reintenta con la misma llave.
        return {'ok': False, 'network_error': True, 'error': _error(status, data) if status else 'Central One no responde'}
    return {'ok': False, 'error': _error(status, data)}


def estado(u, item, player_id, player_id2, servidor):
    """Estado de una recarga en proceso (para el reconciliador). Mismo formato que recargar()."""
    if u.get('co_id'):
        o = consultar_pedido(u['co_id'])
        if o is None:
            return {'ok': False, 'pending': True}
        res = _resultado(o)
        res['co_id'] = u['co_id']
        return res
    # Sin id: repetir la creación con la misma Idempotency-Key devuelve el pedido si ya existía
    res = recargar(u['pkg'], item, player_id, player_id2, servidor, u['ext'], esperar=False)
    if res.get('pending') and res.get('co_id'):
        o = consultar_pedido(res['co_id'])
        if o:
            res = dict(_resultado(o), co_id=res['co_id'])
    return res


def item_de_catalogo(conn, package_uuid):
    """Datos del producto guardados al sincronizar (target_schema, precio...)."""
    row = conn.execute("SELECT raw_json FROM rev_catalog_items WHERE remote_package_id = ? AND remote_product_id LIKE 'co:%' "
                       'ORDER BY updated_at DESC LIMIT 1', (str(package_uuid),)).fetchone()
    try:
        return json.loads(row['raw_json'] or '{}') if row else {}
    except ValueError:
        return {}


def sincronizar_catalogo(get_conn):
    """Copia el catálogo de Central One al catálogo local del Mapeo. Devuelve {'nuevos', 'actualizados', 'retirados'}."""
    items = catalogo()
    conn = get_conn()
    nuevos = actualizados = 0
    vistos = set()
    try:
        for it in items:
            uuid = str(it.get('product_id') or '').strip()
            if not uuid:
                continue
            vistos.add(uuid)
            familia = str(it.get('product_family_id') or 'otros')
            prod_id = PREFIJO + familia
            prod_name = it.get('product_family_name') or familia
            raw = dict(it, price=precio(it))
            activo = str(it.get('status') or 'active').lower() == 'active'
            row = conn.execute("SELECT id FROM rev_catalog_items WHERE remote_package_id = ? AND remote_product_id LIKE 'co:%'",
                               (uuid,)).fetchone()
            if row:
                conn.execute('UPDATE rev_catalog_items SET remote_product_id = ?, remote_product_name = ?, remote_package_name = ?, '
                             'raw_json = ?, active = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                             (prod_id, prod_name, it.get('name') or uuid, json.dumps(raw), activo, row['id']))
                actualizados += 1
            else:
                conn.execute('INSERT INTO rev_catalog_items (remote_product_id, remote_product_name, remote_package_id, '
                             'remote_package_name, raw_json, active) VALUES (?, ?, ?, ?, ?, ?)',
                             (prod_id, prod_name, uuid, it.get('name') or uuid, json.dumps(raw), activo))
                nuevos += 1
        retirados = 0
        for r in conn.execute("SELECT id, remote_package_id FROM rev_catalog_items WHERE remote_product_id LIKE 'co:%' AND active = TRUE").fetchall():
            if r['remote_package_id'] not in vistos:
                conn.execute('UPDATE rev_catalog_items SET active = FALSE WHERE id = ?', (r['id'],))
                retirados += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {'nuevos': nuevos, 'actualizados': actualizados, 'retirados': retirados}

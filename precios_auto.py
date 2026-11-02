"""
Precio automático de los paquetes (pedido del dueño, 2026-11-02).

En los juegos de JUEGOS (por ahora solo «Free fire ID») el precio de venta de cada paquete es lo que cuesta en el proveedor
PRINCIPAL de su Mapeo: la suma del precio de cada paso por su cantidad, redondeada hacia arriba al centavo (nunca se vende
por debajo del costo). Cada 20 minutos se vuelven a leer los precios de ese proveedor y, si cambiaron, se cambia el precio.

Nunca se pone un precio a ciegas: si el paquete no tiene Mapeo activo, si a un paso le falta el precio o si el precio nuevo
sale más de MAX_SALTO veces distinto del actual (casi seguro un catálogo mal leído), ese paquete se deja como estaba y queda
en el registro. Si el proveedor no responde se sigue con el último precio guardado de su catálogo.
"""
import json
import logging
import os
import time
from decimal import ROUND_CEILING, Decimal, InvalidOperation

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)

JUEGOS = ('free fire id',)   # nombre del juego en minúsculas; para sumar otro juego basta con agregarlo aquí
INTERVALO_S = 20 * 60
MAX_SALTO = 3
CLAVE_ULTIMA = 'precios_auto_ultima'   # en configuracion_redeemer: la hora de la última revisión (la comparten los workers)
CENTAVO = Decimal('0.01')


def _dec(valor):
    try:
        d = Decimal(str(valor))
        return d if d.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def es_juego_auto(game):
    """Un juego de la web con precio automático (los juegos propios de un usuario nunca)."""
    return bool(game) and not game.get('usuario_id') and str(game.get('nombre') or '').strip().lower() in JUEGOS


def _juegos(conn):
    rows = conn.execute('SELECT id, nombre, usuario_id FROM juegos_dinamicos').fetchall()
    return [int(r['id']) for r in rows if es_juego_auto(dict(r))]


# ---------------------------------------------------------------------------
# Costo de un paquete en su proveedor principal
# ---------------------------------------------------------------------------

def _pasos(conn, juego_id, paquete_id):
    """[(producto, paquete, cantidad)] del Mapeo activo (los pasos o, si no tiene, la cabeza). [] si no está mapeado."""
    head = conn.execute('SELECT remote_product_id, remote_package_id FROM rev_item_mappings WHERE juego_id = ? AND paquete_id = ? '
                        'AND auto_enabled = TRUE AND active = TRUE', (juego_id, paquete_id)).fetchone()
    if not head:
        return []
    pasos = [(str(p['remote_product_id']), str(p['remote_package_id']), max(1, int(p['cantidad'] or 1)))
             for p in conn.execute('SELECT remote_product_id, remote_package_id, cantidad FROM rev_item_mapping_steps '
                                   'WHERE juego_id = ? AND paquete_id = ? ORDER BY orden', (juego_id, paquete_id)).fetchall()]
    return pasos or [(str(head['remote_product_id']), str(head['remote_package_id']), 1)]


def _precio_catalogo(conn, producto, paquete):
    row = conn.execute('SELECT raw_json FROM rev_catalog_items WHERE remote_product_id = ? AND remote_package_id = ?',
                       (producto, paquete)).fetchone()
    try:
        precio = _dec((json.loads(row['raw_json'] or '{}') or {}).get('price')) if row else None
    except ValueError:
        precio = None
    return precio if precio is not None and precio > 0 else None


def costo_paquete(conn, juego_id, paquete_id):
    """(costo, motivo). costo es None si no se puede calcular exacto (sin Mapeo o un paso sin precio)."""
    pasos = _pasos(conn, juego_id, paquete_id)
    if not pasos:
        return None, 'sin Mapeo activo'
    total = Decimal(0)
    for producto, paquete, cantidad in pasos:
        precio = _precio_catalogo(conn, producto, paquete)
        if precio is None:
            return None, f'sin precio en el catálogo para {producto}/{paquete}'
        total += precio * cantidad
    return total, ''


# ---------------------------------------------------------------------------
# Precios de los proveedores (solo el precio: no se agrega ni se desactiva nada del catálogo)
# ---------------------------------------------------------------------------

def _guardar_precio(conn, fila, precio):
    try:
        raw = json.loads(fila['raw_json'] or '{}') or {}
    except ValueError:
        raw = {}
    if _dec(raw.get('price')) == _dec(precio):
        return False
    raw['price'] = precio
    conn.execute('UPDATE rev_catalog_items SET raw_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?', (json.dumps(raw), fila['id']))
    return True


def _refrescar_central_one(conn):
    import central_one
    if not central_one.configurado():
        raise RuntimeError('Central One no está configurado')
    por_uuid = {str(i.get('product_id')): i for i in central_one.catalogo() if i.get('product_id')}
    n = 0
    for fila in conn.execute("SELECT id, remote_package_id, raw_json FROM rev_catalog_items WHERE remote_product_id LIKE 'co:%'").fetchall():
        item = por_uuid.get(str(fila['remote_package_id']))
        precio = central_one.precio(item) if item else 0
        if precio > 0 and _guardar_precio(conn, fila, precio):
            n += 1
    return n


def _refrescar_inefable(conn):
    import requests
    base = os.environ.get('REVENDEDORES_BASE_URL', '').strip().rstrip('/')
    llave = os.environ.get('REVENDEDORES_API_KEY', '').strip()
    if not base or not llave:
        raise RuntimeError('Inefable no está configurado')
    r = requests.get(f'{base}/api/v1/products', headers={'X-API-Key': llave, 'Accept': 'application/json'}, timeout=15)
    r.raise_for_status()
    data = r.json()
    juegos = data.get('games', data.get('products', data.get('data', []))) if isinstance(data, dict) else []
    n = 0
    for prod in juegos if isinstance(juegos, list) else []:
        prod_id = str(prod.get('game_id', prod.get('id', prod.get('product_id', ''))))
        for pkg in prod.get('packages', prod.get('items', [])) or []:
            precio = _dec(pkg.get('price') if pkg.get('price') is not None else pkg.get('precio'))
            if precio is None or precio <= 0:
                continue
            fila = conn.execute('SELECT id, raw_json FROM rev_catalog_items WHERE remote_product_id = ? AND remote_package_id = ?',
                                (prod_id, str(pkg.get('package_id', pkg.get('id', ''))))).fetchone()
            if fila and _guardar_precio(conn, fila, float(precio)):
                n += 1
    return n


def _refrescar_proveedores(conn, juegos):
    """Vuelve a leer los precios de los proveedores que usan los paquetes de esos juegos. {proveedor: error} de los que fallaron."""
    usados = set()
    for jid in juegos:
        for p in conn.execute('SELECT id FROM paquetes_dinamicos WHERE juego_id = ?', (jid,)).fetchall():
            for producto, _, _ in _pasos(conn, jid, p['id']):
                usados.add('centralone' if producto.startswith('co:') else 'inefable')
    fallos = {}
    for prov, fn in (('centralone', _refrescar_central_one), ('inefable', _refrescar_inefable)):
        if prov not in usados:
            continue
        try:
            n = fn(conn)
            conn.commit()
            if n:
                logger.info(f'[Precio auto] {prov}: {n} precios del catálogo cambiaron')
        except Exception as e:
            conn.rollback()
            fallos[prov] = str(e)[:200]
            logger.warning(f'[Precio auto] No se pudieron leer los precios de {prov}: {e}')
    return fallos


# ---------------------------------------------------------------------------
# Poner al día los precios
# ---------------------------------------------------------------------------

def actualizar(leer_proveedores=True):
    """Pone el precio de cada paquete de los juegos de JUEGOS igual a su costo en el proveedor principal. Devuelve un resumen."""
    conn = get_db_connection()
    resumen = {'cambiados': [], 'iguales': 0, 'saltados': [], 'fallos': {}}
    try:
        juegos = _juegos(conn)
        if not juegos:
            return resumen
        if leer_proveedores:
            resumen['fallos'] = _refrescar_proveedores(conn, juegos)
        for jid in juegos:
            for p in conn.execute('SELECT id, nombre, precio FROM paquetes_dinamicos WHERE juego_id = ? ORDER BY orden, id', (jid,)).fetchall():
                costo, motivo = costo_paquete(conn, jid, p['id'])
                if costo is None:
                    resumen['saltados'].append({'id': p['id'], 'nombre': p['nombre'], 'motivo': motivo})
                    continue
                nuevo = costo.quantize(CENTAVO, rounding=ROUND_CEILING)
                actual = (_dec(p['precio']) or Decimal(0)).quantize(CENTAVO)
                if nuevo == actual:
                    resumen['iguales'] += 1
                    continue
                if actual > 0 and (nuevo > actual * MAX_SALTO or nuevo * MAX_SALTO < actual):
                    motivo = f'el precio nuevo ({nuevo}) es demasiado distinto al actual ({actual}): se deja para revisar a mano'
                    resumen['saltados'].append({'id': p['id'], 'nombre': p['nombre'], 'motivo': motivo})
                    logger.warning(f'[Precio auto] {p["nombre"]}: {motivo}')
                    continue
                conn.execute('UPDATE paquetes_dinamicos SET precio = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ? AND juego_id = ?',
                             (float(nuevo), p['id'], jid))
                conn.commit()
                resumen['cambiados'].append({'id': p['id'], 'nombre': p['nombre'], 'antes': float(actual), 'nuevo': float(nuevo),
                                             'costo': float(costo)})
                logger.info(f'[Precio auto] {p["nombre"]}: {actual} → {nuevo} (costo del proveedor {costo})')
        return resumen
    finally:
        conn.close()


def _tomar_turno(ahora):
    """True si a este proceso le toca revisar ahora (los workers de gunicorn corren el hilo; uno solo trabaja cada 20 min)."""
    marca, limite = f'{int(ahora):012d}', f'{int(ahora - INTERVALO_S + 30):012d}'
    conn = get_db_connection()
    try:
        cur = conn.execute('UPDATE configuracion_redeemer SET valor = ? WHERE clave = ? AND valor < ?', (marca, CLAVE_ULTIMA, limite))
        if cur.rowcount == 1:
            conn.commit()
            return True
        if conn.execute('SELECT 1 FROM configuracion_redeemer WHERE clave = ?', (CLAVE_ULTIMA,)).fetchone():
            conn.rollback()
            return False
        conn.execute('INSERT INTO configuracion_redeemer (clave, valor) VALUES (?, ?)', (CLAVE_ULTIMA, marca))
        conn.commit()
        return True
    except Exception:
        conn.rollback()   # p. ej. el otro worker la insertó a la vez
        return False
    finally:
        conn.close()


def tick():
    """Lo llama cada minuto el hilo de app.py: cada 20 minutos pone al día los precios. Un fallo aquí nunca frena nada."""
    if not _tomar_turno(time.time()):
        return None
    r = actualizar()
    logger.info(f"[Precio auto] Revisión: {len(r['cambiados'])} cambiados, {r['iguales']} iguales, {len(r['saltados'])} sin cambiar")
    return r

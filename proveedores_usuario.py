"""
Proveedores propios de cada usuario (pedido del dueño, 2026-10-25).

Cada usuario puede conectar su propio Central One (uno o varios), ver su saldo allá, sincronizar su catálogo y crear SUS
juegos y paquetes con ese proveedor. Esos juegos solo los ve y los usa él (en la web y en su API de Recargas), se recargan con
SU llave y no tocan su saldo de la web: paga directo a su proveedor. Sus compras quedan en su historial con monto 0 (no cuentan
en las ventas ni en las ganancias de la web) y nunca usan los proveedores, el respaldo ni el Bot de Free Fire de la web.

Tablas y columnas:
  proveedores_usuario                         una conexión por fila (usuario, tipo, nombre, llave cifrada, comienzo visible)
  catalogo_proveedor_usuario                  el catálogo de cada conexión (como rev_catalog_items, pero solo de esa conexión)
  juegos_dinamicos.usuario_id                 dueño del juego (NULL = juego de la web, para todos)
  rev_item_mapping_steps.proveedor_usuario_id la conexión con la que se recarga ese paso (NULL = proveedores de la web)

La llave se guarda cifrada (SecretBox de PyNaCl con una llave derivada de PROVEEDORES_USUARIO_SECRETO o, si no está, de
SECRET_KEY: si esa cambia, el usuario vuelve a pegar su llave) y nunca se vuelve a mostrar, solo su comienzo.
"""
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import threading
from datetime import datetime

from flask import Blueprint, jsonify, redirect, render_template, request, session

import central_one
from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('proveedores_usuario', __name__)

TIPOS = {'centralone': central_one.NOMBRE}
MAX_CONEXIONES = 5
MAX_JUEGOS = 30
MAX_PAQUETES = 60

# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

_tablas_listas = False
_tablas_lock = threading.Lock()


def _suelto(sql, params=()):
    """Una sentencia en su propia conexión; False si falla (por ejemplo, la columna ya existe)."""
    conn = get_db_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


def init_tablas():
    global _tablas_listas
    if _tablas_listas:
        return
    with _tablas_lock:
        if _tablas_listas:
            return
        conn = get_db_connection()
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS proveedores_usuario (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario_id INTEGER NOT NULL,
                    tipo TEXT NOT NULL DEFAULT 'centralone',
                    nombre TEXT NOT NULL DEFAULT '',
                    llave_cifrada TEXT NOT NULL,
                    prefijo TEXT DEFAULT '',
                    activo BOOLEAN DEFAULT TRUE,
                    creado_en TEXT,
                    ultima_sync TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS catalogo_proveedor_usuario (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proveedor_id INTEGER NOT NULL,
                    remote_product_id TEXT NOT NULL,
                    remote_product_name TEXT NOT NULL DEFAULT '',
                    remote_package_id TEXT NOT NULL,
                    remote_package_name TEXT NOT NULL DEFAULT '',
                    raw_json TEXT DEFAULT '{}',
                    active BOOLEAN DEFAULT TRUE,
                    updated_at TEXT
                )
            ''')
            conn.commit()
        finally:
            conn.close()
        _suelto('CREATE INDEX IF NOT EXISTS ix_proveedores_usuario_usuario ON proveedores_usuario (usuario_id)')
        _suelto('CREATE UNIQUE INDEX IF NOT EXISTS ux_catalogo_proveedor_usuario ON catalogo_proveedor_usuario '
                '(proveedor_id, remote_package_id)')
        _suelto('ALTER TABLE juegos_dinamicos ADD COLUMN usuario_id INTEGER')
        _suelto('ALTER TABLE rev_item_mapping_steps ADD COLUMN proveedor_usuario_id INTEGER')
        _suelto('CREATE INDEX IF NOT EXISTS ix_juegos_dinamicos_usuario ON juegos_dinamicos (usuario_id)')
        _tablas_listas = True


# ---------------------------------------------------------------------------
# La llave, cifrada
# ---------------------------------------------------------------------------

def _caja():
    import nacl.secret
    semilla = (os.environ.get('PROVEEDORES_USUARIO_SECRETO') or os.environ.get('SECRET_KEY') or '').encode()
    if not semilla:
        raise RuntimeError('Falta SECRET_KEY en el servidor: no se pueden guardar llaves de proveedores.')
    llave = hashlib.blake2b(semilla, digest_size=32, person=b'prov-usuario-v1').digest()
    return nacl.secret.SecretBox(llave)


def cifrar(texto):
    return base64.b64encode(bytes(_caja().encrypt(str(texto).encode()))).decode()


def descifrar(cifrado):
    return _caja().decrypt(base64.b64decode(cifrado)).decode()


def _comienzo(llave):
    """Lo que se muestra de una llave: sus primeros caracteres (co_live_ab12…)."""
    llave = str(llave or '')
    return llave[:12] if len(llave) > 16 else llave[:4]


def _ahora():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


# ---------------------------------------------------------------------------
# Conexiones
# ---------------------------------------------------------------------------

def _fila_conexion(row, n_catalogo=0):
    return {'id': row['id'], 'tipo': row['tipo'], 'tipo_nombre': TIPOS.get(row['tipo'], row['tipo']),
            'nombre': row['nombre'] or TIPOS.get(row['tipo'], row['tipo']), 'prefijo': row['prefijo'] or '',
            'activo': bool(row['activo']), 'creado_en': row['creado_en'], 'ultima_sync': row['ultima_sync'],
            'catalogo': n_catalogo}


def conexiones(uid):
    """Las conexiones del usuario (sin la llave)."""
    init_tablas()
    conn = get_db_connection()
    try:
        rows = conn.execute('SELECT * FROM proveedores_usuario WHERE usuario_id = ? ORDER BY id', (uid,)).fetchall()
        out = []
        for r in rows:
            n = conn.execute('SELECT COUNT(*) AS n FROM catalogo_proveedor_usuario WHERE proveedor_id = ? AND active = TRUE',
                             (r['id'],)).fetchone()['n']
            out.append(_fila_conexion(r, int(n or 0)))
        return out
    finally:
        conn.close()


def _conexion_de(uid, pid):
    """La fila de una conexión del usuario (con la llave cifrada) o None si no es suya."""
    init_tablas()
    conn = get_db_connection()
    try:
        return conn.execute('SELECT * FROM proveedores_usuario WHERE id = ? AND usuario_id = ?', (pid, uid)).fetchone()
    finally:
        conn.close()


def llave_de(pid):
    """La llave en claro de una conexión activa (para recargar). None si no existe o está apagada. Solo uso interno."""
    init_tablas()
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT llave_cifrada, activo FROM proveedores_usuario WHERE id = ?', (pid,)).fetchone()
    finally:
        conn.close()
    if not row or not row['activo']:
        return None
    try:
        return descifrar(row['llave_cifrada'])
    except Exception as e:
        logger.error(f'[Proveedor propio] No se pudo descifrar la llave de la conexión {pid}: {type(e).__name__}')
        return None


def dueno_de_conexion(pid):
    init_tablas()
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT usuario_id FROM proveedores_usuario WHERE id = ?', (pid,)).fetchone()
        return int(row['usuario_id']) if row else None
    finally:
        conn.close()


def agregar_conexion(uid, nombre, llave, tipo='centralone'):
    """Comprueba la llave con el proveedor (su saldo) y la guarda cifrada; después sincroniza su catálogo."""
    init_tablas()
    llave = str(llave or '').strip()
    nombre = re.sub(r'\s+', ' ', str(nombre or '')).strip()[:60]
    if tipo not in TIPOS:
        raise ValueError('Ese proveedor todavía no se puede conectar.')
    if len(llave) < 12 or len(llave) > 300:
        raise ValueError('Pega la llave completa de tu proveedor.')
    # (lo que no tiene forma de llave no se envía al proveedor: p. ej. una contraseña que el navegador pegó solo)
    if not llave.startswith('co_') or re.search(r'\s', llave):
        raise ValueError(f'Eso no es una llave de {TIPOS[tipo]}: empieza por co_live_… y la sacas de su panel.')
    if len(conexiones(uid)) >= MAX_CONEXIONES:
        raise ValueError(f'Puedes conectar hasta {MAX_CONEXIONES} proveedores.')
    with central_one.con_llave(llave):
        s = central_one.saldo()
    if s.get('error'):
        raise ValueError(f'{TIPOS[tipo]} no aceptó esa llave: {s["error"]}')
    conn = get_db_connection()
    try:
        cur = conn.execute('INSERT INTO proveedores_usuario (usuario_id, tipo, nombre, llave_cifrada, prefijo, activo, creado_en) '
                           'VALUES (?, ?, ?, ?, ?, TRUE, ?) RETURNING id',
                           (uid, tipo, nombre or TIPOS[tipo], cifrar(llave), _comienzo(llave), _ahora()))
        pid = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    res = {'id': pid, 'saldo': s}
    try:
        res['catalogo'] = sincronizar(uid, pid)
    except Exception as e:
        res['catalogo_error'] = str(e)[:200]
    return res


def saldo_de(uid, pid):
    row = _conexion_de(uid, pid)
    if not row:
        return None
    try:
        llave = descifrar(row['llave_cifrada'])
    except Exception:
        return {'error': 'No se pudo leer la llave guardada: vuelve a conectarlo.'}
    with central_one.con_llave(llave):
        return central_one.saldo()


def sincronizar(uid, pid):
    """Copia el catálogo de su proveedor a su catálogo. {'nuevos', 'actualizados', 'retirados'}"""
    row = _conexion_de(uid, pid)
    if not row:
        raise ValueError('Ese proveedor no existe.')
    with central_one.con_llave(descifrar(row['llave_cifrada'])):
        items = central_one.catalogo()
    conn = get_db_connection()
    nuevos = actualizados = 0
    vistos = set()
    try:
        for it in items:
            uuid = str(it.get('product_id') or '').strip()
            if not uuid:
                continue
            vistos.add(uuid)
            familia = str(it.get('product_family_id') or 'otros')
            prod_id = central_one.PREFIJO + familia
            prod_name = it.get('product_family_name') or familia
            raw = json.dumps(dict(it, price=central_one.precio(it)))
            activo = str(it.get('status') or 'active').lower() == 'active'
            hay = conn.execute('SELECT id FROM catalogo_proveedor_usuario WHERE proveedor_id = ? AND remote_package_id = ?',
                               (pid, uuid)).fetchone()
            if hay:
                conn.execute('UPDATE catalogo_proveedor_usuario SET remote_product_id = ?, remote_product_name = ?, '
                             'remote_package_name = ?, raw_json = ?, active = ?, updated_at = ? WHERE id = ?',
                             (prod_id, prod_name, it.get('name') or uuid, raw, activo, _ahora(), hay['id']))
                actualizados += 1
            else:
                conn.execute('INSERT INTO catalogo_proveedor_usuario (proveedor_id, remote_product_id, remote_product_name, '
                             'remote_package_id, remote_package_name, raw_json, active, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                             (pid, prod_id, prod_name, uuid, it.get('name') or uuid, raw, activo, _ahora()))
                nuevos += 1
        retirados = 0
        for r in conn.execute('SELECT id, remote_package_id FROM catalogo_proveedor_usuario WHERE proveedor_id = ? AND active = TRUE',
                              (pid,)).fetchall():
            if r['remote_package_id'] not in vistos:
                conn.execute('UPDATE catalogo_proveedor_usuario SET active = FALSE WHERE id = ?', (r['id'],))
                retirados += 1
        conn.execute('UPDATE proveedores_usuario SET ultima_sync = ? WHERE id = ?', (_ahora(), pid))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {'nuevos': nuevos, 'actualizados': actualizados, 'retirados': retirados}


def quitar_conexion(uid, pid):
    """Borra la conexión y su catálogo. Los paquetes que se recargaban con ella quedan sin proveedor (no se recargan)."""
    if not _conexion_de(uid, pid):
        raise ValueError('Ese proveedor no existe.')
    conn = get_db_connection()
    try:
        afectados = conn.execute('SELECT DISTINCT juego_id, paquete_id FROM rev_item_mapping_steps WHERE proveedor_usuario_id = ?',
                                 (pid,)).fetchall()
        for a in afectados:
            _quitar_mapeo(conn, a['juego_id'], a['paquete_id'])
        conn.execute('DELETE FROM catalogo_proveedor_usuario WHERE proveedor_id = ?', (pid,))
        conn.execute('DELETE FROM proveedores_usuario WHERE id = ? AND usuario_id = ?', (pid, uid))
        conn.commit()
        return {'paquetes_sin_proveedor': len(afectados)}
    finally:
        conn.close()


def catalogo_de(uid, pid):
    """Su catálogo agrupado por juego del proveedor: [{product_id, nombre, paquetes: [{package_id, nombre, precio, ...}]}]"""
    if not _conexion_de(uid, pid):
        return None
    conn = get_db_connection()
    try:
        rows = conn.execute('SELECT remote_product_id, remote_product_name, remote_package_id, remote_package_name, raw_json '
                            'FROM catalogo_proveedor_usuario WHERE proveedor_id = ? AND active = TRUE', (pid,)).fetchall()
    finally:
        conn.close()
    juegos = {}
    for r in rows:
        try:
            raw = json.loads(r['raw_json'] or '{}') or {}
        except ValueError:
            raw = {}
        j = juegos.setdefault(r['remote_product_id'], {'product_id': r['remote_product_id'], 'nombre': r['remote_product_name'],
                                                         'paquetes': []})
        j['paquetes'].append({'package_id': r['remote_package_id'], 'nombre': r['remote_package_name'],
                              'precio': float(raw.get('price') or 0), 'en_stock': raw.get('in_stock', True) is not False,
                              'campos': [f.get('label') or f.get('key') for f in central_one.campos_de(raw)]})
    orden = lambda texto: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(texto))]  # noqa: E731
    out = sorted(juegos.values(), key=lambda j: orden(j['nombre']))
    for j in out:
        j['paquetes'].sort(key=lambda p: orden(p['nombre']))
    return out


def item_de(pid, package_uuid):
    """Los datos guardados de un producto de su catálogo (lo que necesita Central One para armar el pedido)."""
    init_tablas()
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT raw_json FROM catalogo_proveedor_usuario WHERE proveedor_id = ? AND remote_package_id = ?',
                           (pid, str(package_uuid))).fetchone()
    finally:
        conn.close()
    try:
        return json.loads(row['raw_json'] or '{}') if row else {}
    except ValueError:
        return {}


def _linea_catalogo(conn, pid, package_uuid):
    return conn.execute('SELECT remote_product_id, remote_product_name, remote_package_name, raw_json FROM catalogo_proveedor_usuario '
                        'WHERE proveedor_id = ? AND remote_package_id = ? AND active = TRUE', (pid, str(package_uuid))).fetchone()


# ---------------------------------------------------------------------------
# Sus juegos y paquetes (en juegos_dinamicos / paquetes_dinamicos, con usuario_id)
# ---------------------------------------------------------------------------

def juego_de(uid, jid):
    """El juego si es del usuario; None si no existe o es de otro (o de la web)."""
    init_tablas()
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT * FROM juegos_dinamicos WHERE id = ? AND usuario_id = ?', (jid, uid)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def puede_usar_juego(game, uid):
    """Un juego de la web lo usa cualquiera; uno propio, solo su dueño."""
    dueno = (game or {}).get('usuario_id')
    return not dueno or (uid is not None and int(dueno) == int(uid))


def _quitar_mapeo(conn, jid, pkid):
    conn.execute('DELETE FROM rev_item_mapping_steps WHERE juego_id = ? AND paquete_id = ?', (jid, pkid))
    conn.execute('DELETE FROM rev_item_mappings WHERE juego_id = ? AND paquete_id = ?', (jid, pkid))


def _con_historial(conn, columna, valor):
    return conn.execute(f'SELECT 1 FROM transacciones_dinamicas WHERE {columna} = ? LIMIT 1', (valor,)).fetchone() is not None


def juegos(uid):
    """Sus juegos con sus paquetes y con qué se recarga cada uno."""
    init_tablas()
    conn = get_db_connection()
    try:
        out = []
        for g in conn.execute('SELECT * FROM juegos_dinamicos WHERE usuario_id = ? ORDER BY nombre', (uid,)).fetchall():
            g = dict(g)
            try:
                campos = json.loads(g.get('campos_config') or '{}') or {}
            except ValueError:
                campos = {}
            id2 = campos.get('campo_id2') or {}
            paquetes = []
            for p in conn.execute('SELECT * FROM paquetes_dinamicos WHERE juego_id = ? ORDER BY orden, id', (g['id'],)).fetchall():
                pasos = conn.execute('SELECT s.proveedor_usuario_id, s.remote_package_id, s.remote_label, s.cantidad, c.remote_package_name, '
                                     'c.remote_product_name, c.raw_json FROM rev_item_mapping_steps s LEFT JOIN catalogo_proveedor_usuario c '
                                     'ON c.proveedor_id = s.proveedor_usuario_id AND c.remote_package_id = s.remote_package_id '
                                     'WHERE s.juego_id = ? AND s.paquete_id = ? ORDER BY s.orden', (g['id'], p['id'])).fetchall()
                costo = 0.0
                mapeo = []
                for s in pasos:
                    try:
                        precio = float((json.loads(s['raw_json'] or '{}') or {}).get('price') or 0)
                    except ValueError:
                        precio = 0.0
                    costo += precio * int(s['cantidad'] or 1)
                    mapeo.append({'conexion_id': s['proveedor_usuario_id'], 'package_id': s['remote_package_id'],
                                  'nombre': s['remote_package_name'] or s['remote_label'] or s['remote_package_id'],
                                  'juego_proveedor': s['remote_product_name'] or '', 'cantidad': int(s['cantidad'] or 1),
                                  'existe': s['remote_package_name'] is not None})
                paquetes.append({'id': p['id'], 'nombre': p['nombre'], 'precio': float(p['precio'] or 0), 'activo': bool(p['activo']),
                                 'mapeo': mapeo, 'costo': round(costo, 4)})
            out.append({'id': g['id'], 'nombre': g['nombre'], 'slug': g['slug'], 'icono': g.get('icono') or '🎮',
                        'activo': bool(g['activo']), 'zona': bool(id2.get('enabled')), 'zona_label': id2.get('label') or 'Zone ID',
                        'paquetes': paquetes})
        return out
    finally:
        conn.close()


def _slug_libre(conn, nombre, uid):
    from dynamic_games import slugify
    base = f'u{uid}-{slugify(nombre)}'[:70]
    slug = base
    while conn.execute('SELECT 1 FROM juegos_dinamicos WHERE slug = ?', (slug,)).fetchone():
        slug = f'{base}-{secrets.token_hex(2)}'
    return slug


def _campos(zona, zona_label):
    campos = {'campo_id': {'label': 'ID de Jugador', 'placeholder': 'Ingresa tu ID', 'required': True, 'pattern': '', 'pattern_msg': ''}}
    if zona:
        label = re.sub(r'\s+', ' ', str(zona_label or '')).strip()[:30] or 'Zone ID'
        campos['campo_id2'] = {'enabled': True, 'label': label, 'placeholder': f'Ingresa tu {label}', 'required': True,
                               'pattern': '', 'pattern_msg': ''}
    return campos


def _nombre_valido(nombre, maximo=60):
    nombre = re.sub(r'\s+', ' ', str(nombre or '')).strip()[:maximo]
    if not nombre:
        raise ValueError('Escribe un nombre.')
    return nombre


def crear_juego(uid, nombre, icono='🎮', zona=False, zona_label=''):
    init_tablas()
    nombre = _nombre_valido(nombre)
    conn = get_db_connection()
    try:
        n = conn.execute('SELECT COUNT(*) AS n FROM juegos_dinamicos WHERE usuario_id = ?', (uid,)).fetchone()['n']
        if int(n or 0) >= MAX_JUEGOS:
            raise ValueError(f'Puedes crear hasta {MAX_JUEGOS} juegos propios.')
        cur = conn.execute('''
            INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, color_tema, icono, activo, campos_config,
                                          descripcion, ganancia_default, usa_stock_local, usuario_id)
            VALUES (?, ?, 0, 'id', '#a78bfa', ?, TRUE, ?, '', 0, FALSE, ?) RETURNING id
        ''', (nombre, _slug_libre(conn, nombre, uid), (str(icono or '').strip()[:4] or '🎮'), json.dumps(_campos(zona, zona_label)), uid))
        jid = cur.fetchone()[0]
        conn.commit()
        return jid
    finally:
        conn.close()


def editar_juego(uid, jid, datos):
    g = juego_de(uid, jid)
    if not g:
        raise ValueError('Ese juego no existe.')
    accion = datos.get('accion') or 'editar'
    conn = get_db_connection()
    try:
        if accion in ('activar', 'desactivar'):
            conn.execute('UPDATE juegos_dinamicos SET activo = ? WHERE id = ? AND usuario_id = ?', (accion == 'activar', jid, uid))
        elif accion == 'eliminar':
            if _con_historial(conn, 'juego_id', jid):
                raise ValueError('Este juego ya tiene compras: desactívalo en vez de borrarlo (así no se pierde tu historial).')
            for p in conn.execute('SELECT id FROM paquetes_dinamicos WHERE juego_id = ?', (jid,)).fetchall():
                _quitar_mapeo(conn, jid, p['id'])
            conn.execute('DELETE FROM paquetes_dinamicos WHERE juego_id = ?', (jid,))
            conn.execute('DELETE FROM juegos_dinamicos WHERE id = ? AND usuario_id = ?', (jid, uid))
        else:
            conn.execute('UPDATE juegos_dinamicos SET nombre = ?, icono = ?, campos_config = ? WHERE id = ? AND usuario_id = ?',
                         (_nombre_valido(datos.get('nombre') or g['nombre']), (str(datos.get('icono') or g.get('icono') or '🎮').strip()[:4]),
                          json.dumps(_campos(bool(datos.get('zona')), datos.get('zona_label'))), jid, uid))
        conn.commit()
    finally:
        conn.close()


def _precio_valido(precio):
    try:
        precio = round(float(precio), 2)
    except (TypeError, ValueError):
        raise ValueError('El precio debe ser un número.')
    if precio < 0 or precio > 100000:
        raise ValueError('El precio no es válido.')
    return precio


def _guardar_mapeo(conn, uid, jid, pkid, conexion_id, package_id, cantidad):
    """El paquete se recarga con ese producto de su proveedor (cantidad veces)."""
    try:
        conexion_id = int(conexion_id)
    except (TypeError, ValueError):
        raise ValueError('Elige el proveedor.')
    if not conn.execute('SELECT 1 FROM proveedores_usuario WHERE id = ? AND usuario_id = ?', (conexion_id, uid)).fetchone():
        raise ValueError('Ese proveedor no existe.')
    linea = _linea_catalogo(conn, conexion_id, package_id)
    if not linea:
        raise ValueError('Ese producto no está en el catálogo de tu proveedor (sincronízalo).')
    try:
        cantidad = max(1, min(10, int(cantidad or 1)))
    except (TypeError, ValueError):
        cantidad = 1
    etiqueta = f"{TIPOS['centralone']} · {linea['remote_product_name']} – {linea['remote_package_name']}"[:250]
    _quitar_mapeo(conn, jid, pkid)
    conn.execute('INSERT INTO rev_item_mappings (juego_id, paquete_id, remote_product_id, remote_package_id, remote_label, '
                 'auto_enabled, active) VALUES (?, ?, ?, ?, ?, TRUE, TRUE)',
                 (jid, pkid, linea['remote_product_id'], str(package_id), etiqueta))
    conn.execute('INSERT INTO rev_item_mapping_steps (juego_id, paquete_id, orden, remote_product_id, remote_package_id, remote_label, '
                 'cantidad, proveedor_usuario_id) VALUES (?, ?, 0, ?, ?, ?, ?, ?)',
                 (jid, pkid, linea['remote_product_id'], str(package_id), etiqueta, cantidad, conexion_id))


def agregar_paquete(uid, jid, datos):
    if not juego_de(uid, jid):
        raise ValueError('Ese juego no existe.')
    nombre = _nombre_valido(datos.get('nombre'))
    precio = _precio_valido(datos.get('precio', 0))
    conn = get_db_connection()
    try:
        n = conn.execute('SELECT COUNT(*) AS n FROM paquetes_dinamicos WHERE juego_id = ?', (jid,)).fetchone()['n']
        if int(n or 0) >= MAX_PAQUETES:
            raise ValueError(f'Un juego puede tener hasta {MAX_PAQUETES} paquetes.')
        cur = conn.execute('INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) VALUES (?, ?, ?, TRUE, ?) RETURNING id',
                           (jid, nombre, precio, int(n or 0)))
        pkid = cur.fetchone()[0]
        _guardar_mapeo(conn, uid, jid, pkid, datos.get('conexion_id'), datos.get('package_id'), datos.get('cantidad'))
        conn.commit()
        return pkid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _paquete_de(conn, uid, pkid):
    return conn.execute('SELECT p.* FROM paquetes_dinamicos p JOIN juegos_dinamicos g ON g.id = p.juego_id '
                        'WHERE p.id = ? AND g.usuario_id = ?', (pkid, uid)).fetchone()


def editar_paquete(uid, pkid, datos):
    conn = get_db_connection()
    try:
        p = _paquete_de(conn, uid, pkid)
        if not p:
            raise ValueError('Ese paquete no existe.')
        accion = datos.get('accion') or 'editar'
        if accion in ('activar', 'desactivar'):
            conn.execute('UPDATE paquetes_dinamicos SET activo = ? WHERE id = ?', (accion == 'activar', pkid))
        elif accion == 'eliminar':
            if _con_historial(conn, 'paquete_id', pkid):
                raise ValueError('Este paquete ya tiene compras: desactívalo en vez de borrarlo (así no se pierde tu historial).')
            _quitar_mapeo(conn, p['juego_id'], pkid)
            conn.execute('DELETE FROM paquetes_dinamicos WHERE id = ?', (pkid,))
        else:
            conn.execute('UPDATE paquetes_dinamicos SET nombre = ?, precio = ? WHERE id = ?',
                         (_nombre_valido(datos.get('nombre') or p['nombre']), _precio_valido(datos.get('precio', p['precio'])), pkid))
            if datos.get('package_id'):
                _guardar_mapeo(conn, uid, p['juego_id'], pkid, datos.get('conexion_id'), datos.get('package_id'), datos.get('cantidad'))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pantalla «Mis proveedores» (cada usuario, con su sesión)
# ---------------------------------------------------------------------------

def _uid():
    uid = session.get('user_db_id')
    return int(uid) if uid and 'usuario' in session else None


def _json_o_error(fn):
    try:
        return jsonify(ok=True, **(fn() or {}))
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)[:300]), 502
    except Exception as e:
        logger.error(f'[Proveedor propio] Error: {e}', exc_info=True)
        return jsonify(ok=False, error='No se pudo completar. Inténtalo de nuevo.'), 500


def _sesion_json():
    uid = _uid()
    if uid is None:
        return None, (jsonify(ok=False, error='Tu sesión expiró. Vuelve a iniciar sesión.'), 401)
    if request.method == 'POST' and not request.is_json:
        return None, (jsonify(ok=False, error='Se espera JSON'), 415)
    return uid, None


@bp.route('/mis-proveedores')
def pagina():
    uid = _uid()
    if uid is None:
        return redirect('/auth')
    init_tablas()
    return render_template('mis_proveedores.html', user_id=session.get('id'), max_conexiones=MAX_CONEXIONES)


@bp.route('/mis-proveedores/datos')
def datos():
    uid, err = _sesion_json()
    if err:
        return err
    return _json_o_error(lambda: {'conexiones': conexiones(uid), 'juegos': juegos(uid), 'tipos': TIPOS})


@bp.route('/mis-proveedores/conexiones', methods=['POST'])
def r_agregar_conexion():
    uid, err = _sesion_json()
    if err:
        return err
    d = request.get_json(silent=True) or {}
    return _json_o_error(lambda: agregar_conexion(uid, d.get('nombre'), d.get('llave'), d.get('tipo') or 'centralone'))


@bp.route('/mis-proveedores/conexiones/<int:pid>/saldo')
def r_saldo(pid):
    uid, err = _sesion_json()
    if err:
        return err

    def hacer():
        s = saldo_de(uid, pid)
        if s is None:
            raise ValueError('Ese proveedor no existe.')
        return {'saldo': s}
    return _json_o_error(hacer)


@bp.route('/mis-proveedores/conexiones/<int:pid>/sincronizar', methods=['POST'])
def r_sincronizar(pid):
    uid, err = _sesion_json()
    if err:
        return err
    return _json_o_error(lambda: {'catalogo': sincronizar(uid, pid)})


@bp.route('/mis-proveedores/conexiones/<int:pid>/quitar', methods=['POST'])
def r_quitar(pid):
    uid, err = _sesion_json()
    if err:
        return err
    return _json_o_error(lambda: quitar_conexion(uid, pid))


@bp.route('/mis-proveedores/conexiones/<int:pid>/catalogo')
def r_catalogo(pid):
    uid, err = _sesion_json()
    if err:
        return err

    def hacer():
        c = catalogo_de(uid, pid)
        if c is None:
            raise ValueError('Ese proveedor no existe.')
        return {'juegos': c}
    return _json_o_error(hacer)


@bp.route('/mis-proveedores/juegos', methods=['POST'])
def r_crear_juego():
    uid, err = _sesion_json()
    if err:
        return err
    d = request.get_json(silent=True) or {}
    return _json_o_error(lambda: {'id': crear_juego(uid, d.get('nombre'), d.get('icono'), bool(d.get('zona')), d.get('zona_label'))})


@bp.route('/mis-proveedores/juegos/<int:jid>', methods=['POST'])
def r_editar_juego(jid):
    uid, err = _sesion_json()
    if err:
        return err
    d = request.get_json(silent=True) or {}
    return _json_o_error(lambda: editar_juego(uid, jid, d))


@bp.route('/mis-proveedores/juegos/<int:jid>/paquetes', methods=['POST'])
def r_agregar_paquete(jid):
    uid, err = _sesion_json()
    if err:
        return err
    d = request.get_json(silent=True) or {}
    return _json_o_error(lambda: {'id': agregar_paquete(uid, jid, d)})


@bp.route('/mis-proveedores/paquetes/<int:pkid>', methods=['POST'])
def r_editar_paquete(pkid):
    uid, err = _sesion_json()
    if err:
        return err
    d = request.get_json(silent=True) or {}
    return _json_o_error(lambda: editar_paquete(uid, pkid, d))

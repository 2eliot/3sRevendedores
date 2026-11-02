"""
Ganancias (pedido del dueño, 2026-11-02): dentro del Dashboard, cada usuario ve lo que gastó, lo que vendió y lo que ganó
con sus recargas, por paquete y en total; el admin ve el global de todos (cada venta con los precios de su dueño).

  Gasto     lo que se pagó en la web por cada recarga en ese momento (transacciones_dinamicas.monto de las aprobadas).
            En los juegos con su proveedor propio (Mis proveedores) la web no cobra nada: el gasto es el «Mi costo» que
            el usuario escribe (en $), o 0 si no lo escribió.
  Venta     el precio de venta que el usuario pone a cada paquete, en $ o en Bs (queda en la moneda en que lo escribió).
            Lo que está en Bs se pasa a $ con su tasa (Bs por $).
  Ganancia  venta - gasto.

Precios, costos y tasa guardan su historial (columna «desde», UTC): cada venta usa los que había cuando se hizo, así un
cambio solo vale de ahí en adelante y los días pasados (y lo vendido antes en el mismo día) no cambian. Una venta anterior
a la primera vez que se puso un precio (o la tasa) usa ese primer valor.
"""
import bisect
import logging
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request, session

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('ganancias', __name__)

MONEDAS = ('USD', 'VES')
TIPOS = ('venta', 'costo')
HORAS_VE = 4   # Venezuela = UTC-4: el día de Venezuela empieza a las 04:00 UTC

_tablas_listas = False


def _suelto(sql):
    conn = get_db_connection()
    try:
        conn.execute(sql)
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


def init_tablas():
    global _tablas_listas
    if _tablas_listas:
        return
    conn = get_db_connection()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS ganancias_precios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                paquete_id INTEGER NOT NULL,
                tipo TEXT NOT NULL DEFAULT 'venta',
                moneda TEXT NOT NULL DEFAULT 'USD',
                monto REAL,
                desde TEXT NOT NULL
            )''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS ganancias_tasas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                tasa REAL NOT NULL,
                desde TEXT NOT NULL
            )''')
        conn.commit()
    finally:
        conn.close()
    _suelto('CREATE INDEX IF NOT EXISTS ix_ganancias_precios ON ganancias_precios (usuario_id, paquete_id, tipo, desde)')
    _suelto('CREATE INDEX IF NOT EXISTS ix_ganancias_tasas ON ganancias_tasas (usuario_id, desde)')
    _tablas_listas = True


def _ahora():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def _texto_fecha(v):
    """La fecha de la base como texto UTC comparable ('YYYY-MM-DD HH:MM:SS'): SQLite da texto, PostgreSQL datetime."""
    return str(v or '')[:19].replace('T', ' ')


def _rango_utc(inicio, fin):
    """Días de Venezuela [inicio, fin] → [desde, hasta) en UTC."""
    d0 = datetime.strptime(inicio, '%Y-%m-%d') + timedelta(hours=HORAS_VE)
    d1 = datetime.strptime(fin, '%Y-%m-%d') + timedelta(days=1, hours=HORAS_VE)
    return d0.strftime('%Y-%m-%d %H:%M:%S'), d1.strftime('%Y-%m-%d %H:%M:%S')


def _hora_ve(fecha_utc):
    try:
        return (datetime.strptime(fecha_utc, '%Y-%m-%d %H:%M:%S') - timedelta(hours=HORAS_VE)).strftime('%Y-%m-%d %H:%M')
    except ValueError:
        return fecha_utc


# ---------------------------------------------------------------------------
# Historial de precios, costos y tasa
# ---------------------------------------------------------------------------

class _Historial:
    """Valores de cada usuario ordenados por «desde»; vigente(clave, fecha) = el último con desde <= fecha (o el primero)."""

    def __init__(self):
        self._filas = {}

    def agregar(self, clave, desde, valor):
        self._filas.setdefault(clave, []).append((desde, valor))

    def ordenar(self):
        for filas in self._filas.values():
            filas.sort(key=lambda f: f[0])

    def vigente(self, clave, fecha):
        filas = self._filas.get(clave)
        if not filas:
            return None
        i = bisect.bisect_right([f[0] for f in filas], fecha) - 1
        return filas[max(i, 0)][1]

    def actual(self, clave):
        filas = self._filas.get(clave)
        return filas[-1][1] if filas else None


def _cargar_historial(conn, uids):
    precios, tasas = _Historial(), _Historial()
    if uids is not None and not uids:
        return precios, tasas
    filtro, params = ('', ()) if uids is None else (f' WHERE usuario_id IN ({",".join("?" for _ in uids)})', tuple(uids))
    for r in conn.execute(f'SELECT usuario_id, paquete_id, tipo, moneda, monto, desde FROM ganancias_precios{filtro}', params).fetchall():
        valor = None if r['monto'] is None else (r['moneda'], float(r['monto']))
        precios.agregar((int(r['usuario_id']), int(r['paquete_id']), r['tipo']), _texto_fecha(r['desde']), valor)
    for r in conn.execute(f'SELECT usuario_id, tasa, desde FROM ganancias_tasas{filtro}', params).fetchall():
        tasas.agregar(int(r['usuario_id']), _texto_fecha(r['desde']), float(r['tasa']))
    precios.ordenar()
    tasas.ordenar()
    return precios, tasas


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------

def _venta(precio, tasa):
    """(venta en $, venta en Bs) de una unidad; None donde no se puede (sin precio, o en Bs sin tasa)."""
    if not precio:
        return None, None
    moneda, monto = precio
    if moneda == 'VES':
        return (monto / tasa if tasa else None), monto
    return monto, (monto * tasa if tasa else None)


def calcular(conn, inicio, fin, usuario_id=None, paquete_id=None, con_detalle=False):
    """Ganancias de las recargas aprobadas entre inicio y fin (días de Venezuela). usuario_id=None → todos (el admin)."""
    init_tablas()
    desde, hasta = _rango_utc(inicio, fin)
    sql = ('SELECT t.id, t.usuario_id, t.paquete_id, t.monto, t.fecha, p.nombre AS paquete, g.nombre AS juego, '
           'g.usuario_id AS juego_dueno, u.nombre AS u_nombre, u.apellido AS u_apellido, u.correo AS u_correo '
           'FROM transacciones_dinamicas t LEFT JOIN paquetes_dinamicos p ON p.id = t.paquete_id '
           'LEFT JOIN juegos_dinamicos g ON g.id = t.juego_id LEFT JOIN usuarios u ON u.id = t.usuario_id '
           "WHERE t.estado = 'aprobado' AND t.fecha >= ? AND t.fecha < ?")
    params = [desde, hasta]
    if usuario_id is not None:
        sql += ' AND t.usuario_id = ?'
        params.append(int(usuario_id))
    if paquete_id is not None:
        sql += ' AND t.paquete_id = ?'
        params.append(int(paquete_id))
    ventas = conn.execute(sql + ' ORDER BY t.fecha, t.id', tuple(params)).fetchall()
    precios, tasas = _cargar_historial(conn, sorted({int(v['usuario_id']) for v in ventas}))

    paquetes, detalle = {}, []
    total = {'vendidos': 0, 'gasto': 0.0, 'venta': 0.0, 'venta_bs': 0.0, 'ganancia': 0.0, 'ganancia_bs': 0.0, 'sin_precio': 0}
    for v in ventas:
        uid, pid, fecha = int(v['usuario_id']), int(v['paquete_id']), _texto_fecha(v['fecha'])
        tasa = tasas.vigente(uid, fecha)
        propio = bool(v['juego_dueno'])
        if propio:   # con su proveedor: la web no cobró nada; el gasto es «Mi costo» (en $)
            costo = precios.vigente((uid, pid, 'costo'), fecha)
            gasto = costo[1] if costo else 0.0
        else:
            gasto = float(v['monto'] or 0)
        venta, venta_bs = _venta(precios.vigente((uid, pid, 'venta'), fecha), tasa)
        ganancia = None if venta is None else venta - gasto
        ganancia_bs = None if ganancia is None or not tasa else ganancia * tasa

        fila = paquetes.setdefault(pid, {'paquete_id': pid, 'juego': v['juego'] or '', 'paquete': v['paquete'] or f'#{pid}',
                                         'propio': propio, 'vendidos': 0, 'gasto': 0.0, 'venta': 0.0, 'venta_bs': 0.0,
                                         'ganancia': 0.0, 'ganancia_bs': 0.0, 'sin_precio': 0})
        for d in (fila, total):
            d['vendidos'] += 1
            d['gasto'] += gasto
            if ganancia is None:
                d['sin_precio'] += 1
            else:
                d['venta'] += venta
                d['ganancia'] += ganancia
                d['venta_bs'] += venta_bs or 0.0
                d['ganancia_bs'] += ganancia_bs or 0.0
        if con_detalle:
            detalle.append({'id': v['id'], 'hora': _hora_ve(fecha), 'usuario_id': uid,
                            'usuario': ' '.join(x for x in (v['u_nombre'], v['u_apellido']) if x) or v['u_correo'] or '',
                            'paquete': fila['paquete'], 'gasto': round(gasto, 4), 'tasa': tasa,
                            'venta': None if venta is None else round(venta, 4),
                            'venta_bs': None if venta_bs is None else round(venta_bs, 2),
                            'ganancia': None if ganancia is None else round(ganancia, 4),
                            'ganancia_bs': None if ganancia_bs is None else round(ganancia_bs, 2)})

    def redondear(d):
        for k in ('gasto', 'venta', 'ganancia'):
            d[k] = round(d[k], 4)
        for k in ('venta_bs', 'ganancia_bs'):
            d[k] = round(d[k], 2)
        return d
    filas = sorted((redondear(f) for f in paquetes.values()), key=lambda f: (f['juego'].lower(), -f['vendidos'], f['paquete']))
    res = {'paquetes': filas, 'total': redondear(total), 'inicio': inicio, 'fin': fin}
    if con_detalle:
        res['ventas'] = detalle
    return res


# ---------------------------------------------------------------------------
# Precios de venta, costos y tasa de un usuario
# ---------------------------------------------------------------------------

def _juegos_visibles(uid):
    """Juegos que el usuario puede comprar (los de la web y los suyos), con sus paquetes activos."""
    import dynamic_games
    conn = get_db_connection()
    try:
        out = []
        for g in dynamic_games.juegos_para(uid, only_active=True):
            pkgs = conn.execute('SELECT id, nombre, precio FROM paquetes_dinamicos WHERE juego_id = ? AND activo = TRUE ORDER BY orden, id',
                                (g['id'],)).fetchall()
            if pkgs:
                out.append({'id': g['id'], 'nombre': g['nombre'], 'propio': bool(g.get('usuario_id')),
                            'paquetes': [{'id': p['id'], 'nombre': p['nombre'], 'precio_web': float(p['precio'] or 0)} for p in pkgs]})
        return out
    finally:
        conn.close()


def mis_precios(uid):
    init_tablas()
    juegos = _juegos_visibles(uid)
    conn = get_db_connection()
    try:
        precios, tasas = _cargar_historial(conn, [uid])
    finally:
        conn.close()
    for j in juegos:
        for p in j['paquetes']:
            venta = precios.actual((uid, p['id'], 'venta'))
            p['venta'] = {'moneda': venta[0], 'monto': venta[1]} if venta else None
            if j['propio']:
                costo = precios.actual((uid, p['id'], 'costo'))
                p['costo'] = costo[1] if costo else None
    return {'tasa': tasas.actual(uid), 'juegos': juegos}


def _paquete_permitido(uid, paquete_id, tipo):
    for j in _juegos_visibles(uid):
        for p in j['paquetes']:
            if p['id'] == paquete_id:
                return tipo == 'venta' or j['propio']
    return False


def guardar_precio(uid, paquete_id, tipo, moneda, monto):
    """Un precio de venta (en $ o Bs) o un costo (en $, solo paquetes con su proveedor). monto None = quitarlo. Desde ahora."""
    init_tablas()
    if tipo not in TIPOS:
        raise ValueError('Tipo no válido')
    moneda = 'USD' if tipo == 'costo' else (moneda if moneda in MONEDAS else 'USD')
    try:
        paquete_id = int(paquete_id)
    except (TypeError, ValueError):
        raise ValueError('Paquete no válido')
    if monto in (None, ''):
        monto = None
    else:
        try:
            monto = round(float(str(monto).replace(',', '.')), 4)
        except ValueError:
            raise ValueError('Escribe un número')
        if monto < 0 or monto > 100000000:
            raise ValueError('El monto no es válido')
    if not _paquete_permitido(uid, paquete_id, tipo):
        raise ValueError('Ese paquete no existe')
    conn = get_db_connection()
    try:
        ultimo = conn.execute('SELECT moneda, monto FROM ganancias_precios WHERE usuario_id = ? AND paquete_id = ? AND tipo = ? '
                              'ORDER BY desde DESC, id DESC LIMIT 1', (uid, paquete_id, tipo)).fetchone()
        igual = ultimo and ultimo['moneda'] == moneda and (
            (ultimo['monto'] is None and monto is None) or
            (ultimo['monto'] is not None and monto is not None and abs(float(ultimo['monto']) - monto) < 1e-9))
        if not igual and (ultimo or monto is not None):
            conn.execute('INSERT INTO ganancias_precios (usuario_id, paquete_id, tipo, moneda, monto, desde) VALUES (?, ?, ?, ?, ?, ?)',
                         (uid, paquete_id, tipo, moneda, monto, _ahora()))
            conn.commit()
    finally:
        conn.close()


def guardar_tasa(uid, tasa):
    init_tablas()
    try:
        tasa = round(float(str(tasa).replace(',', '.')), 4)
    except (TypeError, ValueError):
        raise ValueError('Escribe la tasa en Bs por $')
    if tasa <= 0 or tasa > 100000000:
        raise ValueError('La tasa no es válida')
    conn = get_db_connection()
    try:
        ultima = conn.execute('SELECT tasa FROM ganancias_tasas WHERE usuario_id = ? ORDER BY desde DESC, id DESC LIMIT 1', (uid,)).fetchone()
        if not ultima or abs(float(ultima['tasa']) - tasa) > 1e-9:
            conn.execute('INSERT INTO ganancias_tasas (usuario_id, tasa, desde) VALUES (?, ?, ?)', (uid, tasa, _ahora()))
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Rutas (con la sesión de la web)
# ---------------------------------------------------------------------------

def _uid():
    uid = session.get('user_db_id')
    return int(uid) if uid and 'usuario' in session else None


def _respuesta(fn):
    try:
        return jsonify(ok=True, **(fn() or {}))
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    except Exception as e:
        logger.error(f'[Ganancias] Error: {e}', exc_info=True)
        return jsonify(ok=False, error='No se pudo completar. Inténtalo de nuevo.'), 500


def _fecha(valor, defecto):
    try:
        return datetime.strptime(str(valor), '%Y-%m-%d').strftime('%Y-%m-%d')
    except (TypeError, ValueError):
        return defecto


@bp.route('/ganancias/datos')
def r_datos():
    uid = _uid()
    if uid is None:
        return jsonify(ok=False, error='Tu sesión expiró. Vuelve a iniciar sesión.'), 401
    hoy = (datetime.utcnow() - timedelta(hours=HORAS_VE)).strftime('%Y-%m-%d')
    fin = _fecha(request.args.get('fin'), hoy)
    inicio = _fecha(request.args.get('inicio'), fin)
    if inicio > fin:
        inicio, fin = fin, inicio
    admin = bool(session.get('is_admin'))
    quien = request.args.get('usuario') or ('todos' if admin else 'yo')

    def hacer():
        # el usuario solo ve lo suyo; el admin, todo o un usuario
        try:
            objetivo = uid if (not admin or quien == 'yo') else (None if quien == 'todos' else int(quien))
            paquete = int(request.args['paquete']) if request.args.get('paquete') else None
        except ValueError:
            raise ValueError('Filtro no válido')
        conn = get_db_connection()
        try:
            res = calcular(conn, inicio, fin, objetivo, paquete, con_detalle=paquete is not None)
            if admin:
                d, h = _rango_utc(inicio, fin)
                res['usuarios'] = [{'id': r['id'], 'nombre': ' '.join(x for x in (r['nombre'], r['apellido']) if x) or r['correo']}
                                   for r in conn.execute("SELECT DISTINCT u.id, u.nombre, u.apellido, u.correo FROM transacciones_dinamicas t "
                                                         "JOIN usuarios u ON u.id = t.usuario_id WHERE t.estado = 'aprobado' "
                                                         'AND t.fecha >= ? AND t.fecha < ? ORDER BY u.nombre', (d, h)).fetchall()]
        finally:
            conn.close()
        res.update(admin=admin, quien='yo' if objetivo == uid else ('todos' if objetivo is None else str(objetivo)),
                   puede_editar=objetivo == uid)
        return res
    return _respuesta(hacer)


@bp.route('/ganancias/precios')
def r_precios():
    uid = _uid()
    if uid is None:
        return jsonify(ok=False, error='Tu sesión expiró. Vuelve a iniciar sesión.'), 401
    return _respuesta(lambda: mis_precios(uid))


@bp.route('/ganancias/precio', methods=['POST'])
def r_precio():
    uid = _uid()
    if uid is None:
        return jsonify(ok=False, error='Tu sesión expiró. Vuelve a iniciar sesión.'), 401
    d = request.get_json(silent=True) or {}
    return _respuesta(lambda: guardar_precio(uid, d.get('paquete_id'), d.get('tipo') or 'venta', d.get('moneda'), d.get('monto')))


@bp.route('/ganancias/tasa', methods=['POST'])
def r_tasa():
    uid = _uid()
    if uid is None:
        return jsonify(ok=False, error='Tu sesión expiró. Vuelve a iniciar sesión.'), 401
    d = request.get_json(silent=True) or {}
    return _respuesta(lambda: guardar_tasa(uid, d.get('tasa')))

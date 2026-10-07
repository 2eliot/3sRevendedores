"""
Bot de Free Fire (Blueprint Flask) — respaldo del proveedor para el juego de Free Fire.

Canjea PINs de Free Fire (tabla pines_freefire_global, una fila por PIN y monto_id =
denominación) en el ID del jugador a través del VPS (redeem_hype_vps.redeem_pin_vps).

No tiene precios: el cliente paga el precio del paquete del juego creado por el admin.
Si el bot está activo y un paquete de ese juego está vinculado a una denominación,
la compra intenta primero con PINs del bot y, si no hay o el canje falla sin entregar
nada, sigue con el proveedor (Mapeo) como siempre.

Configuración (configuracion_redeemer, clave 'bot_ff_config', JSON):
    {"activo": bool, "juego_id": int|null,
     "paquetes": {"<paquete_id>": {"monto_id": int, "cantidad": int}}}
"""
import json
import logging
import re
import threading
from datetime import datetime

from flask import Blueprint, flash, jsonify, redirect, render_template, request, session

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('bot_freefire', __name__)

NOMBRE = 'Bot de Free Fire'
CONFIG_KEY = 'bot_ff_config'
LOG_KEY = 'bot_ff_log'
DUDOSOS_KEY = 'bot_ff_dudosos'
LOG_MAX = 60
CANTIDAD_MAX = 10
_lock = threading.Lock()

# Fallos en los que es seguro que el PIN NO se usó: vuelve al stock.
# Cualquier otro fallo (sin respuesta, PIN rechazado, error del VPS…) aparta el PIN para revisión.
_PIN_INTACTO = ('no se pudo conectar', 'player id', 'id de jugador', 'id inválido', 'jugador no encontrado')


def _ahora():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def _leer(clave, default):
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT valor FROM configuracion_redeemer WHERE clave = ?', (clave,)).fetchone()
        return json.loads(row['valor']) if row and row['valor'] else default
    except Exception:
        return default
    finally:
        conn.close()


def _guardar(clave, valor):
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion) VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion",
            (clave, json.dumps(valor)))
        conn.commit()
    finally:
        conn.close()


def get_config():
    cfg = _leer(CONFIG_KEY, {}) or {}
    return {'activo': bool(cfg.get('activo')), 'juego_id': cfg.get('juego_id'),
            'paquetes': cfg.get('paquetes') or {}}


def denominaciones():
    """[{monto_id, nombre, stock}] de los PINs del bot (sin precios)."""
    conn = get_db_connection()
    try:
        rows = conn.execute('SELECT id, nombre FROM precios_freefire_global ORDER BY id').fetchall()
        stock = {r['monto_id']: r['n'] for r in conn.execute(
            'SELECT monto_id, COUNT(*) AS n FROM pines_freefire_global WHERE usado = FALSE GROUP BY monto_id').fetchall()}
    finally:
        conn.close()
    return [{'monto_id': r['id'], 'nombre': r['nombre'], 'stock': int(stock.get(r['id'], 0))} for r in rows]


def plan_para(juego_id, paquete_id):
    """{'monto_id', 'cantidad'} si el bot debe intentar este paquete; None si no aplica."""
    try:
        cfg = get_config()
    except Exception:
        return None
    if not cfg['activo'] or not cfg['juego_id'] or int(cfg['juego_id']) != int(juego_id):
        return None
    p = cfg['paquetes'].get(str(paquete_id))
    if not p or not p.get('monto_id'):
        return None
    return {'monto_id': int(p['monto_id']), 'cantidad': max(1, min(CANTIDAD_MAX, int(p.get('cantidad') or 1)))}


def hay_stock(plan):
    conn = get_db_connection()
    try:
        n = conn.execute('SELECT COUNT(*) AS n FROM pines_freefire_global WHERE monto_id = ? AND usado = FALSE',
                         (plan['monto_id'],)).fetchone()['n']
    finally:
        conn.close()
    return int(n) >= plan['cantidad']


# ---------------------------------------------------------------------------
# PINs
# ---------------------------------------------------------------------------

def _sacar_pin(monto_id):
    """Saca un PIN del stock de forma atómica (dos compras nunca reciben el mismo)."""
    for _ in range(5):
        conn = get_db_connection()
        try:
            row = conn.execute('SELECT id, pin_codigo FROM pines_freefire_global WHERE monto_id = ? AND usado = FALSE '
                               'ORDER BY id LIMIT 1', (monto_id,)).fetchone()
            if not row:
                return None
            cur = conn.execute('DELETE FROM pines_freefire_global WHERE id = ? AND usado = FALSE', (row['id'],))
            conn.commit()
            if cur.rowcount == 1:
                return row['pin_codigo']
        finally:
            conn.close()
    return None


def _devolver_pin(monto_id, pin):
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado) VALUES (?, ?, FALSE)', (monto_id, pin))
        conn.commit()
    except Exception as e:
        logger.error(f'[Bot FF] No se pudo devolver un PIN al stock: {e}')
    finally:
        conn.close()


def _registrar(entrada, dudoso=None):
    with _lock:
        log = _leer(LOG_KEY, []) or []
        log.insert(0, dict(entrada, fecha=_ahora()))
        _guardar(LOG_KEY, log[:LOG_MAX])
        if dudoso:
            dud = _leer(DUDOSOS_KEY, []) or []
            dud.append(dict(dudoso, fecha=_ahora()))
            _guardar(DUDOSOS_KEY, dud)


def _canjear(pin, player_id, ref):
    from pin_redeemer import get_redeemer_config_from_db
    from redeem_hype_vps import redeem_pin_vps
    cfg = get_redeemer_config_from_db(get_db_connection)
    return redeem_pin_vps(pin, player_id, cfg, request_id=ref)


def intentar_bot(plan, player_id, ref):
    """Canjea `cantidad` PINs en orden. Nunca lanza.

    Devuelve {'usado': bool, 'ok': n, 'total': N, 'name': str, 'err': str}.
    'usado' es False si no entregó nada (sin stock o primer canje fallido): se sigue con el proveedor.
    """
    total, monto_id = plan['cantidad'], plan['monto_id']
    player_id = str(player_id or '').strip()
    res = {'usado': False, 'ok': 0, 'total': total, 'name': '', 'err': ''}
    if not player_id.isdigit():
        res['err'] = 'El bot solo recarga IDs numéricos'
        return res
    pines = []
    for _ in range(total):
        pin = _sacar_pin(monto_id)
        if not pin:
            break
        pines.append(pin)
    if len(pines) < total:
        for pin in pines:
            _devolver_pin(monto_id, pin)
        res['err'] = 'Sin PINs suficientes en el bot'
        return res

    for i, pin in enumerate(pines):
        try:
            r = _canjear(pin, player_id, f'{ref}-B{i + 1}')
            ok, msg, name = bool(r and r.success), (r.message if r else 'Sin respuesta del VPS'), (r.player_name if r else '')
        except Exception as e:
            ok, msg, name = False, f'Error de conexión: {e}', ''
        if ok:
            res['ok'] += 1
            res['name'] = res['name'] or (name or '')
            _registrar({'ref': ref, 'player_id': player_id, 'monto_id': monto_id, 'resultado': 'ok', 'mensaje': msg or ''})
            continue
        # Fallo: devolver los PINs no intentados
        for resto in pines[i + 1:]:
            _devolver_pin(monto_id, resto)
        m = (msg or '').lower()
        if any(k in m for k in _PIN_INTACTO):
            _devolver_pin(monto_id, pin)
            _registrar({'ref': ref, 'player_id': player_id, 'monto_id': monto_id, 'resultado': 'fallo', 'mensaje': msg})
        else:
            # Estado del PIN desconocido: no vuelve al stock, queda apartado para que el admin decida
            _registrar({'ref': ref, 'player_id': player_id, 'monto_id': monto_id, 'resultado': 'apartado', 'mensaje': msg},
                       dudoso={'pin': pin, 'monto_id': monto_id, 'player_id': player_id, 'ref': ref, 'mensaje': msg})
        res['err'] = msg or 'El canje con el bot falló'
        break
    res['usado'] = res['ok'] > 0
    logger.info(f"[Bot FF] {ref} {res['ok']}/{total} PINs canjeados" + (f" — {res['err']}" if res['err'] else ''))
    return res


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

def _juegos_free_fire():
    from dynamic_games import get_all_dynamic_games
    return [g for g in get_all_dynamic_games() if (g.get('modo') or 'id') == 'id']


def _solo_admin_json():
    if not session.get('is_admin'):
        return jsonify(ok=False, error='Acceso denegado'), 403
    if request.method == 'POST' and not request.is_json:
        return jsonify(ok=False, error='Se espera JSON'), 415
    return None


@bp.route('/admin/bot-freefire')
def admin_bot():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    juegos = _juegos_free_fire()
    cfg = get_config()
    if not cfg['juego_id']:
        sugerido = next((g for g in juegos if 'freefire' in re.sub(r'[^a-z]', '', g['nombre'].lower())), None)
        cfg['juego_id'] = sugerido['id'] if sugerido else None
    return render_template('admin_bot_freefire.html', nombre=NOMBRE, juegos=juegos, cfg=cfg)


@bp.route('/admin/bot-freefire/datos')
def admin_bot_datos():
    err = _solo_admin_json()
    if err:
        return err
    from dynamic_games import get_dynamic_packages
    try:
        juego_id = int(request.args.get('juego_id') or 0)
    except ValueError:
        juego_id = 0
    paquetes = [{'id': p['id'], 'nombre': p['nombre'], 'activo': bool(p['activo'])}
                for p in get_dynamic_packages(juego_id)] if juego_id else []
    return jsonify(ok=True, config=get_config(), denominaciones=denominaciones(), paquetes=paquetes,
                   log=_leer(LOG_KEY, []) or [], dudosos=_leer(DUDOSOS_KEY, []) or [])


@bp.route('/admin/bot-freefire/config', methods=['POST'])
def admin_bot_config():
    err = _solo_admin_json()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    validos = {d['monto_id'] for d in denominaciones()}
    paquetes = {}
    for pid, v in (data.get('paquetes') or {}).items():
        try:
            monto_id, cantidad = int((v or {}).get('monto_id') or 0), int((v or {}).get('cantidad') or 1)
        except (TypeError, ValueError):
            continue
        if monto_id in validos and str(pid).isdigit():
            paquetes[str(pid)] = {'monto_id': monto_id, 'cantidad': max(1, min(CANTIDAD_MAX, cantidad))}
    try:
        juego_id = int(data.get('juego_id')) if data.get('juego_id') else None
    except (TypeError, ValueError):
        juego_id = None
    cfg = {'activo': bool(data.get('activo')), 'juego_id': juego_id, 'paquetes': paquetes}
    _guardar(CONFIG_KEY, cfg)
    return jsonify(ok=True, config=cfg)


@bp.route('/admin/bot-freefire/recarga-manual', methods=['POST'])
def admin_bot_manual():
    """Recarga manual del admin con un PIN del bot (no cobra saldo)."""
    err = _solo_admin_json()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    try:
        monto_id = int(data.get('monto_id'))
    except (TypeError, ValueError):
        return jsonify(ok=False, error='Elige la denominación'), 400
    player_id = str(data.get('player_id') or '').strip()
    if not player_id.isdigit():
        return jsonify(ok=False, error='Escribe un ID de jugador numérico'), 400
    ref = 'MANUAL-' + datetime.now().strftime('%H%M%S')
    r = intentar_bot({'monto_id': monto_id, 'cantidad': 1}, player_id, ref)
    if r['ok']:
        return jsonify(ok=True, mensaje='Recarga hecha' + (f" para {r['name']}" if r['name'] else ''))
    return jsonify(ok=False, error=r['err'] or 'No se pudo recargar')


@bp.route('/admin/bot-freefire/dudosos', methods=['POST'])
def admin_bot_dudosos():
    """Resuelve un PIN dudoso: 'devolver' al stock o 'descartar'."""
    err = _solo_admin_json()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    pin, accion = str(data.get('pin') or ''), data.get('accion')
    with _lock:
        dud = _leer(DUDOSOS_KEY, []) or []
        item = next((d for d in dud if d.get('pin') == pin), None)
        if not item:
            return jsonify(ok=False, error='Ese PIN ya no está en la lista'), 404
        dud = [d for d in dud if d.get('pin') != pin]
        _guardar(DUDOSOS_KEY, dud)
    if accion == 'devolver':
        _devolver_pin(int(item['monto_id']), pin)
        return jsonify(ok=True, mensaje='PIN devuelto al stock')
    return jsonify(ok=True, mensaje='PIN descartado')

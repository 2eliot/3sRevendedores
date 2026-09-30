"""
Verificación de ID de jugador — API configurable por juego (Blueprint Flask).

Cada juego por ID (Free Fire ID y juegos dinámicos en modo 'id') puede tener su
propia API de verificación, configurada desde /admin/verificacion-ids y guardada
en configuracion_redeemer con la clave 'verify_api:<game_key>':
    game_key = 'freefire_id'  |  'dyn_<slug>'

Configuración (JSON):
    enabled     bool
    method      'GET' | 'POST'
    url         p.ej. https://api.proveedor.com/check?uid={player_id}&zone={player_id2}
    headers     objeto JSON, p.ej. {"X-API-Key": "..."}
    body        (POST) plantilla JSON, p.ej. {"id": "{player_id}", "server": "{servidor}"}
    name_path   ruta al nombre en la respuesta, p.ej. data.nickname  (índices: data.0.name)
    error_path  (opcional) ruta a un mensaje de error del proveedor

La API se llama desde el servidor (/api/verificar-id): las claves nunca llegan al navegador.

Proveedor Inefable (preconfigurado): basta con guardar la API key una vez
(clave 'verify_inefable'); cada juego usa cfg = {enabled, provider: 'inefable',
inefable_game: 'freefire' | 'Inefable-bloodstriker' | 'Inefable-mobilelegends'}.
"""
import json
import logging
import os
import re
import threading
import time
import urllib.parse

import requests
from flask import Blueprint, jsonify, redirect, render_template, request, session, flash

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('id_verify', __name__)

KEY_PREFIX = 'verify_api:'
TIMEOUT_S = 10
PLAYER_RE = re.compile(r'^[A-Za-z0-9_\-.#]{1,40}$')
RATE_MAX, RATE_WINDOW = 20, 300  # 20 verificaciones cada 5 min por usuario
_rate = {}
_rate_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def get_verify_config(game_key):
    try:
        conn = get_db_connection()
        row = conn.execute('SELECT valor FROM configuracion_redeemer WHERE clave = ?',
                           (KEY_PREFIX + game_key,)).fetchone()
        conn.close()
        return json.loads(row['valor']) if row and row['valor'] else {}
    except Exception as e:
        logger.warning(f'[VerifyID] No se pudo leer config de {game_key}: {e}')
        return {}


def set_verify_config(game_key, cfg):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion) "
        "VALUES (?, ?, datetime('now')) "
        "ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion",
        (KEY_PREFIX + game_key, json.dumps(cfg))
    )
    conn.commit()
    conn.close()


# --- Inefable -------------------------------------------------------------
INEFABLE_KEY = 'verify_inefable'
INEFABLE_URL = os.environ.get('INEFABLE_VERIFY_URL', 'https://inefablerevendedores.co/api/v1/verify-name')
INEFABLE_GAMES = [
    ('freefire', 'Free Fire'),
    ('Inefable-bloodstriker', 'Blood Strike'),
    ('Inefable-mobilelegends', 'Mobile Legends (requiere Zone ID)'),
]
INEFABLE_GAME_CODES = {c for c, _ in INEFABLE_GAMES}


def get_inefable_key():
    try:
        conn = get_db_connection()
        row = conn.execute('SELECT valor FROM configuracion_redeemer WHERE clave = ?', (INEFABLE_KEY,)).fetchone()
        conn.close()
        return (json.loads(row['valor']).get('api_key') or '').strip() if row and row['valor'] else ''
    except Exception as e:
        logger.warning(f'[VerifyID] No se pudo leer la API key de Inefable: {e}')
        return ''


def set_inefable_key(api_key):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion) "
        "VALUES (?, ?, datetime('now')) "
        "ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion",
        (INEFABLE_KEY, json.dumps({'api_key': api_key}))
    )
    conn.commit()
    conn.close()


def guess_inefable_game(nombre):
    """Juego de Inefable que corresponde a un nombre de juego local (o None)."""
    n = re.sub(r'[^a-z]', '', (nombre or '').lower())
    if 'freefire' in n:
        return 'freefire'
    if 'bloodstrike' in n:
        return 'Inefable-bloodstriker'
    if 'mobilelegend' in n:
        return 'Inefable-mobilelegends'
    return None


def cfg_ready(cfg):
    """La configuración tiene lo necesario para verificar (activa o no)."""
    if cfg.get('provider') == 'inefable':
        return bool(cfg.get('inefable_game') in INEFABLE_GAME_CODES and get_inefable_key())
    return bool(cfg.get('url') and cfg.get('name_path'))


def verify_api_enabled(game_key):
    cfg = get_verify_config(game_key)
    return bool(cfg.get('enabled') and cfg_ready(cfg))


def list_id_games():
    """Juegos por ID configurables: Free Fire ID + dinámicos en modo 'id'."""
    games = [{'key': 'freefire_id', 'nombre': 'Bot Free Fire', 'dual': False, 'servidor': False}]
    try:
        from dynamic_games import get_all_dynamic_games, parse_campos_config
        for g in get_all_dynamic_games():
            if (g.get('modo') or 'id') != 'id':
                continue
            campos = parse_campos_config(g)
            games.append({
                'key': f"dyn_{g['slug']}", 'nombre': g['nombre'],
                'dual': bool((campos.get('campo_id2') or {}).get('enabled')),
                'servidor': bool((campos.get('servidor') or {}).get('enabled')),
            })
    except Exception as e:
        logger.warning(f'[VerifyID] No se pudieron listar juegos dinámicos: {e}')
    return games


# ---------------------------------------------------------------------------
# Llamada a la API del proveedor
# ---------------------------------------------------------------------------

def _dig(data, path):
    cur = data
    for part in [p for p in (path or '').split('.') if p]:
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _fill(template, values, encode):
    def rep(m):
        v = values.get(m.group(1), '')
        return encode(v)
    return re.sub(r'\{(player_id2|player_id|servidor)\}', rep, template or '')


INEFABLE_ERRORES = {
    400: 'Datos incompletos: revisa el ID (Mobile Legends necesita también el Zone ID)',
    401: 'La API key de Inefable no es válida',
    403: 'La verificación de nombres no está activa en tu cuenta de Inefable',
    404: 'ID de jugador no encontrado',
    429: 'Demasiadas verificaciones seguidas. Espera un minuto.',
    503: 'El servicio de verificación no está disponible. Inténtalo más tarde.',
}


def call_inefable(cfg, player_id, player_id2=''):
    api_key = get_inefable_key()
    game = cfg.get('inefable_game')
    if not api_key:
        return False, 'Falta la API key de Inefable'
    if game not in INEFABLE_GAME_CODES:
        return False, 'Juego de Inefable no configurado'
    if game == 'Inefable-mobilelegends' and not player_id2:
        return False, 'Escribe también el Zone ID'
    payload = {'game': game, 'player_id': player_id}
    if player_id2:
        payload['player_id2'] = player_id2
    headers = {'X-API-Key': api_key, 'Authorization': f'Bearer {api_key}',
               'Accept': 'application/json', 'User-Agent': '3sRevendedores-VerifyID/1.0'}
    try:
        resp = requests.post(INEFABLE_URL, json=payload, headers=headers, timeout=TIMEOUT_S)
    except requests.Timeout:
        return False, 'El proveedor tardó demasiado en responder'
    except requests.RequestException as e:
        logger.warning(f'[VerifyID] Error de conexión con Inefable: {e}')
        return False, 'No se pudo conectar con el proveedor'
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if resp.ok and isinstance(data, dict) and data.get('ok') and data.get('player_name'):
        return True, str(data['player_name'])[:60]
    if resp.status_code in INEFABLE_ERRORES:
        return False, INEFABLE_ERRORES[resp.status_code]
    if resp.ok:
        return False, 'ID de jugador no encontrado'
    return False, f'El proveedor respondió con error ({resp.status_code})'


def call_verify_api(cfg, player_id, player_id2='', servidor=''):
    """Devuelve (ok, nombre_o_mensaje)."""
    if cfg.get('provider') == 'inefable':
        return call_inefable(cfg, player_id, player_id2)
    values = {'player_id': player_id, 'player_id2': player_id2, 'servidor': servidor}
    url = _fill(cfg.get('url', ''), values, lambda v: urllib.parse.quote(str(v), safe=''))
    if not url.lower().startswith(('http://', 'https://')):
        return False, 'La URL de verificación no es válida'
    headers = {'Accept': 'application/json', 'User-Agent': '3sRevendedores-VerifyID/1.0'}
    try:
        headers.update({str(k): str(v) for k, v in (json.loads(cfg.get('headers') or '{}') or {}).items()})
    except Exception:
        return False, 'Las cabeceras configuradas no son JSON válido'
    method = (cfg.get('method') or 'GET').upper()
    try:
        if method == 'POST':
            body = _fill(cfg.get('body') or '{}', values, lambda v: json.dumps(str(v))[1:-1])
            try:
                payload = json.loads(body)
            except Exception:
                return False, 'El cuerpo configurado no es JSON válido'
            resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT_S)
        else:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT_S)
    except requests.Timeout:
        return False, 'El proveedor tardó demasiado en responder'
    except requests.RequestException as e:
        logger.warning(f'[VerifyID] Error de conexión: {e}')
        return False, 'No se pudo conectar con el proveedor'

    try:
        data = resp.json()
    except ValueError:
        data = None
    name = _dig(data, cfg.get('name_path')) if data is not None else None
    if resp.ok and name not in (None, '', [], {}):
        return True, str(name)[:60]
    msg = _dig(data, cfg.get('error_path')) if data is not None and cfg.get('error_path') else None
    if msg:
        return False, str(msg)[:120]
    if resp.status_code in (400, 404, 422) or resp.ok:
        return False, 'ID de jugador no encontrado'
    return False, f'El proveedor respondió con error ({resp.status_code})'


def _rate_ok(user_key):
    now = time.time()
    with _rate_lock:
        hits = [t for t in _rate.get(user_key, []) if now - t < RATE_WINDOW]
        if len(hits) >= RATE_MAX:
            _rate[user_key] = hits
            return False
        hits.append(now)
        _rate[user_key] = hits
        return True


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@bp.route('/api/verificar-id', methods=['POST'])
def api_verificar_id():
    if 'usuario' not in session:
        return jsonify(ok=False, error='Sesión expirada. Vuelve a iniciar sesión.'), 401
    data = request.get_json(silent=True) or {}
    game_key = str(data.get('game') or '')
    player_id = str(data.get('player_id') or '').strip()
    player_id2 = str(data.get('player_id2') or '').strip()
    servidor = str(data.get('servidor') or '').strip()[:60]

    if not PLAYER_RE.match(player_id) or (player_id2 and not PLAYER_RE.match(player_id2)):
        return jsonify(ok=False, error='Escribe un ID válido (solo letras, números y - _ .)'), 400
    cfg = get_verify_config(game_key)
    if not (cfg.get('enabled') and cfg_ready(cfg)):
        return jsonify(ok=False, configured=False, error='La verificación no está disponible para este juego.'), 404
    if not _rate_ok(session.get('user_db_id') or session.get('usuario')):
        return jsonify(ok=False, error='Demasiadas verificaciones. Espera unos minutos.'), 429

    ok, result = call_verify_api(cfg, player_id, player_id2, servidor)
    logger.info(f'[VerifyID] {game_key} player={player_id} -> {"OK" if ok else "FAIL"}')
    return jsonify(ok=True, name=result) if ok else (jsonify(ok=False, error=result), 200)


@bp.route('/admin/verificacion-ids', methods=['GET', 'POST'])
def admin_verificacion_ids():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    games = list_id_games()
    valid_keys = {g['key'] for g in games}

    if request.method == 'POST':
        if request.form.get('accion') == 'inefable_key':
            api_key = request.form.get('api_key', '').strip()
            if api_key == '********':
                api_key = get_inefable_key()
            set_inefable_key(api_key)
            activados = []
            if api_key and request.form.get('auto') == 'on':
                for g in games:
                    code = guess_inefable_game(g['nombre'])
                    cur = get_verify_config(g['key'])
                    if code and (not cur.get('url') or cur.get('provider') == 'inefable'):
                        set_verify_config(g['key'], {'enabled': True, 'provider': 'inefable', 'inefable_game': code})
                        activados.append(g['nombre'])
            if not api_key:
                flash('API key de Inefable eliminada.', 'success')
            else:
                flash('API key de Inefable guardada.' + (f' Activada en: {", ".join(activados)}.' if activados else ''), 'success')
            return redirect('/admin/verificacion-ids')

        key = request.form.get('game_key', '')
        if key not in valid_keys:
            flash('Juego no válido', 'error')
            return redirect('/admin/verificacion-ids')
        if request.form.get('provider') == 'inefable':
            code = request.form.get('inefable_game', '')
            cfg = {'enabled': request.form.get('enabled') == 'on', 'provider': 'inefable', 'inefable_game': code}
            if code not in INEFABLE_GAME_CODES:
                flash('Elige el juego de Inefable.', 'error')
            elif cfg['enabled'] and not get_inefable_key():
                flash('No se guardó: primero guarda la API key de Inefable (arriba).', 'error')
            else:
                set_verify_config(key, cfg)
                flash(f'Configuración guardada ({"activa" if cfg["enabled"] else "desactivada"}).', 'success')
            return redirect(f'/admin/verificacion-ids#{key}')
        cfg = {
            'enabled': request.form.get('enabled') == 'on',
            'method': 'POST' if request.form.get('method') == 'POST' else 'GET',
            'url': request.form.get('url', '').strip(),
            'headers': request.form.get('headers', '').strip(),
            'body': request.form.get('body', '').strip(),
            'name_path': request.form.get('name_path', '').strip(),
            'error_path': request.form.get('error_path', '').strip(),
        }
        errores = []
        if cfg['url'] and not cfg['url'].lower().startswith(('http://', 'https://')):
            errores.append('la URL debe empezar por http:// o https://')
        for campo in ('headers', 'body'):
            if cfg[campo]:
                try:
                    json.loads(cfg[campo])
                except ValueError:
                    errores.append('las cabeceras no son JSON válido' if campo == 'headers' else 'el cuerpo no es JSON válido')
        if cfg['enabled'] and not (cfg['url'] and cfg['name_path']):
            errores.append('para activarla hacen falta la URL y la ruta del nombre')
        if errores:
            flash('No se guardó: ' + '; '.join(errores) + '.', 'error')
            return redirect(f'/admin/verificacion-ids#{key}')
        set_verify_config(key, cfg)
        flash(f'Configuración guardada ({"activa" if cfg["enabled"] else "desactivada"}).', 'success')
        return redirect(f'/admin/verificacion-ids#{key}')

    for g in games:
        g['cfg'] = get_verify_config(g['key'])
        g['inefable_guess'] = guess_inefable_game(g['nombre'])
    return render_template('admin_verify_ids.html', games=games, inefable_has_key=bool(get_inefable_key()),
                           inefable_games=INEFABLE_GAMES)


@bp.route('/admin/verificacion-ids/probar', methods=['POST'])
def admin_probar_verificacion():
    if not session.get('is_admin'):
        return jsonify(ok=False, error='Acceso denegado'), 403
    data = request.get_json(silent=True) or {}
    cfg = get_verify_config(str(data.get('game') or ''))
    if not cfg_ready(cfg):
        return jsonify(ok=False, error='Guarda primero la configuración de este juego.')
    ok, result = call_verify_api(cfg, str(data.get('player_id') or '').strip(),
                                 str(data.get('player_id2') or '').strip(), str(data.get('servidor') or '').strip())
    return jsonify(ok=ok, name=result) if ok else jsonify(ok=False, error=result)

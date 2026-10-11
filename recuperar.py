"""
Recuperar la contraseña por correo (Blueprint Flask).

  GET/POST /recuperar            → pide el correo y envía un enlace (la respuesta es siempre la misma: no revela qué correos existen)
  GET/POST /restablecer/<token>  → pone la contraseña nueva; el enlace vale 30 minutos y una sola vez
  GET      /admin/correo         → estado del correo del servidor y botón de prueba
  POST     /admin/correo/prueba  → envía un correo de prueba (JSON)

Variables del servidor (las mismas del correo de los respaldos, más SITE_URL):
  MAIL_USERNAME  cuenta de Gmail que envía
  MAIL_PASSWORD  contraseña de aplicación de esa cuenta (16 letras; los espacios se ignoran)
  SITE_URL       dirección pública de la web (https://…); los enlaces se arman con ella y nunca con la cabecera
                 Host de la petición, para que nadie pueda hacer llegar un enlace a otro sitio. En local (127.0.0.1)
                 puede faltar.

Solo se guarda el hash del enlace. Las cuentas de admin (ADMIN_EMAIL / ADMIN_EMAILS) no se recuperan aquí:
su contraseña es ADMIN_PASSWORD en el servidor.
"""
import hashlib
import logging
import os
import re
import secrets
import smtplib
import socket
import threading
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr
from urllib.parse import urlsplit

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, session
from markupsafe import escape
from werkzeug.security import generate_password_hash

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('recuperar', __name__)

NOMBRE_SITIO = '3SRevendedores'
VALIDEZ_S = 30 * 60
MAX_POR_CUENTA_HORA = 3
MAX_POR_IP_HORA = 10
CLAVE_MIN, CLAVE_MAX = 8, 128
SMTP_TIMEOUT_S = 20
ENVIAR_EN_HILO = True   # las pruebas lo apagan para enviar en el momento
limpiar_fallos = None   # app.py lo apunta a limpiar_login_fallidos(correo)

_CORREO_RE = re.compile(r'^[^@\s<>"\'`]+@[^@\s<>"\'`]+\.[^@\s<>"\'`]+$')
_HOSTS_LOCALES = ('127.0.0.1', 'localhost', '::1')
_ips = {}
_ips_lock = threading.Lock()
_tabla_lista = False
_tabla_lock = threading.Lock()


def init_tabla():
    global _tabla_lista
    if _tabla_lista:
        return
    with _tabla_lock:
        if _tabla_lista:
            return
        conn = get_db_connection()
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recuperar_clave (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    expira BIGINT NOT NULL,
                    usado BOOLEAN DEFAULT FALSE,
                    creado BIGINT NOT NULL,
                    ip TEXT
                )''')
            conn.execute('CREATE INDEX IF NOT EXISTS ix_recuperar_clave_usuario ON recuperar_clave (usuario_id, creado)')
            conn.commit()
            _tabla_lista = True
        finally:
            conn.close()


# ---------------------------------------------------------------- configuración

def _cfg_correo():
    c = current_app.config
    return {
        'servidor': c.get('MAIL_SERVER') or 'smtp.gmail.com',
        'puerto': int(c.get('MAIL_PORT') or 587),
        'ssl': bool(c.get('MAIL_USE_SSL')),
        'usuario': (c.get('MAIL_USERNAME') or '').strip(),
        'clave': (c.get('MAIL_PASSWORD') or '').replace(' ', ''),
    }


def url_sitio():
    """Base de los enlaces, o None si no hay una segura (falta SITE_URL fuera de local)."""
    base = os.environ.get('SITE_URL', '').strip().rstrip('/')
    if base:
        return base if urlsplit(base).scheme in ('http', 'https') and urlsplit(base).netloc else None
    if request and (request.host.split(':')[0] in _HOSTS_LOCALES):
        return request.host_url.rstrip('/')
    return None


def disponible():
    cfg = _cfg_correo()
    return bool(cfg['usuario'] and cfg['clave'] and url_sitio())


def _correos_admin():
    lista = [os.environ.get('ADMIN_EMAIL', '')] + os.environ.get('ADMIN_EMAILS', '').split(',')
    return {c.strip().lower() for c in lista if c.strip()}


def _ip():
    return (request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or request.remote_addr or '?')


def _ip_bloqueada():
    ahora = time.time()
    with _ips_lock:
        if len(_ips) > 20000:
            _ips.clear()
        recientes = [t for t in _ips.get(_ip(), []) if ahora - t < 3600]
        recientes.append(ahora)
        _ips[_ip()] = recientes
        return len(recientes) > MAX_POR_IP_HORA


def _hash(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- envío

def _smtp_enviar(cfg, destino, asunto, texto, html):
    msg = EmailMessage()
    msg['Subject'] = asunto
    msg['From'] = formataddr((NOMBRE_SITIO, cfg['usuario']))
    msg['To'] = destino
    msg.set_content(texto)
    msg.add_alternative(html, subtype='html')
    if cfg['ssl']:
        smtp = smtplib.SMTP_SSL(cfg['servidor'], cfg['puerto'], timeout=SMTP_TIMEOUT_S)
    else:
        smtp = smtplib.SMTP(cfg['servidor'], cfg['puerto'], timeout=SMTP_TIMEOUT_S)
        smtp.starttls()
    try:
        smtp.login(cfg['usuario'], cfg['clave'])
        smtp.send_message(msg)
    finally:
        try:
            smtp.quit()
        except Exception:
            pass


def _despachar(destino, asunto, texto, html):
    cfg = _cfg_correo()

    def trabajo():
        try:
            _smtp_enviar(cfg, destino, asunto, texto, html)
            logger.info('[Recuperar] Correo enviado: %s', asunto)
        except Exception as e:
            logger.error('[Recuperar] No se pudo enviar el correo (%s): %s', asunto, type(e).__name__)

    if ENVIAR_EN_HILO:
        threading.Thread(target=trabajo, daemon=True).start()
    else:
        trabajo()


def _html_correo(titulo, parrafos, boton=None, pie=''):
    cuerpo = ''.join(f'<p style="margin:0 0 14px;font-size:15px;line-height:1.55;color:#333;">{p}</p>' for p in parrafos)
    if boton:
        cuerpo += (f'<p style="margin:22px 0;text-align:center;"><a href="{escape(boton[1])}" '
                   'style="display:inline-block;padding:12px 22px;border-radius:10px;background:#111;color:#fff;'
                   f'text-decoration:none;font-weight:600;font-size:15px;">{escape(boton[0])}</a></p>'
                   f'<p style="margin:0 0 14px;font-size:12.5px;color:#777;word-break:break-all;">Si el botón no funciona, '
                   f'copia esta dirección en tu navegador:<br>{escape(boton[1])}</p>')
    if pie:
        cuerpo += f'<p style="margin:18px 0 0;font-size:12.5px;color:#777;">{pie}</p>'
    return ('<div style="font-family:Arial,Helvetica,sans-serif;background:#f4f4f5;padding:24px 12px;">'
            '<div style="max-width:520px;margin:0 auto;background:#fff;border-radius:14px;padding:26px 24px;">'
            f'<p style="margin:0 0 6px;font-size:13px;font-weight:700;color:#d11;">{NOMBRE_SITIO}</p>'
            f'<h1 style="margin:0 0 16px;font-size:20px;color:#111;">{escape(titulo)}</h1>{cuerpo}</div></div>')


def _correo_enlace(usuario, enlace):
    nombre = escape(usuario['nombre'] or '')
    texto = (f'Hola {usuario["nombre"] or ""}:\n\nPediste cambiar tu contraseña de {NOMBRE_SITIO}. '
             f'Abre este enlace para elegir una nueva (vale 30 minutos y sirve una sola vez):\n\n{enlace}\n\n'
             'Si no lo pediste, ignora este correo: tu contraseña no cambia.')
    html = _html_correo('Cambia tu contraseña', [
        f'Hola {nombre}:',
        f'Pediste cambiar tu contraseña de {NOMBRE_SITIO}. Pulsa el botón para elegir una nueva. '
        'El enlace vale <b>30 minutos</b> y sirve <b>una sola vez</b>.'],
        boton=('Elegir contraseña nueva', enlace),
        pie='Si no lo pediste, ignora este correo: tu contraseña no cambia.')
    _despachar(usuario['correo'], f'Cambia tu contraseña de {NOMBRE_SITIO}', texto, html)


def _correo_aviso_cambio(usuario):
    texto = (f'Hola {usuario["nombre"] or ""}:\n\nLa contraseña de tu cuenta de {NOMBRE_SITIO} se acaba de cambiar.\n\n'
             'Si no fuiste tú, pide un enlace nuevo en «¿Olvidaste tu contraseña?» y escríbenos a soporte.')
    html = _html_correo('Tu contraseña se cambió', [
        f'Hola {escape(usuario["nombre"] or "")}:',
        f'La contraseña de tu cuenta de {NOMBRE_SITIO} se acaba de cambiar.'],
        pie='Si no fuiste tú, pide un enlace nuevo en «¿Olvidaste tu contraseña?» y escríbenos a soporte.')
    _despachar(usuario['correo'], f'Tu contraseña de {NOMBRE_SITIO} se cambió', texto, html)


# ---------------------------------------------------------------- tokens

def _token_valido(conn, token):
    if not token or len(token) > 100:
        return None
    return conn.execute(
        'SELECT r.id, r.usuario_id, u.correo, u.nombre FROM recuperar_clave r JOIN usuarios u ON u.id = r.usuario_id '
        'WHERE r.token_hash = ? AND r.usado = FALSE AND r.expira > ?', (_hash(token), int(time.time()))).fetchone()


def _crear_enlace(usuario):
    """Crea un enlace nuevo (anula los anteriores) o None si esa cuenta ya pidió demasiados en la última hora."""
    ahora = int(time.time())
    conn = get_db_connection()
    try:
        n = conn.execute('SELECT COUNT(*) AS n FROM recuperar_clave WHERE usuario_id = ? AND creado > ?',
                         (usuario['id'], ahora - 3600)).fetchone()['n']
        if int(n or 0) >= MAX_POR_CUENTA_HORA:
            return None
        token = secrets.token_urlsafe(32)
        conn.execute('UPDATE recuperar_clave SET usado = TRUE WHERE usuario_id = ? AND usado = FALSE', (usuario['id'],))
        conn.execute('INSERT INTO recuperar_clave (usuario_id, token_hash, expira, usado, creado, ip) VALUES (?, ?, ?, FALSE, ?, ?)',
                     (usuario['id'], _hash(token), ahora + VALIDEZ_S, ahora, _ip()[:64]))
        conn.commit()
        return f'{url_sitio()}/restablecer/{token}'
    finally:
        conn.close()


# ---------------------------------------------------------------- páginas

def _pagina(modo, **extra):
    return render_template('recuperar.html', modo=modo, clave_min=CLAVE_MIN, **extra)


@bp.route('/recuperar', methods=['GET', 'POST'])
def recuperar():
    init_tabla()
    if not disponible():
        return _pagina('no_disponible')
    if request.method == 'GET':
        return _pagina('pedir')

    correo = (request.form.get('correo') or '').strip()[:120]
    if not _CORREO_RE.match(correo):
        return _pagina('pedir', error='Escribe un correo electrónico válido.', correo=correo)
    if _ip_bloqueada():
        return _pagina('pedir', error='Pediste demasiados enlaces. Espera una hora e inténtalo de nuevo.', correo=correo)

    if correo.lower() not in _correos_admin():
        conn = get_db_connection()
        try:
            usuario = conn.execute('SELECT id, correo, nombre FROM usuarios WHERE LOWER(correo) = LOWER(?) ORDER BY id LIMIT 1',
                                   (correo,)).fetchone()
        finally:
            conn.close()
        if usuario:
            enlace = _crear_enlace(usuario)
            if enlace:
                _correo_enlace(usuario, enlace)
                logger.info('[Recuperar] Enlace creado para el usuario %s', usuario['id'])
    # Misma respuesta exista o no la cuenta
    return _pagina('enviado', correo=correo)


@bp.route('/restablecer/<token>', methods=['GET', 'POST'])
def restablecer(token):
    init_tabla()
    conn = get_db_connection()
    try:
        fila = _token_valido(conn, token)
        if not fila:
            return _pagina('invalido')
        if request.method == 'GET':
            return _pagina('nueva', token=token)

        clave = request.form.get('clave') or ''
        repetir = request.form.get('repetir') or ''
        if len(clave) < CLAVE_MIN or len(clave) > CLAVE_MAX:
            return _pagina('nueva', token=token, error=f'La contraseña debe tener entre {CLAVE_MIN} y {CLAVE_MAX} caracteres.')
        if clave != repetir:
            return _pagina('nueva', token=token, error='Las dos contraseñas no coinciden.')

        cur = conn.execute('UPDATE recuperar_clave SET usado = TRUE WHERE id = ? AND usado = FALSE', (fila['id'],))
        if cur.rowcount != 1:  # otro envío lo usó al mismo tiempo
            conn.rollback()
            return _pagina('invalido')
        conn.execute('UPDATE usuarios SET contraseña = ? WHERE id = ?',
                     (generate_password_hash(clave, method='pbkdf2:sha256', salt_length=16), fila['usuario_id']))
        conn.execute('UPDATE recuperar_clave SET usado = TRUE WHERE usuario_id = ?', (fila['usuario_id'],))
        conn.commit()
    finally:
        conn.close()

    logger.info('[Recuperar] Contraseña cambiada para el usuario %s', fila['usuario_id'])
    if limpiar_fallos:
        try:
            limpiar_fallos(fila['correo'])
        except Exception:
            pass
    _correo_aviso_cambio(fila)
    return _pagina('listo')


# ---------------------------------------------------------------- admin

def _ocultar(correo):
    if '@' not in correo:
        return correo[:2] + '…' if correo else ''
    nombre, dominio = correo.split('@', 1)
    return (nombre[:3] + '…' if len(nombre) > 3 else nombre) + '@' + dominio


def _estado():
    cfg = _cfg_correo()
    base = os.environ.get('SITE_URL', '').strip().rstrip('/')
    return {
        'usuario': _ocultar(cfg['usuario']),
        'clave': bool(cfg['clave']),
        'site_url': base,
        'site_url_ok': bool(url_sitio()),
        'activo': disponible(),
        'servidor': f"{cfg['servidor']}:{cfg['puerto']}",
    }


@bp.route('/admin/correo')
def admin_correo():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    return render_template('admin_correo.html', estado=_estado(), destino=os.environ.get('ADMIN_EMAIL', '').strip())


@bp.route('/admin/correo/prueba', methods=['POST'])
def admin_correo_prueba():
    if not session.get('is_admin'):
        return jsonify(ok=False, error='Acceso denegado'), 403
    if not request.is_json:
        return jsonify(ok=False, error='Se espera JSON'), 415
    cfg = _cfg_correo()
    if not (cfg['usuario'] and cfg['clave']):
        return jsonify(ok=False, error='Falta MAIL_USERNAME o MAIL_PASSWORD en el servidor.'), 400
    destino = str((request.get_json(silent=True) or {}).get('destino') or '').strip()[:120]
    if not _CORREO_RE.match(destino):
        return jsonify(ok=False, error='Escribe un correo válido para la prueba.'), 400

    hora = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    texto = f'Correo de prueba de {NOMBRE_SITIO} ({hora}). Si te llegó, el correo del servidor funciona.'
    html = _html_correo('Correo de prueba', [f'Si te llegó este correo, el correo del servidor funciona.', f'Enviado: {hora}'])
    try:
        _smtp_enviar(cfg, destino, f'Prueba de correo de {NOMBRE_SITIO}', texto, html)
    except smtplib.SMTPAuthenticationError:
        return jsonify(ok=False, error='Gmail rechazó el usuario o la contraseña. MAIL_PASSWORD debe ser la contraseña '
                                       'de aplicación de 16 letras, no la contraseña normal de Gmail.'), 502
    except (socket.timeout, TimeoutError, ConnectionError, OSError) as e:
        if isinstance(e, smtplib.SMTPException):
            return jsonify(ok=False, error=f'El servidor de correo respondió con un error ({type(e).__name__}).'), 502
        return jsonify(ok=False, error=f'No se pudo conectar con {cfg["servidor"]}:{cfg["puerto"]}. Puede que el VPS '
                                       'bloquee ese puerto: pídele a tu proveedor que lo abra.'), 502
    except Exception as e:
        return jsonify(ok=False, error=f'No se pudo enviar ({type(e).__name__}).'), 502
    logger.info('[Recuperar] Correo de prueba enviado por el admin')
    return jsonify(ok=True, mensaje=f'Correo de prueba enviado a {destino}. Revisa también la carpeta de spam.')

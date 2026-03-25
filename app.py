from dotenv import load_dotenv
load_dotenv()

def get_games_active():
    """Devuelve dict con flags de juegos activos segun tablas de precios."""
    flags = {'freefire': False, 'freefire_global': False, 'bloodstriker': False, 'freefire_id': False}
    try:
        conn = get_db_connection()
        flags['freefire'] = conn.execute("SELECT COUNT(1) FROM precios_paquetes WHERE activo = TRUE").fetchone()[0] > 0
        flags['freefire_global'] = conn.execute("SELECT COUNT(1) FROM precios_freefire_global WHERE activo = TRUE").fetchone()[0] > 0
        flags['bloodstriker'] = conn.execute("SELECT COUNT(1) FROM precios_bloodstriker WHERE activo = TRUE").fetchone()[0] > 0
        flags['freefire_id'] = conn.execute("SELECT COUNT(1) FROM precios_freefire_id WHERE activo = TRUE").fetchone()[0] > 0
        conn.close()
    except:
        pass
    return flags

import logging
logger = logging.getLogger(__name__)

from flask import Flask, render_template, render_template_string, request, redirect, session, flash, jsonify
import json
import csv
import re
from pg_compat import get_db_connection, get_db_connection_optimized, PgRow, table_exists as pg_table_exists
import pytz
from datetime import datetime
import hashlib
import base64
import os
import secrets
from datetime import timedelta, datetime
import pytz
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import threading
import hmac as hmac_module
import time as time_module
import urllib.parse
import requests
from pin_manager import create_pin_manager
from pin_redeemer import PinRedeemResult, get_redeemer_config_from_db
from redeem_hype_vps import redeem_pin_vps
from functools import lru_cache
import random
import string
import zipfile
import io
from admin_stats import bp as admin_stats_bp
from dynamic_games import bp as dynamic_games_bp, get_all_dynamic_games as get_dynamic_games_list, get_dynamic_game_by_slug, get_dynamic_packages, get_dynamic_package_by_id, sync_all_dynamic_games_prices
from api_whitelabel import bp as whitelabel_bp, init_whitelabel_tables
from update_monthly_spending import update_monthly_spending


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')


def _jwt_hs256(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    sig = hmac_module.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def _gameclub_config():
    base_url = (os.environ.get('GAMECLUB_BASE_URL') or 'https://api.gamepointclub.net').strip().rstrip('/')
    partnerid = (os.environ.get('GAMECLUB_PARTNERID') or '').strip()
    secret = (os.environ.get('GAMECLUB_SECRET') or '').strip()
    return base_url, partnerid, secret


def _gameclub_build_proxies():
    """Construye dict proxies para requests desde env vars.

    Soporta:
    - GAMECLUB_PROXY: host:port:user:pass  (o URL completa)
    - GAMECLUB_PROXY_HTTP / GAMECLUB_PROXY_HTTPS
    """
    def normalize_proxy(val: str):
        if not val:
            return None
        v = str(val).strip()
        if not v:
            return None
        if '://' in v:
            return v
        parts = v.split(':')
        if len(parts) == 4:
            host, port, user, pwd = parts
            return f"http://{user}:{pwd}@{host}:{port}"
        if len(parts) == 2:
            host, port = parts
            return f"http://{host}:{port}"
        return v

    raw = os.environ.get('GAMECLUB_PROXY')
    http_raw = os.environ.get('GAMECLUB_PROXY_HTTP') or raw
    https_raw = os.environ.get('GAMECLUB_PROXY_HTTPS') or raw
    http_p = normalize_proxy(http_raw)
    https_p = normalize_proxy(https_raw)
    proxies = {}
    if http_p:
        proxies['http'] = http_p
    if https_p:
        proxies['https'] = https_p
    return proxies or None


def _gameclub_post(endpoint_path: str, payload: dict):
    base_url, partnerid, secret = _gameclub_config()
    if not partnerid or not secret:
        return None, {
            'code': 400,
            'message': 'GameClub no configurado: faltan GAMECLUB_PARTNERID o GAMECLUB_SECRET'
        }

    ts = int(time_module.time())
    full_payload = dict(payload or {})
    full_payload.setdefault('timestamp', ts)

    jwt_token = _jwt_hs256(full_payload, secret)
    url = f"{base_url}/{endpoint_path.lstrip('/')}"
    headers = {
        'Content-Type': 'application/json',
        'partnerid': partnerid,
        'Accept': 'application/json,text/plain,*/*',
        'User-Agent': os.environ.get('GAMECLUB_USER_AGENT') or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    body = {'payload': jwt_token}
    try:
        proxies = _gameclub_build_proxies()
        res = requests.post(url, json=body, headers=headers, timeout=25, proxies=proxies)
        try:
            data = res.json()
        except Exception:
            data = {'code': res.status_code, 'message': res.text}
        return res, data
    except Exception as e:
        return None, {'code': 500, 'message': f'Error conectando a GameClub: {str(e)}'}


def _gameclub_get_token():
    _, data = _gameclub_post('merchant/token', {})
    if (data or {}).get('code') == 200 and (data or {}).get('token'):
        return data.get('token'), None
    return None, data


def _gameclub_order_validate(token, product_id, fields):
    """Paso 1 de compra: valida la orden y devuelve validation_token (expira en 30s)."""
    payload = {
        'token': token,
        'productid': int(product_id),
        'fields': fields,
    }
    _, data = _gameclub_post('order/validate', payload)
    return data


def _gameclub_order_create(token, validate_token, package_id, merchant_code, price=None):
    """Paso 2 de compra: crea la orden real. Devuelve referenceno si code 100/101."""
    payload = {
        'token': token,
        'validate_token': validate_token,
        'packageid': int(package_id),
        'merchantcode': str(merchant_code),
    }
    if price is not None:
        payload['price'] = round(float(price), 2)
    _, data = _gameclub_post('order/create', payload)
    return data


def _gameclub_order_inquiry(token, reference_no):
    """Consulta el estado de una orden por su referenceno."""
    payload = {
        'token': token,
        'referenceno': str(reference_no),
    }
    _, data = _gameclub_post('order/inquiry', payload)
    return data


def get_revendedores_balance():
    """Consulta el saldo disponible en la cuenta del revendedor externo."""
    base_url = (os.environ.get('REVENDEDORES_BASE_URL') or '').strip().rstrip('/')
    api_key = (os.environ.get('REVENDEDORES_API_KEY') or '').strip()
    if not base_url or not api_key:
        return 0.0

    headers = {
        'X-API-Key': api_key,
        'Accept': 'application/json',
        'User-Agent': '3sRecargas-Web/1.0',
    }

    endpoints = (
        '/api/v1/balance',
        '/api/v1/account',
    )

    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json() if response.content else {}
        except Exception as exc:
            logger.warning(f"No se pudo consultar saldo en {endpoint}: {exc}")
            continue

        candidates = [
            data.get('balance'),
            data.get('saldo'),
            (data.get('account') or {}).get('balance') if isinstance(data.get('account'), dict) else None,
            (data.get('data') or {}).get('balance') if isinstance(data.get('data'), dict) else None,
            (data.get('user') or {}).get('balance') if isinstance(data.get('user'), dict) else None,
        ]
        for value in candidates:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    return 0.0


def _generate_batch_id():
    return datetime.utcnow().strftime('%Y%m%d%H%M%S%f') + '-' + secrets.token_hex(4)


def _extract_pin_codes_from_csv_bytes(content: bytes):
    """Extrae códigos de PIN desde un CSV, ignorando texto/columnas sobrantes."""
    if not content:
        return []
    text = content.decode('utf-8', errors='ignore')
    pins = []
    uuid_re = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
    try:
        # Intentar primero como CSV con encabezados (p.ej. Id,Serial,Clave)
        dict_reader = csv.DictReader(text.splitlines())
        fieldnames = [str(f or '').strip().lstrip('\ufeff') for f in (dict_reader.fieldnames or [])]
        key_field = None
        for f in fieldnames:
            if f.lower() == 'clave':
                key_field = f
                break
        if key_field:
            for row in dict_reader:
                raw_val = (row or {}).get(key_field)
                if not raw_val:
                    continue
                m = uuid_re.search(str(raw_val))
                if m:
                    pins.append(m.group(0))
                else:
                    # Fallback por si la clave no es UUID
                    m2 = re.search(r"[A-Za-z0-9]{6,}", str(raw_val))
                    if m2:
                        pins.append(m2.group(0))
        else:
            # Sin encabezados reconocibles: buscar UUID por fila; si no hay, tomar token largo
            reader = csv.reader(text.splitlines())
            for row in reader:
                if not row:
                    continue
                joined = ' '.join(str(c) for c in row if c is not None)
                m = uuid_re.search(joined)
                if m:
                    pins.append(m.group(0))
                    continue
                # Si no hay UUID, intentar con la última columna (frecuente que el PIN esté al final)
                last = str(row[-1]) if row else ''
                m2 = re.search(r"[A-Za-z0-9]{6,}", last)
                if m2:
                    pins.append(m2.group(0))
    except Exception:
        # Fallback: extraer tokens en bruto por líneas
        for line in text.splitlines():
            m = uuid_re.search(line)
            if m:
                pins.append(m.group(0))
                continue
            m = re.search(r"[A-Za-z0-9]{6,}", line)
            if m:
                pins.append(m.group(0))
    return [p.strip() for p in pins if p and str(p).strip()]

app = Flask(__name__)

# Configuración de seguridad
# IMPORTANTE: En producción con múltiples workers, SECRET_KEY DEBE estar en env vars.
# Si no está definido, se genera uno fijo basado en el path del app para que todos
# los workers compartan la misma clave (secrets.token_hex genera uno distinto por proceso).
_env_secret = os.environ.get('SECRET_KEY', '').strip()
if _env_secret:
    app.secret_key = _env_secret
else:
    import hashlib as _hlib
    app.secret_key = _hlib.sha256(f"3srecargas-fallback-{os.path.abspath(__file__)}".encode()).hexdigest()
    logger.warning("[SEGURIDAD] SECRET_KEY no configurado. Usando clave derivada. Configura SECRET_KEY en produccion.")

# Configuración de cookies seguras
# SESSION_COOKIE_SECURE=True requiere HTTPS. En VPS sin HTTPS usar FORCE_HTTPS=false.
_force_https = os.environ.get('FORCE_HTTPS', '').strip().lower()
if _force_https in ('false', '0', 'no'):
    _cookie_secure = False
elif _force_https in ('true', '1', 'yes'):
    _cookie_secure = True
else:
    # Por defecto: solo activar Secure si se detecta HTTPS (Render)
    _cookie_secure = os.environ.get('RENDER') == 'true'
app.config['SESSION_COOKIE_SECURE'] = _cookie_secure
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevenir XSS
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Protección CSRF

# Configuración de duración de sesión (30 minutos)
app.permanent_session_lifetime = timedelta(minutes=30)

@app.template_filter('format_date')
def format_date_filter(value, fmt='%Y-%m-%d %H:%M:%S'):
    """Format a date value that may be a datetime object or a string."""
    if value is None:
        return ''
    if isinstance(value, datetime):
        return value.strftime(fmt)
    s = str(value)
    # Strip microseconds from string dates
    if '.' in s:
        s = s.split('.')[0]
    return s

# Configuración de correo electrónico (solo 2 variables necesarias)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME']

# Inicializar Flask-Mail
mail = Mail(app)

# Registrar blueprint de estadísticas de administración
app.register_blueprint(admin_stats_bp, url_prefix='/admin/stats')
app.register_blueprint(dynamic_games_bp)
app.register_blueprint(whitelabel_bp)

@app.context_processor
def inject_dynamic_games_menu():
    """Inyecta la lista de juegos dinámicos activos en todas las plantillas."""
    try:
        games = get_dynamic_games_list(only_active=True)
        games = [g for g in games if g.get('slug') != 'bloodstriker']
        id_games = [g for g in games if (g.get('modo') or 'id') == 'id']
        pin_games = [g for g in games if (g.get('modo') or 'id') != 'id']
        return {
            'dynamic_games_menu': games,
            'dynamic_games_id_menu': id_games,
            'dynamic_games_pin_menu': pin_games,
        }
    except Exception:
        return {'dynamic_games_menu': [], 'dynamic_games_id_menu': [], 'dynamic_games_pin_menu': []}

# PostgreSQL: la URL se lee de DATABASE_URL en .env
# get_db_connection y get_db_connection_optimized vienen de pg_compat

# ===== Helpers de persistencia de profit (legacy) =====
def record_profit_for_transaction(conn, usuario_id, is_admin, juego, paquete_id, cantidad, precio_unitario, transaccion_id=None):
    try:
        if is_admin:
            return
        # No registrar profit para cuentas marcadas sin_ganancia
        sg = conn.execute('SELECT sin_ganancia FROM usuarios WHERE id = ?', (int(usuario_id),)).fetchone()
        if sg and sg['sin_ganancia']:
            return
        if juego is None or paquete_id is None or cantidad is None or precio_unitario is None:
            return
        cur = conn.cursor()
        # Buscar costo activo en precios_compra
        row = cur.execute(
            """
            SELECT precio_compra FROM precios_compra
            WHERE juego = ? AND paquete_id = ? AND activo = 1
            """,
            (juego, int(paquete_id))
        ).fetchone()
        costo_unit = float(row[0]) if row else 0.0
        precio_venta_unit = float(precio_unitario)
        profit_unit = round(precio_venta_unit - costo_unit, 6)
        total = round(profit_unit * int(cantidad), 6)
        # Insertar en ledger
        cur.execute(
            """
            INSERT INTO profit_ledger (usuario_id, juego, paquete_id, cantidad, precio_venta_unit, costo_unit, profit_unit, profit_total, transaccion_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(usuario_id), str(juego), int(paquete_id), int(cantidad), precio_venta_unit, costo_unit, profit_unit, total, transaccion_id)
        )
        # Upsert agregado diario (usar día en zona horaria local para coincidir con UI)
        tz_name = os.environ.get('DEFAULT_TZ', 'America/Caracas')
        try:
            tz = pytz.timezone(tz_name)
            day = datetime.now(tz).date().isoformat()
        except Exception:
            day = datetime.utcnow().date().isoformat()
        existing = cur.execute("SELECT profit_total FROM profit_daily_aggregate WHERE day = ?", (day,)).fetchone()
        if existing:
            cur.execute(
                "UPDATE profit_daily_aggregate SET profit_total = ?, updated_at = datetime('now') WHERE day = ?",
                (round(float(existing[0]) + total, 6), day)
            )
        else:
            cur.execute(
                "INSERT INTO profit_daily_aggregate (day, profit_total) VALUES (?, ?)",
                (day, total)
            )
    except Exception:
        # No interrumpir la compra por error de estadística
        pass

def return_db_connection(conn):
    """Cierra la conexión (sin pool para evitar problemas de threading)"""
    conn.close()

def init_db():
    """Inicializa la base de datos con las tablas necesarias - Compatible con Render"""
    conn = None
    try:
        conn = get_db_connection_optimized()
        cursor = conn.cursor()
        
        # Tabla de usuarios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                apellido TEXT NOT NULL,
                telefono TEXT NOT NULL,
                correo TEXT UNIQUE NOT NULL,
                contraseña TEXT NOT NULL,
                saldo REAL DEFAULT 0.0,
                fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de transacciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transacciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER,
                numero_control TEXT NOT NULL,
                pin TEXT NOT NULL,
                transaccion_id TEXT NOT NULL,
                monto REAL DEFAULT 0.0,
                fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
            )
        ''')
        # Intentar agregar columna paquete_nombre si no existe (SQLite no soporta IF NOT EXISTS en ADD COLUMN)
        try:
            cursor.execute("ALTER TABLE transacciones ADD COLUMN paquete_nombre TEXT")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE transacciones ADD COLUMN duracion_segundos REAL")
        except Exception:
            pass
        
        # Tabla historial_compras: registro permanente independiente de transacciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historial_compras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                monto REAL NOT NULL,
                fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
                paquete_nombre TEXT,
                pin TEXT,
                tipo_evento TEXT DEFAULT 'compra',
                duracion_segundos REAL,
                saldo_antes REAL DEFAULT 0,
                saldo_despues REAL DEFAULT 0,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_historial_fecha ON historial_compras(fecha DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_historial_usuario ON historial_compras(usuario_id, fecha DESC)')
        
        # Columna sin_ganancia: cuentas marcadas no generan profit, no suman a saldo activo, no compiten en top
        try:
            cursor.execute("ALTER TABLE usuarios ADD COLUMN sin_ganancia BOOLEAN DEFAULT FALSE")
        except Exception:
            pass

        # Columna bono_activo: si True y recarga Binance >= 1000$, se aplica bono del 1.5%
        try:
            cursor.execute("ALTER TABLE usuarios ADD COLUMN bono_activo BOOLEAN DEFAULT FALSE")
        except Exception:
            pass
    
        # Tabla de pines de Free Fire LATAM
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pines_freefire (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                monto_id INTEGER NOT NULL,
                pin_codigo TEXT NOT NULL,
                usado BOOLEAN DEFAULT FALSE,
                fecha_agregado DATETIME DEFAULT CURRENT_TIMESTAMP,
                fecha_usado DATETIME NULL,
                usuario_id INTEGER NULL,
                batch_id TEXT NULL,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
            )
        ''')
        
        # Tabla de pines de Free Fire (nuevo juego)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pines_freefire_global (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                monto_id INTEGER NOT NULL,
                pin_codigo TEXT NOT NULL,
                usado BOOLEAN DEFAULT FALSE,
                fecha_agregado DATETIME DEFAULT CURRENT_TIMESTAMP,
                fecha_usado DATETIME NULL,
                usuario_id INTEGER NULL,
                batch_id TEXT NULL,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
            )
        ''')

        # Migración suave: agregar batch_id si la tabla existía antes
        try:
            cursor.execute("ALTER TABLE pines_freefire ADD COLUMN batch_id TEXT")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE pines_freefire_global ADD COLUMN batch_id TEXT")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE transacciones_freefire_id ADD COLUMN pin_codigo TEXT")
        except Exception:
            pass
    
        # Tabla de precios de Free Fire (nuevo juego)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS precios_freefire_global (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL,
                precio REAL NOT NULL,
                descripcion TEXT NOT NULL,
                activo BOOLEAN DEFAULT TRUE,
                fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de precios de paquetes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS precios_paquetes (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL,
                precio REAL NOT NULL,
                descripcion TEXT NOT NULL,
                activo BOOLEAN DEFAULT TRUE,
                fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de precios de Blood Striker
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS precios_bloodstriker (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL,
                precio REAL NOT NULL,
                descripcion TEXT NOT NULL,
                activo BOOLEAN DEFAULT TRUE,
                fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migración: agregar columna gamepoint_package_id a precios_bloodstriker
        try:
            cursor.execute("ALTER TABLE precios_bloodstriker ADD COLUMN gamepoint_package_id INTEGER DEFAULT NULL")
        except Exception:
            pass
        
        # Tabla de transacciones de Blood Striker
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transacciones_bloodstriker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER,
                player_id TEXT NOT NULL,
                paquete_id INTEGER NOT NULL,
                numero_control TEXT NOT NULL,
                transaccion_id TEXT NOT NULL,
                monto REAL DEFAULT 0.0,
                estado TEXT DEFAULT 'pendiente',
                fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
                fecha_procesado DATETIME NULL,
                admin_id INTEGER NULL,
                notas TEXT NULL,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id),
                FOREIGN KEY (admin_id) REFERENCES usuarios (id)
            )
        ''')
        
        # Migración: agregar columna gamepoint_referenceno a transacciones_bloodstriker
        try:
            cursor.execute("ALTER TABLE transacciones_bloodstriker ADD COLUMN gamepoint_referenceno TEXT DEFAULT NULL")
        except Exception:
            pass
        
        # Tabla de precios de Free Fire ID
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS precios_freefire_id (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL,
                precio REAL NOT NULL,
                descripcion TEXT NOT NULL,
                activo BOOLEAN DEFAULT TRUE,
                fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de transacciones de Free Fire ID
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transacciones_freefire_id (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER,
                player_id TEXT NOT NULL,
                paquete_id INTEGER NOT NULL,
                numero_control TEXT NOT NULL,
                transaccion_id TEXT NOT NULL,
                monto REAL DEFAULT 0.0,
                estado TEXT DEFAULT 'pendiente',
                fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
                fecha_procesado DATETIME NULL,
                admin_id INTEGER NULL,
                notas TEXT NULL,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id),
                FOREIGN KEY (admin_id) REFERENCES usuarios (id)
            )
        ''')
        
        # Tabla de configuración de fuentes de pines por monto
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS configuracion_fuentes_pines (
                monto_id INTEGER PRIMARY KEY,
                fuente TEXT NOT NULL DEFAULT 'local',
                activo BOOLEAN DEFAULT TRUE,
                fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP,
                CHECK (fuente IN ('local', 'api_externa'))
            )
        ''')
        
        # Tabla de créditos de billetera
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS creditos_billetera (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER,
                monto REAL DEFAULT 0.0,
                saldo_anterior REAL DEFAULT 0.0,
                fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
                visto BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
            )
        ''')
        
        # Tabla de noticias
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS noticias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titulo TEXT NOT NULL,
                contenido TEXT NOT NULL,
                importante BOOLEAN DEFAULT FALSE,
                fecha DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Tabla para evitar re-importar el mismo archivo CSV por nombre
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_imported_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                imported_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de noticias vistas por usuario
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS noticias_vistas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER,
                noticia_id INTEGER,
                fecha_vista DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id),
                FOREIGN KEY (noticia_id) REFERENCES noticias (id),
                UNIQUE(usuario_id, noticia_id)
            )
        ''')
        
        # Tabla de notificaciones personalizadas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notificaciones_personalizadas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER,
                titulo TEXT NOT NULL,
                mensaje TEXT NOT NULL,
                tipo TEXT DEFAULT 'info',
                visto BOOLEAN DEFAULT FALSE,
                fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
            )
        ''')
        
        # Insertar configuración por defecto si no existe (todos en local)
        cursor.execute('SELECT COUNT(*) FROM configuracion_fuentes_pines')
        if cursor.fetchone()[0] == 0:
            configuracion_default = [(i, 'local', True) for i in range(1, 10)]
            cursor.executemany('''
                INSERT INTO configuracion_fuentes_pines (monto_id, fuente, activo)
                VALUES (?, ?, ?)
            ''', configuracion_default)
    
        # Insertar precios por defecto si no existen
        cursor.execute('SELECT COUNT(*) FROM precios_paquetes')
        if cursor.fetchone()[0] == 0:
            precios_default = [
                (1, '110 💎', 0.66, '110 Diamantes Free Fire', True),
                (2, '341 💎', 2.25, '341 Diamantes Free Fire', True),
                (3, '572 💎', 3.66, '572 Diamantes Free Fire', True),
                (4, '1.166 💎', 7.10, '1.166 Diamantes Free Fire', True),
                (5, '2.376 💎', 14.44, '2.376 Diamantes Free Fire', True),
                (6, '6.138 💎', 33.10, '6.138 Diamantes Free Fire', True),
                (7, 'Tarjeta básica', 0.50, 'Tarjeta básica Free Fire', True),
                (8, 'Tarjeta semanal', 1.55, 'Tarjeta semanal Free Fire', True),
                (9, 'Tarjeta mensual', 7.10, 'Tarjeta mensual Free Fire', True)
            ]
            cursor.executemany('''
                INSERT INTO precios_paquetes (id, nombre, precio, descripcion, activo)
                VALUES (?, ?, ?, ?, ?)
            ''', precios_default)
        
        # Insertar precios de Blood Striker por defecto si no existen
        cursor.execute('SELECT COUNT(*) FROM precios_bloodstriker')
        if cursor.fetchone()[0] == 0:
            precios_bloodstriker = [
                (1, '100+16 🪙', 0.82, '100+16 Monedas Blood Striker', True),
                (2, '300+52 🪙', 2.60, '300+52 Monedas Blood Striker', True),
                (3, '500+94 🪙', 4.30, '500+94 Monedas Blood Striker', True),
                (4, '1,000+210 🪙', 8.65, '1,000+210 Monedas Blood Striker', True),
                (5, '2,000+486 🪙', 17.30, '2,000+486 Monedas Blood Striker', True),
                (6, '5,000+1,380 🪙', 43.15, '5,000+1,380 Monedas Blood Striker', True),
                (7, 'Pase Elite 🎖️', 3.50, 'Pase Elite Blood Striker', True),
                (8, 'Pase Elite (Plus) 🎖️', 8.00, 'Pase Elite Plus Blood Striker', True),
                (9, 'Pase de Mejora 🔫', 1.85, 'Pase de Mejora Blood Striker', True),
                (10, 'Cofre Camuflaje Ultra 💼', 0.50, 'Cofre Camuflaje Ultra Blood Striker', True)
            ]
            cursor.executemany('''
                INSERT INTO precios_bloodstriker (id, nombre, precio, descripcion, activo)
                VALUES (?, ?, ?, ?, ?)
            ''', precios_bloodstriker)
        
        # Insertar precios de Free Fire Global por defecto si no existen
        cursor.execute('SELECT COUNT(*) FROM precios_freefire_global')
        if cursor.fetchone()[0] == 0:
            precios_freefire_global = [
                (1, '100+10 💎', 0.86, '100+10 Diamantes Free Fire', True),
                (2, '310+31 💎', 2.90, '310+31 Diamantes Free Fire', True),
                (3, '520+52 💎', 4.00, '520+52 Diamantes Free Fire', True),
                (4, '1.060+106 💎', 7.75, '1.060+106 Diamantes Free Fire', True),
                (5, '2.180+218 💎', 15.30, '2.180+218 Diamantes Free Fire', True),
                (6, '5.600+560 💎', 38.00, '5.600+560 Diamantes Free Fire', True)
            ]
            cursor.executemany('''
                INSERT INTO precios_freefire_global (id, nombre, precio, descripcion, activo)
                VALUES (?, ?, ?, ?, ?)
            ''', precios_freefire_global)
        
        # Insertar precios de Free Fire ID por defecto si no existen
        cursor.execute('SELECT COUNT(*) FROM precios_freefire_id')
        if cursor.fetchone()[0] == 0:
            precios_freefire_id = [
                (1, '100+10 💎', 0.90, '100+10 Diamantes Free Fire ID', True),
                (2, '310+31 💎', 2.95, '310+31 Diamantes Free Fire ID', True),
                (3, '520+52 💎', 4.10, '520+52 Diamantes Free Fire ID', True),
                (4, '1.060+106 💎', 7.90, '1.060+106 Diamantes Free Fire ID', True),
                (5, '2.180+218 💎', 15.50, '2.180+218 Diamantes Free Fire ID', True),
                (6, '5.600+560 💎', 38.50, '5.600+560 Diamantes Free Fire ID', True)
            ]
            cursor.executemany('''
                INSERT INTO precios_freefire_id (id, nombre, precio, descripcion, activo)
                VALUES (?, ?, ?, ?, ?)
            ''', precios_freefire_id)
        
        # Tabla de configuración del redeemer automático
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS configuracion_redeemer (
                clave TEXT PRIMARY KEY,
                valor TEXT NOT NULL,
                fecha_actualizacion TEXT DEFAULT (datetime('now'))
            )
        ''')
        
        # Insertar configuración por defecto del redeemer si no existe
        cursor.execute('SELECT COUNT(*) FROM configuracion_redeemer')
        if cursor.fetchone()[0] == 0:
            redeemer_defaults = [
                ('nombre_completo', 'Usuario Revendedor'),
                ('fecha_nacimiento', '01/01/1995'),
                ('nacionalidad', 'Chile'),
                ('url_base', 'https://redeem.hype.games/'),
                ('headless', 'true'),
                ('timeout_ms', '30000'),
                ('auto_redeem', 'false'),
            ]
            cursor.executemany('''
                INSERT INTO configuracion_redeemer (clave, valor) VALUES (?, ?)
            ''', redeemer_defaults)
        
        # Tabla de precios de compra (legacy)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS precios_compra (
                juego TEXT NOT NULL,
                paquete_id INTEGER NOT NULL,
                precio_compra REAL NOT NULL,
                fecha_actualizacion TEXT DEFAULT (datetime('now')),
                activo INTEGER DEFAULT 1,
                UNIQUE(juego, paquete_id)
            )
        ''')

        # Tabla de ledger de profit (legacy persistente)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profit_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                juego TEXT NOT NULL,
                paquete_id INTEGER NOT NULL,
                cantidad INTEGER NOT NULL,
                precio_venta_unit REAL NOT NULL,
                costo_unit REAL NOT NULL,
                profit_unit REAL NOT NULL,
                profit_total REAL NOT NULL,
                transaccion_id TEXT,
                fecha TEXT DEFAULT (datetime('now'))
            )
        ''')

        # Tabla de agregados diarios de profit (legacy persistente)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profit_daily_aggregate (
                day TEXT PRIMARY KEY,
                profit_total REAL NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        ''')
        
        # Tabla de gastos mensuales por usuario (persistente para top clientes)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monthly_user_spending (
                usuario_id INTEGER NOT NULL,
                year_month TEXT NOT NULL,
                total_spent REAL NOT NULL DEFAULT 0.0,
                purchases_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (usuario_id, year_month)
            )
        ''')
        
        # Tabla de estadísticas de ventas semanales
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ventas_semanales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                juego TEXT NOT NULL,
                paquete_id INTEGER NOT NULL,
                paquete_nombre TEXT NOT NULL,
                precio_venta REAL NOT NULL,
                precio_compra REAL NOT NULL DEFAULT 0.0,
                ganancia_unitaria REAL NOT NULL DEFAULT 0.0,
                cantidad_vendida INTEGER NOT NULL DEFAULT 1,
                ganancia_total REAL NOT NULL DEFAULT 0.0,
                fecha_venta DATETIME DEFAULT CURRENT_TIMESTAMP,
                semana_year TEXT NOT NULL,
                CHECK (cantidad_vendida > 0)
            )
        ''')
        
        # Insertar precios de compra por defecto si no existen
        cursor.execute('SELECT COUNT(*) FROM precios_compra')
        if cursor.fetchone()[0] == 0:
            precios_compra_default = [
                # Free Fire LATAM
                ('freefire_latam', 1, 0.59),  # 110 💎 - costo $0.59, venta $0.66
                ('freefire_latam', 2, 2.00),  # 341 💎 - costo $2.00, venta $2.25
                ('freefire_latam', 3, 3.20),  # 572 💎 - costo $3.20, venta $3.66
                ('freefire_latam', 4, 6.50),  # 1.166 💎 - costo $6.50, venta $7.10
                ('freefire_latam', 5, 13.00), # 2.376 💎 - costo $13.00, venta $14.44
                ('freefire_latam', 6, 30.00), # 6.138 💎 - costo $30.00, venta $33.10
                ('freefire_latam', 7, 0.40),  # Tarjeta básica - costo $0.40, venta $0.50
                ('freefire_latam', 8, 1.30),  # Tarjeta semanal - costo $1.30, venta $1.55
                ('freefire_latam', 9, 6.50),  # Tarjeta mensual - costo $6.50, venta $7.10
                
                # Free Fire Global
                ('freefire_global', 1, 0.75), # 100+10 💎 - costo $0.75, venta $0.86
                ('freefire_global', 2, 2.50), # 310+31 💎 - costo $2.50, venta $2.90
                ('freefire_global', 3, 3.50), # 520+52 💎 - costo $3.50, venta $4.00
                ('freefire_global', 4, 7.00), # 1.060+106 💎 - costo $7.00, venta $7.75
                ('freefire_global', 5, 14.00), # 2.180+218 💎 - costo $14.00, venta $15.30
                ('freefire_global', 6, 35.00), # 5.600+560 💎 - costo $35.00, venta $38.00
                
                # Blood Striker
                ('bloodstriker', 1, 0.70),   # 100+16 🪙 - costo $0.70, venta $0.82
                ('bloodstriker', 2, 2.30),   # 300+52 🪙 - costo $2.30, venta $2.60
                ('bloodstriker', 3, 3.80),   # 500+94 🪙 - costo $3.80, venta $4.30
                ('bloodstriker', 4, 7.80),   # 1,000+210 🪙 - costo $7.80, venta $8.65
                ('bloodstriker', 5, 15.50),  # 2,000+486 🪙 - costo $15.50, venta $17.30
                ('bloodstriker', 6, 39.00),  # 5,000+1,380 🪙 - costo $39.00, venta $43.15
                ('bloodstriker', 7, 3.00),   # Pase Elite - costo $3.00, venta $3.50
                ('bloodstriker', 8, 7.20),   # Pase Elite Plus - costo $7.20, venta $8.00
                ('bloodstriker', 9, 1.60),   # Pase de Mejora - costo $1.60, venta $1.85
                ('bloodstriker', 10, 0.40),  # Cofre Camuflaje - costo $0.40, venta $0.50
            ]
            cursor.executemany('''
                INSERT INTO precios_compra (juego, paquete_id, precio_compra)
                VALUES (?, ?, ?)
            ''', precios_compra_default)
    
        # === Tablas para sistema de juegos dinámicos ===
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS juegos_dinamicos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                slug TEXT NOT NULL UNIQUE,
                gamepoint_product_id INTEGER NOT NULL,
                modo TEXT NOT NULL DEFAULT 'id',
                color_tema TEXT DEFAULT '#a78bfa',
                icono TEXT DEFAULT '🎮',
                activo BOOLEAN DEFAULT FALSE,
                campos_config TEXT DEFAULT '{}',
                descripcion TEXT DEFAULT '',
                ganancia_default REAL DEFAULT 0.10,
                fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
                fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        try:
            cursor.execute("ALTER TABLE juegos_dinamicos ADD COLUMN usa_stock_local BOOLEAN DEFAULT FALSE")
        except Exception:
            pass

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paquetes_dinamicos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                juego_id INTEGER NOT NULL,
                nombre TEXT NOT NULL,
                precio REAL NOT NULL,
                descripcion TEXT DEFAULT '',
                gamepoint_package_id INTEGER,
                activo BOOLEAN DEFAULT TRUE,
                orden INTEGER DEFAULT 0,
                fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (juego_id) REFERENCES juegos_dinamicos(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transacciones_dinamicas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                juego_id INTEGER NOT NULL,
                usuario_id INTEGER NOT NULL,
                player_id TEXT,
                player_id2 TEXT,
                servidor TEXT,
                paquete_id INTEGER NOT NULL,
                numero_control TEXT NOT NULL,
                transaccion_id TEXT NOT NULL,
                monto REAL DEFAULT 0.0,
                estado TEXT DEFAULT 'pendiente',
                gamepoint_referenceno TEXT,
                ingame_name TEXT,
                pin_entregado TEXT,
                fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
                fecha_procesado DATETIME,
                notas TEXT,
                FOREIGN KEY (juego_id) REFERENCES juegos_dinamicos(id),
                FOREIGN KEY (usuario_id) REFERENCES usuarios(id),
                FOREIGN KEY (paquete_id) REFERENCES paquetes_dinamicos(id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tx_din_usuario ON transacciones_dinamicas(usuario_id, fecha DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tx_din_juego ON transacciones_dinamicas(juego_id, fecha DESC)')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pines_dinamicos_stock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                juego_id INTEGER NOT NULL,
                paquete_id INTEGER NOT NULL,
                pin_codigo TEXT NOT NULL,
                usado BOOLEAN DEFAULT FALSE,
                usuario_id INTEGER,
                batch_id TEXT,
                fecha_agregado DATETIME DEFAULT CURRENT_TIMESTAMP,
                fecha_uso DATETIME,
                FOREIGN KEY (juego_id) REFERENCES juegos_dinamicos(id),
                FOREIGN KEY (paquete_id) REFERENCES paquetes_dinamicos(id),
                FOREIGN KEY (usuario_id) REFERENCES usuarios(id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dyn_pins_pkg_usado ON pines_dinamicos_stock(juego_id, paquete_id, usado)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dyn_pins_batch ON pines_dinamicos_stock(batch_id)')

        # Tabla de log de recargas via API (Inefable Store)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_recharges_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                package_id INTEGER NOT NULL,
                success BOOLEAN NOT NULL,
                player_name TEXT DEFAULT '',
                error_msg TEXT DEFAULT '',
                duration_seconds REAL DEFAULT 0,
                game_name TEXT DEFAULT '',
                package_name TEXT DEFAULT '',
                fecha DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_log_fecha ON api_recharges_log(fecha DESC)')
        # Migración: agregar columnas game_name y package_name si no existen
        try:
            cursor.execute("ALTER TABLE api_recharges_log ADD COLUMN game_name TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE api_recharges_log ADD COLUMN package_name TEXT DEFAULT ''")
        except Exception:
            pass

        # === Tablas para sistema de mapeo de revendedores ===
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rev_catalog_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                remote_product_id TEXT NOT NULL,
                remote_product_name TEXT NOT NULL DEFAULT '',
                remote_package_id TEXT NOT NULL,
                remote_package_name TEXT NOT NULL DEFAULT '',
                active BOOLEAN DEFAULT TRUE,
                raw_json TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(remote_product_id, remote_package_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rev_item_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                juego_id INTEGER NOT NULL,
                paquete_id INTEGER NOT NULL,
                remote_product_id TEXT NOT NULL,
                remote_package_id TEXT NOT NULL,
                remote_label TEXT DEFAULT '',
                auto_enabled BOOLEAN DEFAULT FALSE,
                active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(juego_id, paquete_id),
                FOREIGN KEY (juego_id) REFERENCES juegos_dinamicos(id),
                FOREIGN KEY (paquete_id) REFERENCES paquetes_dinamicos(id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rev_mappings_paquete ON rev_item_mappings(paquete_id)')

        # Tablas para API de marca blanca (WebService accounts + órdenes)
        init_whitelabel_tables(cursor)

        # Crear índices optimizados para mejor rendimiento
        create_optimized_indexes(cursor)
        
        conn.commit()
        
    except Exception as e:
        print(f"Error al inicializar la base de datos: {e}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        raise e
    finally:
        if conn:
            return_db_connection(conn)

def create_optimized_indexes(cursor):
    """Crea índices optimizados para consultas frecuentes"""
    indexes = [
        'CREATE INDEX IF NOT EXISTS idx_usuarios_correo ON usuarios(correo)',
        'CREATE INDEX IF NOT EXISTS idx_transacciones_usuario_fecha ON transacciones(usuario_id, fecha DESC)',
        'CREATE INDEX IF NOT EXISTS idx_transacciones_fecha ON transacciones(fecha DESC)',
        'CREATE INDEX IF NOT EXISTS idx_pines_monto_usado ON pines_freefire(monto_id, usado)',
        'CREATE INDEX IF NOT EXISTS idx_pines_global_monto_usado ON pines_freefire_global(monto_id, usado)',
        'CREATE INDEX IF NOT EXISTS idx_ventas_semanales_juego_semana ON ventas_semanales(juego, semana_year)',
        'CREATE INDEX IF NOT EXISTS idx_precios_compra_juego_paquete ON precios_compra(juego, paquete_id, activo)',
        'CREATE INDEX IF NOT EXISTS idx_bloodstriker_estado ON transacciones_bloodstriker(estado, fecha DESC)',
        'CREATE INDEX IF NOT EXISTS idx_creditos_usuario_visto ON creditos_billetera(usuario_id, visto)',
        'CREATE INDEX IF NOT EXISTS idx_noticias_fecha ON noticias(fecha DESC)'
    ]
    
    for index_sql in indexes:
        try:
            cursor.execute(index_sql)
        except Exception as e:
            print(f"Error creando índice: {e}")

# Cache en memoria para datos frecuentes
@lru_cache(maxsize=128)
def get_package_info_with_prices_cached():
    """Versión cacheada de información de paquetes Free Fire LATAM"""
    conn = get_db_connection_optimized()
    try:
        packages = conn.execute('''
            SELECT id, nombre, precio, descripcion 
            FROM precios_paquetes 
            WHERE activo = TRUE 
            ORDER BY id
        ''').fetchall()
        
        package_dict = {}
        for package in packages:
            package_dict[package['id']] = {
                'nombre': package['nombre'],
                'precio': package['precio'],
                'descripcion': package['descripcion']
            }
        return package_dict
    finally:
        return_db_connection(conn)

@lru_cache(maxsize=128)
def get_bloodstriker_prices_cached():
    """Versión cacheada de precios de Blood Striker"""
    conn = get_db_connection_optimized()
    try:
        packages = conn.execute('''
            SELECT id, nombre, precio, descripcion 
            FROM precios_bloodstriker 
            WHERE activo = TRUE 
            ORDER BY id
        ''').fetchall()
        
        package_dict = {}
        for package in packages:
            package_dict[package['id']] = {
                'nombre': package['nombre'],
                'precio': package['precio'],
                'descripcion': package['descripcion']
            }
        return package_dict
    finally:
        return_db_connection(conn)

@lru_cache(maxsize=128)
def get_freefire_global_prices_cached():
    """Versión cacheada de precios de Free Fire Global"""
    conn = get_db_connection_optimized()
    try:
        packages = conn.execute('''
            SELECT id, nombre, precio, descripcion 
            FROM precios_freefire_global 
            WHERE activo = TRUE 
            ORDER BY id
        ''').fetchall()
        
        package_dict = {}
        for package in packages:
            package_dict[package['id']] = {
                'nombre': package['nombre'],
                'precio': package['precio'],
                'descripcion': package['descripcion']
            }
        return package_dict
    finally:
        return_db_connection(conn)

def clear_price_cache():
    """Limpia el cache de precios cuando se actualizan"""
    get_package_info_with_prices_cached.cache_clear()
    get_bloodstriker_prices_cached.cache_clear()
    get_freefire_global_prices_cached.cache_clear()
    get_freefire_id_prices_cached.cache_clear()

@lru_cache(maxsize=1000)
def convert_to_venezuela_time_cached(utc_datetime_str):
    """Versión optimizada con cache de conversión de zona horaria"""
    try:
        utc_dt = datetime.strptime(utc_datetime_str, '%Y-%m-%d %H:%M:%S')
        utc_dt = pytz.utc.localize(utc_dt)
        venezuela_tz = pytz.timezone('America/Caracas')
        venezuela_dt = utc_dt.astimezone(venezuela_tz)
        return venezuela_dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return utc_datetime_str

# Funciones de stock optimizadas
def get_pin_stock_optimized():
    """Versión optimizada que usa una sola query en lugar de 9"""
    conn = get_db_connection_optimized()
    try:
        results = conn.execute('''
            SELECT monto_id, COUNT(*) as count 
            FROM pines_freefire 
            WHERE usado = FALSE 
            GROUP BY monto_id
        ''').fetchall()
        
        stock = {i: 0 for i in range(1, 10)}
        for result in results:
            stock[result['monto_id']] = result['count']
        return stock
    finally:
        return_db_connection(conn)

def get_pin_stock_freefire_global_optimized():
    """Versión optimizada para Free Fire Global"""
    conn = get_db_connection_optimized()
    try:
        results = conn.execute('''
            SELECT monto_id, COUNT(*) as count 
            FROM pines_freefire_global 
            WHERE usado = FALSE 
            GROUP BY monto_id
        ''').fetchall()
        
        stock = {i: 0 for i in range(1, 7)}
        for result in results:
            stock[result['monto_id']] = result['count']
        return stock
    finally:
        return_db_connection(conn)

def hash_password(password):
    """Hashea la contraseña usando Werkzeug (más seguro que SHA256)"""
    return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

def verify_password(password, hashed):
    """Verifica la contraseña hasheada (compatible con métodos antiguos y nuevos)"""
    # Intentar con Werkzeug (maneja pbkdf2, scrypt, etc.)
    if hashed.startswith(('pbkdf2:', 'scrypt:')):
        return check_password_hash(hashed, password)
    
    # Si no es un hash de Werkzeug, verificar con SHA256 (método anterior)
    sha256_hash = hashlib.sha256(password.encode()).hexdigest()
    return hashed == sha256_hash



def registrar_historial_compra(conn_existente, usuario_id, monto, paquete_nombre, pin='', tipo_evento='compra', duracion_segundos=None, saldo_antes=0, saldo_despues=0):
    """Registra una compra en el historial permanente (no se borra con transacciones). Usa la conexión existente para evitar bloqueo."""
    try:
        conn_existente.execute('''
            INSERT INTO historial_compras (usuario_id, monto, paquete_nombre, pin, tipo_evento, duracion_segundos, saldo_antes, saldo_despues)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (usuario_id, monto, paquete_nombre, pin, tipo_evento, duracion_segundos, saldo_antes, saldo_despues))
    except Exception as e:
        logger.error(f"Error registrando historial_compra: {e}")

def convert_to_venezuela_time(utc_datetime_val):
    """Convierte una fecha UTC a la zona horaria de Venezuela (UTC-4).
    Acepta tanto objetos datetime (PostgreSQL) como strings (legacy)."""
    try:
        venezuela_tz = pytz.timezone('America/Caracas')
        if isinstance(utc_datetime_val, datetime):
            utc_dt = utc_datetime_val
            if utc_dt.tzinfo is None:
                utc_dt = pytz.utc.localize(utc_dt)
            return utc_dt.astimezone(venezuela_tz).strftime('%Y-%m-%d %H:%M:%S')
        # Fallback: string
        utc_dt = datetime.strptime(str(utc_datetime_val), '%Y-%m-%d %H:%M:%S')
        utc_dt = pytz.utc.localize(utc_dt)
        return utc_dt.astimezone(venezuela_tz).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(utc_datetime_val) if utc_datetime_val else ''

def get_user_by_email(email):
    """Obtiene un usuario por su email"""
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM usuarios WHERE correo = ?', (email,)).fetchone()
    conn.close()
    return user

def create_user(nombre, apellido, telefono, correo, contraseña):
    """Crea un nuevo usuario en la base de datos"""
    conn = get_db_connection()
    hashed_password = hash_password(contraseña)
    try:
        cursor = conn.execute('''
            INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo)
            VALUES (?, ?, ?, ?, ?, ?) RETURNING id
        ''', (nombre, apellido, telefono, correo, hashed_password, 0.0))
        row = cursor.fetchone()
        user_id = row['id'] if row else None
        conn.commit()
        conn.close()
        return user_id
    except Exception as _ie:
        if 'unique' in str(_ie).lower() or 'duplicate' in str(_ie).lower():
            conn.close()
            return None
        raise

def get_user_transactions(user_id, is_admin=False, page=1, per_page=10):
    """Obtiene las transacciones de un usuario con información del paquete y paginación"""
    conn = get_db_connection()
    
    # Calcular offset para paginación
    offset = (page - 1) * per_page
    
    if is_admin:
        # Admin ve todas las transacciones de todos los usuarios (incluyendo las propias)
        transactions = conn.execute('''
            SELECT t.*, u.nombre, u.apellido
            FROM transacciones t
            JOIN usuarios u ON t.usuario_id = u.id
            ORDER BY t.fecha DESC
            LIMIT ? OFFSET ?
        ''', (per_page, offset)).fetchall()
        
        # Obtener total de transacciones para paginación
        total_count = conn.execute('''
            SELECT COUNT(*) FROM transacciones t
            JOIN usuarios u ON t.usuario_id = u.id
        ''').fetchone()[0]
    else:
        # Usuario normal ve solo sus transacciones
        if user_id:
            transactions = conn.execute('''
                SELECT t.*, u.nombre, u.apellido
                FROM transacciones t
                JOIN usuarios u ON t.usuario_id = u.id
                WHERE t.usuario_id = ? 
                ORDER BY t.fecha DESC
                LIMIT ? OFFSET ?
            ''', (user_id, per_page, offset)).fetchall()
            
            # Obtener total de transacciones del usuario para paginación
            total_count = conn.execute('''
                SELECT COUNT(*) FROM transacciones t
                JOIN usuarios u ON t.usuario_id = u.id
                WHERE t.usuario_id = ?
            ''', (user_id,)).fetchone()[0]
        else:
            transactions = []
            total_count = 0
    
    # Obtener precios dinámicos de la base de datos (Free Fire LATAM, Free Fire Global y Blood Striker)
    packages_info = get_package_info_with_prices()
    freefire_global_packages_info = get_freefire_global_prices()
    bloodstriker_packages_info = get_bloodstriker_prices()
    
    # Agregar información del paquete: usar paquete_nombre si existe; si no, resolver por PIN
    transactions_with_package = []
    for transaction in transactions:
        transaction_dict = dict(transaction)
        monto = abs(transaction['monto'])  # Usar valor absoluto para comparar

        # Marcar transacciones de Free Fire ID (completadas) por prefijo de transaccion_id
        try:
            txid = str(transaction_dict.get('transaccion_id') or '')
            if txid.startswith('FFID-'):
                transaction_dict['is_freefire_id'] = True
                transaction_dict['estado'] = transaction_dict.get('estado') or 'completado'

                # Extraer player_id / player_name desde el texto guardado en `pin`
                raw_pin_info = str(transaction_dict.get('pin') or '')
                m_id = re.search(r"ID:\s*([0-9]{4,})", raw_pin_info)
                if m_id:
                    transaction_dict['player_id'] = m_id.group(1)
                m_name = re.search(r"Jugador:\s*([^\n\r\-]+)", raw_pin_info)
                if m_name:
                    transaction_dict['player_name'] = m_name.group(1).strip()

            elif txid.startswith('BS-'):
                transaction_dict['is_bloodstriker'] = True
                transaction_dict['estado'] = transaction_dict.get('estado') or 'completado'

                # Extraer player_id / player_name / gamepoint_ref desde el texto guardado en `pin`
                raw_pin_info = str(transaction_dict.get('pin') or '')
                m_id = re.search(r"ID:\s*([0-9]{4,})", raw_pin_info)
                if m_id:
                    transaction_dict['player_id'] = m_id.group(1)
                m_name = re.search(r"Jugador:\s*([^\n\r\-]+)", raw_pin_info)
                if m_name:
                    transaction_dict['player_name'] = m_name.group(1).strip()
                m_ref = re.search(r"Ref:\s*(\S+)", raw_pin_info)
                if m_ref:
                    transaction_dict['gamepoint_ref'] = m_ref.group(1).strip()

            elif txid.startswith('DG'):
                transaction_dict['is_dynamic_game'] = True

                # Extraer nombre de juego (antes del primer " - ") para el encabezado
                raw_pkg = str(transaction_dict.get('paquete_nombre') or '')
                transaction_dict['juego_nombre'] = raw_pkg.split(' - ')[0].strip() if ' - ' in raw_pkg else raw_pkg

                raw_pin_info = str(transaction_dict.get('pin') or '')
                # Detectar Gift Card pendiente (sin serial todavía)
                if raw_pin_info.startswith('⏳'):
                    transaction_dict['estado'] = 'pendiente'
                    m_ref = re.search(r"Ref:\s*(\S+)", raw_pin_info)
                    if m_ref:
                        transaction_dict['gamepoint_ref'] = m_ref.group(1).strip()
                elif raw_pin_info.startswith('Código:'):
                    transaction_dict['estado'] = 'completado'
                    # Extraer serial: "Código: XXXX - Ref: ..."
                    m_serial = re.match(r"C[oó]digo:\s*(.+?)(?:\s+-\s+Ref:|$)", raw_pin_info)
                    if m_serial:
                        transaction_dict['serial_key'] = m_serial.group(1).strip()
                    m_ref = re.search(r"Ref:\s*(\S+)", raw_pin_info)
                    if m_ref:
                        transaction_dict['gamepoint_ref'] = m_ref.group(1).strip()
                else:
                    transaction_dict['estado'] = transaction_dict.get('estado') or 'completado'
                    m_id = re.search(r"ID:\s*([^\s\-/]+(?:\s*/\s*[^\s\-]+)?)", raw_pin_info)
                    if m_id:
                        transaction_dict['player_id'] = m_id.group(1).strip()
                    m_name = re.search(r"Jugador:\s*([^\n\r\-]+)", raw_pin_info)
                    if m_name:
                        transaction_dict['player_name'] = m_name.group(1).strip()
                    m_ref = re.search(r"Ref:\s*(\S+)", raw_pin_info)
                    if m_ref:
                        transaction_dict['gamepoint_ref'] = m_ref.group(1).strip()
        except Exception:
            pass
        
        # Si la transacción ya trae paquete_nombre, úsalo y sigue
        if transaction_dict.get('paquete_nombre'):
            transaction_dict['paquete'] = transaction_dict['paquete_nombre']
            transaction_dict['fecha'] = convert_to_venezuela_time(transaction_dict['fecha'])
            transactions_with_package.append(transaction_dict)
            continue

        # 1) Resolver por PIN exacto (mejor precisión)
        paquete_encontrado = False
        try:
            raw_pin = transaction_dict.get('pin') or ''
            pins_list = [p.strip() for p in (raw_pin.replace('\r','').split('\n') if '\n' in raw_pin else [raw_pin]) if p.strip()]
            cantidad_pines = len(pins_list)
            pin_sample = pins_list[0] if pins_list else None
            if pin_sample:
                c2 = get_db_connection()
                try:
                    row_latam = c2.execute('SELECT monto_id FROM pines_freefire WHERE pin_codigo = ? LIMIT 1', (pin_sample,)).fetchone()
                    row_global = None if row_latam else c2.execute('SELECT monto_id FROM pines_freefire_global WHERE pin_codigo = ? LIMIT 1', (pin_sample,)).fetchone()
                finally:
                    c2.close()
                if row_latam:
                    mid = int(row_latam['monto_id'])
                    nombre = packages_info.get(mid, {}).get('nombre')
                    if nombre:
                        transaction_dict['paquete'] = f"{nombre}{'' if cantidad_pines <= 1 else f' x{cantidad_pines}'}"
                        paquete_encontrado = True
                elif row_global:
                    mid = int(row_global['monto_id'])
                    nombre = freefire_global_packages_info.get(mid, {}).get('nombre')
                    if nombre:
                        transaction_dict['paquete'] = f"{nombre}{'' if cantidad_pines <= 1 else f' x{cantidad_pines}'}"
                        paquete_encontrado = True
        except Exception:
            # Ignorar errores de lookup por PIN y continuar con fallback por monto
            paquete_encontrado = False or paquete_encontrado

        # 2) Blood Striker: resolver por transaccion_id -> paquete_id (nombre exacto de precios)
        if not paquete_encontrado:
            try:
                c3 = get_db_connection()
                try:
                    row_bs = c3.execute('SELECT paquete_id FROM transacciones_bloodstriker WHERE transaccion_id = ? LIMIT 1', (transaction_dict.get('transaccion_id'),)).fetchone()
                finally:
                    c3.close()
                if row_bs:
                    pid = int(row_bs['paquete_id'])
                    nombre_bs = bloodstriker_packages_info.get(pid, {}).get('nombre')
                    if nombre_bs:
                        transaction_dict['paquete'] = nombre_bs
                        paquete_encontrado = True
            except Exception:
                paquete_encontrado = False or paquete_encontrado
            if not paquete_encontrado:
                transaction_dict['paquete'] = 'Paquete'
        
        # Convertir fecha a zona horaria de Venezuela
        transaction_dict['fecha'] = convert_to_venezuela_time(transaction_dict['fecha'])
        
        transactions_with_package.append(transaction_dict)
    
    conn.close()
    
    # Calcular información de paginación
    total_pages = (total_count + per_page - 1) // per_page  # Redondear hacia arriba
    has_prev = page > 1
    has_next = page < total_pages
    
    return {
        'transactions': transactions_with_package,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total_count,
            'total_pages': total_pages,
            'has_prev': has_prev,
            'has_next': has_next,
            'prev_num': page - 1 if has_prev else None,
            'next_num': page + 1 if has_next else None
        }
    }

def get_user_wallet_credits(user_id):
    """Obtiene los créditos de billetera de un usuario"""
    conn = get_db_connection()
    # Crear tabla si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS creditos_billetera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            monto REAL DEFAULT 0.0,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            visto BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    
    credits = conn.execute('''
        SELECT * FROM creditos_billetera 
        WHERE usuario_id = ? 
        ORDER BY fecha DESC
    ''', (user_id,)).fetchall()
    conn.close()
    return credits

def get_all_wallet_credits():
    """Obtiene todos los créditos de billetera del sistema para el admin"""
    conn = get_db_connection()
    # Crear tabla si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS creditos_billetera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            monto REAL DEFAULT 0.0,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            visto BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    
    # Agregar columna 'visto' si no existe (para compatibilidad con datos existentes)
    try:
        conn.execute('ALTER TABLE creditos_billetera ADD COLUMN visto BOOLEAN DEFAULT FALSE')
        conn.commit()
    except:
        pass  # La columna ya existe
    
    # Agregar columna 'saldo_anterior' si no existe (para compatibilidad con datos existentes)
    try:
        conn.execute('ALTER TABLE creditos_billetera ADD COLUMN saldo_anterior REAL DEFAULT 0.0')
        conn.commit()
    except:
        pass  # La columna ya existe
    
    try:
        credits = conn.execute('''
            SELECT cb.*, u.nombre, u.apellido, u.correo 
            FROM creditos_billetera cb
            JOIN usuarios u ON cb.usuario_id = u.id
            ORDER BY cb.fecha DESC
            LIMIT 100
        ''').fetchall()
    except Exception as e:
        print(f"Error al obtener créditos de billetera: {e}")
        credits = []
    
    conn.close()
    return credits

def get_wallet_credits_stats():
    """Obtiene estadísticas de créditos de billetera para el admin"""
    conn = get_db_connection()
    # Crear tabla si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS creditos_billetera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            monto REAL DEFAULT 0.0,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            visto BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    
    try:
        # Total de créditos agregados
        total_credits = conn.execute('''
            SELECT COALESCE(SUM(monto), 0) as total FROM creditos_billetera
        ''').fetchone()['total']
        
        # Créditos agregados hoy
        today_credits = conn.execute('''
            SELECT COALESCE(SUM(monto), 0) as today_total 
            FROM creditos_billetera 
            WHERE DATE(fecha) = DATE('now')
        ''').fetchone()['today_total']
        
        # Número de usuarios que han recibido créditos
        users_with_credits = conn.execute('''
            SELECT COUNT(DISTINCT usuario_id) as count FROM creditos_billetera
        ''').fetchone()['count']
        
        conn.close()
        return {
            'total_credits': total_credits,
            'today_credits': today_credits,
            'users_with_credits': users_with_credits
        }
    except Exception as e:
        print(f"Error al obtener estadísticas de créditos: {e}")
        conn.close()
        return {
            'total_credits': 0,
            'today_credits': 0,
            'users_with_credits': 0
        }

def get_unread_wallet_credits_count(user_id):
    """Obtiene si hay créditos de billetera no vistos (retorna 1 si hay, 0 si no hay)"""
    conn = get_db_connection()
    # Crear tabla si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS creditos_billetera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            monto REAL DEFAULT 0.0,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            visto BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    
    # Agregar columna 'visto' si no existe (para compatibilidad con datos existentes)
    try:
        conn.execute('ALTER TABLE creditos_billetera ADD COLUMN visto BOOLEAN DEFAULT FALSE')
        conn.commit()
    except:
        pass  # La columna ya existe
    
    count = conn.execute('''
        SELECT COUNT(*) FROM creditos_billetera 
        WHERE usuario_id = ? AND (visto = FALSE OR visto IS NULL)
    ''', (user_id,)).fetchone()[0]
    conn.close()
    
    # Retornar 1 si hay créditos no vistos, 0 si no hay
    return 1 if count > 0 else 0

def mark_wallet_credits_as_read(user_id):
    """Marca todos los créditos de billetera como vistos"""
    conn = get_db_connection()
    # Crear tabla si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS creditos_billetera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            monto REAL DEFAULT 0.0,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            visto BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    
    # Agregar columna 'visto' si no existe
    try:
        conn.execute('ALTER TABLE creditos_billetera ADD COLUMN visto BOOLEAN DEFAULT FALSE')
        conn.commit()
    except:
        pass  # La columna ya existe
    
    conn.execute('''
        UPDATE creditos_billetera 
        SET visto = TRUE 
        WHERE usuario_id = ?
    ''', (user_id,))
    conn.commit()
    conn.close()


@app.route('/api/news/unread', methods=['GET'])
def api_unread_news():
    if 'usuario' not in session:
        return jsonify({'news': []})

    user_id = session.get('user_db_id')
    is_admin = session.get('is_admin', False)
    
    # Para admin, usar ID 0 para que pueda ver noticias (no hay vistas registradas para admin)
    if is_admin:
        user_id = 0
    elif not user_id:
        return jsonify({'news': []})

    create_news_table()
    create_news_views_table()

    conn = get_db_connection()
    rows = conn.execute('''
        SELECT n.id, n.titulo, n.contenido, n.importante, n.fecha, n.imagen_url
        FROM noticias n
        WHERE n.id NOT IN (
            SELECT nv.noticia_id FROM noticias_vistas nv
            WHERE nv.usuario_id = ?
        )
        ORDER BY n.fecha DESC
        LIMIT 5
    ''', (user_id,)).fetchall()
    conn.close()

    news = [dict(r) for r in rows]
    return jsonify({'news': news})


@app.route('/api/news/dismiss/<int:noticia_id>', methods=['POST'])
def api_dismiss_news(noticia_id):
    if 'usuario' not in session:
        return jsonify({'status': 'error'}), 401

    user_id = session.get('user_db_id')
    is_admin = session.get('is_admin', False)
    
    # Para admin, usar ID 0
    if is_admin:
        user_id = 0
    elif not user_id:
        return jsonify({'status': 'error'}), 400

    create_news_table()
    create_news_views_table()

    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO noticias_vistas (usuario_id, noticia_id)
            VALUES (%s, %s) ON CONFLICT DO NOTHING
        ''', (user_id, noticia_id))
        conn.commit()
    finally:
        conn.close()

    return jsonify({'status': 'ok'})

# ===== Sistema de Recargas por Binance Pay =====
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET', '')
BINANCE_PAY_ID = os.environ.get('BINANCE_PAY_ID', '')
RECARGA_BONUS_PERCENT = float(os.environ.get('RECARGA_BONUS_PERCENT', '3'))
RECARGA_EXPIRATION_MINUTES = int(os.environ.get('RECARGA_EXPIRATION_MINUTES', '30'))
RECARGA_MIN_USDT = float(os.environ.get('RECARGA_MIN_USDT', '10'))
RECARGA_MAX_USDT = float(os.environ.get('RECARGA_MAX_USDT', '50000'))
BINANCE_PROXY = os.environ.get('BINANCE_PROXY', '')
BINANCE_REQUEST_TIMEOUT_SECONDS = float(os.environ.get('BINANCE_REQUEST_TIMEOUT_SECONDS', '4'))
BINANCE_TOTAL_TIMEOUT_SECONDS = float(os.environ.get('BINANCE_TOTAL_TIMEOUT_SECONDS', '8'))

def binance_create_signature(query_string):
    """Genera firma HMAC SHA256 para autenticación con Binance API"""
    return hmac_module.new(
        BINANCE_API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

BINANCE_API_ENDPOINTS = [
    'https://api1.binance.com',
    'https://api2.binance.com',
    'https://api3.binance.com',
    'https://api4.binance.com',
    'https://api.binance.com',
]

def binance_get_pay_transactions(start_time=None, end_time=None, limit=100, req_timeout=None, total_timeout_override=None):
    """Consulta historial de transacciones de Binance Pay via GET /sapi/v1/pay/transactions"""
    import requests as req_lib
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.error("Binance API keys no configuradas")
        return None
    
    timestamp = str(int(time_module.time() * 1000))
    params = {'timestamp': timestamp, 'limit': str(limit)}
    
    if start_time:
        params['startTime'] = str(int(start_time))  # ya viene en milisegundos
    if end_time:
        params['endTime'] = str(int(end_time))  # ya viene en milisegundos
    
    query_string = urllib.parse.urlencode(params)
    signature = binance_create_signature(query_string)
    params['signature'] = signature
    
    headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
    proxies = {'https': BINANCE_PROXY, 'http': BINANCE_PROXY} if BINANCE_PROXY else None
    request_timeout = max(1.0, req_timeout or BINANCE_REQUEST_TIMEOUT_SECONDS)
    total_timeout = max(request_timeout, total_timeout_override or BINANCE_TOTAL_TIMEOUT_SECONDS)
    started_at = time_module.monotonic()
    
    last_error = None
    for base_url in BINANCE_API_ENDPOINTS:
        elapsed = time_module.monotonic() - started_at
        if elapsed >= total_timeout:
            logger.warning(f"Timeout total alcanzado consultando Binance Pay ({elapsed:.2f}s)")
            break

        url = f'{base_url}/sapi/v1/pay/transactions'
        try:
            resp = req_lib.get(url, params=params, headers=headers, timeout=request_timeout, proxies=proxies)
            data = resp.json()
            code = str(data.get('code', ''))
            if code == '000000' or code == '0' or data.get('success') == True:
                return data.get('data', [])
            else:
                logger.error(f"Binance Pay API error ({base_url}): {data}")
                return None
        except Exception as e:
            last_error = e
            logger.warning(f"Binance endpoint {base_url} falló: {type(e).__name__}")
            continue
    
    logger.error(f"Todos los endpoints de Binance fallaron. Último error: {last_error}")
    return None

def generar_codigo_recarga():
    """Genera un código de referencia único para la recarga"""
    chars = string.ascii_uppercase + string.digits
    while True:
        code = 'REC-' + ''.join(random.choices(chars, k=6))
        conn = get_db_connection()
        exists = conn.execute(
            'SELECT 1 FROM recargas_binance WHERE codigo_referencia = ?', (code,)
        ).fetchone()
        conn.close()
        if not exists:
            return code

def crear_orden_recarga(user_id, monto):
    """Crea una nueva orden de recarga pendiente"""
    codigo = generar_codigo_recarga()
    
    # Usar UTC para que coincida con CURRENT_TIMESTAMP de SQLite
    ahora_utc = datetime.utcnow()
    expiracion_utc = ahora_utc + timedelta(minutes=RECARGA_EXPIRATION_MINUTES)
    
    conn = get_db_connection()
    try:
        # Verificar que no haya otra recarga pendiente del mismo usuario
        pendiente = conn.execute('''
            SELECT id FROM recargas_binance 
            WHERE usuario_id = ? AND estado = 'pendiente'
        ''', (user_id,)).fetchone()
        
        if pendiente:
            # Expirar la anterior
            conn.execute('''
                UPDATE recargas_binance SET estado = 'expirada' 
                WHERE usuario_id = ? AND estado = 'pendiente'
            ''', (user_id,))
        
        conn.execute('''
            INSERT INTO recargas_binance (usuario_id, codigo_referencia, monto_solicitado, monto_unico, fecha_creacion, fecha_expiracion)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, codigo, monto, monto, ahora_utc.strftime('%Y-%m-%d %H:%M:%S'), expiracion_utc.strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        
        # Convertir expiración a hora local para mostrar al usuario
        tz = pytz.timezone(os.environ.get('DEFAULT_TZ', 'America/Caracas'))
        expiracion_local = pytz.utc.localize(expiracion_utc).astimezone(tz)
        
        return {
            'codigo': codigo,
            'monto': monto,
            'expiracion': expiracion_local.strftime('%Y-%m-%d %H:%M:%S'),
            'binance_pay_id': BINANCE_PAY_ID
        }
    except Exception as e:
        logger.error(f"Error creando orden de recarga: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def verificar_recarga_binance(recarga_id, _binance_tx_kwargs=None):
    """Verifica si una recarga pendiente fue pagada consultando Binance Pay API"""
    if _binance_tx_kwargs is None:
        _binance_tx_kwargs = {}
    conn = get_db_connection()
    recarga = conn.execute('''
        SELECT * FROM recargas_binance WHERE id = ? AND estado = 'pendiente'
    ''', (recarga_id,)).fetchone()
    
    if not recarga:
        conn.close()
        return {'status': 'error', 'message': 'Recarga no encontrada o ya procesada'}
    
    def _to_datetime(val):
        if isinstance(val, datetime):
            return val
        if val is None:
            return None
        sval = str(val).strip()
        if not sval:
            return None
        try:
            # Formato legacy SQLite
            return datetime.strptime(sval, '%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                # ISO/otros formatos que psycopg pueda devolver serializados
                return datetime.fromisoformat(sval.replace('Z', '+00:00')).replace(tzinfo=None)
            except Exception:
                return None

    # Verificar expiración (fechas almacenadas en UTC)
    ahora_utc = datetime.utcnow()
    fecha_exp = _to_datetime(recarga['fecha_expiracion'])
    if not fecha_exp:
        conn.close()
        return {'status': 'error', 'message': 'fecha_expiracion inválida'}
    
    if ahora_utc > fecha_exp:
        conn.execute('UPDATE recargas_binance SET estado = ? WHERE id = ?', ('expirada', recarga_id))
        conn.commit()
        conn.close()
        return {'status': 'expirada', 'message': 'La orden de recarga ha expirado'}
    
    conn.close()
    
    # Consultar transacciones de Binance Pay
    fecha_creacion = _to_datetime(recarga['fecha_creacion']) or ahora_utc
    start_ts = int(fecha_creacion.timestamp() * 1000) - 60000  # 1 min antes
    
    transactions = binance_get_pay_transactions(start_time=start_ts, **_binance_tx_kwargs)
    if transactions is None:
        return {'status': 'error', 'message': 'Error al consultar Binance Pay API'}
    
    codigo_ref = recarga['codigo_referencia']
    monto_esperado = float(recarga['monto_unico'])
    
    # Guardar datos antes de buscar (la conexión original ya se cerró)
    usuario_id = recarga['usuario_id']
    
    logger.info(f"Verificando recarga {recarga_id}: codigo={codigo_ref}, monto={monto_esperado}, transacciones encontradas={len(transactions)}")
    
    # Buscar transacción que coincida: note contiene el código Y amount coincide
    # La API de Binance Pay devuelve: orderMemo/remark para la nota, fundsDetail para currency/amount
    for tx in transactions:
        # La nota puede venir en distintos campos según la versión de la API
        tx_note = str(tx.get('orderMemo') or tx.get('remark') or tx.get('note') or '').strip().upper()
        
        # El monto y currency vienen en fundsDetail (array) o directamente
        tx_currency = ''
        tx_amount = 0.0
        funds = tx.get('fundsDetail') or []
        if funds and isinstance(funds, list):
            for f in funds:
                if str(f.get('currency', '')).upper() == 'USDT':
                    tx_currency = 'USDT'
                    tx_amount = abs(float(f.get('amount', 0)))
                    break
        if not tx_currency:
            tx_currency = str(tx.get('currency', '')).upper()
            tx_amount = abs(float(tx.get('amount', 0)))
        
        tx_order_type = str(tx.get('orderType', '')).upper()
        
        logger.info(f"  TX: note='{tx_note}', amount={tx_amount}, currency={tx_currency}, orderType={tx_order_type}")
        
        # Solo procesar transacciones recibidas - aceptar todos los tipos de ingreso
        if tx_order_type and tx_order_type not in ('PAY', 'C2C', 'C2C_TRANSFER', 'CRYPTO_BOX', ''):
            continue
        
        # Verificar: nota contiene el código de referencia Y monto coincide Y es USDT
        if codigo_ref.upper() in tx_note and abs(tx_amount - monto_esperado) < 0.01 and tx_currency == 'USDT':
            logger.info(f"  ¡Match encontrado! TX ID: {tx.get('transactionId', '')}")
            # ¡Match encontrado! Acreditar saldo
            bonus = 0.0
            monto_total = monto_esperado
            tx_id = tx.get('transactionId', '')
            
            try:
                # === Transacción atómica: idempotencia + bono + crédito ===
                conn2 = get_db_connection()

                # Verificar que no se haya procesado ya (idempotencia por binance_transaction_id)
                ya_procesada = conn2.execute(
                    'SELECT 1 FROM recargas_binance WHERE binance_transaction_id = ? AND estado = ?',
                    (tx_id, 'completada')
                ).fetchone()

                if ya_procesada:
                    conn2.close()
                    return {'status': 'ya_procesada', 'message': 'Esta transacción ya fue procesada'}

                # Calcular bono 1.5% si el usuario tiene bono_activo y monto >= 1000
                try:
                    user_row = conn2.execute(
                        'SELECT bono_activo FROM usuarios WHERE id = ?', (usuario_id,)
                    ).fetchone()
                    if user_row and user_row['bono_activo'] and monto_esperado >= 1000:
                        bonus = round(monto_esperado * 0.015, 2)
                        monto_total = monto_esperado + bonus
                        logger.info(f"Recarga {recarga_id}: bono_activo=True, monto={monto_esperado} >= 1000, bono={bonus}, total={monto_total}")
                except Exception as e_bonus:
                    logger.warning(f"Recarga {recarga_id}: error consultando bono_activo: {e_bonus}")

                # Actualizar recarga como completada
                conn2.execute('''
                    UPDATE recargas_binance 
                    SET estado = 'completada', binance_transaction_id = ?, fecha_completada = CURRENT_TIMESTAMP, bonus = ?
                    WHERE id = ?
                ''', (tx_id, bonus, recarga_id))

                # Acreditar saldo al usuario (atómico, misma transacción)
                saldo_row = conn2.execute('SELECT saldo FROM usuarios WHERE id = ?', (usuario_id,)).fetchone()
                saldo_anterior = saldo_row['saldo'] if saldo_row else 0.0
                conn2.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (monto_total, usuario_id))
                conn2.execute('''
                    INSERT INTO creditos_billetera (usuario_id, monto, saldo_anterior)
                    VALUES (?, ?, ?)
                ''', (usuario_id, monto_total, saldo_anterior))

                conn2.commit()
                conn2.close()

                logger.info(f"Recarga {recarga_id} completada: usuario={usuario_id}, monto={monto_esperado}, bonus={bonus}, total={monto_total}")

                return {
                    'status': 'completada',
                    'message': f'Recarga completada exitosamente',
                    'monto': monto_esperado,
                    'bonus': bonus,
                    'total_acreditado': monto_total,
                    'transaction_id': tx_id
                }
            except Exception as e:
                try:
                    conn2.rollback()
                    conn2.close()
                except Exception:
                    pass
                logger.error(f"Error acreditando recarga {recarga_id}: {e}", exc_info=True)
                return {'status': 'error', 'message': 'Error al acreditar saldo'}
    
    if len(transactions) > 0:
        logger.info(f"Recarga {recarga_id}: {len(transactions)} transacciones revisadas, ninguna coincide con codigo={codigo_ref} monto={monto_esperado}")
    
    return {'status': 'pendiente', 'message': 'Pago no detectado aún. Asegúrate de enviar el monto exacto con el código como nota y espera unos segundos.'}

def _ensure_recargas_table():
    """Crea la tabla recargas_binance si no existe"""
    conn = get_db_connection()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS recargas_binance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                codigo_referencia TEXT NOT NULL UNIQUE,
                monto_solicitado REAL NOT NULL,
                monto_unico REAL NOT NULL,
                estado TEXT DEFAULT 'pendiente',
                binance_transaction_id TEXT,
                fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
                fecha_expiracion DATETIME NOT NULL,
                fecha_completada DATETIME,
                bonus REAL DEFAULT 0.0,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_recargas_usuario ON recargas_binance(usuario_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_recargas_estado ON recargas_binance(estado)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_recargas_codigo ON recargas_binance(codigo_referencia)')
        conn.commit()
    except Exception as e:
        logger.error(f"Error creando tabla recargas_binance: {e}")
    finally:
        conn.close()

def expirar_recargas_vencidas():
    """Marca como expiradas las recargas que pasaron su tiempo límite"""
    _ensure_recargas_table()
    conn = get_db_connection()
    try:
        conn.execute('''
            UPDATE recargas_binance 
            SET estado = 'expirada' 
            WHERE estado = 'pendiente' AND fecha_expiracion < CURRENT_TIMESTAMP
        ''')
        conn.commit()
    except Exception as e:
        logger.error(f"Error expirando recargas: {e}")
    finally:
        conn.close()

def _utc_to_local(utc_str):
    """Convierte fecha UTC string o datetime a fecha local string"""
    if not utc_str:
        return utc_str
    try:
        tz = pytz.timezone(os.environ.get('DEFAULT_TZ', 'America/Caracas'))
        if isinstance(utc_str, datetime):
            utc_dt = utc_str if utc_str.tzinfo else pytz.utc.localize(utc_str)
        else:
            utc_dt = datetime.strptime(str(utc_str).split('.')[0], '%Y-%m-%d %H:%M:%S')
            utc_dt = pytz.utc.localize(utc_dt)
        local_dt = utc_dt.astimezone(tz)
        return local_dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(utc_str)

def _recarga_to_dict(row):
    """Convierte una fila de recarga a dict con fechas en hora local"""
    if not row:
        return None
    d = dict(row)
    d['fecha_creacion'] = _utc_to_local(d.get('fecha_creacion', ''))
    d['fecha_expiracion'] = _utc_to_local(d.get('fecha_expiracion', ''))
    d['fecha_completada'] = _utc_to_local(d.get('fecha_completada', ''))
    return d

def get_all_recargas_admin(limit=50):
    """Obtiene todas las recargas de todos los usuarios (para admin)"""
    _ensure_recargas_table()
    conn = get_db_connection()
    recargas = conn.execute('''
        SELECT r.*, u.nombre, u.apellido, u.correo
        FROM recargas_binance r
        LEFT JOIN usuarios u ON r.usuario_id = u.id
        ORDER BY r.fecha_creacion DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()
    result = []
    for r in recargas:
        d = _recarga_to_dict(r)
        d['nombre'] = r['nombre'] if r['nombre'] else ''
        d['apellido'] = r['apellido'] if r['apellido'] else ''
        d['correo'] = r['correo'] if r['correo'] else ''
        result.append(d)
    return result

def get_recargas_usuario(user_id, limit=20):
    """Obtiene el historial de recargas de un usuario"""
    conn = get_db_connection()
    # Crear tabla si no existe (compatibilidad)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS recargas_binance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER NOT NULL,
            codigo_referencia TEXT NOT NULL UNIQUE,
            monto_solicitado REAL NOT NULL,
            monto_unico REAL NOT NULL,
            estado TEXT DEFAULT 'pendiente',
            binance_transaction_id TEXT,
            fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
            fecha_expiracion DATETIME NOT NULL,
            fecha_completada DATETIME,
            bonus REAL DEFAULT 0.0,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    recargas = conn.execute('''
        SELECT * FROM recargas_binance 
        WHERE usuario_id = ? 
        ORDER BY fecha_creacion DESC 
        LIMIT ?
    ''', (user_id, limit)).fetchall()
    conn.close()
    return [_recarga_to_dict(r) for r in recargas]

def get_recarga_pendiente(user_id):
    """Obtiene la recarga pendiente activa de un usuario (si existe)"""
    expirar_recargas_vencidas()
    conn = get_db_connection()
    recarga = conn.execute('''
        SELECT * FROM recargas_binance 
        WHERE usuario_id = ? AND estado = 'pendiente'
        ORDER BY fecha_creacion DESC LIMIT 1
    ''', (user_id,)).fetchone()
    conn.close()
    return _recarga_to_dict(recarga)

def _binance_verification_loop():
    """Background thread que verifica periódicamente recargas pendientes"""
    while True:
        try:
            time_module.sleep(30)  # Verificar cada 30 segundos
            expirar_recargas_vencidas()
            
            if not BINANCE_API_KEY or not BINANCE_API_SECRET:
                continue
            
            conn = get_db_connection()
            pendientes = conn.execute('''
                SELECT id FROM recargas_binance WHERE estado = 'pendiente'
            ''').fetchall()
            conn.close()
            
            for rec in pendientes:
                try:
                    verificar_recarga_binance(rec['id'], _binance_tx_kwargs={'req_timeout': 15, 'total_timeout_override': 30})
                    time_module.sleep(2)  # Rate limit
                except Exception as e:
                    logger.error(f"Error verificando recarga {rec['id']}: {e}")
        except Exception as e:
            logger.error(f"Error en binance verification loop: {e}")
            time_module.sleep(60)

# Iniciar thread de verificación automática
_binance_verify_thread = threading.Thread(target=_binance_verification_loop, daemon=True)
_binance_verify_thread.start()


# === Juegos Dinámicos: Sincronización automática de precios cada 6 horas ===
_DYN_SYNC_INTERVAL_HOURS = float(os.environ.get('DYN_SYNC_INTERVAL_HOURS', '6'))

def _dyngame_price_sync_loop():
    """Hilo en background que sincroniza precios de juegos dinámicos cada N horas.
    Convierte precios GP (MYR) a USD usando la tasa configurada en el admin,
    manteniendo la ganancia fija por paquete (precio_venta - costo_compra)."""
    time_module.sleep(60)  # Esperar 60s al iniciar para que la app esté lista
    while True:
        try:
            from dynamic_games import sync_all_dynamic_games_prices
            results = sync_all_dynamic_games_prices()
            for dr in results:
                r = dr.get('result') or {}
                if r.get('error') or dr.get('error'):
                    logger.warning(f"[DynPrice AutoSync] {dr.get('game', '?')}: {r.get('error') or dr.get('error')}")
                else:
                    logger.info(f"[DynPrice AutoSync] {dr.get('game', '?')}: {r.get('packages_updated', 0)}/{r.get('total_gp', 0)} actualizados")
        except Exception as e:
            logger.error(f"[DynPrice AutoSync] Error: {e}")
        time_module.sleep(_DYN_SYNC_INTERVAL_HOURS * 3600)

_dyn_price_sync_thread = threading.Thread(target=_dyngame_price_sync_loop, daemon=True)
_dyn_price_sync_thread.start()
logger.info(f"[DynPrice AutoSync] Thread iniciado — sincronización cada {_DYN_SYNC_INTERVAL_HOURS}h")


# === Gift Cards: Polling de seriales pendientes cada 60s ===
def _dyngame_serial_poll_loop():
    """Hilo que verifica transacciones de Gift Cards pendientes y actualiza el serial."""
    time_module.sleep(45)  # Esperar al inicio
    while True:
        try:
            from dynamic_games import poll_pending_dynamic_transactions
            poll_pending_dynamic_transactions()
        except Exception as e:
            logger.error(f"[DynGame Poll Loop] Error: {e}")
        time_module.sleep(60)

_dyngame_poll_thread = threading.Thread(target=_dyngame_serial_poll_loop, daemon=True)
_dyngame_poll_thread.start()
logger.info("[DynGame Poll] Thread iniciado — verificación de Gift Cards pendientes cada 60s")

# Funciones para sistema de noticias
def create_news_table():
    """Crea la tabla de noticias si no existe"""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS noticias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT NOT NULL,
            contenido TEXT NOT NULL,
            importante BOOLEAN DEFAULT FALSE,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    # Migración: agregar columna imagen_url si no existe
    try:
        conn.execute('ALTER TABLE noticias ADD COLUMN imagen_url TEXT')
        conn.commit()
    except:
        pass
    conn.close()

def create_news_views_table():
    """Crea la tabla para rastrear noticias vistas por usuario"""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS noticias_vistas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            noticia_id INTEGER,
            fecha_vista DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id),
            FOREIGN KEY (noticia_id) REFERENCES noticias (id),
            UNIQUE(usuario_id, noticia_id)
        )
    ''')
    conn.commit()
    conn.close()

def create_news(titulo, contenido, importante=False, imagen_url=None):
    """Crea una nueva noticia"""
    create_news_table()
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO noticias (titulo, contenido, importante, imagen_url)
        VALUES (?, ?, ?, ?)
    ''', (titulo, contenido, importante, imagen_url))
    conn.commit()
    row = conn.execute('SELECT id FROM noticias ORDER BY id DESC LIMIT 1').fetchone()
    news_id = row['id'] if row else None
    conn.close()
    return news_id

def get_all_news():
    """Obtiene todas las noticias ordenadas por fecha (más recientes primero)"""
    create_news_table()
    conn = get_db_connection()
    news = conn.execute('''
        SELECT * FROM noticias 
        ORDER BY fecha DESC
    ''').fetchall()
    conn.close()
    return news

def get_user_news(user_id):
    """Obtiene las noticias para un usuario específico"""
    create_news_table()
    create_news_views_table()
    conn = get_db_connection()
    news = conn.execute('''
        SELECT * FROM noticias 
        ORDER BY fecha DESC
        LIMIT 20
    ''').fetchall()
    conn.close()
    return news

def get_unread_news_count(user_id):
    """Obtiene el número de noticias no leídas por un usuario"""
    create_news_table()
    create_news_views_table()
    conn = get_db_connection()
    
    # Contar noticias que el usuario no ha visto
    count = conn.execute('''
        SELECT COUNT(*) FROM noticias n
        WHERE n.id NOT IN (
            SELECT nv.noticia_id FROM noticias_vistas nv 
            WHERE nv.usuario_id = ?
        )
    ''', (user_id,)).fetchone()[0]
    conn.close()
    
    # Retornar 1 si hay noticias no leídas, 0 si no hay
    return 1 if count > 0 else 0

def mark_news_as_read(user_id):
    """Marca todas las noticias como leídas para un usuario"""
    create_news_table()
    create_news_views_table()
    conn = get_db_connection()
    
    # Obtener todas las noticias que el usuario no ha visto
    unread_news = conn.execute('''
        SELECT id FROM noticias 
        WHERE id NOT IN (
            SELECT noticia_id FROM noticias_vistas 
            WHERE usuario_id = ?
        )
    ''', (user_id,)).fetchall()
    
    # Marcar como vistas
    for news in unread_news:
        conn.execute('''
            INSERT INTO noticias_vistas (usuario_id, noticia_id)
            VALUES (%s, %s) ON CONFLICT DO NOTHING
        ''', (user_id, news['id']))
    
    conn.commit()
    conn.close()

def delete_news(news_id):
    """Elimina una noticia y sus registros de vistas"""
    conn = get_db_connection()
    # Eliminar registros de vistas
    conn.execute('DELETE FROM noticias_vistas WHERE noticia_id = ?', (news_id,))
    # Eliminar noticia
    conn.execute('DELETE FROM noticias WHERE id = ?', (news_id,))
    conn.commit()
    conn.close()

# Funciones para notificaciones personalizadas
def create_personal_notification(user_id, titulo, mensaje, tipo='success'):
    """Crea una notificación personalizada para un usuario específico"""
    conn = get_db_connection()
    # Crear tabla si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS notificaciones_personalizadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            titulo TEXT NOT NULL,
            mensaje TEXT NOT NULL,
            tipo TEXT DEFAULT 'info',
            tag TEXT,
            visto BOOLEAN DEFAULT FALSE,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')

    # Migración suave: agregar columna tag si la tabla ya existía
    try:
        conn.execute("ALTER TABLE notificaciones_personalizadas ADD COLUMN tag TEXT")
    except Exception:
        pass
    
    conn.execute('''
        INSERT INTO notificaciones_personalizadas (usuario_id, titulo, mensaje, tipo, tag)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, titulo, mensaje, tipo, None))
    conn.commit()
    row = conn.execute('SELECT id FROM notificaciones_personalizadas WHERE usuario_id = ? ORDER BY id DESC LIMIT 1', (user_id,)).fetchone()
    notification_id = row['id'] if row else None
    conn.close()
    return notification_id

def get_user_personal_notifications(user_id):
    """Obtiene las notificaciones personalizadas de un usuario"""
    conn = get_db_connection()
    # Crear tabla si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS notificaciones_personalizadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            titulo TEXT NOT NULL,
            mensaje TEXT NOT NULL,
            tipo TEXT DEFAULT 'info',
            tag TEXT,
            visto BOOLEAN DEFAULT FALSE,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')

    # Migración suave: agregar columna tag si la tabla ya existía
    try:
        conn.execute("ALTER TABLE notificaciones_personalizadas ADD COLUMN tag TEXT")
    except Exception:
        pass
    
    notifications = conn.execute('''
        SELECT * FROM notificaciones_personalizadas
        WHERE usuario_id = ?
          AND visto = FALSE
          AND (tag IS NULL OR tag != 'bloodstriker_reload')
        ORDER BY fecha DESC
        LIMIT 10
    ''', (user_id,)).fetchall()
    conn.close()
    return notifications

def get_unread_personal_notifications_count(user_id):
    """Obtiene el número de notificaciones personalizadas no leídas"""
    conn = get_db_connection()
    # Crear tabla si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS notificaciones_personalizadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            titulo TEXT NOT NULL,
            mensaje TEXT NOT NULL,
            tipo TEXT DEFAULT 'info',
            tag TEXT,
            visto BOOLEAN DEFAULT FALSE,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')

    # Migración suave: agregar columna tag si la tabla ya existía
    try:
        conn.execute("ALTER TABLE notificaciones_personalizadas ADD COLUMN tag TEXT")
    except Exception:
        pass
    
    count = conn.execute('''
        SELECT COUNT(*) FROM notificaciones_personalizadas
        WHERE usuario_id = ?
          AND visto = FALSE
          AND (tag IS NULL OR tag != 'bloodstriker_reload')
    ''', (user_id,)).fetchone()[0]
    conn.close()
    
    return 1 if count > 0 else 0

def mark_personal_notifications_as_read(user_id):
    """Marca todas las notificaciones personalizadas como leídas y las elimina"""
    conn = get_db_connection()
    # Crear tabla si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS notificaciones_personalizadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            titulo TEXT NOT NULL,
            mensaje TEXT NOT NULL,
            tipo TEXT DEFAULT 'info',
            tag TEXT,
            visto BOOLEAN DEFAULT FALSE,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')

    # Migración suave: agregar columna tag si la tabla ya existía
    try:
        conn.execute("ALTER TABLE notificaciones_personalizadas ADD COLUMN tag TEXT")
    except Exception:
        pass
    
    # Eliminar todas las notificaciones del usuario EXCEPTO las de recarga Blood Striker
    conn.execute('''
        DELETE FROM notificaciones_personalizadas
        WHERE usuario_id = ?
          AND (tag IS NULL OR tag != 'bloodstriker_reload')
    ''', (user_id,))
    conn.commit()
    conn.close()


# Función de debug para mostrar información de la base de datos
def debug_database_info():
    """Muestra información de debug sobre la base de datos"""
    print("=" * 50)
    print("[DEBUG] INFORMACION DE BASE DE DATOS")
    print("=" * 50)
    
    # Variables de entorno
    db_url = os.environ.get('DATABASE_URL', '').strip()
    db_path = os.environ.get('DATABASE_PATH', '').strip()
    print(f"RENDER: {os.environ.get('RENDER', 'No configurado')}")
    print(f"DATABASE_PATH: {db_path or 'No configurado'}")
    print(f"DATABASE_URL: {'Configurado' if db_url else 'No configurado'}")
    print(f"Directorio actual: {os.getcwd()}")

    # Si hay DB URL, asumimos PostgreSQL
    if db_url:
        try:
            conn = get_db_connection_optimized()
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
            ).fetchall()
            print(f"[INFO] Tablas PostgreSQL encontradas ({len(tables)}):")
            for t in tables[:25]:
                tname = t['table_name']
                try:
                    cnt = conn.execute(f'SELECT COUNT(*) AS c FROM "{tname}"').fetchone()['c']
                    print(f"   - {tname}: {cnt} registros")
                except Exception:
                    print(f"   - {tname}: Error al contar")
            conn.close()
        except Exception as e:
            print(f"[ERROR] Error conectando a PostgreSQL: {e}")
    elif db_path:
        print(f"Ruta de BD configurada: {db_path}")
        print(f"Ruta absoluta: {os.path.abspath(db_path)}")
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"[OK] Base de datos existe: {file_size} bytes")
            try:
                conn = get_db_connection_optimized()
                tables = conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
                ).fetchall()
                print(f"[INFO] Tablas encontradas ({len(tables)}):")
                for t in tables[:25]:
                    tname = t['table_name']
                    try:
                        cnt = conn.execute(f'SELECT COUNT(*) AS c FROM "{tname}"').fetchone()['c']
                        print(f"   - {tname}: {cnt} registros")
                    except Exception:
                        print(f"   - {tname}: Error al contar")
                conn.close()
            except Exception as e:
                print(f"[ERROR] Error conectando a BD: {e}")
        else:
            print(f"[ERROR] Base de datos NO existe en: {db_path}")
            db_dir = os.path.dirname(db_path)
            if db_dir:
                print(f"[DIR] Directorio padre: {db_dir}")
                print(f"   Existe: {os.path.exists(db_dir)}")
                if os.path.exists(db_dir):
                    try:
                        files = os.listdir(db_dir)
                        print(f"   Archivos: {files}")
                    except Exception:
                        print("   Error listando archivos")
    else:
        print("[WARN] No hay DATABASE_URL ni DATABASE_PATH configurados")
    
    print("=" * 50)

# Inicializar la base de datos al iniciar la aplicación
debug_database_info()
init_db()

@app.route('/')
def index():
    if 'usuario' not in session:
        return redirect('/auth')
    
    # Ejecutar limpieza automática de transacciones antiguas (solo en la primera carga)
    if request.args.get('page', 1, type=int) == 1:
        try:
            clean_old_transactions()
        except Exception as e:
            print(f"Error en limpieza automática de transacciones: {e}")
    
    # Obtener parámetros de paginación
    page = request.args.get('page', 1, type=int)
    per_page = 30  # Transacciones por página
    
    user_id = session.get('id', '00000')
    transactions_data = {}
    is_admin = session.get('is_admin', False)

    def _tx_fecha_sort_key(tx):
        v = (tx or {}).get('fecha', '')
        if isinstance(v, datetime):
            return v.isoformat(sep=' ')
        return str(v or '')
    
    if is_admin:
        # Admin ve todas las transacciones de todos los usuarios con paginación
        transactions_data = get_user_transactions(None, is_admin=True, page=page, per_page=per_page)
        
        # Para admin, también agregar transacciones pendientes de Blood Striker solo en la primera página
        # (Free Fire ID es 100% automático, no requiere aprobación manual)
        if page == 1:
            bloodstriker_transactions = get_pending_bloodstriker_transactions()
            # Combinar transacciones normales con las de Blood Striker
            all_transactions = list(transactions_data['transactions']) + list(bloodstriker_transactions)
            # Ordenar por fecha
            all_transactions.sort(key=_tx_fecha_sort_key, reverse=True)
            # Tomar solo las primeras per_page transacciones
            transactions_data['transactions'] = all_transactions[:per_page]
        
        balance = get_revendedores_balance()
    else:
        # Usuario normal ve solo sus transacciones
        if 'user_db_id' in session:
            # Actualizar saldo desde la base de datos SIEMPRE
            conn = get_db_connection()
            user = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (session['user_db_id'],)).fetchone()
            if user:
                session['saldo'] = user['saldo']
                balance = user['saldo']
            else:
                balance = 0
            conn.close()
            
            # Obtener transacciones normales del usuario con paginación
            transactions_data = get_user_transactions(session['user_db_id'], is_admin=False, page=page, per_page=per_page)
            
            # Para usuario normal, también agregar transacciones pendientes de Blood Striker solo en la primera página
            # (Free Fire ID es 100% automático, no requiere aprobación manual)
            if page == 1:
                user_bloodstriker_transactions = get_user_pending_bloodstriker_transactions(session['user_db_id'])
                # Combinar transacciones normales con las de Blood Striker del usuario
                all_user_transactions = list(transactions_data['transactions']) + list(user_bloodstriker_transactions)
                # Ordenar por fecha
                all_user_transactions.sort(key=_tx_fecha_sort_key, reverse=True)
                # Tomar solo las primeras per_page transacciones
                transactions_data['transactions'] = all_user_transactions[:per_page]
        else:
            balance = 0
            transactions_data = {'transactions': [], 'pagination': {'page': 1, 'total_pages': 0, 'has_prev': False, 'has_next': False}}
    
    # Obtener contador de notificaciones de cartera para usuarios normales
    wallet_notification_count = 0
    if not is_admin and 'user_db_id' in session:
        wallet_notification_count = get_unread_wallet_credits_count(session['user_db_id'])
    
    # Obtener contador de notificaciones de noticias
    news_notification_count = 0
    personal_notification_count = 0
    if 'user_db_id' in session:
        news_notification_count = get_unread_news_count(session['user_db_id'])
        # Obtener contador de notificaciones personalizadas (solo para usuarios normales)
        if not is_admin:
            personal_notification_count = get_unread_personal_notifications_count(session['user_db_id'])
    
    # Combinar notificaciones de noticias y personalizadas
    total_notification_count = news_notification_count + personal_notification_count
    
    return render_template('index.html', 
                         user_id=user_id, 
                         balance=balance, 
                         transactions=transactions_data['transactions'],
                         pagination=transactions_data['pagination'],
                         is_admin=is_admin,
                         wallet_notification_count=wallet_notification_count,
                         news_notification_count=news_notification_count,
                         personal_notification_count=personal_notification_count,
                         total_notification_count=total_notification_count,
                         games_active=get_games_active())

def _get_aviso_config():
    """Lee config del banner de redirección desde la BD."""
    try:
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT clave, valor FROM configuracion_redeemer WHERE clave IN ('aviso_activo', 'aviso_url')"
        ).fetchall()
        conn.close()
        cfg = {r['clave']: r['valor'] for r in rows}
        return {
            'activo': cfg.get('aviso_activo', '0') == '1',
            'url': cfg.get('aviso_url', ''),
        }
    except Exception:
        return {'activo': False, 'url': ''}


@app.route('/auth')
def auth():
    aviso = _get_aviso_config()
    return render_template('auth.html', aviso=aviso)


@app.route('/control-aviso')
def control_aviso():
    aviso = _get_aviso_config()
    return render_template('redirect_panel.html', aviso=aviso)


@app.route('/control-aviso/guardar', methods=['POST'])
def control_aviso_guardar():
    activo = '1' if request.form.get('activo') == '1' else '0'
    url = request.form.get('url', '').strip()
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion) VALUES ('aviso_activo', %s, NOW()) ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion",
            (activo,))
        conn.execute(
            "INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion) VALUES ('aviso_url', %s, NOW()) ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion",
            (url,))
        conn.commit()
        conn.close()
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
    return jsonify({'ok': True, 'activo': activo == '1', 'url': url})

@app.route('/login', methods=['POST'])
def login():
    correo = request.form.get('correo', '').strip()
    contraseña = request.form.get('contraseña', '')
    
    if not correo or not contraseña:
        flash('Por favor, complete todos los campos', 'error')
        return redirect('/auth')
    
    # Verificar credenciales de administrador (desde variables de entorno)
    admin_email = os.environ.get('ADMIN_EMAIL', 'admin@inefable.com').strip()
    admin_password = os.environ.get('ADMIN_PASSWORD', 'InefableAdmin2024!')
    
    _is_prod = bool(os.environ.get('DATABASE_URL'))
    dev_login = not _is_prod and correo == 'admin' and contraseña == '123456'
    if dev_login or (correo.lower() == admin_email.lower() and contraseña == admin_password):
        logger.info(f"[Login] Admin login OK para {admin_email}")
        # Buscar o crear usuario admin en la base de datos
        conn = get_db_connection()
        admin_user = conn.execute('SELECT * FROM usuarios WHERE correo = ?', (admin_email,)).fetchone()
        
        if not admin_user:
            # Crear usuario admin si no existe
            hashed_password = hash_password(admin_password)
            conn.execute('''
                INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ('Administrador', 'Sistema', '00000000000', admin_email, hashed_password, 0))
            conn.commit()
            admin_user = conn.execute('SELECT * FROM usuarios WHERE correo = ?', (admin_email,)).fetchone()
        
        conn.close()
        
        session.permanent = True  # Activar duración de sesión de 30 minutos
        session['usuario'] = admin_email
        session['nombre'] = admin_user['nombre']
        session['apellido'] = admin_user['apellido']
        session['id'] = str(admin_user['id']).zfill(5)
        session['user_db_id'] = admin_user['id']
        session['saldo'] = 0
        session['is_admin'] = True
        return redirect('/')
    
    # Buscar usuario en la base de datos
    user = get_user_by_email(correo)
    
    if user and verify_password(contraseña, user['contraseña']):
        # Migrar contraseña antigua a nuevo formato si es necesario
        if not user['contraseña'].startswith('pbkdf2:'):
            # Actualizar contraseña al nuevo formato seguro
            new_hashed_password = hash_password(contraseña)
            conn = get_db_connection()
            conn.execute('UPDATE usuarios SET contraseña = ? WHERE id = ?', 
                        (new_hashed_password, user['id']))
            conn.commit()
            conn.close()
            print(f"Contraseña migrada para usuario: {user['correo']}")
        
        # Login exitoso
        session.permanent = True  # Activar duración de sesión de 30 minutos
        session['usuario'] = user['correo']
        session['nombre'] = user['nombre']
        session['apellido'] = user['apellido']
        session['id'] = str(user['id']).zfill(5)
        session['user_db_id'] = user['id']
        session['saldo'] = user['saldo']
        session['is_admin'] = False
        return redirect('/')
    else:
        flash('Credenciales incorrectas', 'error')
        return redirect('/auth')


@app.route('/api/notifications/bloodstriker_reload', methods=['GET'])
def api_bloodstriker_reload_notifications():
    if 'usuario' not in session or session.get('is_admin'):
        return jsonify({'notifications': []})

    user_id = session.get('user_db_id')
    if not user_id:
        return jsonify({'notifications': []})

    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS notificaciones_personalizadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            titulo TEXT NOT NULL,
            mensaje TEXT NOT NULL,
            tipo TEXT DEFAULT 'info',
            tag TEXT,
            visto BOOLEAN DEFAULT FALSE,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    try:
        conn.execute("ALTER TABLE notificaciones_personalizadas ADD COLUMN tag TEXT")
    except Exception:
        pass

    rows = conn.execute('''
        SELECT id, titulo, mensaje, tipo, fecha
        FROM notificaciones_personalizadas
        WHERE usuario_id = ? AND visto = FALSE AND tag = 'bloodstriker_reload'
        ORDER BY fecha DESC
        LIMIT 5
    ''', (user_id,)).fetchall()
    conn.close()

    notifications = [dict(r) for r in rows]
    return jsonify({'notifications': notifications})


@app.route('/api/notifications/dismiss/<int:notification_id>', methods=['POST'])
def api_dismiss_notification(notification_id):
    if 'usuario' not in session or session.get('is_admin'):
        return jsonify({'status': 'error'}), 401

    user_id = session.get('user_db_id')
    if not user_id:
        return jsonify({'status': 'error'}), 400

    conn = get_db_connection()
    try:
        conn.execute('''
            DELETE FROM notificaciones_personalizadas
            WHERE id = ? AND usuario_id = ? AND tag = 'bloodstriker_reload'
        ''', (notification_id, user_id))
        conn.commit()
    finally:
        conn.close()

    return jsonify({'status': 'ok'})

@app.route('/register', methods=['POST'])
def register():
    nombre = request.form.get('nombre')
    apellido = request.form.get('apellido')
    telefono = request.form.get('telefono')
    correo = request.form.get('correo')
    contraseña = request.form.get('contraseña')
    
    # Validar que todos los campos estén completos
    if not all([nombre, apellido, telefono, correo, contraseña]):
        flash('Por favor, complete todos los campos', 'error')
        return redirect('/auth')
    
    # Verificar si el usuario ya existe
    if get_user_by_email(correo):
        flash('El correo electrónico ya está registrado', 'error')
        return redirect('/auth')
    
    # Crear nuevo usuario
    user_id = create_user(nombre, apellido, telefono, correo, contraseña)
    
    if user_id:
        # Registro exitoso, iniciar sesión automáticamente
        session.permanent = True  # Activar duración de sesión de 30 minutos
        session['usuario'] = correo
        session['nombre'] = nombre
        session['apellido'] = apellido
        session['id'] = str(user_id).zfill(5)
        session['user_db_id'] = user_id
        session['saldo'] = 0.0  # Saldo inicial
        flash('Registro exitoso. ¡Bienvenido!', 'success')
        return redirect('/')
    else:
        flash('Error al crear la cuenta. Intente nuevamente.', 'error')
        return redirect('/auth')


# Funciones de administrador
def get_all_users():
    """Obtiene todos los usuarios registrados"""
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM usuarios ORDER BY fecha_registro DESC').fetchall()
    conn.close()
    return users

def update_user_balance(user_id, new_balance):
    """Actualiza el saldo de un usuario"""
    conn = get_db_connection()
    conn.execute('UPDATE usuarios SET saldo = ? WHERE id = ?', (new_balance, user_id))
    conn.commit()
    conn.close()

def delete_user(user_id):
    """Elimina un usuario y todos sus datos relacionados"""
    conn = get_db_connection()
    # Eliminar transacciones del usuario
    conn.execute('DELETE FROM transacciones WHERE usuario_id = ?', (user_id,))
    # Eliminar créditos de billetera del usuario
    conn.execute('DELETE FROM creditos_billetera WHERE usuario_id = ?', (user_id,))
    # Eliminar usuario
    conn.execute('DELETE FROM usuarios WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()

def add_credit_to_user(user_id, amount):
    """Añade crédito al saldo de un usuario y registra en billetera"""
    conn = get_db_connection()
    
    # Crear tabla de créditos de billetera si no existe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS creditos_billetera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            monto REAL DEFAULT 0.0,
            saldo_anterior REAL DEFAULT 0.0,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')
    
    # Agregar columna 'saldo_anterior' si no existe (para compatibilidad con datos existentes)
    try:
        conn.execute('ALTER TABLE creditos_billetera ADD COLUMN saldo_anterior REAL DEFAULT 0.0')
        conn.commit()
    except:
        pass  # La columna ya existe
    
    # Obtener saldo actual del usuario antes de agregar el crédito
    user_data = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
    saldo_anterior = user_data['saldo'] if user_data else 0.0
    
    # Actualizar saldo del usuario
    conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (amount, user_id))
    
    # Registrar en créditos de billetera (monto, fecha y saldo anterior)
    conn.execute('''
        INSERT INTO creditos_billetera (usuario_id, monto, saldo_anterior)
        VALUES (?, ?, ?)
    ''', (user_id, amount, saldo_anterior))
    
    # Limitar créditos de billetera a 10 por usuario - eliminar los más antiguos si hay más de 10
    conn.execute('''
        DELETE FROM creditos_billetera 
        WHERE usuario_id = ? AND id NOT IN (
            SELECT id FROM creditos_billetera 
            WHERE usuario_id = ? 
            ORDER BY fecha DESC 
            LIMIT 10
        )
    ''', (user_id, user_id))
    
    conn.commit()
    conn.close()

# Funciones para pines de Free Fire
def add_pin_freefire(monto_id, pin_codigo):
    """Añade un pin de Free Fire al stock"""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO pines_freefire (monto_id, pin_codigo, batch_id)
        VALUES (?, ?, NULL)
    ''', (monto_id, pin_codigo))
    conn.commit()
    conn.close()

def add_pins_batch(monto_id, pins_list):
    """Añade múltiples pines de Free Fire al stock en lote"""
    conn = get_db_connection()
    try:
        batch_id = _generate_batch_id()
        for pin_codigo in pins_list:
            pin_codigo = pin_codigo.strip()
            if pin_codigo:  # Solo agregar si el pin no está vacío
                conn.execute('''
                    INSERT INTO pines_freefire (monto_id, pin_codigo, batch_id)
                    VALUES (?, ?, ?)
                ''', (monto_id, pin_codigo, batch_id))
        conn.commit()
        return len([p for p in pins_list if p.strip()])  # Retornar cantidad agregada
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_pin_stock():
    """Obtiene el stock de pines por monto_id"""
    conn = get_db_connection()
    stock = {}
    for i in range(1, 10):  # monto_id del 1 al 9
        count = conn.execute('''
            SELECT COUNT(*) FROM pines_freefire 
            WHERE monto_id = ? AND usado = FALSE
        ''', (i,)).fetchone()[0]
        stock[i] = count
    conn.close()
    return stock

def get_available_pin(monto_id):
    """Obtiene un pin disponible para el monto especificado"""
    conn = get_db_connection()
    pin = conn.execute('''
        SELECT * FROM pines_freefire 
        WHERE monto_id = ? AND usado = FALSE 
        LIMIT 1
    ''', (monto_id,)).fetchone()
    conn.close()
    return pin


def get_all_pins():
    """Obtiene todos los pines para el admin"""
    conn = get_db_connection()
    pins = conn.execute('''
        SELECT p.*, u.nombre, u.apellido 
        FROM pines_freefire p
        LEFT JOIN usuarios u ON p.usuario_id = u.id
        ORDER BY p.fecha_agregado DESC
    ''').fetchall()
    conn.close()
    return pins

def get_pins_by_game(game_type, only_unused=True, monto_id=None):
    """Obtiene pines por juego (freefire_latam o freefire_global), con filtro opcional por monto_id."""
    if game_type.startswith('dyn_'):
        game = get_dynamic_game_by_slug(game_type[4:])
        if not game or not game.get('usa_stock_local'):
            return []
        conn = get_db_connection()
        try:
            params = [game['id']]
            where_clauses = ['juego_id = ?']
            if only_unused:
                where_clauses.append('usado = FALSE')
            if monto_id is not None:
                where_clauses.append('paquete_id = ?')
                params.append(monto_id)
            where_sql = 'WHERE ' + ' AND '.join(where_clauses)
            query = f'''
                SELECT id, paquete_id AS monto_id, pin_codigo, usado, fecha_agregado, batch_id
                FROM pines_dinamicos_stock
                {where_sql}
                ORDER BY fecha_agregado DESC
            '''
            return conn.execute(query, tuple(params)).fetchall()
        finally:
            conn.close()

    table = 'pines_freefire' if game_type == 'freefire_latam' else 'pines_freefire_global'
    conn = get_db_connection()
    try:
        params = []
        where_clauses = []
        if only_unused:
            where_clauses.append('usado = FALSE')
        if monto_id is not None:
            where_clauses.append('monto_id = ?')
            params.append(monto_id)
        where_sql = ('WHERE ' + ' AND '.join(where_clauses)) if where_clauses else ''
        query = f'''
            SELECT id, monto_id, pin_codigo, usado, fecha_agregado, batch_id
            FROM {table}
            {where_sql}
            ORDER BY fecha_agregado DESC
        '''
        pins = conn.execute(query, tuple(params)).fetchall()
        return pins
    finally:
        conn.close()


def delete_pins_by_batch_id(game_type, batch_id, only_unused=True):
    """Elimina pines de un lote específico (batch_id) para el juego indicado."""
    if not batch_id:
        return 0
    if game_type.startswith('dyn_'):
        game = get_dynamic_game_by_slug(game_type[4:])
        if not game or not game.get('usa_stock_local'):
            return 0
        conn = get_db_connection()
        try:
            where_sql = 'juego_id = ? AND batch_id = ?'
            params = [game['id'], str(batch_id)]
            if only_unused:
                where_sql += ' AND usado = FALSE'
            cursor = conn.execute(f'DELETE FROM pines_dinamicos_stock WHERE {where_sql}', tuple(params))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
    table = 'pines_freefire' if game_type == 'freefire_latam' else 'pines_freefire_global'
    conn = get_db_connection()
    try:
        where_sql = "batch_id = ?"
        params = [str(batch_id)]
        if only_unused:
            where_sql += " AND usado = FALSE"
        cursor = conn.execute(f"DELETE FROM {table} WHERE {where_sql}", tuple(params))
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()

def delete_pins_by_ids(game_type, pin_ids):
    """Elimina pines por IDs para el juego indicado."""
    if not pin_ids:
        return 0
    if game_type.startswith('dyn_'):
        game = get_dynamic_game_by_slug(game_type[4:])
        if not game or not game.get('usa_stock_local'):
            return 0
        placeholders = ','.join('?' for _ in pin_ids)
        conn = get_db_connection()
        try:
            cursor = conn.execute(
                f'DELETE FROM pines_dinamicos_stock WHERE juego_id = ? AND id IN ({placeholders})',
                (game['id'], *pin_ids)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
    table = 'pines_freefire' if game_type == 'freefire_latam' else 'pines_freefire_global'
    placeholders = ','.join('?' for _ in pin_ids)
    conn = get_db_connection()
    try:
        cursor = conn.execute(f"DELETE FROM {table} WHERE id IN ({placeholders})", tuple(pin_ids))
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()

def remove_duplicate_pins():
    """Elimina pines duplicados de la base de datos, manteniendo el más reciente de cada código"""
    conn = get_db_connection()
    try:
        # Encontrar pines duplicados y eliminar los más antiguos
        duplicates_removed = conn.execute('''
            DELETE FROM pines_freefire 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM pines_freefire 
                GROUP BY pin_codigo, monto_id
            )
        ''').rowcount
        
        conn.commit()
        return duplicates_removed
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def add_dynamic_stock_pin(game_id, package_id, pin_codigo):
    """Añade un pin al stock local de un juego dinámico de tarjeta."""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO pines_dinamicos_stock (juego_id, paquete_id, pin_codigo, batch_id)
        VALUES (?, ?, ?, NULL)
    ''', (game_id, package_id, pin_codigo))
    conn.commit()
    conn.close()


def add_dynamic_stock_pins_batch(game_id, package_id, pins_list):
    """Añade múltiples pines al stock local de un juego dinámico."""
    conn = get_db_connection()
    try:
        batch_id = _generate_batch_id()
        added = 0
        for pin_codigo in pins_list:
            pin_codigo = (pin_codigo or '').strip()
            if not pin_codigo:
                continue
            conn.execute('''
                INSERT INTO pines_dinamicos_stock (juego_id, paquete_id, pin_codigo, batch_id)
                VALUES (?, ?, ?, ?)
            ''', (game_id, package_id, pin_codigo, batch_id))
            added += 1
        conn.commit()
        return added
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_dynamic_stock_counts():
    """Devuelve conteos de stock local por juego y paquete dinámico."""
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            SELECT juego_id, paquete_id, COUNT(*) AS count
            FROM pines_dinamicos_stock
            WHERE usado = FALSE
            GROUP BY juego_id, paquete_id
        ''').fetchall()
        data = {}
        for row in rows:
            juego_id = int(row['juego_id'])
            paquete_id = int(row['paquete_id'])
            data.setdefault(juego_id, {})[paquete_id] = int(row['count'])
        return data
    finally:
        conn.close()


def get_dynamic_stock_games():
    """Lista de juegos dinámicos que usan stock local de tarjetas."""
    return [g for g in get_dynamic_games_list(only_active=False) if g.get('usa_stock_local')]

def get_duplicate_pins_count():
    """Obtiene el número de pines duplicados en la base de datos"""
    conn = get_db_connection()
    try:
        # Contar pines duplicados
        result = conn.execute('''
            SELECT COUNT(*) - COUNT(DISTINCT pin_codigo || '-' || monto_id) as duplicates
            FROM pines_freefire
            WHERE usado = FALSE
        ''').fetchone()
        
        return result[0] if result else 0
    finally:
        conn.close()

# Funciones para gestión de precios
def get_all_prices():
    """Obtiene todos los precios de paquetes"""
    conn = get_db_connection()
    prices = conn.execute('''
        SELECT * FROM precios_paquetes 
        ORDER BY id
    ''').fetchall()
    conn.close()
    return prices

def get_price_by_id(monto_id):
    """Obtiene el precio de un paquete específico"""
    conn = get_db_connection()
    price = conn.execute('''
        SELECT precio FROM precios_paquetes 
        WHERE id = ? AND activo = TRUE
    ''', (monto_id,)).fetchone()
    conn.close()
    return price['precio'] if price else 0


def get_price_by_id_any(monto_id):
    """Obtiene el precio de un paquete específico (incluye inactivos)"""
    conn = get_db_connection()
    price = conn.execute('''
        SELECT precio FROM precios_paquetes
        WHERE id = ?
    ''', (monto_id,)).fetchone()
    conn.close()
    return price['precio'] if price else 0

def update_package_price(package_id, new_price):
    """Actualiza el precio de un paquete"""
    conn = get_db_connection_optimized()
    try:
        conn.execute('''
            UPDATE precios_paquetes 
            SET precio = ?, fecha_actualizacion = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_price, package_id))
        conn.commit()
        # Limpiar cache después de actualizar precios
        clear_price_cache()
    finally:
        return_db_connection(conn)

def update_package_name(package_id, new_name):
    """Actualiza el nombre de un paquete"""
    conn = get_db_connection_optimized()
    try:
        conn.execute('''
            UPDATE precios_paquetes 
            SET nombre = ?, fecha_actualizacion = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_name, package_id))
        conn.commit()
        # Limpiar cache después de actualizar nombres
        clear_price_cache()
    finally:
        return_db_connection(conn)

def get_package_info_with_prices():
    """Obtiene información de paquetes con precios dinámicos"""
    conn = get_db_connection()
    packages = conn.execute('''
        SELECT id, nombre, precio, descripcion 
        FROM precios_paquetes 
        WHERE activo = TRUE 
        ORDER BY id
    ''').fetchall()
    conn.close()
    
    # Convertir a diccionario para fácil acceso
    package_dict = {}
    for package in packages:
        package_dict[package['id']] = {
            'nombre': package['nombre'],
            'precio': package['precio'],
            'descripcion': package['descripcion']
        }
    
    return package_dict

# Funciones para Blood Striker
def get_bloodstriker_prices():
    """Obtiene información de paquetes de Blood Striker con precios dinámicos"""
    conn = get_db_connection()
    packages = conn.execute('''
        SELECT id, nombre, precio, descripcion, gamepoint_package_id 
        FROM precios_bloodstriker 
        WHERE activo = TRUE AND gamepoint_package_id IS NOT NULL 
        ORDER BY id
    ''').fetchall()
    conn.close()
    
    # Convertir a diccionario para fácil acceso
    package_dict = {}
    for package in packages:
        package_dict[package['id']] = {
            'nombre': package['nombre'],
            'precio': package['precio'],
            'descripcion': package['descripcion'],
            'gamepoint_package_id': package['gamepoint_package_id']
        }
    
    return package_dict

def get_bloodstriker_price_by_id(package_id):
    """Obtiene el precio de un paquete específico de Blood Striker"""
    conn = get_db_connection()
    price = conn.execute('''
        SELECT precio FROM precios_bloodstriker 
        WHERE id = ? AND activo = TRUE
    ''', (package_id,)).fetchone()
    conn.close()
    return price['precio'] if price else 0


def get_bloodstriker_price_by_id_any(package_id):
    """Obtiene el precio de un paquete específico de Blood Striker (incluye inactivos)"""
    conn = get_db_connection()
    price = conn.execute('''
        SELECT precio FROM precios_bloodstriker
        WHERE id = ?
    ''', (package_id,)).fetchone()
    conn.close()
    return price['precio'] if price else 0

def create_bloodstriker_transaction(user_id, player_id, package_id, precio, estado='pendiente', gamepoint_referenceno=None):
    """Crea una transacción de Blood Striker"""
    import random
    import string
    
    conn = get_db_connection()
    try:
        # Idempotencia: si ya existe una transacción con este gamepoint_referenceno, no duplicar
        if gamepoint_referenceno:
            dup = conn.execute(
                'SELECT id, numero_control, transaccion_id FROM transacciones_bloodstriker WHERE gamepoint_referenceno = ?',
                (gamepoint_referenceno,)
            ).fetchone()
            if dup:
                conn.close()
                return {
                    'id': dup['id'],
                    'numero_control': dup['numero_control'],
                    'transaccion_id': dup['transaccion_id']
                }

        # Generar datos de la transacción
        numero_control = ''.join(random.choices(string.digits, k=10))
        transaccion_id = 'BS-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

        conn.execute('''
            INSERT INTO transacciones_bloodstriker 
            (usuario_id, player_id, paquete_id, numero_control, transaccion_id, monto, estado, gamepoint_referenceno)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, player_id, package_id, numero_control, transaccion_id, -precio, estado, gamepoint_referenceno))
        conn.commit()
        row = conn.execute('SELECT id FROM transacciones_bloodstriker WHERE transaccion_id = ?', (transaccion_id,)).fetchone()
        transaction_id = row['id'] if row else None
        return {
            'id': transaction_id,
            'numero_control': numero_control,
            'transaccion_id': transaccion_id
        }
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_pending_bloodstriker_transactions():
    """Obtiene todas las transacciones pendientes de Blood Striker para el admin"""
    conn = get_db_connection()
    transactions = conn.execute('''
        SELECT bs.*, u.nombre, u.apellido, u.correo, p.nombre as paquete_nombre
        FROM transacciones_bloodstriker bs
        JOIN usuarios u ON bs.usuario_id = u.id
        JOIN precios_bloodstriker p ON bs.paquete_id = p.id
        WHERE bs.estado = 'pendiente'
        ORDER BY bs.fecha DESC
    ''').fetchall()
    
    # Formatear las transacciones de Blood Striker para que sean compatibles con el template
    formatted_transactions = []
    for transaction in transactions:
        formatted_transaction = {
            'id': transaction['id'],
            'usuario_id': transaction['usuario_id'],
            'numero_control': transaction['numero_control'],
            'transaccion_id': transaction['transaccion_id'],
            'monto': transaction['monto'],
            'fecha': transaction['fecha'],
            'nombre': transaction['nombre'],
            'apellido': transaction['apellido'],
            'paquete': transaction['paquete_nombre'],
            'pin': f"ID: {transaction['player_id']}",  # Mostrar Player ID en lugar de PIN
            'estado': transaction['estado'],
            'is_bloodstriker': True  # Marcar como transacción de Blood Striker
        }
        formatted_transactions.append(formatted_transaction)
    
    conn.close()
    return formatted_transactions

def get_user_pending_bloodstriker_transactions(user_id):
    """Obtiene las transacciones pendientes de Blood Striker de un usuario específico"""
    conn = get_db_connection()
    transactions = conn.execute('''
        SELECT bs.*, u.nombre, u.apellido, p.nombre as paquete_nombre
        FROM transacciones_bloodstriker bs
        JOIN usuarios u ON bs.usuario_id = u.id
        JOIN precios_bloodstriker p ON bs.paquete_id = p.id
        WHERE bs.usuario_id = ? AND bs.estado = 'pendiente'
        ORDER BY bs.fecha DESC
    ''', (user_id,)).fetchall()
    
    # Formatear las transacciones de Blood Striker para que sean compatibles con el template
    formatted_transactions = []
    for transaction in transactions:
        formatted_transaction = {
            'id': transaction['id'],
            'usuario_id': transaction['usuario_id'],
            'numero_control': transaction['numero_control'],
            'transaccion_id': transaction['transaccion_id'],
            'monto': transaction['monto'],
            'fecha': convert_to_venezuela_time(transaction['fecha']),  # Convertir a zona horaria de Venezuela
            'nombre': transaction['nombre'],
            'apellido': transaction['apellido'],
            'paquete': transaction['paquete_nombre'],
            'pin': f"ID: {transaction['player_id']}",  # Mostrar Player ID del usuario
            'estado': transaction['estado'],
            'is_bloodstriker': True  # Marcar como transacción de Blood Striker
        }
        formatted_transactions.append(formatted_transaction)
    
    conn.close()
    return formatted_transactions

def update_bloodstriker_transaction_status(transaction_id, new_status, admin_id, notas=None):
    """Actualiza el estado de una transacción de Blood Striker"""
    conn = get_db_connection()
    conn.execute('''
        UPDATE transacciones_bloodstriker 
        SET estado = ?, admin_id = ?, notas = ?, fecha_procesado = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (new_status, admin_id, notas, transaction_id))
    conn.commit()
    conn.close()

def update_bloodstriker_price(package_id, new_price):
    """Actualiza el precio de un paquete de Blood Striker"""
    conn = get_db_connection_optimized()
    try:
        conn.execute('''
            UPDATE precios_bloodstriker 
            SET precio = ?, fecha_actualizacion = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_price, package_id))
        conn.commit()
        # Limpiar cache después de actualizar precios
        clear_price_cache()
    finally:
        return_db_connection(conn)

def update_bloodstriker_name(package_id, new_name):
    """Actualiza el nombre de un paquete de Blood Striker"""
    conn = get_db_connection_optimized()
    try:
        conn.execute('''
            UPDATE precios_bloodstriker 
            SET nombre = ?, fecha_actualizacion = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_name, package_id))
        conn.commit()
        # Limpiar cache después de actualizar nombres
        clear_price_cache()
    finally:
        return_db_connection(conn)

def get_all_bloodstriker_prices():
    """Obtiene todos los precios de paquetes de Blood Striker"""
    conn = get_db_connection()
    prices = conn.execute('''
        SELECT * FROM precios_bloodstriker 
        ORDER BY id
    ''').fetchall()
    conn.close()
    return prices

# Funciones para Free Fire ID
def get_freefire_id_prices():
    """Obtiene información de paquetes de Free Fire ID con precios dinámicos"""
    conn = get_db_connection()
    packages = conn.execute('''
        SELECT id, nombre, precio, descripcion 
        FROM precios_freefire_id 
        WHERE activo = TRUE 
        ORDER BY id
    ''').fetchall()
    conn.close()
    
    package_dict = {}
    for package in packages:
        package_dict[package['id']] = {
            'nombre': package['nombre'],
            'precio': package['precio'],
            'descripcion': package['descripcion']
        }
    
    return package_dict

def get_freefire_id_price_by_id(package_id):
    """Obtiene el precio de un paquete específico de Free Fire ID"""
    conn = get_db_connection()
    price = conn.execute('''
        SELECT precio FROM precios_freefire_id 
        WHERE id = ? AND activo = TRUE
    ''', (package_id,)).fetchone()
    conn.close()
    return price['precio'] if price else 0


def get_freefire_id_price_by_id_any(package_id):
    """Obtiene el precio de un paquete específico de Free Fire ID (incluye inactivos)"""
    conn = get_db_connection()
    price = conn.execute('''
        SELECT precio FROM precios_freefire_id
        WHERE id = ?
    ''', (package_id,)).fetchone()
    conn.close()
    return price['precio'] if price else 0

@lru_cache(maxsize=128)
def get_freefire_id_prices_cached():
    """Versión cacheada de precios de Free Fire ID"""
    conn = get_db_connection_optimized()
    try:
        packages = conn.execute('''
            SELECT id, nombre, precio, descripcion 
            FROM precios_freefire_id 
            WHERE activo = TRUE 
            ORDER BY id
        ''').fetchall()
        
        package_dict = {}
        for package in packages:
            package_dict[package['id']] = {
                'nombre': package['nombre'],
                'precio': package['precio'],
                'descripcion': package['descripcion']
            }
        return package_dict
    finally:
        return_db_connection(conn)

def create_freefire_id_transaction(user_id, player_id, package_id, precio, pin_codigo=None):
    """Crea una transacción pendiente de Free Fire ID"""
    import random
    import string
    
    numero_control = ''.join(random.choices(string.digits, k=10))
    transaccion_id = 'FFID-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO transacciones_freefire_id 
            (usuario_id, player_id, paquete_id, numero_control, transaccion_id, monto, estado, pin_codigo)
            VALUES (?, ?, ?, ?, ?, ?, 'pendiente', ?)
        ''', (user_id, player_id, package_id, numero_control, transaccion_id, -precio, pin_codigo))
        conn.commit()
        row = conn.execute('SELECT id FROM transacciones_freefire_id WHERE transaccion_id = ?', (transaccion_id,)).fetchone()
        transaction_id = row['id'] if row else None
        return {
            'id': transaction_id,
            'numero_control': numero_control,
            'transaccion_id': transaccion_id
        }
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_pending_freefire_id_transactions():
    """Obtiene transacciones FFID pendientes para revisión manual (no las recientes en auto-proceso)."""
    conn = get_db_connection()
    transactions = conn.execute('''
        SELECT fi.*, u.nombre, u.apellido, u.correo, p.nombre as paquete_nombre
        FROM transacciones_freefire_id fi
        JOIN usuarios u ON fi.usuario_id = u.id
        JOIN precios_freefire_id p ON fi.paquete_id = p.id
        WHERE fi.estado = 'pendiente'
          AND datetime(fi.fecha) <= datetime('now', '-3 minutes')
        ORDER BY fi.fecha DESC
    ''').fetchall()
    
    formatted_transactions = []
    for transaction in transactions:
        formatted_transaction = {
            'id': transaction['id'],
            'usuario_id': transaction['usuario_id'],
            'numero_control': transaction['numero_control'],
            'transaccion_id': transaction['transaccion_id'],
            'monto': transaction['monto'],
            'fecha': transaction['fecha'],
            'nombre': transaction['nombre'],
            'apellido': transaction['apellido'],
            'correo': transaction['correo'],
            'paquete_nombre': transaction['paquete_nombre'],
            'paquete': transaction['paquete_nombre'],
            'pin': f"ID: {transaction['player_id']}",
            'player_id': transaction['player_id'],
            'estado': transaction['estado'],
            'is_freefire_id': True
        }
        formatted_transactions.append(formatted_transaction)
    
    conn.close()
    return formatted_transactions

def get_user_pending_freefire_id_transactions(user_id):
    """Obtiene las transacciones pendientes de Free Fire ID de un usuario específico"""
    conn = get_db_connection()
    transactions = conn.execute('''
        SELECT fi.*, u.nombre, u.apellido, p.nombre as paquete_nombre
        FROM transacciones_freefire_id fi
        JOIN usuarios u ON fi.usuario_id = u.id
        JOIN precios_freefire_id p ON fi.paquete_id = p.id
        WHERE fi.usuario_id = ? AND fi.estado = 'pendiente'
        ORDER BY fi.fecha DESC
    ''', (user_id,)).fetchall()
    
    formatted_transactions = []
    for transaction in transactions:
        formatted_transaction = {
            'id': transaction['id'],
            'usuario_id': transaction['usuario_id'],
            'numero_control': transaction['numero_control'],
            'transaccion_id': transaction['transaccion_id'],
            'monto': transaction['monto'],
            'fecha': convert_to_venezuela_time(transaction['fecha']),
            'nombre': transaction['nombre'],
            'apellido': transaction['apellido'],
            'paquete': transaction['paquete_nombre'],
            'pin': f"ID: {transaction['player_id']}",
            'estado': transaction['estado'],
            'is_freefire_id': True
        }
        formatted_transactions.append(formatted_transaction)
    
    conn.close()
    return formatted_transactions

def verify_pin_already_redeemed(pin_code, player_id, config=None):
    """
    Verifica si un PIN ya fue redimido exitosamente en el VPS.
    Esto previene devolver PINS que ya fueron usados.
    """
    try:
        cfg = dict(config or {})
        vps_url = cfg.get("vps_url") or os.environ.get("VPS_REDEEM_URL", "http://74.208.158.70:5000/redeem")
        
        # Intentar 1: Verificar si el VPS tiene endpoint de verificación
        try:
            payload = {
                "pin_key": str(pin_code).strip(),
                "player_id": str(player_id).strip(),
                "verify_only": True
            }
            
            timeout = 30
            
            resp = requests.post(
                vps_url + "/verify",
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("already_redeemed") or data.get("used") or data.get("status") == "used":
                    logger.warning(f"[FreeFire ID] PIN {pin_code[:8]}... ya fue redimido (verify endpoint)")
                    return True
                    
        except requests.exceptions.RequestException:
            # Si no hay endpoint /verify, continuar con método alternativo
            logger.info(f"[FreeFire ID] Endpoint /verify no disponible, usando método alternativo")
        
        # Intento 2: Verificar en base de datos local si hay transacciones exitosas recientes
        conn = get_db_connection()
        recent_success = conn.execute('''
            SELECT COUNT(*) as count FROM transacciones_freefire_id 
            WHERE pin_codigo = ? AND player_id = ? AND estado = 'aprobado'
            AND fecha > datetime('now', '-5 minutes')
        ''', (pin_code, player_id)).fetchone()
        conn.close()
        
        if recent_success and recent_success['count'] > 0:
            logger.warning(f"[FreeFire ID] PIN {pin_code[:8]}... ya fue redimido (verificación local)")
            return True
        
        # Intento 3: Verificar si el PIN ya no está en el stock (fue removido)
        conn = get_db_connection()
        pin_in_stock = conn.execute('''
            SELECT COUNT(*) as count FROM pines_freefire_global 
            WHERE pin_codigo = ? AND usado = FALSE
        ''', (pin_code,)).fetchone()
        conn.close()
        
        if pin_in_stock and pin_in_stock['count'] == 0:
            logger.warning(f"[FreeFire ID] PIN {pin_code[:8]}... no está en stock (probablemente usado)")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"[FreeFire ID] Error verificando PIN {pin_code[:8]}...: {str(e)}")
        # En caso de error, ser conservador y asumir que no fue usado
        return False

def audit_freefire_id_inconsistent_transactions():
    """
    Audita y detecta transacciones inconsistentes de Free Fire ID.
    Busca casos donde el saldo fue reembolsado pero el PIN fue usado.
    """
    conn = get_db_connection()
    
    # Buscar transacciones rechazadas con PIN que ya no está en stock
    inconsistent = conn.execute('''
        SELECT t.*, p.pin_codigo
        FROM transacciones_freefire_id t
        LEFT JOIN pines_freefire_global p ON t.pin_codigo = p.pin_codigo AND p.usado = FALSE
        WHERE t.estado = 'rechazado' 
        AND p.pin_codigo IS NULL  -- El PIN no está disponible en stock
        AND t.fecha > datetime('now', '-24 hours')
        ORDER BY t.fecha DESC
        LIMIT 50
    ''').fetchall()
    
    results = []
    for trans in inconsistent:
        # Verificación adicional: buscar si hay transacciones generales correspondientes
        general_trans = conn.execute('''
            SELECT COUNT(*) as count FROM transacciones 
            WHERE usuario_id = ? AND numero_control = ? 
            AND monto = ?
        ''', (trans['usuario_id'], trans['numero_control'], trans['monto'])).fetchone()
        
        # Si no hay transacción general, probablemente fue reembolsado
        was_refunded = general_trans and general_trans['count'] == 0
        
        results.append({
            'id': trans['id'],
            'usuario_id': trans['usuario_id'],
            'player_id': trans['player_id'],
            'pin_codigo': trans['pin_codigo'],
            'numero_control': trans['numero_control'],
            'monto': abs(trans['monto']),  # Convertir a positivo
            'fecha': trans['fecha'],
            'notas': trans['notas'],
            'was_refunded': was_refunded,
            'severity': 'HIGH' if was_refunded else 'MEDIUM'
        })
    
    conn.close()
    return results

def update_freefire_id_transaction_status(transaction_id, new_status, admin_id, notas=None, register_general_tx=True):
    """Actualiza el estado de una transacción de Free Fire ID"""
    conn = get_db_connection()
    conn.execute('''
        UPDATE transacciones_freefire_id 
        SET estado = ?, admin_id = ?, notas = ?, fecha_procesado = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (new_status, admin_id, notas, transaction_id))
    conn.commit()

    # Si se aprueba, asegurar que exista registro en transacciones generales (historial)
    # Puede desactivarse en el flujo automático que ya inserta un registro más completo
    if new_status == 'aprobado' and register_general_tx:
        try:
            tx = conn.execute('''
                SELECT fi.usuario_id, fi.numero_control, fi.transaccion_id, fi.player_id,
                       fi.monto, p.nombre as paquete_nombre
                FROM transacciones_freefire_id fi
                JOIN precios_freefire_id p ON fi.paquete_id = p.id
                WHERE fi.id = ?
            ''', (transaction_id,)).fetchone()
            if tx:
                ya_existe = conn.execute(
                    'SELECT 1 FROM transacciones WHERE transaccion_id = ?',
                    (tx['transaccion_id'],)
                ).fetchone()
                if not ya_existe:
                    pin_info = f"ID: {tx['player_id']}"
                    conn.execute('''
                        INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (tx['usuario_id'], tx['numero_control'], pin_info,
                          tx['transaccion_id'], tx['paquete_nombre'], -abs(tx['monto'])))
                    conn.commit()
        except Exception as e:
            logger.error(f"[FFID] Error registrando en historial general: {e}")

    conn.close()

def update_freefire_id_price(package_id, new_price):
    """Actualiza el precio de un paquete de Free Fire ID"""
    conn = get_db_connection_optimized()
    try:
        conn.execute('''
            UPDATE precios_freefire_id 
            SET precio = ?, fecha_actualizacion = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_price, package_id))
        conn.commit()
        clear_price_cache()
    finally:
        return_db_connection(conn)

def update_freefire_id_name(package_id, new_name):
    """Actualiza el nombre de un paquete de Free Fire ID"""
    conn = get_db_connection_optimized()
    try:
        conn.execute('''
            UPDATE precios_freefire_id 
            SET nombre = ?, fecha_actualizacion = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_name, package_id))
        conn.commit()
        clear_price_cache()
    finally:
        return_db_connection(conn)

def get_all_freefire_id_prices():
    """Obtiene todos los precios de paquetes de Free Fire ID"""
    conn = get_db_connection()
    prices = conn.execute('''
        SELECT * FROM precios_freefire_id 
        ORDER BY id
    ''').fetchall()
    conn.close()
    return prices

def send_freefire_id_notification(transaction_data):
    """Envía notificación por correo cuando hay una nueva transacción de Free Fire ID"""
    if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
        return
    
    try:
        admin_email = os.environ.get('ADMIN_EMAIL', '')
        if not admin_email:
            return
        
        subject = f"🔥 Free Fire ID - Nueva solicitud de recarga"
        body = f"""
        <h2>🔥 Nueva solicitud de recarga de Free Fire ID</h2>
        <p><strong>Usuario:</strong> {transaction_data['nombre']} {transaction_data['apellido']}</p>
        <p><strong>Correo:</strong> {transaction_data['correo']}</p>
        <p><strong>ID de Jugador:</strong> {transaction_data['player_id']}</p>
        <p><strong>Paquete:</strong> {transaction_data['paquete_nombre']}</p>
        <p><strong>Precio:</strong> ${transaction_data['precio']:.2f}</p>
        <p><strong>Número de Control:</strong> {transaction_data['numero_control']}</p>
        <p><strong>ID de Transacción:</strong> {transaction_data['transaccion_id']}</p>
        <p><strong>Fecha:</strong> {transaction_data['fecha']}</p>
        <br>
        <p><a href="#">Ir al panel de administración para procesar</a></p>
        """
        
        msg = Message(subject, recipients=[admin_email], html=body)
        
        def send_async():
            with app.app_context():
                try:
                    mail.send(msg)
                except Exception as e:
                    print(f"Error al enviar correo: {str(e)}")
        
        thread = threading.Thread(target=send_async)
        thread.start()
    except Exception as e:
        print(f"Error al enviar correo: {str(e)}")

# Funciones para configuración de fuentes de pines
def get_pin_source_config():
    """Obtiene la configuración de fuentes de pines por monto"""
    conn = get_db_connection()
    config = {}
    for i in range(1, 10):
        result = conn.execute('''
            SELECT fuente FROM configuracion_fuentes_pines 
            WHERE monto_id = ? AND activo = TRUE
        ''', (i,)).fetchone()
        config[i] = result['fuente'] if result else 'local'
    conn.close()
    return config

def update_pin_source_config(monto_id, fuente):
    """Actualiza la configuración de fuente para un monto específico"""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO configuracion_fuentes_pines (monto_id, fuente, activo, fecha_actualizacion)
        VALUES (%s, %s, TRUE, CURRENT_TIMESTAMP)
        ON CONFLICT (monto_id) DO UPDATE SET fuente = EXCLUDED.fuente, activo = EXCLUDED.activo, fecha_actualizacion = EXCLUDED.fecha_actualizacion
    ''', (monto_id, fuente))
    conn.commit()
    conn.close()

# Funciones de notificación por correo
def send_email_async(app, msg):
    """Envía correo de forma asíncrona"""
    with app.app_context():
        try:
            mail.send(msg)
            print("Correo de notificación enviado exitosamente")
        except Exception as e:
            print(f"Error al enviar correo: {str(e)}")

def send_bloodstriker_notification(transaction_data):
    """Envía notificación por correo cuando hay una nueva transacción de Blood Striker"""
    # Verificar si las credenciales de correo están configuradas
    if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
        print("Credenciales de correo no configuradas. Notificación omitida.")
        return
    
    try:
        # Obtener correo del administrador
        admin_email = os.environ.get('ADMIN_EMAIL', 'admin@inefable.com')
        
        # Crear mensaje
        msg = Message(
            subject='🎯 Nueva Transacción Blood Striker Pendiente',
            recipients=[admin_email],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        
        # Contenido del correo
        msg.html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #667eea; text-align: center;">🎯 Nueva Transacción Blood Striker</h2>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="color: #333; margin-top: 0;">Detalles de la Transacción:</h3>
                    
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold;">Usuario:</td>
                            <td style="padding: 8px 0;">{transaction_data['nombre']} {transaction_data['apellido']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold;">Correo:</td>
                            <td style="padding: 8px 0;">{transaction_data['correo']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold;">Player ID:</td>
                            <td style="padding: 8px 0; font-family: monospace; background: #e9ecef; padding: 4px 8px; border-radius: 4px;">{transaction_data['player_id']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold;">Paquete:</td>
                            <td style="padding: 8px 0;">{transaction_data['paquete_nombre']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold;">Monto:</td>
                            <td style="padding: 8px 0; color: #dc3545; font-weight: bold;">${transaction_data['precio']:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold;">Número de Control:</td>
                            <td style="padding: 8px 0; font-family: monospace;">{transaction_data['numero_control']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold;">ID de Transacción:</td>
                            <td style="padding: 8px 0; font-family: monospace;">{transaction_data['transaccion_id']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold;">Fecha:</td>
                            <td style="padding: 8px 0;">{transaction_data['fecha']}</td>
                        </tr>
                    </table>
                </div>
                
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 6px; margin: 20px 0;">
                    <p style="margin: 0; color: #856404;">
                        <strong>⏳ Acción Requerida:</strong> Esta transacción está pendiente de aprobación. 
                        Ingresa al panel de administración para aprobar o rechazar la solicitud.
                    </p>
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <p style="color: #6c757d; font-size: 14px;">
                        Este es un correo automático del sistema de notificaciones.<br>
                        No responder a este correo.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Enviar correo de forma asíncrona
        thread = threading.Thread(target=send_email_async, args=(app, msg))
        thread.daemon = True
        thread.start()
        
    except Exception as e:
        print(f"Error al preparar notificación por correo: {str(e)}")

# Rutas de administrador
@app.route('/admin')
def admin_panel():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    users = get_all_users()
    pin_stock = get_pin_stock_optimized()
    pin_stock_freefire_global = get_pin_stock_freefire_global_optimized()
    prices = get_all_prices()
    freefire_global_prices = get_all_freefire_global_prices()
    bloodstriker_prices = get_all_bloodstriker_prices()
    freefire_id_prices = get_all_freefire_id_prices()
    pin_sources_config = get_pin_source_config()
    noticias = get_all_news()
    games_active = get_games_active()
    redeemer_config = get_redeemer_config_from_db(get_db_connection)

    # Dynamic games for Precios + GameClub tabs
    from dynamic_games import get_all_dynamic_games as _dg_all, get_dynamic_packages as _dg_pkgs
    dyn_games = _dg_all()
    dynamic_stock = get_dynamic_stock_counts()
    for dg in dyn_games:
        dg['_packages'] = _dg_pkgs(dg['id'])
    dynamic_stock_games = [dg for dg in dyn_games if dg.get('usa_stock_local')]
    
    return render_template('admin.html', 
                         users=users, 
                         pin_stock=pin_stock, 
                         pin_stock_freefire_global=pin_stock_freefire_global,
                         prices=prices, 
                         freefire_global_prices=freefire_global_prices,
                         bloodstriker_prices=bloodstriker_prices,
                         freefire_id_prices=freefire_id_prices,
                         pin_sources_config=pin_sources_config,
                         noticias=noticias,
                         games_active=games_active,
                         redeemer_config=redeemer_config,
                         dyn_games=dyn_games,
                         dynamic_stock=dynamic_stock,
                         dynamic_stock_games=dynamic_stock_games)


@app.route('/admin/gameclub/products', methods=['GET'])
def admin_gameclub_products():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Acceso denegado'}), 403

    token, err = _gameclub_get_token()
    if not token:
        msg = (err or {}).get('message') if isinstance(err, dict) else None
        return jsonify({'success': False, 'error': msg or 'No se pudo obtener token de GameClub', 'raw': err}), 400

    _, data = _gameclub_post('product/list', {'token': token})
    if (data or {}).get('code') != 200:
        return jsonify({'success': False, 'error': (data or {}).get('message') or 'Error consultando product/list', 'raw': data}), 400

    return jsonify({'success': True, 'data': data})


@app.route('/admin/gameclub/product/<int:product_id>', methods=['GET'])
def admin_gameclub_product_detail(product_id: int):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Acceso denegado'}), 403

    token, err = _gameclub_get_token()
    if not token:
        msg = (err or {}).get('message') if isinstance(err, dict) else None
        return jsonify({'success': False, 'error': msg or 'No se pudo obtener token de GameClub', 'raw': err}), 400

    _, data = _gameclub_post('product/detail', {'token': token, 'productid': product_id})
    if (data or {}).get('code') != 200:
        return jsonify({'success': False, 'error': (data or {}).get('message') or 'Error consultando product/detail', 'raw': data}), 400

    return jsonify({'success': True, 'data': data})

@app.route('/admin/add_credit', methods=['POST'])
def admin_add_credit():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    user_id = request.form.get('user_id')
    amount = float(request.form.get('amount', 0))
    
    if user_id and amount > 0:
        add_credit_to_user(user_id, amount)
        flash(f'Se agregaron ${amount:.2f} al usuario ID {user_id}', 'success')
    else:
        flash('Datos inválidos para agregar crédito', 'error')
    
    return redirect('/admin')


@app.route('/admin/import_pins_csv', methods=['POST'])
def admin_import_pins_csv():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')

    monto_id = request.form.get('batch_monto_id')
    game_type = request.form.get('game_type')
    f = request.files.get('csv_file')

    if not monto_id or not game_type or not f:
        flash('Datos inválidos para importar CSV', 'error')
        return redirect('/admin')

    # Bloquear re-import por nombre (global, sin importar monto/juego)
    original_name = (getattr(f, 'filename', None) or '').strip()
    if not original_name:
        flash('El archivo CSV debe tener un nombre válido', 'error')
        return redirect('/admin')
    normalized_name = original_name.lower()

    try:
        # Verificar si ya se importó este nombre
        conn = get_db_connection_optimized()
        try:
            ex = conn.execute("SELECT 1 FROM admin_imported_files WHERE filename = ?", (normalized_name,)).fetchone()
            if ex:
                flash(f'Este archivo ya fue importado antes: {original_name}', 'warning')
                return redirect('/admin')
        finally:
            return_db_connection(conn)

        raw = f.read()
        pins_list = _extract_pin_codes_from_csv_bytes(raw)
        if not pins_list:
            flash('No se encontraron códigos de pin válidos en el CSV', 'warning')
            return redirect('/admin')

        if game_type == 'freefire_latam':
            added_count = add_pins_batch(int(monto_id), pins_list)
            packages_info = get_package_info_with_prices()
            package_info = packages_info.get(int(monto_id), {})
            juego_nombre = "Free Fire Latam"
        elif game_type == 'freefire_global':
            added_count = add_pins_batch_freefire_global(int(monto_id), pins_list)
            packages_info = get_freefire_global_prices()
            package_info = packages_info.get(int(monto_id), {})
            juego_nombre = "Free Fire"
        elif game_type.startswith('dyn_'):
            dyn_game = get_dynamic_game_by_slug(game_type[4:])
            pkg = get_dynamic_package_by_id(int(monto_id)) if monto_id else None
            if not dyn_game or not dyn_game.get('usa_stock_local') or not pkg or pkg['juego_id'] != dyn_game['id']:
                flash('Juego o paquete dinámico inválido', 'error')
                return redirect('/admin')
            added_count = add_dynamic_stock_pins_batch(dyn_game['id'], int(monto_id), pins_list)
            juego_nombre = dyn_game['nombre']
            package_info = {'nombre': pkg['nombre'], 'precio': float(pkg['precio'])}
        else:
            flash('Tipo de juego inválido', 'error')
            return redirect('/admin')

        if package_info:
            paquete_nombre = f"{package_info['nombre']} / ${package_info['precio']:.2f}"
        else:
            paquete_nombre = "Paquete desconocido"

        # Registrar nombre de archivo como importado (evita duplicados futuros)
        conn2 = get_db_connection_optimized()
        try:
            conn2.execute("INSERT INTO admin_imported_files (filename) VALUES (?)", (normalized_name,))
            conn2.commit()
        finally:
            return_db_connection(conn2)

        flash(f'Se importaron {added_count} pines desde CSV para {juego_nombre} - {paquete_nombre}', 'success')
    except Exception as e:
        if 'unique' in str(e).lower() or 'duplicate' in str(e).lower():
            flash(f'Este archivo ya fue importado antes: {original_name}', 'warning')
        else:
            flash(f'Error al importar CSV: {str(e)}', 'error')

    return redirect('/admin')

# ======= Batch update de nombres y precios =======
@app.route('/admin/save_prices_batch', methods=['POST'])
def admin_save_prices_batch():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')

    game = request.form.get('game')
    payload_raw = request.form.get('payload', '')
    if not game or not payload_raw:
        flash('Datos incompletos para guardar cambios.', 'error')
        return redirect('/admin')

    try:
        items = json.loads(payload_raw)
        if not isinstance(items, list):
            raise ValueError('Formato inválido')
    except Exception:
        flash('Formato de datos inválido.', 'error')
        return redirect('/admin')

    # Determinar tabla por juego
    if game == 'freefire':
        table = 'precios_paquetes'
    elif game == 'freefire_global':
        table = 'precios_freefire_global'
    elif game == 'bloodstriker':
        table = 'precios_bloodstriker'
    elif game == 'freefire_id':
        table = 'precios_freefire_id'
    elif game.startswith('dyn_'):
        table = 'paquetes_dinamicos'
    else:
        flash('Juego no soportado.', 'error')
        return redirect('/admin')

    conn = get_db_connection()
    try:
        dyn_game_id = None
        if game.startswith('dyn_'):
            try:
                slug = game.replace('dyn_', '', 1)
                row = conn.execute('SELECT id FROM juegos_dinamicos WHERE slug = ?', (slug,)).fetchone()
                dyn_game_id = int(row['id']) if row and row.get('id') is not None else None
            except Exception:
                dyn_game_id = None

        updated = 0
        for it in items:
            try:
                pid = int(it.get('id'))
                name = str(it.get('nombre', '')).strip()
                price = float(it.get('precio'))
            except Exception:
                continue

            if game.startswith('dyn_'):
                # Activate priced packages so they are visible to users (public pages filter activo=1)
                activo = True if price > 0 else False
                if dyn_game_id is not None:
                    conn.execute(
                        "UPDATE paquetes_dinamicos SET nombre = ?, precio = ?, activo = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ? AND juego_id = ?",
                        (name, price, activo, pid, dyn_game_id)
                    )
                else:
                    conn.execute(
                        "UPDATE paquetes_dinamicos SET nombre = ?, precio = ?, activo = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ?",
                        (name, price, activo, pid)
                    )
            else:
                conn.execute(
                    f"UPDATE {table} SET nombre = ?, precio = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ?",
                    (name, price, pid)
                )
            updated += 1
        conn.commit()
        flash(f'Se guardaron {updated} cambios correctamente.', 'success')
    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
        flash(f'Error al guardar cambios: {str(e)}', 'error')
    finally:
        conn.close()

    return redirect('/admin')

@app.route('/admin/toggle_game', methods=['POST'])
def admin_toggle_game():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    game = request.form.get('game')
    active = request.form.get('active')
    if active not in ['0','1']:
        flash('Parámetros inválidos.', 'error')
        return redirect('/admin')

    static_tables = {
        'freefire': 'precios_paquetes',
        'freefire_global': 'precios_freefire_global',
        'bloodstriker': 'precios_bloodstriker',
        'freefire_id': 'precios_freefire_id'
    }

    try:
        conn = get_db_connection()
        active_sql = 'TRUE' if active == '1' else 'FALSE'
        if game in static_tables:
            cur = conn.execute(f"UPDATE {static_tables[game]} SET activo = {active_sql}")
            affected = getattr(cur, 'rowcount', -1)
        elif game and game.startswith('dyn_'):
            slug = game[4:]
            cur = conn.execute(f"UPDATE juegos_dinamicos SET activo = {active_sql} WHERE slug = ?", (slug,))
            affected = getattr(cur, 'rowcount', -1)
            if affected == 0:
                conn.close()
                flash(f'No se encontró juego dinámico con slug: {slug}', 'warning')
                return redirect('/admin')
        else:
            conn.close()
            flash('Juego no soportado.', 'error')
            return redirect('/admin')
        conn.commit()
        conn.close()
        estado = 'activado' if active == '1' else 'desactivado'
        logger.info(f"[admin_toggle_game] game={game} active={active} sql={active_sql} affected={affected}")
        flash(f'Juego {game} {estado} correctamente.', 'success')
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        logger.exception(f"[admin_toggle_game] Error game={game} active={active}")
        flash(f'Error al actualizar estado del juego: {str(e)}', 'error')
    return redirect('/admin')

@app.route('/admin/pins')
def admin_pins_list():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    game = request.args.get('game', 'freefire_latam')
    only_unused = request.args.get('estado', 'unused') == 'unused'
    # Filtro opcional por paquete/monto
    try:
        monto_filter = int(request.args.get('monto')) if request.args.get('monto') else None
    except ValueError:
        monto_filter = None
    pins = get_pins_by_game(game, only_unused=only_unused, monto_id=monto_filter)

    game_display_name = 'Free Fire Latam' if game == 'freefire_latam' else 'Free Fire'
    pin_game_options = [
        {'value': 'freefire_latam', 'label': 'Free Fire Latam'},
        {'value': 'freefire_global', 'label': 'Free Fire'},
    ]

    if game == 'freefire_latam':
        package_dict = get_package_info_with_prices()
    elif game == 'freefire_global':
        package_dict = get_freefire_global_prices()
    elif game.startswith('dyn_'):
        dyn_game = get_dynamic_game_by_slug(game[4:])
        if not dyn_game or not dyn_game.get('usa_stock_local'):
            flash('Juego dinámico de tarjetas no encontrado.', 'error')
            return redirect('/admin')
        game_display_name = dyn_game['nombre']
        package_dict = {pkg['id']: {'nombre': pkg['nombre']} for pkg in get_dynamic_packages(dyn_game['id'])}
    else:
        package_dict = {}

    for dyn_game in get_dynamic_stock_games():
        pin_game_options.append({'value': f"dyn_{dyn_game['slug']}", 'label': dyn_game['nombre']})

    # Convertir a estructura simple para template
    pins_view = []
    for p in pins:
        pins_view.append({
            'id': p['id'],
            'monto_id': p['monto_id'],
            'paquete': package_dict.get(p['monto_id'], {}).get('nombre', f'Paquete {p["monto_id"]}'),
            'pin_codigo': p['pin_codigo'],
            'usado': bool(p['usado']),
            'fecha_agregado': p['fecha_agregado'],
            'batch_id': p['batch_id']
        })

    # Agrupar por lote (batch_id). Pines individuales (batch_id NULL) van en su propio grupo.
    pin_groups = []
    groups_map = {}
    for pin in pins_view:
        bid = pin.get('batch_id')
        key = str(bid) if bid else f"single-{pin['id']}"
        if key not in groups_map:
            groups_map[key] = {
                'key': key,
                'batch_id': bid,
                'fecha_agregado': pin.get('fecha_agregado'),
                'monto_id': pin.get('monto_id'),
                'paquete': pin.get('paquete'),
                'pins': []
            }
        groups_map[key]['pins'].append(pin)

    # Mantener orden: ya vienen ordenados por fecha_agregado desc, respetar primer aparición
    seen = set()
    for pin in pins_view:
        bid = pin.get('batch_id')
        key = str(bid) if bid else f"single-{pin['id']}"
        if key in seen:
            continue
        seen.add(key)
        grp = groups_map.get(key)
        if grp:
            pin_groups.append(grp)

    # Nombre del paquete seleccionado (si aplica)
    selected_package_name = None
    if 'monto_filter' in locals() and monto_filter:
        selected_package_name = package_dict.get(monto_filter, {}).get('nombre')

    return render_template(
        'admin_pins.html',
        pins=pins_view,
        pin_groups=pin_groups,
        game=game,
        game_display_name=game_display_name,
        pin_game_options=pin_game_options,
        only_unused=only_unused,
        monto=monto_filter,
        selected_package_name=selected_package_name,
    )


@app.route('/admin/delete_pins_batch', methods=['POST'])
def admin_delete_pins_batch():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    game = request.form.get('game')
    batch_id = request.form.get('batch_id')
    estado = request.form.get('estado', 'unused')
    only_unused = estado == 'unused'
    deleted = delete_pins_by_batch_id(game, batch_id, only_unused=only_unused)
    if deleted > 0:
        flash(f'Se eliminó el lote ({batch_id}) con {deleted} pines', 'success')
    else:
        flash('No se eliminaron pines del lote (puede que no existan o estén usados)', 'warning')
    return redirect(f'/admin/pins?game={game}&estado={"unused" if only_unused else "all"}')

@app.route('/admin/delete_pins', methods=['POST'])
def admin_delete_pins():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    game = request.form.get('game')
    ids_raw = request.form.getlist('pin_ids')
    try:
        pin_ids = [int(x) for x in ids_raw if str(x).isdigit()]
    except Exception:
        pin_ids = []
    deleted = delete_pins_by_ids(game, pin_ids)
    if deleted > 0:
        flash(f'Se eliminaron {deleted} pines correctamente', 'success')
    else:
        flash('No se eliminaron pines (lista vacía)', 'warning')
    return redirect(f'/admin/pins?game={game}')

@app.route('/admin/update_balance', methods=['POST'])
def admin_update_balance():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    user_id = request.form.get('user_id')
    new_balance = float(request.form.get('new_balance', 0))
    
    if user_id and new_balance >= 0:
        update_user_balance(user_id, new_balance)
        flash(f'Saldo actualizado a ${new_balance:.2f} para usuario ID {user_id}', 'success')
    else:
        flash('Datos inválidos para actualizar saldo', 'error')
    
    return redirect('/admin')

@app.route('/admin/delete_user', methods=['POST'])
def admin_delete_user():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    user_id = request.form.get('user_id')
    
    if user_id:
        delete_user(user_id)
        flash(f'Usuario ID {user_id} eliminado exitosamente', 'success')
    else:
        flash('ID de usuario inválido', 'error')
    
    return redirect('/admin')

@app.route('/admin/toggle_sin_ganancia', methods=['POST'])
def admin_toggle_sin_ganancia():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    user_id = request.form.get('user_id')
    if user_id:
        conn = get_db_connection()
        user = conn.execute('SELECT sin_ganancia FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            new_val = False if user['sin_ganancia'] else True
            conn.execute('UPDATE usuarios SET sin_ganancia = ? WHERE id = ?', (new_val, user_id))
            conn.commit()
            estado = 'SIN ganancia' if new_val else 'CON ganancia'
            flash(f'Usuario ID {user_id} ahora está {estado}', 'success')
        else:
            flash('Usuario no encontrado', 'error')
        conn.close()
    else:
        flash('ID de usuario inválido', 'error')
    
    return redirect('/admin')

@app.route('/admin/toggle_bono_activo', methods=['POST'])
def admin_toggle_bono_activo():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    user_id = request.form.get('user_id')
    if user_id:
        conn = get_db_connection()
        user = conn.execute('SELECT bono_activo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            new_val = False if user['bono_activo'] else True
            conn.execute('UPDATE usuarios SET bono_activo = ? WHERE id = ?', (new_val, user_id))
            conn.commit()
            estado = 'ACTIVADO' if new_val else 'DESACTIVADO'
            flash(f'Bono 1.5% para usuario ID {user_id}: {estado}', 'success')
        else:
            flash('Usuario no encontrado', 'error')
        conn.close()
    else:
        flash('ID de usuario inválido', 'error')
    
    return redirect('/admin')

@app.route('/admin/add_pin', methods=['POST'])
def admin_add_pin():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    monto_id = request.form.get('monto_id')
    pin_codigo = request.form.get('pin_codigo')
    game_type = request.form.get('game_type')
    
    if monto_id and pin_codigo and game_type:
        if game_type == 'freefire_latam':
            add_pin_freefire(int(monto_id), pin_codigo)
            juego_nombre = "Free Fire Latam"
            table = 'precios_paquetes'
        elif game_type == 'freefire_global':
            add_pin_freefire_global(int(monto_id), pin_codigo)
            juego_nombre = "Free Fire"
            table = 'precios_freefire_global'
        elif game_type.startswith('dyn_'):
            dyn_game = get_dynamic_game_by_slug(game_type[4:])
            pkg = get_dynamic_package_by_id(int(monto_id)) if monto_id else None
            if not dyn_game or not dyn_game.get('usa_stock_local') or not pkg or pkg['juego_id'] != dyn_game['id']:
                if request.headers.get('Accept') == 'application/json':
                    return jsonify({'success': False, 'error': 'Juego o paquete dinámico inválido'}), 400
                flash('Juego o paquete dinámico inválido', 'error')
                return redirect('/admin')
            add_dynamic_stock_pin(dyn_game['id'], int(monto_id), pin_codigo)
            paquete_nombre = f"{pkg['nombre']} / ${float(pkg['precio']):.2f}"
            if request.headers.get('Accept') == 'application/json':
                return jsonify({'success': True, 'message': f'Pin agregado para {dyn_game["nombre"]} - {paquete_nombre}'})
            flash(f'Pin agregado exitosamente para {dyn_game["nombre"]} - {paquete_nombre}', 'success')
            return redirect('/admin')
        else:
            flash('Tipo de juego inválido', 'error')
            return redirect('/admin')
        
        conn_pkg = get_db_connection()
        row = conn_pkg.execute(f'SELECT nombre, precio FROM {table} WHERE id = ?', (int(monto_id),)).fetchone()
        conn_pkg.close()
        paquete_nombre = f"{row['nombre']} / ${row['precio']:.2f}" if row else f"Monto #{monto_id}"
        
        if request.headers.get('Accept') == 'application/json':
            return jsonify({'success': True, 'message': f'Pin agregado para {juego_nombre} - {paquete_nombre}'})
        flash(f'Pin agregado exitosamente para {juego_nombre} - {paquete_nombre}', 'success')
    else:
        if request.headers.get('Accept') == 'application/json':
            return jsonify({'success': False, 'error': 'Datos inválidos'}), 400
        flash('Datos inválidos para agregar pin', 'error')
    
    return redirect('/admin')

@app.route('/admin/add_pins_batch', methods=['POST'])
def admin_add_pins_batch():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    monto_id = request.form.get('batch_monto_id')
    pins_text = request.form.get('pins_batch')
    game_type = request.form.get('game_type')
    
    if not monto_id or not pins_text or not game_type:
        flash('Por favor complete todos los campos para el lote de pines', 'error')
        return redirect('/admin')
    
    # Procesar los pines (separados por líneas o comas)
    pins_list = []
    for line in pins_text.replace(',', '\n').split('\n'):
        pin = line.strip()
        if pin:
            pins_list.append(pin)
    
    if not pins_list:
        flash('No se encontraron pines válidos en el texto', 'error')
        return redirect('/admin')
    
    try:
        if game_type == 'freefire_latam':
            added_count = add_pins_batch(int(monto_id), pins_list)
            juego_nombre = "Free Fire Latam"
            table = 'precios_paquetes'
        elif game_type == 'freefire_global':
            added_count = add_pins_batch_freefire_global(int(monto_id), pins_list)
            juego_nombre = "Free Fire"
            table = 'precios_freefire_global'
        elif game_type.startswith('dyn_'):
            dyn_game = get_dynamic_game_by_slug(game_type[4:])
            pkg = get_dynamic_package_by_id(int(monto_id)) if monto_id else None
            if not dyn_game or not dyn_game.get('usa_stock_local') or not pkg or pkg['juego_id'] != dyn_game['id']:
                if request.headers.get('Accept') == 'application/json':
                    return jsonify({'success': False, 'error': 'Juego o paquete dinámico inválido'}), 400
                flash('Juego o paquete dinámico inválido', 'error')
                return redirect('/admin')
            added_count = add_dynamic_stock_pins_batch(dyn_game['id'], int(monto_id), pins_list)
            paquete_nombre = f"{pkg['nombre']} / ${float(pkg['precio']):.2f}"
            if request.headers.get('Accept') == 'application/json':
                return jsonify({'success': True, 'added': added_count, 'message': f'{added_count} pines agregados para {dyn_game["nombre"]} - {paquete_nombre}'})
            flash(f'Se agregaron {added_count} pines exitosamente para {dyn_game["nombre"]} - {paquete_nombre}', 'success')
            return redirect('/admin')
        else:
            flash('Tipo de juego inválido', 'error')
            return redirect('/admin')
        
        conn_pkg = get_db_connection()
        row = conn_pkg.execute(f'SELECT nombre, precio FROM {table} WHERE id = ?', (int(monto_id),)).fetchone()
        conn_pkg.close()
        paquete_nombre = f"{row['nombre']} / ${row['precio']:.2f}" if row else f"Monto #{monto_id}"
        
        if request.headers.get('Accept') == 'application/json':
            return jsonify({'success': True, 'added': added_count, 'message': f'{added_count} pines agregados para {juego_nombre} - {paquete_nombre}'})
        flash(f'Se agregaron {added_count} pines exitosamente para {juego_nombre} - {paquete_nombre}', 'success')
        
    except Exception as e:
        if request.headers.get('Accept') == 'application/json':
            return jsonify({'success': False, 'error': str(e)}), 500
        flash(f'Error al agregar pines en lote: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/remove_duplicates', methods=['POST'])
def admin_remove_duplicates():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        duplicates_removed = remove_duplicate_pins()
        if duplicates_removed > 0:
            flash(f'Se eliminaron {duplicates_removed} pines duplicados exitosamente', 'success')
        else:
            flash('No se encontraron pines duplicados para eliminar', 'success')
    except Exception as e:
        flash(f'Error al eliminar pines duplicados: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/update_price', methods=['POST'])
def admin_update_price():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    package_id = request.form.get('package_id')
    new_price = request.form.get('new_price')
    
    if not package_id or not new_price:
        flash('Datos inválidos para actualizar precio', 'error')
        return redirect('/admin')
    
    try:
        new_price = float(new_price)
        if new_price < 0:
            flash('El precio no puede ser negativo', 'error')
            return redirect('/admin')
        
        # Obtener información del paquete antes de actualizar
        conn = get_db_connection()
        package = conn.execute('SELECT nombre FROM precios_paquetes WHERE id = ?', (package_id,)).fetchone()
        conn.close()
        
        if not package:
            flash('Paquete no encontrado', 'error')
            return redirect('/admin')
        
        # Actualizar precio
        update_package_price(int(package_id), new_price)
        flash(f'Precio actualizado exitosamente para {package["nombre"]}: ${new_price:.2f}', 'success')
        
    except ValueError:
        flash('Precio inválido. Debe ser un número válido.', 'error')
    except Exception as e:
        flash(f'Error al actualizar precio: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/update_name', methods=['POST'])
def admin_update_name():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    package_id = request.form.get('package_id')
    new_name = request.form.get('new_name')
    
    if not package_id or not new_name:
        flash('Datos inválidos para actualizar nombre', 'error')
        return redirect('/admin')
    
    try:
        new_name = new_name.strip()
        if len(new_name) < 1:
            flash('El nombre no puede estar vacío', 'error')
            return redirect('/admin')
        
        if len(new_name) > 50:
            flash('El nombre no puede exceder 50 caracteres', 'error')
            return redirect('/admin')
        
        # Obtener información del paquete antes de actualizar
        conn = get_db_connection()
        package = conn.execute('SELECT nombre FROM precios_paquetes WHERE id = ?', (package_id,)).fetchone()
        conn.close()
        
        if not package:
            flash('Paquete no encontrado', 'error')
            return redirect('/admin')
        
        old_name = package['nombre']
        
        # Actualizar nombre
        update_package_name(int(package_id), new_name)
        flash(f'Nombre actualizado exitosamente: "{old_name}" → "{new_name}"', 'success')
        
    except Exception as e:
        flash(f'Error al actualizar nombre: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/billetera')
def billetera():
    if 'usuario' not in session:
        return redirect('/auth')
    
    is_admin = session.get('is_admin', False)
    
    if is_admin:
        # Admin ve todos los créditos agregados a usuarios
        wallet_credits = get_all_wallet_credits()
        recargas_admin = get_all_recargas_admin()
        
        return render_template('billetera.html', 
                             wallet_credits=wallet_credits,
                             recargas_admin=recargas_admin,
                             user_id=session.get('id', '00000'),
                             balance=0,
                             is_admin=True,
                             recarga_pendiente=None,
                             recargas_historial=[],
                             binance_pay_id=BINANCE_PAY_ID,
                             recarga_min=RECARGA_MIN_USDT,
                             recarga_max=RECARGA_MAX_USDT,
                             recarga_bonus=RECARGA_BONUS_PERCENT)
    else:
        # Usuario normal ve solo sus créditos de billetera
        user_id = session.get('user_db_id')
        if not user_id:
            flash('Error al acceder a la billetera', 'error')
            return redirect('/')
        
        # Marcar todas las notificaciones de cartera como vistas
        mark_wallet_credits_as_read(user_id)
        
        # Obtener créditos de billetera del usuario
        wallet_credits = get_user_wallet_credits(user_id)
        
        # Obtener recarga pendiente y historial de recargas
        recarga_pendiente = get_recarga_pendiente(user_id)
        recargas_historial = get_recargas_usuario(user_id)
        
        # Actualizar saldo
        conn = get_db_connection()
        user = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            session['saldo'] = user['saldo']
        conn.close()
        
        return render_template('billetera.html', 
                             wallet_credits=wallet_credits, 
                             user_id=session.get('id', '00000'),
                             balance=session.get('saldo', 0),
                             is_admin=False,
                             recarga_pendiente=recarga_pendiente,
                             recargas_historial=recargas_historial,
                             binance_pay_id=BINANCE_PAY_ID,
                             recarga_min=RECARGA_MIN_USDT,
                             recarga_max=RECARGA_MAX_USDT,
                             recarga_bonus=RECARGA_BONUS_PERCENT)

@app.route('/billetera/crear-recarga', methods=['POST'])
def crear_recarga():
    """Crea una nueva orden de recarga por Binance Pay"""
    if 'usuario' not in session or session.get('is_admin'):
        return redirect('/auth')
    
    user_id = session.get('user_db_id')
    if not user_id:
        flash('Error de sesión', 'error')
        return redirect('/billetera')


@app.route('/admin/revendedores/balance', methods=['GET'])
def admin_revendedores_balance():
    if not session.get('is_admin'):
        return jsonify({'ok': False, 'error': 'Acceso denegado'}), 403

    return jsonify({
        'ok': True,
        'balance': round(float(get_revendedores_balance()), 2),
    })
    
    try:
        monto = float(request.form.get('monto', 0))
    except (ValueError, TypeError):
        flash('Monto inválido', 'error')
        return redirect('/billetera')
    
    if monto < RECARGA_MIN_USDT or monto > RECARGA_MAX_USDT:
        flash(f'El monto debe estar entre {RECARGA_MIN_USDT} y {RECARGA_MAX_USDT} USDT', 'error')
        return redirect('/billetera')
    
    resultado = crear_orden_recarga(user_id, monto)
    if resultado:
        flash(f'Orden de recarga creada. Envía exactamente {monto:.2f} USDT con el código {resultado["codigo"]} como nota.', 'success')
    else:
        flash('Error al crear la orden de recarga. Intenta de nuevo.', 'error')
    
    return redirect('/billetera')

@app.route('/billetera/verificar-recarga', methods=['POST'])
def verificar_recarga():
    """Verifica manualmente si una recarga fue pagada"""
    if 'usuario' not in session or session.get('is_admin'):
        return redirect('/auth')
    
    user_id = session.get('user_db_id')
    if not user_id:
        flash('Error de sesión', 'error')
        return redirect('/billetera')
    
    recarga = get_recarga_pendiente(user_id)
    if not recarga:
        flash('No tienes una recarga pendiente', 'error')
        return redirect('/billetera')
    
    resultado = verificar_recarga_binance(recarga['id'])
    
    if resultado['status'] == 'completada':
        total = resultado.get('total_acreditado', 0)
        flash(f'¡Recarga completada! {total:.2f}$ acreditados a tu saldo', 'success')
    elif resultado['status'] == 'expirada':
        flash('La orden de recarga ha expirado. Crea una nueva.', 'error')
    elif resultado['status'] == 'pendiente':
        flash('Pago no detectado aún. Asegúrate de enviar el monto exacto con el código como nota y espera unos segundos.', 'warning')
    elif resultado['status'] == 'ya_procesada':
        flash('Esta transacción ya fue procesada anteriormente.', 'warning')
    else:
        flash(resultado.get('message', 'Error al verificar'), 'error')
    
    return redirect('/billetera')

@app.route('/billetera/verificar-recarga-api', methods=['POST'])
def verificar_recarga_api():
    """API JSON para verificación automática desde el frontend (polling)"""
    if 'usuario' not in session or session.get('is_admin'):
        return jsonify({'status': 'error', 'message': 'No autorizado'}), 401
    
    user_id = session.get('user_db_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'Error de sesión'}), 400
    
    recarga = get_recarga_pendiente(user_id)
    if not recarga:
        return jsonify({'status': 'no_pendiente', 'message': 'No hay recarga pendiente'})
    
    resultado = verificar_recarga_binance(recarga['id'])
    return jsonify(resultado)

@app.route('/billetera/cancelar-recarga', methods=['POST'])
def cancelar_recarga():
    """Cancela una recarga pendiente"""
    if 'usuario' not in session or session.get('is_admin'):
        return redirect('/auth')
    
    user_id = session.get('user_db_id')
    if not user_id:
        return redirect('/billetera')
    
    conn = get_db_connection()
    conn.execute('''
        UPDATE recargas_binance SET estado = 'expirada' 
        WHERE usuario_id = ? AND estado = 'pendiente'
    ''', (user_id,))
    conn.commit()
    conn.close()
    
    flash('Orden de recarga cancelada', 'info')
    return redirect('/billetera')


@app.route('/validar/freefire_latam', methods=['POST'])
def validar_freefire_latam():
    return redirect('/juego/freefire')

    if 'usuario' not in session:
        return redirect('/auth')

    is_admin = session.get('is_admin', False)
    if not is_admin:
        ga = get_games_active()
        if not ga.get('freefire', False):
            flash('Este juego está desactivado temporalmente.', 'error')
            return redirect('/')
    
    monto_id = request.form.get('monto')
    cantidad = request.form.get('cantidad')
    
    if not monto_id or not cantidad:
        flash('Por favor selecciona un paquete y cantidad', 'error')
        return redirect('/juego/freefire_latam')
    
    monto_id = int(monto_id)
    cantidad = int(cantidad)
    user_id = session.get('user_db_id')
    is_admin = session.get('is_admin', False)
    
    # Validar cantidad (entre 1 y 10)
    if cantidad < 1 or cantidad > 10:
        flash('La cantidad debe estar entre 1 y 10 pines', 'error')
        return redirect('/juego/freefire_latam')
    
    # Obtener precio dinámico de la base de datos
    if session.get('is_admin'):
        precio_unitario = get_price_by_id_any(monto_id)
    else:
        precio_unitario = get_price_by_id(monto_id)
    precio_total = precio_unitario * cantidad
    
    # Obtener información del paquete usando cache
    packages_info = get_package_info_with_prices_cached()
    package_info = packages_info.get(monto_id, {})
    
    paquete_nombre = f"{package_info.get('nombre', 'Paquete')} x{cantidad}"
    
    if precio_unitario == 0:
        flash('Paquete no encontrado o inactivo', 'error')
        return redirect('/juego/freefire_latam')
    
    saldo_actual = session.get('saldo', 0)
    
    # Solo verificar saldo para usuarios normales, admin puede comprar sin saldo
    if not is_admin and saldo_actual < precio_total:
        flash(f'Saldo insuficiente. Necesitas ${precio_total:.2f} pero tienes ${saldo_actual:.2f}', 'error')
        return redirect('/juego/freefire_latam')
    
    # CRÍTICO: Usar pin manager para obtener pines ANTES de descontar saldo
    pin_manager = create_pin_manager(DATABASE)
    
    try:
        # PASO 1: Intentar obtener los pines SIN descontar saldo aún
        if cantidad == 1:
            # Para un solo pin
            result = pin_manager.request_pin(monto_id)
            
            if result.get('status') == 'success':
                pines_codigos = [result.get('pin_code')]
                sources_used = ['local_stock']
            else:
                flash('Sin stock disponible para este paquete.', 'error')
                return redirect('/juego/freefire_latam')
        else:
            # Para múltiples pines
            result = pin_manager.request_multiple_pins(monto_id, cantidad)
            
            if result.get('status') == 'success':
                pines_data = result.get('pins', [])
                pines_codigos = [pin['pin_code'] for pin in pines_data]
                # Determinar fuentes usadas basado en el resultado
                source = result.get('source', 'local_stock')
                sources_used = [source]
            elif result.get('status') == 'partial_success':
                # Algunos pines obtenidos, pero no todos
                pines_data = result.get('pins', [])
                pines_codigos = [pin['pin_code'] for pin in pines_data]
                source = result.get('source', 'local_stock')
                sources_used = [source]
                
                # Actualizar cantidad y precio total para los pines realmente obtenidos
                cantidad_original = cantidad
                cantidad = len(pines_codigos)
                precio_total = precio_unitario * cantidad  # Recalcular precio
                
                flash(f'Advertencia: Solo se obtuvieron {cantidad} pines de los {cantidad_original} solicitados. Precio ajustado a ${precio_total:.2f}', 'warning')
            else:
                flash(f'Error al obtener pines. {result.get("message", "Error desconocido")}', 'error')
                return redirect('/juego/freefire_latam')
        
        # PASO 2: Verificar que se obtuvieron pines exitosamente
        if not pines_codigos:
            flash('No se pudieron obtener pines. Intente nuevamente.', 'error')
            return redirect('/juego/freefire_latam')
        
        # PASO 3: AHORA SÍ verificar saldo final después de saber cuántos pines obtuvimos
        if not is_admin and saldo_actual < precio_total:
            flash(f'Saldo insuficiente para la cantidad obtenida. Necesitas ${precio_total:.2f} pero tienes ${saldo_actual:.2f}', 'error')
            return redirect('/juego/freefire_latam')
        
        # Generar datos de la transacción
        import random
        import string
        numero_control = ''.join(random.choices(string.digits, k=10))
        transaccion_id = 'FF-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        
        # Procesar la transacción
        conn = get_db_connection()
        try:
            # Solo actualizar saldo si no es admin
            if not is_admin:
                conn.execute('UPDATE usuarios SET saldo = saldo - ? WHERE id = ?', (precio_total, user_id))
            
            # Registrar la transacción
            pines_texto = '\n'.join(pines_codigos)
            
            # Para admin, registrar con monto negativo pero agregar etiqueta [ADMIN]
            if is_admin:
                pines_texto = f"[ADMIN - PRUEBA/GESTIÓN]\n{pines_texto}"
                monto_transaccion = -precio_total  # Registrar monto real para mostrar en historial
            else:
                monto_transaccion = -precio_total
                
                # Agregar información de fuente en el pin si viene de API externa
                if 'inefable_api' in sources_used:
                    pines_texto += f"\n[Fuente: {', '.join(sources_used)}]"
            
            conn.execute('''
                INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, numero_control, pines_texto, transaccion_id, paquete_nombre, monto_transaccion))
            
            # Registrar en historial permanente
            new_saldo_row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
            _saldo_despues = new_saldo_row['saldo'] if new_saldo_row else 0
            registrar_historial_compra(conn, user_id, abs(monto_transaccion), paquete_nombre, pines_texto, 'compra', None, _saldo_despues + abs(monto_transaccion), _saldo_despues)
            
            # Actualizar gastos mensuales persistentes (para top clientes)
            if not is_admin:
                try:
                    update_monthly_spending(conn, user_id, precio_total)
                except Exception:
                    pass
            
            # Persistir profit (legacy)
            try:
                record_profit_for_transaction(conn, user_id, is_admin, 'freefire_latam', monto_id, cantidad, precio_unitario, transaccion_id)
            except Exception:
                pass
            
            # Limitar transacciones a 100 por usuario (aumentado de 30 para evitar eliminaciones frecuentes)
            conn.execute('''
                DELETE FROM transacciones 
                WHERE usuario_id = ? AND id NOT IN (
                    SELECT id FROM (SELECT id FROM transacciones 
                    WHERE usuario_id = ? 
                    ORDER BY fecha DESC 
                    LIMIT 100) AS keep_ids
                )
            ''', (user_id, user_id))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            flash('Error al procesar la transacción. Intente nuevamente.', 'error')
            return redirect('/juego/freefire_latam')
        finally:
            conn.close()
        
        # Actualizar saldo en sesión solo si no es admin
        if not is_admin:
            session['saldo'] = saldo_actual - precio_total
        
        # Registrar venta en estadísticas semanales (solo para usuarios normales)
        if not is_admin:
            register_weekly_sale('freefire_latam', monto_id, package_info.get('nombre', 'Paquete'), precio_unitario, cantidad)
        
        # Guardar datos de la compra en la sesión para mostrar después del redirect
        if cantidad == 1:
            # Para un solo pin
            session['compra_exitosa'] = {
                'paquete_nombre': paquete_nombre,
                'monto_compra': precio_total,
                'numero_control': numero_control,
                'pin': pines_codigos[0],
                'transaccion_id': transaccion_id,
                'cantidad_comprada': cantidad,
                'source': sources_used[0] if sources_used else 'local_stock'
            }
        else:
            # Para múltiples pines
            session['compra_exitosa'] = {
                'paquete_nombre': paquete_nombre,
                'monto_compra': precio_total,
                'numero_control': numero_control,
                'pines_list': pines_codigos,
                'transaccion_id': transaccion_id,
                'cantidad_comprada': cantidad,
                'sources_used': sources_used
            }
        
        # Redirect para evitar reenvío del formulario (patrón POST-Redirect-GET)
        return redirect('/juego/freefire_latam?compra=exitosa')
        
    except Exception as e:
        flash(f'Error inesperado al procesar la compra: {str(e)}', 'error')
        return redirect('/juego/freefire_latam')

@app.route('/juego/freefire_latam')
def freefire_latam():
    return redirect('/juego/freefire')

    if 'usuario' not in session:
        return redirect('/auth')

    is_admin = session.get('is_admin', False)
    if not is_admin:
        ga = get_games_active()
        if not ga.get('freefire', False):
            flash('Este juego está desactivado temporalmente.', 'error')
            return redirect('/')
    
    # Actualizar saldo desde la base de datos
    user_id = session.get('user_db_id')
    if user_id:
        conn = get_db_connection()
        user = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            session['saldo'] = user['saldo']
        conn.close()
    
    # Obtener stock local y configuración de fuentes
    pin_manager = create_pin_manager(DATABASE)
    local_stock = pin_manager.get_local_stock()
    pin_sources_config = get_pin_source_config()
    
    # Preparar información de stock considerando la configuración de fuentes
    stock = {}
    for monto_id in range(1, 10):
        local_count = local_stock.get(monto_id, 0)
        source_config = pin_sources_config.get(monto_id, 'local')
        
        # Si está configurado para API externa, siempre mostrar disponible
        if source_config == 'api_externa':
            stock[monto_id] = {
                'local': local_count,
                'external_available': True,  # Siempre True para API externa
                'total_available': True,     # Siempre disponible cuando usa API externa
            }
        else:
            # Si está configurado para stock local, mostrar según stock real
            stock[monto_id] = {
                'local': local_count,
                'external_available': False,
                'total_available': local_count > 0,  # Solo disponible si hay stock local
            }
    
    # Obtener precios
    if is_admin:
        prices = {}
        for row in get_all_prices():
            try:
                prices[int(row['id'])] = {
                    'nombre': row['nombre'],
                    'precio': row['precio'],
                    'descripcion': row.get('descripcion') if hasattr(row, 'get') else row['descripcion']
                }
            except Exception:
                continue
    else:
        prices = get_package_info_with_prices()
    
    # Verificar si hay una compra exitosa para mostrar (solo una vez)
    compra_exitosa = False
    compra_data = {}
    
    # Solo mostrar compra exitosa si viene del redirect POST y hay datos en sesión
    if request.args.get('compra') == 'exitosa' and 'compra_exitosa' in session:
        compra_exitosa = True
        compra_data = session.pop('compra_exitosa')  # Remover después de usar para evitar mostrar de nuevo
    
    return render_template('freefire_latam.html', 
                         user_id=session.get('id', '00000'),
                         balance=session.get('saldo', 0),
                         stock=stock,
                         prices=prices,
                         compra_exitosa=compra_exitosa,
                         is_admin=is_admin,
                         games_active=get_games_active(),
                         **compra_data)  # Desempaquetar los datos de la compra

# Rutas para Blood Striker
@app.route('/juego/bloodstriker')
def bloodstriker():
    if 'usuario' not in session:
        return redirect('/auth')

    is_admin = session.get('is_admin', False)
    if not is_admin:
        ga = get_games_active()
        if not ga.get('bloodstriker', False):
            flash('Este juego está desactivado temporalmente.', 'error')
            return redirect('/')
    
    # Actualizar saldo desde la base de datos
    user_id = session.get('user_db_id')
    if user_id:
        conn = get_db_connection()
        user = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            session['saldo'] = user['saldo']
        conn.close()
    
    # Obtener precios dinámicos de Blood Striker
    if is_admin:
        prices = {}
        for row in get_all_bloodstriker_prices():
            try:
                prices[int(row['id'])] = {
                    'nombre': row['nombre'],
                    'precio': row['precio'],
                    'descripcion': row.get('descripcion') if hasattr(row, 'get') else row['descripcion']
                }
            except Exception:
                continue
    else:
        prices = get_bloodstriker_prices()
    
    # Verificar si hay una compra exitosa para mostrar (solo una vez)
    compra_exitosa = False
    compra_data = {}
    
    # Solo mostrar compra exitosa si viene del redirect POST y hay datos en sesión
    if request.args.get('compra') == 'exitosa' and 'compra_bloodstriker_exitosa' in session:
        compra_exitosa = True
        compra_data = session.pop('compra_bloodstriker_exitosa')  # Remover después de usar
    
    return render_template('bloodstriker.html', 
                         user_id=session.get('id', '00000'),
                         balance=session.get('saldo', 0),
                         prices=prices,
                         compra_exitosa=compra_exitosa,
                         is_admin=is_admin,
                         games_active=get_games_active(),
                         **compra_data)

@app.route('/validar/bloodstriker', methods=['POST'])
def validar_bloodstriker():
    if 'usuario' not in session:
        return redirect('/auth')

    is_admin = session.get('is_admin', False)
    if not is_admin:
        ga = get_games_active()
        if not ga.get('bloodstriker', False):
            flash('Este juego está desactivado temporalmente.', 'error')
            return redirect('/')
    
    package_id = request.form.get('monto')
    player_id = request.form.get('player_id')
    
    if not package_id or not player_id:
        flash('Por favor complete todos los campos', 'error')
        return redirect('/juego/bloodstriker')
    
    package_id = int(package_id)
    user_id = session.get('user_db_id')
    is_admin = session.get('is_admin', False)
    
    # Obtener precio dinámico de la base de datos
    if is_admin:
        precio = get_bloodstriker_price_by_id_any(package_id)
    else:
        precio = get_bloodstriker_price_by_id(package_id)
    
    # Obtener información del paquete (necesitamos gamepoint_package_id)
    conn_pkg = get_db_connection()
    pkg_row = conn_pkg.execute(
        'SELECT id, nombre, precio, descripcion, gamepoint_package_id FROM precios_bloodstriker WHERE id = ?',
        (package_id,)
    ).fetchone()
    conn_pkg.close()
    
    if not pkg_row or precio == 0:
        flash('Paquete no encontrado o inactivo', 'error')
        return redirect('/juego/bloodstriker')
    
    gp_package_id = pkg_row['gamepoint_package_id']
    paquete_nombre = f"{pkg_row['nombre']} / ${precio:.2f}"
    
    if not gp_package_id:
        flash('Este paquete no tiene configurado el ID de GamePoint. Contacta al administrador.', 'error')
        return redirect('/juego/bloodstriker')
    
    # === PROTECCIÓN: Verificar saldo desde DB (no session) ===
    if not is_admin:
        conn_check = get_db_connection()
        row = conn_check.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        conn_check.close()
        saldo_actual = row['saldo'] if row else 0
        session['saldo'] = saldo_actual
    else:
        saldo_actual = session.get('saldo', 0)
    
    if not is_admin and saldo_actual < precio:
        flash(f'Saldo insuficiente. Necesitas ${precio:.2f} pero tienes ${saldo_actual:.2f}', 'error')
        return redirect('/juego/bloodstriker')
    
    # Check for recent duplicate pending transaction (prevents double recharge on page refresh)
    try:
        conn_dup = get_db_connection()
        dup = conn_dup.execute(
            '''SELECT id FROM transacciones_bloodstriker
               WHERE usuario_id = ? AND paquete_id = ?
                 AND estado IN ('procesando', 'pendiente', 'aprobado')
                 AND fecha >= (NOW() - INTERVAL '2 minutes')
               LIMIT 1''',
            (user_id, package_id)
        ).fetchone()
        conn_dup.close()
        if dup:
            flash('Ya se est\u00e1 procesando tu recarga. Espera unos segundos y revisa tu historial.', 'error')
            return redirect('/juego/bloodstriker')
    except Exception:
        pass
    
    # === COMPRA AUTOMÁTICA VIA GAMEPOINT CLUB ===
    import time as _time
    _bs_start = _time.time()
    bloodstrike_product_id = int(os.environ.get('BLOODSTRIKE_PRODUCT_ID', '155'))
    _bs_tx_id = None
    
    try:
        # 1. Cobrar al usuario (atómico)
        if not is_admin:
            conn = get_db_connection()
            cursor = conn.execute(
                'UPDATE usuarios SET saldo = saldo - ? WHERE id = ? AND saldo >= ?',
                (precio, user_id, precio))
            if cursor.rowcount == 0:
                conn.close()
                flash('Saldo insuficiente al momento de procesar. Recarga tu saldo e intenta de nuevo.', 'error')
                return redirect('/juego/bloodstriker')
            conn.commit()
            new_saldo = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
            conn.close()
            session['saldo'] = new_saldo['saldo'] if new_saldo else 0
        
        # 1b. Insert 'procesando' record BEFORE GamePoint calls (prevents duplicate on refresh)
        import random as _bs_random
        import string as _bs_string
        _bs_numero_control = ''.join(_bs_random.choices(_bs_string.digits, k=10))
        _bs_transaccion_id = 'BS-' + ''.join(_bs_random.choices(_bs_string.ascii_uppercase + _bs_string.digits, k=8))
        try:
            conn_proc = get_db_connection()
            dup_proc = conn_proc.execute(
                '''SELECT id FROM transacciones_bloodstriker
                   WHERE usuario_id = ? AND paquete_id = ?
                     AND estado IN ('procesando', 'pendiente', 'aprobado')
                     AND fecha >= (NOW() - INTERVAL '2 minutes')
                   LIMIT 1''',
                (user_id, package_id)
            ).fetchone()
            if dup_proc:
                conn_proc.close()
                if not is_admin:
                    conn_r = get_db_connection()
                    conn_r.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, user_id))
                    conn_r.commit()
                    conn_r.close()
                    session['saldo'] = session.get('saldo', 0) + precio
                flash('Ya se est\u00e1 procesando tu recarga. Espera unos segundos y revisa tu historial.', 'error')
                return redirect('/juego/bloodstriker')
            conn_proc.execute('''
                INSERT INTO transacciones_bloodstriker
                (usuario_id, player_id, paquete_id, numero_control, transaccion_id, monto, estado)
                VALUES (?, ?, ?, ?, ?, ?, 'procesando')
            ''', (user_id, player_id, package_id, _bs_numero_control, _bs_transaccion_id, -precio))
            conn_proc.commit()
            _bs_tx_row = conn_proc.execute('SELECT id FROM transacciones_bloodstriker WHERE transaccion_id = ?', (_bs_transaccion_id,)).fetchone()
            _bs_tx_id = _bs_tx_row['id'] if _bs_tx_row else None
            conn_proc.close()
        except Exception as e_proc:
            logger.error(f"[BloodStrike] Error insertando procesando: {e_proc}")
            if not is_admin:
                conn_r = get_db_connection()
                conn_r.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, user_id))
                conn_r.commit()
                conn_r.close()
                session['saldo'] = session.get('saldo', 0) + precio
            flash('Error al procesar. Tu saldo ha sido devuelto.', 'error')
            return redirect('/juego/bloodstriker')
        
        # 2. Obtener token de GamePoint
        gc_token, gc_err = _gameclub_get_token()
        if not gc_token:
            # Mark procesando as error
            if _bs_tx_id:
                try:
                    conn_e = get_db_connection()
                    conn_e.execute('UPDATE transacciones_bloodstriker SET estado = ?, notas = ?, fecha_procesado = CURRENT_TIMESTAMP WHERE id = ?', ('error', 'No se pudo obtener token GP', _bs_tx_id))
                    conn_e.commit()
                    conn_e.close()
                except Exception:
                    pass
            # Reembolsar saldo
            if not is_admin:
                conn = get_db_connection()
                conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, user_id))
                conn.commit()
                conn.close()
                session['saldo'] = session.get('saldo', 0) + precio
            err_msg = (gc_err or {}).get('message', 'No se pudo conectar con GamePoint')
            flash(f'Error de conexión con proveedor: {err_msg}. Tu saldo ha sido devuelto.', 'error')
            return redirect('/juego/bloodstriker')
        
        # 3. Validar orden (order/validate) — obtiene validation_token (30s de vida)
        validate_data = _gameclub_order_validate(gc_token, bloodstrike_product_id, {'input1': str(player_id)})
        validate_code = (validate_data or {}).get('code')
        
        if validate_code != 200 or not (validate_data or {}).get('validation_token'):
            # Mark procesando as error
            if _bs_tx_id:
                try:
                    conn_e = get_db_connection()
                    conn_e.execute('UPDATE transacciones_bloodstriker SET estado = ?, notas = ?, fecha_procesado = CURRENT_TIMESTAMP WHERE id = ?', ('error', f'validate failed: code={validate_code}', _bs_tx_id))
                    conn_e.commit()
                    conn_e.close()
                except Exception:
                    pass
            # Reembolsar saldo
            if not is_admin:
                conn = get_db_connection()
                conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, user_id))
                conn.commit()
                conn.close()
                session['saldo'] = session.get('saldo', 0) + precio
            err_msg = (validate_data or {}).get('message', 'Error validando orden')
            logger.error(f"[BloodStrike] order/validate failed: code={validate_code} msg={err_msg} player={player_id}")
            flash(f'Error al validar la recarga: {err_msg}. Tu saldo ha sido devuelto.', 'error')
            return redirect('/juego/bloodstriker')
        
        validation_token = validate_data['validation_token']
        
        # 3.1 validate NO devuelve ingamename según docs — solo code, message, validation_token
        ingame_name = ''
        logger.info(f"[BloodStrike] validate OK | player_id={player_id}")
        
        # 4. Crear orden (order/create) — la compra real
        merchant_code = _bs_transaccion_id
        create_data = _gameclub_order_create(gc_token, validation_token, gp_package_id, merchant_code)
        create_code = (create_data or {}).get('code')
        reference_no = (create_data or {}).get('referenceno', '')

        # 4.1 order/inquiry — extraer ingamename (docs: campo opcional, aparece cuando el pedido está procesado)
        item_name = ''
        try:
            if reference_no:
                # Retry inquiry con delay para dar tiempo a que el pedido se procese
                for _attempt in range(3):
                    if _attempt > 0:
                        _time.sleep(1.5)
                    inquiry_data = _gameclub_order_inquiry(gc_token, reference_no)
                    ingame_name = (inquiry_data or {}).get('ingamename') or ''
                    item_name = (inquiry_data or {}).get('item') or ''
                    logger.info(f"[BloodStrike] inquiry attempt {_attempt+1}: ingamename='{ingame_name}' | item='{item_name}'")
                    if ingame_name:
                        break
                # Limpiar HTML del item (ej: "Blood Strike<br />300 + 20 Gold")
                if item_name:
                    import re as _re
                    item_name = _re.sub(r'<[^>]+>', ' ', item_name).strip()
                    item_name = ' '.join(item_name.split())
        except Exception as e:
            logger.error(f"[BloodStrike] inquiry FAILED: {e}")
        
        _bs_duration = round(_time.time() - _bs_start, 1)
        
        if create_code in (100, 101):
            # === ÉXITO: Recarga completada o enviada ===
            # Update the 'procesando' record with final results
            conn_upd = get_db_connection()
            conn_upd.execute('''
                UPDATE transacciones_bloodstriker
                SET estado = 'aprobado', gamepoint_referenceno = ?,
                    fecha_procesado = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (reference_no, _bs_tx_id))
            conn_upd.commit()
            conn_upd.close()
            transaction_data = {
                'id': _bs_tx_id,
                'numero_control': _bs_numero_control,
                'transaccion_id': _bs_transaccion_id
            }
            
            # Registrar en transacciones generales
            conn = get_db_connection()
            # Mantener estilo similar a Free Fire ID: incluir ID y nombre si existe
            if ingame_name:
                pin_info = f"ID: {player_id} - Jugador: {ingame_name} - Ref: {reference_no}"
            else:
                pin_info = f"ID: {player_id} - Ref: {reference_no}"

            # Si inquiry devolvió item, úsalo como nombre mostrado del paquete (no reemplaza tu nombre local)
            display_package_name = pkg_row['nombre']
            if item_name:
                display_package_name = f"{pkg_row['nombre']} ({item_name})"
            conn.execute('''
                INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto, duracion_segundos)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, transaction_data['numero_control'], pin_info,
                  transaction_data['transaccion_id'], display_package_name, -precio, _bs_duration))
            
            # Registrar en historial permanente
            _bs_saldo_row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
            _bs_saldo = _bs_saldo_row['saldo'] if _bs_saldo_row else 0
            registrar_historial_compra(conn, user_id, precio, display_package_name, pin_info, 'compra', _bs_duration, _bs_saldo + precio, _bs_saldo)
            
            # Registrar profit
            try:
                admin_ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
                admin_ids = [int(x.strip()) for x in admin_ids_env.split(',') if x.strip().isdigit()]
                is_admin_target = user_id in admin_ids
                record_profit_for_transaction(conn, user_id, is_admin_target, 'bloodstriker', package_id, 1, precio, transaction_data['transaccion_id'])
            except Exception:
                pass
            
            conn.commit()
            conn.close()
            
            # Actualizar gastos mensuales
            if not is_admin:
                try:
                    conn = get_db_connection()
                    update_monthly_spending(conn, user_id, precio)
                    conn.commit()
                    conn.close()
                except Exception:
                    pass
            
            # Registrar venta semanal
            if not is_admin:
                register_weekly_sale('bloodstriker', package_id, pkg_row['nombre'], precio, 1)
            
            estado_txt = 'completado' if create_code == 100 else 'procesando'
            logger.info(f"[BloodStrike] storing in session: player_name='{ingame_name}' | player_id={player_id} | ref={reference_no}")
            session['compra_bloodstriker_exitosa'] = {
                'paquete_nombre': paquete_nombre,
                'monto_compra': precio,
                'numero_control': transaction_data['numero_control'],
                'transaccion_id': transaction_data['transaccion_id'],
                'player_id': player_id,
                'player_name': ingame_name,
                'estado': estado_txt,
                'gamepoint_ref': reference_no,
            }
            
            return redirect('/juego/bloodstriker?compra=exitosa')
        
        else:
            # === FALLO: code 102 u otro error ===
            err_msg = (create_data or {}).get('message', 'Error creando orden')
            logger.error(
                f"[BloodStrike] order/create failed: code={create_code} msg={err_msg} | "
                f"user={user_id} player={player_id} pkg={package_id} gp_pkg={gp_package_id} ref={reference_no}"
            )
            
            # Update procesando record to rechazado
            conn_rej = get_db_connection()
            conn_rej.execute('''
                UPDATE transacciones_bloodstriker
                SET estado = 'rechazado', gamepoint_referenceno = ?, notas = ?,
                    fecha_procesado = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (reference_no, err_msg, _bs_tx_id))
            conn_rej.commit()
            conn_rej.close()
            
            # Reembolsar saldo
            if not is_admin:
                conn = get_db_connection()
                conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, user_id))
                conn.commit()
                conn.close()
                session['saldo'] = session.get('saldo', 0) + precio
                logger.info(f"[BloodStrike] Saldo ${precio} reembolsado al usuario {user_id}")
            
            flash(f'La recarga falló: {err_msg}. Tu saldo ha sido devuelto.', 'error')
            return redirect('/juego/bloodstriker')
    
    except Exception as e:
        logger.error(f"[BloodStrike] Error general: {str(e)}")
        # Mark procesando as error
        if _bs_tx_id:
            try:
                conn_err = get_db_connection()
                conn_err.execute(
                    'UPDATE transacciones_bloodstriker SET estado = ?, notas = ?, fecha_procesado = CURRENT_TIMESTAMP WHERE id = ?',
                    ('error', str(e)[:500], _bs_tx_id)
                )
                conn_err.commit()
                conn_err.close()
            except Exception:
                pass
        # Intentar reembolsar en caso de error inesperado
        try:
            if not is_admin:
                conn = get_db_connection()
                conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, user_id))
                conn.commit()
                conn.close()
                session['saldo'] = session.get('saldo', 0) + precio
        except Exception:
            pass
        flash('Error al procesar la compra. Tu saldo ha sido devuelto. Intente nuevamente.', 'error')
        return redirect('/juego/bloodstriker')

# Rutas de administrador para Blood Striker
@app.route('/admin/bloodstriker_transactions')
def admin_bloodstriker_transactions():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    pending_transactions = get_pending_bloodstriker_transactions()
    return render_template('admin_bloodstriker.html', transactions=pending_transactions)

@app.route('/admin/bloodstriker_approve', methods=['POST'])
def admin_bloodstriker_approve():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    transaction_id = request.form.get('transaction_id')
    notas = request.form.get('notas', '')
    
    if transaction_id:
        update_bloodstriker_transaction_status(int(transaction_id), 'aprobado', session.get('user_db_id'), notas)
        flash('Transacción aprobada exitosamente', 'success')
    else:
        flash('ID de transacción inválido', 'error')
    
    return redirect('/admin/bloodstriker_transactions')

@app.route('/admin/bloodstriker_reject', methods=['POST'])
def admin_bloodstriker_reject():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    transaction_id = request.form.get('transaction_id')
    notas = request.form.get('notas', '')
    
    if transaction_id:
        # Obtener información de la transacción para devolver el saldo
        conn = get_db_connection()
        transaction = conn.execute('''
            SELECT usuario_id, monto FROM transacciones_bloodstriker 
            WHERE id = ?
        ''', (transaction_id,)).fetchone()
        
        if transaction:
            # Devolver saldo al usuario (monto es negativo, así que sumamos el valor absoluto)
            conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', 
                        (abs(transaction['monto']), transaction['usuario_id']))
            conn.commit()
        conn.close()
        
        # Actualizar estado de la transacción
        update_bloodstriker_transaction_status(int(transaction_id), 'rechazado', session.get('user_db_id'), notas)
        flash('Transacción rechazada y saldo devuelto al usuario', 'success')
    else:
        flash('ID de transacción inválido', 'error')
    
    return redirect('/admin/bloodstriker_transactions')

@app.route('/admin/update_bloodstriker_price', methods=['POST'])
def admin_update_bloodstriker_price():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    package_id = request.form.get('package_id')
    new_price = request.form.get('new_price')
    
    if not package_id or not new_price:
        flash('Datos inválidos para actualizar precio', 'error')
        return redirect('/admin')
    
    try:
        new_price = float(new_price)
        if new_price < 0:
            flash('El precio no puede ser negativo', 'error')
            return redirect('/admin')
        
        # Obtener información del paquete antes de actualizar
        conn = get_db_connection()
        package = conn.execute('SELECT nombre FROM precios_bloodstriker WHERE id = ?', (package_id,)).fetchone()
        conn.close()
        
        if not package:
            flash('Paquete no encontrado', 'error')
            return redirect('/admin')
        
        # Actualizar precio
        update_bloodstriker_price(int(package_id), new_price)
        flash(f'Precio de Blood Striker actualizado exitosamente para {package["nombre"]}: ${new_price:.2f}', 'success')
        
    except ValueError:
        flash('Precio inválido. Debe ser un número válido.', 'error')
    except Exception as e:
        flash(f'Error al actualizar precio: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/update_bloodstriker_name', methods=['POST'])
def admin_update_bloodstriker_name():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    package_id = request.form.get('package_id')
    new_name = request.form.get('new_name')
    
    if not package_id or not new_name:
        flash('Datos inválidos para actualizar nombre', 'error')
        return redirect('/admin')
    
    try:
        new_name = new_name.strip()
        if len(new_name) < 1:
            flash('El nombre no puede estar vacío', 'error')
            return redirect('/admin')
        
        if len(new_name) > 50:
            flash('El nombre no puede exceder 50 caracteres', 'error')
            return redirect('/admin')
        
        # Obtener información del paquete antes de actualizar
        conn = get_db_connection()
        package = conn.execute('SELECT nombre FROM precios_bloodstriker WHERE id = ?', (package_id,)).fetchone()
        conn.close()
        
        if not package:
            flash('Paquete no encontrado', 'error')
            return redirect('/admin')
        
        old_name = package['nombre']
        
        # Actualizar nombre
        update_bloodstriker_name(int(package_id), new_name)
        flash(f'Nombre de Blood Striker actualizado exitosamente: "{old_name}" → "{new_name}"', 'success')
        
    except Exception as e:
        flash(f'Error al actualizar nombre: {str(e)}', 'error')
    
    return redirect('/admin')

def _bloodstrike_sync_prices_internal(deactivate_missing=False, deactivate_unmapped=False):
    """Sincroniza precios de Blood Strike desde GamePoint Club (función interna, sin request context).
    Mantiene la ganancia por paquete: ganancia = precio_venta_actual - costo_actual.
    Nuevo precio = nuevo_costo + ganancia_existente.
    Retorna dict con resultado o error."""
    default_profit_usd = float(os.environ.get('BLOODSTRIKE_PROFIT_USD', '0.11'))
    from dynamic_games import get_gp_myr_rate as _get_gp_myr_rate
    myr_to_usd = _get_gp_myr_rate()
    usd_to_myr = round((1.0 / float(myr_to_usd)), 6) if float(myr_to_usd) > 0 else None
    product_id = int(os.environ.get('BLOODSTRIKE_PRODUCT_ID', '155'))
    
    # 1. Obtener token
    gc_token, gc_err = _gameclub_get_token()
    if not gc_token:
        return {'error': (gc_err or {}).get('message', 'No se pudo obtener token de GamePoint')}
    
    # 2. Obtener detalle del producto
    _, detail_data = _gameclub_post('product/detail', {'token': gc_token, 'productid': product_id})
    if (detail_data or {}).get('code') != 200:
        return {'error': (detail_data or {}).get('message', 'Error obteniendo detalle del producto')}
    
    gp_packages = (detail_data or {}).get('package', [])
    if not gp_packages:
        return {'error': 'No se encontraron paquetes en GamePoint para este producto'}

    gp_ids = set()
    for gp_pkg in gp_packages:
        try:
            gp_ids.add(int(gp_pkg.get('id')))
        except Exception:
            continue
    
    # 3. Leer paquetes locales actuales con su costo de compra
    conn = get_db_connection()
    local_packages = conn.execute('SELECT id, nombre, precio, gamepoint_package_id FROM precios_bloodstriker ORDER BY id').fetchall()
    
    # Cargar costos actuales desde precios_compra
    local_costs = {}
    for row in conn.execute("SELECT paquete_id, precio_compra FROM precios_compra WHERE juego = 'bloodstriker'").fetchall():
        local_costs[int(row['paquete_id'])] = float(row['precio_compra'])
    
    # Crear mapeo gamepoint_package_id -> local row
    gp_to_local = {}
    for lp in local_packages:
        if lp['gamepoint_package_id']:
            gp_to_local[int(lp['gamepoint_package_id'])] = dict(lp)

    # Desactivar paquetes locales sin mapeo o con mapeo no existente (si se pide)
    if deactivate_unmapped:
        conn.execute('UPDATE precios_bloodstriker SET activo = FALSE, fecha_actualizacion = CURRENT_TIMESTAMP WHERE gamepoint_package_id IS NULL')

    if deactivate_missing:
        for lp in local_packages:
            try:
                gp_id = lp['gamepoint_package_id']
                if gp_id is None:
                    continue
                if int(gp_id) not in gp_ids:
                    conn.execute('UPDATE precios_bloodstriker SET activo = FALSE, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ?', (lp['id'],))
            except Exception:
                continue
    
    report = []
    updated = 0
    
    for gp_pkg in gp_packages:
        gp_id = int(gp_pkg['id'])
        gp_name = gp_pkg.get('name', '')
        gp_price_myr = float(gp_pkg.get('price', 0))
        nuevo_costo_usd = round(gp_price_myr * myr_to_usd, 4)
        
        local = gp_to_local.get(gp_id)
        
        if local:
            # Calcular ganancia actual de ESTE paquete: precio_venta - costo_compra
            costo_actual = local_costs.get(local['id'], 0)
            if costo_actual > 0:
                ganancia_paquete = round(local['precio'] - costo_actual, 4)
            else:
                # Si falta costo histórico, inferir margen desde precio actual para
                # mantener el precio base y que próximos cambios sigan el delta GP.
                ganancia_paquete = round(float(local['precio']) - float(nuevo_costo_usd), 4)
            
            nuevo_precio_venta = round(nuevo_costo_usd + ganancia_paquete, 2)
            
            entry = {
                'gamepoint_id': gp_id,
                'gamepoint_name': gp_name,
                'gamepoint_price_myr': gp_price_myr,
                'costo_usd_anterior': round(costo_actual, 4),
                'costo_usd_nuevo': round(nuevo_costo_usd, 4),
                'ganancia_paquete': round(ganancia_paquete, 4),
                'nuevo_precio_venta_usd': nuevo_precio_venta,
                'local_id': local['id'],
                'local_nombre': local['nombre'],
                'precio_anterior': local['precio'],
                'cambio': round(nuevo_precio_venta - local['precio'], 4),
            }
            
            conn.execute(
                'UPDATE precios_bloodstriker SET precio = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ?',
                (nuevo_precio_venta, local['id'])
            )
            conn.execute(
                '''
                INSERT INTO precios_compra (juego, paquete_id, precio_compra, activo)
                VALUES (?, ?, ?, TRUE)
                ON CONFLICT (juego, paquete_id) DO UPDATE
                SET precio_compra = EXCLUDED.precio_compra,
                    activo = EXCLUDED.activo,
                    fecha_actualizacion = CURRENT_TIMESTAMP
                ''',
                ('bloodstriker', local['id'], nuevo_costo_usd)
            )
            updated += 1
        else:
            nuevo_precio_venta = round(nuevo_costo_usd + default_profit_usd, 2)
            entry = {
                'gamepoint_id': gp_id,
                'gamepoint_name': gp_name,
                'gamepoint_price_myr': gp_price_myr,
                'costo_usd_nuevo': round(nuevo_costo_usd, 4),
                'ganancia_paquete': default_profit_usd,
                'nuevo_precio_venta_usd': nuevo_precio_venta,
                'local_id': None,
                'nota': 'Sin mapeo local (gamepoint_package_id no asignado a ningún paquete)',
            }
        
        report.append(entry)
    
    conn.commit()
    conn.close()
    
    # Limpiar caches
    try:
        get_bloodstriker_prices_cached.cache_clear()
    except Exception:
        pass
    
    return {
        'success': True,
        'product_id': product_id,
        'default_profit_usd': default_profit_usd,
        'usd_to_myr_rate': usd_to_myr,
        'myr_to_usd_rate': myr_to_usd,
        'packages_updated': updated,
        'total_gamepoint_packages': len(gp_packages),
        'report': report
    }


@app.route('/admin/bloodstrike/sync_prices', methods=['POST'])
def admin_bloodstrike_sync_prices():
    """Sincroniza precios de Blood Strike desde GamePoint Club (endpoint admin)."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403

    deactivate_missing = str(request.args.get('deactivate_missing', '')).strip() == '1'
    deactivate_unmapped = str(request.args.get('deactivate_unmapped', '')).strip() == '1'
    
    result = _bloodstrike_sync_prices_internal(deactivate_missing=deactivate_missing, deactivate_unmapped=deactivate_unmapped)
    
    if result.get('error'):
        return jsonify(result), 500
    return jsonify(result)


@app.route('/admin/bloodstrike/set_gamepoint_id', methods=['POST'])
def admin_bloodstrike_set_gamepoint_id():
    """Asigna un gamepoint_package_id a un paquete local de Blood Strike."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    
    data = request.get_json() or request.form
    local_id = data.get('local_id') or data.get('package_id')
    gp_id = data.get('gamepoint_package_id')
    
    if not local_id:
        return jsonify({'error': 'Falta local_id'}), 400
    
    conn = get_db_connection()
    pkg = conn.execute('SELECT id, nombre FROM precios_bloodstriker WHERE id = ?', (local_id,)).fetchone()
    if not pkg:
        conn.close()
        return jsonify({'error': 'Paquete local no encontrado'}), 404
    
    gp_val = int(gp_id) if gp_id and str(gp_id).strip() else None
    conn.execute(
        'UPDATE precios_bloodstriker SET gamepoint_package_id = ?, fecha_actualizacion = CURRENT_TIMESTAMP WHERE id = ?',
        (gp_val, local_id)
    )
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'local_id': int(local_id), 'nombre': pkg['nombre'], 'gamepoint_package_id': gp_val})


@app.route('/admin/bloodstrike/gamepoint_packages')
def admin_bloodstrike_gamepoint_packages():
    """Devuelve los paquetes de GamePoint para Blood Strike (productid=155) para el mapeo en admin."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    
    product_id = int(os.environ.get('BLOODSTRIKE_PRODUCT_ID', '155'))
    gc_token, gc_err = _gameclub_get_token()
    if not gc_token:
        return jsonify({'error': (gc_err or {}).get('message', 'No se pudo obtener token')}), 500
    
    _, detail_data = _gameclub_post('product/detail', {'token': gc_token, 'productid': product_id})
    if (detail_data or {}).get('code') != 200:
        return jsonify({'error': (detail_data or {}).get('message', 'Error')}), 500
    
    # Leer mapeo local actual
    conn = get_db_connection()
    local_packages = conn.execute('SELECT id, nombre, precio, gamepoint_package_id FROM precios_bloodstriker ORDER BY id').fetchall()
    conn.close()
    
    return jsonify({
        'gamepoint_packages': (detail_data or {}).get('package', []),
        'local_packages': [dict(lp) for lp in local_packages],
        'fields': (detail_data or {}).get('fields', []),
        'product_id': product_id,
    })


@app.route('/admin/gameclub/price_health')
def admin_gameclub_price_health():
    """Estadísticas de salud de precios: GP actual vs precio local esperado."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    try:
        from dynamic_games import get_gp_myr_rate as _get_gp_myr_rate
        myr_to_usd = float(_get_gp_myr_rate())
        usd_to_myr = round((1.0 / myr_to_usd), 6) if myr_to_usd > 0 else None

        gc_token, gc_err = _gameclub_get_token()
        if not gc_token:
            return jsonify({'error': (gc_err or {}).get('message', 'No se pudo obtener token de GamePoint')}), 500

        conn = get_db_connection()
        items = []
        api_errors = []

        def _process_game(game_key, game_name, product_id, local_rows, cost_map):
            try:
                _, detail_data = _gameclub_post('product/detail', {'token': gc_token, 'productid': int(product_id)})
                if (detail_data or {}).get('code') != 200:
                    api_errors.append({'game': game_name, 'error': (detail_data or {}).get('message', 'Error obteniendo detalle')})
                    return

                gp_map = {}
                for p in (detail_data or {}).get('package', []) or []:
                    try:
                        gp_map[int(p.get('id'))] = float(p.get('price', 0))
                    except Exception:
                        continue

                for lp in local_rows:
                    gp_id = lp.get('gamepoint_package_id')
                    if not gp_id:
                        continue
                    try:
                        gp_id_int = int(gp_id)
                    except Exception:
                        continue
                    if gp_id_int not in gp_map:
                        continue

                    gp_myr_now = float(gp_map[gp_id_int])
                    gp_usd_now = round(gp_myr_now * myr_to_usd, 4)
                    my_price_now = float(lp.get('precio') or 0)

                    cost_before = cost_map.get(int(lp['id']))
                    my_updated_now = None
                    diff = None
                    ok = None
                    if cost_before is not None:
                        my_price_before = round(float(cost_before), 4)
                        margin = my_price_now - my_price_before
                        my_updated_now = round(gp_usd_now + margin, 2)
                        diff = round(my_price_now - my_updated_now, 4)
                        ok = abs(diff) <= 0.01

                    items.append({
                        'row_key': f"{game_key}:{lp['id']}",
                        'game_key': game_key,
                        'game_name': game_name,
                        'local_id': int(lp['id']),
                        'local_name': lp.get('nombre'),
                        'gp_package_id': gp_id_int,
                        'gp_price_before_usd': round(float(cost_before), 4) if cost_before is not None else None,
                        'gp_price_now_usd': gp_usd_now,
                        'my_price_now': round(my_price_now, 2),
                        'my_price_updated_now': my_updated_now,
                        'diff': diff,
                        'ok': ok,
                    })
            except Exception as e:
                api_errors.append({'game': game_name, 'error': str(e)})

        # Blood Strike
        if pg_table_exists(conn, 'precios_bloodstriker') and pg_table_exists(conn, 'precios_compra'):
            bs_locals = conn.execute(
                'SELECT id, nombre, precio, gamepoint_package_id FROM precios_bloodstriker WHERE gamepoint_package_id IS NOT NULL'
            ).fetchall()
            bs_costs = {
                int(r['paquete_id']): float(r['precio_compra'])
                for r in conn.execute("SELECT paquete_id, precio_compra FROM precios_compra WHERE juego = 'bloodstriker'").fetchall()
            }
            bs_product_id = int(os.environ.get('BLOODSTRIKE_PRODUCT_ID', '155'))
            _process_game('bloodstriker', 'Blood Striker', bs_product_id, [dict(x) for x in bs_locals], bs_costs)
        else:
            api_errors.append({'game': 'Blood Striker', 'error': 'Tablas de Blood Striker no disponibles'})

        # Juegos dinámicos
        if pg_table_exists(conn, 'juegos_dinamicos') and pg_table_exists(conn, 'paquetes_dinamicos') and pg_table_exists(conn, 'precios_compra'):
            dyn_games_rows = conn.execute(
                'SELECT id, nombre, slug, gamepoint_product_id FROM juegos_dinamicos ORDER BY nombre'
            ).fetchall()
            for g in dyn_games_rows:
                g = dict(g)
                local_rows = conn.execute(
                    'SELECT id, nombre, precio, gamepoint_package_id FROM paquetes_dinamicos WHERE juego_id = ? AND gamepoint_package_id IS NOT NULL',
                    (g['id'],)
                ).fetchall()
                cost_map = {
                    int(r['paquete_id']): float(r['precio_compra'])
                    for r in conn.execute(
                        'SELECT paquete_id, precio_compra FROM precios_compra WHERE juego = ?',
                        (f"dyn_{g['slug']}",)
                    ).fetchall()
                }
                _process_game(f"dyn_{g['slug']}", g['nombre'], int(g['gamepoint_product_id']), [dict(x) for x in local_rows], cost_map)
        else:
            api_errors.append({'game': 'Juegos dinámicos', 'error': 'Tablas dinámicas no disponibles'})

        conn.close()

        comparable = [it for it in items if it.get('ok') is not None]
        ok_count = sum(1 for it in comparable if it.get('ok'))
        bad_count = sum(1 for it in comparable if it.get('ok') is False)
        avg_abs_diff = round((sum(abs(float(it.get('diff') or 0)) for it in comparable) / len(comparable)), 4) if comparable else 0.0

        return jsonify({
            'success': True,
            'usd_to_myr_rate': usd_to_myr,
            'myr_to_usd_rate': myr_to_usd,
            'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_rows': len(items),
                'comparable_rows': len(comparable),
                'ok_rows': ok_count,
                'out_of_sync_rows': bad_count,
                'avg_abs_diff': avg_abs_diff,
                'api_errors': len(api_errors),
            },
            'items': items,
            'errors': api_errors,
        })
    except Exception as e:
        logger.exception('[GameClub Price Health] Error no controlado')
        return jsonify({'error': f'No se pudo calcular estadísticas: {str(e)}'}), 500


@app.route('/admin/update_freefire_global_price', methods=['POST'])
def admin_update_freefire_global_price():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    package_id = request.form.get('package_id')
    new_price = request.form.get('new_price')
    
    if not package_id or not new_price:
        flash('Datos inválidos para actualizar precio', 'error')
        return redirect('/admin')
    
    try:
        new_price = float(new_price)
        if new_price < 0:
            flash('El precio no puede ser negativo', 'error')
            return redirect('/admin')
        
        # Obtener información del paquete antes de actualizar
        conn = get_db_connection()
        package = conn.execute('SELECT nombre FROM precios_freefire_global WHERE id = ?', (package_id,)).fetchone()
        conn.close()
        
        if not package:
            flash('Paquete no encontrado', 'error')
            return redirect('/admin')
        
        # Actualizar precio
        update_freefire_global_price(int(package_id), new_price)
        flash(f'Precio de Free Fire actualizado exitosamente para {package["nombre"]}: ${new_price:.2f}', 'success')
        
    except ValueError:
        flash('Precio inválido. Debe ser un número válido.', 'error')
    except Exception as e:
        flash(f'Error al actualizar precio: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/update_freefire_global_name', methods=['POST'])
def admin_update_freefire_global_name():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    package_id = request.form.get('package_id')
    new_name = request.form.get('new_name')
    
    if not package_id or not new_name:
        flash('Datos inválidos para actualizar nombre', 'error')
        return redirect('/admin')
    
    try:
        new_name = new_name.strip()
        if len(new_name) < 1:
            flash('El nombre no puede estar vacío', 'error')
            return redirect('/admin')
        
        if len(new_name) > 50:
            flash('El nombre no puede exceder 50 caracteres', 'error')
            return redirect('/admin')
        
        # Obtener información del paquete antes de actualizar
        conn = get_db_connection()
        package = conn.execute('SELECT nombre FROM precios_freefire_global WHERE id = ?', (package_id,)).fetchone()
        conn.close()
        
        if not package:
            flash('Paquete no encontrado', 'error')
            return redirect('/admin')
        
        old_name = package['nombre']
        
        # Actualizar nombre
        update_freefire_global_name(int(package_id), new_name)
        flash(f'Nombre de Free Fire actualizado exitosamente: "{old_name}" → "{new_name}"', 'success')
        
    except Exception as e:
        flash(f'Error al actualizar nombre: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/approve_bloodstriker/<int:transaction_id>', methods=['POST'])
def approve_bloodstriker_transaction(transaction_id):
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        # Obtener información de la transacción de Blood Striker
        conn = get_db_connection()
        bs_transaction = conn.execute('''
            SELECT bs.*, u.nombre, u.apellido, p.nombre as paquete_nombre, p.precio
            FROM transacciones_bloodstriker bs
            JOIN usuarios u ON bs.usuario_id = u.id
            JOIN precios_bloodstriker p ON bs.paquete_id = p.id
            WHERE bs.id = ?
        ''', (transaction_id,)).fetchone()
        
        if bs_transaction:
            # Obtener el ID del admin que está validando
            admin_user_id = session.get('user_db_id')
            
            # Crear transacción normal en el historial del admin (quien valida)
            conn.execute('''
                INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                admin_user_id,  # Usar ID del admin en lugar del usuario que compró
                bs_transaction['numero_control'],
                f"ID: {bs_transaction['player_id']} - Usuario: {bs_transaction['nombre']} {bs_transaction['apellido']}",
                bs_transaction['transaccion_id'],
                bs_transaction['paquete_nombre'],
                bs_transaction['monto']
            ))
            # Registrar en historial permanente
            _bs_precio = abs(bs_transaction['monto'])
            _bs_saldo_row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (bs_transaction['usuario_id'],)).fetchone()
            _bs_saldo = _bs_saldo_row['saldo'] if _bs_saldo_row else 0
            registrar_historial_compra(conn, bs_transaction['usuario_id'], _bs_precio, bs_transaction['paquete_nombre'], f"ID: {bs_transaction['player_id']}", 'compra', None, _bs_saldo + _bs_precio, _bs_saldo)
            
            # Persistir profit (legacy) para Blood Striker (cantidad=1)
            try:
                admin_ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
                admin_ids = [int(x.strip()) for x in admin_ids_env.split(',') if x.strip().isdigit()]
                is_admin_target = bs_transaction['usuario_id'] in admin_ids
                record_profit_for_transaction(conn, bs_transaction['usuario_id'], is_admin_target, 'bloodstriker', bs_transaction['paquete_id'], 1, bs_transaction['precio'], bs_transaction['transaccion_id'])
            except Exception:
                pass
            
            # Limitar transacciones a 100 por usuario (aumentado de 30 para evitar eliminaciones frecuentes)
            conn.execute('''
                DELETE FROM transacciones 
                WHERE usuario_id = ? AND id NOT IN (
                    SELECT id FROM (SELECT id FROM transacciones 
                    WHERE usuario_id = ? 
                    ORDER BY fecha DESC 
                    LIMIT 100) AS keep_ids
                )
            ''', (bs_transaction['usuario_id'], bs_transaction['usuario_id']))
            
            conn.commit()
            
            # Registrar venta en estadísticas semanales (solo para usuarios normales)
            admin_ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
            admin_ids = [int(x.strip()) for x in admin_ids_env.split(',') if x.strip().isdigit()]
            is_admin_user = bs_transaction['usuario_id'] in admin_ids
            
            if not is_admin_user:
                register_weekly_sale(
                    'bloodstriker', 
                    bs_transaction['paquete_id'], 
                    bs_transaction['paquete_nombre'], 
                    bs_transaction['precio'], 
                    1
                )
            
            # Crear notificación personalizada para el usuario
            titulo = "🎯 Blood Striker - Recarga realizada con éxito"
            mensaje = f"Blood Striker: Recarga realizada con éxito. {bs_transaction['paquete_nombre']} por ${bs_transaction['precio']:.2f}. ID: {bs_transaction['player_id']}"
            conn.execute('''
                CREATE TABLE IF NOT EXISTS notificaciones_personalizadas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario_id INTEGER,
                    titulo TEXT NOT NULL,
                    mensaje TEXT NOT NULL,
                    tipo TEXT DEFAULT 'info',
                    tag TEXT,
                    visto BOOLEAN DEFAULT FALSE,
                    fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
                )
            ''')
            try:
                conn.execute("ALTER TABLE notificaciones_personalizadas ADD COLUMN tag TEXT")
            except Exception:
                pass
            try:
                conn.execute('''
                    INSERT INTO notificaciones_personalizadas (usuario_id, titulo, mensaje, tipo, tag)
                    VALUES (?, ?, ?, ?, ?)
                ''', (bs_transaction['usuario_id'], titulo, mensaje, 'success', 'bloodstriker_reload'))
                conn.commit()
            except Exception:
                conn.execute('''
                    INSERT INTO notificaciones_personalizadas (usuario_id, titulo, mensaje, tipo)
                    VALUES (?, ?, ?, ?)
                ''', (bs_transaction['usuario_id'], titulo, mensaje, 'success'))
                conn.commit()
        
        conn.close()
        
        # Actualizar estado de la transacción de Blood Striker
        update_bloodstriker_transaction_status(transaction_id, 'aprobado', session.get('user_db_id'))
        flash('Transacción aprobada exitosamente', 'success')
    except Exception as e:
        flash(f'Error al aprobar transacción: {str(e)}', 'error')
    
    return redirect('/')

@app.route('/admin/reject_bloodstriker/<int:transaction_id>', methods=['POST'])
def reject_bloodstriker_transaction(transaction_id):
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        # Obtener información de la transacción para devolver el saldo
        conn = get_db_connection()
        transaction = conn.execute('''
            SELECT usuario_id, monto FROM transacciones_bloodstriker 
            WHERE id = ?
        ''', (transaction_id,)).fetchone()
        
        if transaction:
            # Devolver saldo al usuario (monto es negativo, así que sumamos el valor absoluto)
            conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', 
                        (abs(transaction['monto']), transaction['usuario_id']))
            conn.commit()
        conn.close()
        
        # Actualizar estado de la transacción
        update_bloodstriker_transaction_status(transaction_id, 'rechazado', session.get('user_db_id'))
        flash('Transacción rechazada y saldo devuelto al usuario', 'success')
    except Exception as e:
        flash(f'Error al rechazar transacción: {str(e)}', 'error')

@app.route('/juego/freefire_id')
def freefire_id():
    if 'usuario' not in session:
        return redirect('/auth')

    is_admin = session.get('is_admin', False)
    if not is_admin:
        ga = get_games_active()
        if not ga.get('freefire_id', False):
            flash('Este juego está desactivado temporalmente.', 'error')
            return redirect('/')
    
    user_id = session.get('user_db_id')
    if user_id:
        conn = get_db_connection()
        user = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            session['saldo'] = user['saldo']
        conn.close()
    
    if is_admin:
        prices = {}
        for row in get_all_freefire_id_prices():
            try:
                prices[int(row['id'])] = {
                    'nombre': row['nombre'],
                    'precio': row['precio'],
                    'descripcion': row.get('descripcion') if hasattr(row, 'get') else row['descripcion']
                }
            except Exception:
                continue
    else:
        prices = get_freefire_id_prices()
    
    compra_exitosa = False
    compra_data = {}
    
    if 'compra_freefire_id_exitosa' in session:
        compra_exitosa = True
        compra_data = session.pop('compra_freefire_id_exitosa')
    
    # Nonce de un solo uso para evitar doble submit/reintentos del navegador
    session['ffid_form_nonce'] = secrets.token_urlsafe(16)

    return render_template('freefire_id.html', 
                         user_id=session.get('id', '00000'),
                         balance=session.get('saldo', 0),
                         prices=prices,
                         compra_exitosa=compra_exitosa,
                         is_admin=is_admin,
                         games_active=get_games_active(),
                         ffid_form_nonce=session.get('ffid_form_nonce'),
                         **compra_data)

@app.route('/validar/freefire_id', methods=['POST'])
def validar_freefire_id():
    if 'usuario' not in session:
        return redirect('/auth')

    is_admin = session.get('is_admin', False)
    if not is_admin:
        ga = get_games_active()
        if not ga.get('freefire_id', False):
            flash('Este juego está desactivado temporalmente.', 'error')
            return redirect('/')
    
    package_id = request.form.get('monto')
    player_id = request.form.get('player_id')

    # Validar nonce (un solo uso) para prevenir reintentos/doble submit en caídas
    # Excepción: solicitudes API automáticas (Inefablestore) usan WEBB_API_KEY en su lugar
    _webb_api_key_env = os.environ.get('WEBB_API_KEY', '').strip()
    _webb_api_key_req = request.form.get('webb_api_key', '').strip()
    _is_api_call = bool(_webb_api_key_env and _webb_api_key_req and _webb_api_key_env == _webb_api_key_req)

    if not _is_api_call:
        nonce_form = request.form.get('ffid_form_nonce')
        nonce_session = session.pop('ffid_form_nonce', None)
        if not nonce_form or not nonce_session or nonce_form != nonce_session:
            flash('Solicitud duplicada o expirada. Recarga la página e intenta nuevamente.', 'error')
            return redirect('/juego/freefire_id')
    else:
        session.pop('ffid_form_nonce', None)
    
    if not package_id or not player_id:
        flash('Por favor complete todos los campos', 'error')
        return redirect('/juego/freefire_id')
    
    package_id = int(package_id)
    user_id = session.get('user_db_id')
    is_admin = session.get('is_admin', False)
    
    precio = get_freefire_id_price_by_id(package_id)
    
    packages_info = get_freefire_id_prices_cached()
    package_info = packages_info.get(package_id, {})
    
    paquete_nombre = f"{package_info.get('nombre', 'Paquete')} / ${precio:.2f}"
    
    if precio == 0:
        flash('Paquete no encontrado o inactivo', 'error')
        return redirect('/juego/freefire_id')

    # Evitar doble envío: solo bloquear si hay una transacción PENDIENTE reciente
    # (no bloquear si la anterior ya fue completada o rechazada)
    try:
        conn_dup = get_db_connection()
        dup = conn_dup.execute(
            '''
            SELECT id, estado, fecha
            FROM transacciones_freefire_id
            WHERE usuario_id = ?
              AND player_id = ?
              AND paquete_id = ?
              AND estado = 'pendiente'
              AND datetime(fecha) >= datetime('now', '-2 minutes')
            ORDER BY id DESC
            LIMIT 1
            ''',
            (user_id, player_id, package_id)
        ).fetchone()
        conn_dup.close()
        if dup:
            flash('Ya se está procesando tu recarga. Espera unos segundos y revisa tu dashboard.', 'error')
            return redirect('/juego/freefire_id')
    except Exception:
        pass
    
    # === PROTECCIÓN 1: Verificar saldo desde DB (no session) ===
    if not is_admin:
        conn_check = get_db_connection()
        row = conn_check.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        conn_check.close()
        saldo_actual = row['saldo'] if row else 0
        session['saldo'] = saldo_actual  # sincronizar session
    else:
        saldo_actual = session.get('saldo', 0)
    
    if not is_admin and saldo_actual < precio:
        flash(f'Saldo insuficiente. Necesitas ${precio:.2f} pero tienes ${saldo_actual:.2f}', 'error')
        return redirect('/juego/freefire_id')
    
    transaction_data = None
    pin_codigo = None
    saldo_cobrado = False
    redencion_exitosa = False

    try:
        # 1. Verificar si hay PIN disponible en stock de FF Global ANTES de cobrar
        pin_disponible = get_available_pin_freefire_global(package_id)
        if not pin_disponible:
            flash(f'No hay stock disponible para este paquete en este momento. Intenta más tarde.', 'error')
            return redirect('/juego/freefire_id')
        
        pin_codigo = pin_disponible['pin_codigo']
        
        # 2. Cobrar al usuario (atómico: solo si saldo >= precio)
        if not is_admin:
            conn = get_db_connection()
            cursor = conn.execute(
                'UPDATE usuarios SET saldo = saldo - ? WHERE id = ? AND saldo >= ?',
                (precio, user_id, precio))
            if cursor.rowcount == 0:
                conn.close()
                # Devolver PIN al stock
                try:
                    conn2 = get_db_connection()
                    conn2.execute('INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado) VALUES (?, ?, FALSE)', (package_id, pin_codigo))
                    conn2.commit()
                    conn2.close()
                except Exception:
                    pass
                flash(f'Saldo insuficiente al momento de procesar. Recarga tu saldo e intenta de nuevo.', 'error')
                return redirect('/juego/freefire_id')
            conn.commit()
            saldo_cobrado = True
            # Leer saldo actualizado desde DB
            new_saldo = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
            conn.close()
            session['saldo'] = new_saldo['saldo'] if new_saldo else 0
        
        # 3. Crear transacción (con PIN incluido directamente)
        transaction_data = create_freefire_id_transaction(user_id, player_id, package_id, precio, pin_codigo=pin_codigo)
        
        # 4. Actualizar gastos mensuales
        if not is_admin:
            try:
                conn = get_db_connection()
                update_monthly_spending(conn, user_id, precio)
                conn.commit()
                conn.close()
            except Exception:
                pass
        
        # 5. Ejecutar redención automática (medir duración)
        import time as _time
        redeemer_config = get_redeemer_config_from_db(get_db_connection)
        
        redeem_result = None
        _redeem_start = _time.time()
        try:
            redeem_result = redeem_pin_vps(pin_codigo, player_id, redeemer_config, request_id=transaction_data['transaccion_id'])
        except Exception as e:
            logger.error(f"[FreeFire ID] Error en redencion automatica (VPS): {str(e)}")
            redeem_result = None
        _redeem_duration = round(_time.time() - _redeem_start, 1)
        
        # 6. Evaluar resultado
        if redeem_result and redeem_result.success:
            redencion_exitosa = True
            # === ÉXITO: Recarga completada ===
            player_name = redeem_result.player_name or ''
            
            # Actualizar estado de transacción a aprobado
            # register_general_tx=False para evitar duplicado: este flujo inserta abajo el registro completo
            update_freefire_id_transaction_status(transaction_data['id'], 'aprobado', user_id, register_general_tx=False)
            
            # Registrar en transacciones generales (con duración de redención)
            conn = get_db_connection()
            pin_info = f"ID: {player_id} - Jugador: {player_name}"
            conn.execute('''
                INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto, duracion_segundos)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, transaction_data['numero_control'], pin_info, 
                  transaction_data['transaccion_id'], package_info.get('nombre', 'FF ID'), -precio, _redeem_duration))
            
            # Registrar en historial permanente
            _ff_saldo_row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
            _ff_saldo = _ff_saldo_row['saldo'] if _ff_saldo_row else 0
            registrar_historial_compra(conn, user_id, precio, package_info.get('nombre', 'FF ID'), pin_info, 'compra', _redeem_duration, _ff_saldo + precio, _ff_saldo)
            
            # Registrar profit
            try:
                admin_ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
                admin_ids = [int(x.strip()) for x in admin_ids_env.split(',') if x.strip().isdigit()]
                is_admin_target = user_id in admin_ids
                record_profit_for_transaction(conn, user_id, is_admin_target, 'freefire_id', package_id, 1, precio, transaction_data['transaccion_id'])
            except Exception:
                pass
            
            conn.commit()
            
            # Registrar venta semanal
            if not is_admin:
                register_weekly_sale('freefire_id', package_id, package_info.get('nombre', 'FF ID'), precio, 1)
            
            session['compra_freefire_id_exitosa'] = {
                'paquete_nombre': paquete_nombre,
                'monto_compra': precio,
                'numero_control': transaction_data['numero_control'],
                'transaccion_id': transaction_data['transaccion_id'],
                'player_id': player_id,
                'player_name': player_name,
                'estado': 'completado'
            }
            
            return redirect('/juego/freefire_id?compra=exitosa')
        
        else:
            # === FALLO: Devolver PIN al stock y reembolsar saldo ===
            error_msg = redeem_result.message if redeem_result else 'Error desconocido en la redención'
            logger.error(
                f"[FreeFire ID] Redencion fallida: {error_msg} | "
                f"usuario_id={user_id} player_id={player_id} package_id={package_id} precio={precio} | "
                f"pin={pin_codigo} numero_control={transaction_data.get('numero_control')} transaccion_id={transaction_data.get('transaccion_id')}"
            )
            
            # Devolver PIN al inventario
            try:
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado)
                    VALUES (?, ?, FALSE)
                ''', (package_id, pin_codigo))
                conn.commit()
                conn.close()
                logger.info(f"[FreeFire ID] PIN {pin_codigo[:8]}... devuelto al stock")
            except Exception as e:
                logger.error(f"[FreeFire ID] Error devolviendo PIN al stock: {str(e)}")
            
            # Reembolsar saldo al usuario
            if not is_admin:
                conn = get_db_connection()
                conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, user_id))
                conn.commit()
                conn.close()
                saldo_cobrado = False
                session['saldo'] = session.get('saldo', 0) + precio
                logger.info(f"[FreeFire ID] Saldo ${precio} reembolsado al usuario {user_id}")
            
            # Actualizar transacción como rechazada
            update_freefire_id_transaction_status(transaction_data['id'], 'rechazado', user_id, f'Auto-redención fallida: {error_msg}')
            
            flash(f'La recarga falló. Tu saldo ha sido devuelto. Puedes intentar nuevamente con otro PIN. Error: {error_msg}', 'error')
            return redirect('/juego/freefire_id')
        
    except Exception as e:
        logger.error(f"[FreeFire ID] Error general: {str(e)}")

        # Fallback anti-transacciones atascadas en "pendiente"
        # Si ya se creó la transacción y sigue pendiente, cerrarla como rechazada
        # y compensar saldo/pin cuando aplique.
        if transaction_data and transaction_data.get('id'):
            try:
                conn_fix = get_db_connection()
                row_fix = conn_fix.execute(
                    'SELECT estado FROM transacciones_freefire_id WHERE id = ? LIMIT 1',
                    (transaction_data['id'],)
                ).fetchone()
                estado_actual = row_fix['estado'] if row_fix else None

                if estado_actual == 'pendiente':
                    if redencion_exitosa:
                        # Si el PIN ya se redimió exitosamente, NO devolver PIN ni saldo.
                        # Cerrar como aprobado para evitar pérdida por doble compensación.
                        conn_fix.commit()
                        update_freefire_id_transaction_status(
                            transaction_data['id'],
                            'aprobado',
                            user_id,
                            f'Auto-aprobado: redención exitosa con error posterior ({str(e)[:180]})'
                        )
                    else:
                        if pin_codigo:
                            try:
                                conn_fix.execute(
                                    '''
                                    INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado)
                                    VALUES (?, ?, FALSE)
                                    ''',
                                    (package_id, pin_codigo)
                                )
                            except Exception:
                                pass

                        if saldo_cobrado and not is_admin:
                            conn_fix.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', (precio, user_id))
                            try:
                                saldo_row = conn_fix.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
                                session['saldo'] = saldo_row['saldo'] if saldo_row else session.get('saldo', 0)
                            except Exception:
                                pass

                        conn_fix.commit()
                        update_freefire_id_transaction_status(
                            transaction_data['id'],
                            'rechazado',
                            user_id,
                            f'Auto-rechazo por excepción: {str(e)[:200]}'
                        )
            except Exception as fix_err:
                logger.error(f"[FreeFire ID] Error en fallback anti-pendiente: {str(fix_err)}")
            finally:
                try:
                    conn_fix.close()
                except Exception:
                    pass

        flash('Error al procesar la compra. Intente nuevamente.', 'error')
        return redirect('/juego/freefire_id')

# Rutas de administrador para Free Fire ID
@app.route('/admin/freefire_id_transactions')
def admin_freefire_id_transactions():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    pending_transactions = get_pending_freefire_id_transactions()
    return render_template('admin_freefire_id.html', transactions=pending_transactions)

@app.route('/admin/freefire_id_approve', methods=['POST'])
def admin_freefire_id_approve():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    transaction_id = request.form.get('transaction_id')
    notas = request.form.get('notas', '')
    
    if transaction_id:
        update_freefire_id_transaction_status(int(transaction_id), 'aprobado', session.get('user_db_id'), notas)
        flash('Transacción aprobada exitosamente', 'success')
    else:
        flash('ID de transacción inválido', 'error')
    
    return redirect('/admin/freefire_id_transactions')

@app.route('/admin/freefire_id_reject', methods=['POST'])
def admin_freefire_id_reject():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    transaction_id = request.form.get('transaction_id')
    notas = request.form.get('notas', '')
    
    if transaction_id:
        conn = get_db_connection()
        transaction = conn.execute('''
            SELECT usuario_id, monto FROM transacciones_freefire_id 
            WHERE id = ?
        ''', (transaction_id,)).fetchone()
        
        if transaction:
            conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', 
                        (abs(transaction['monto']), transaction['usuario_id']))
            conn.commit()
        conn.close()
        
        update_freefire_id_transaction_status(int(transaction_id), 'rechazado', session.get('user_db_id'), notas)
        flash('Transacción rechazada y saldo devuelto al usuario', 'success')
    else:
        flash('ID de transacción inválido', 'error')
    
    return redirect('/admin/freefire_id_transactions')

@app.route('/admin/update_freefire_id_price', methods=['POST'])
def admin_update_freefire_id_price():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    package_id = request.form.get('package_id')
    new_price = request.form.get('new_price')
    
    if not package_id or not new_price:
        flash('Datos inválidos para actualizar precio', 'error')
        return redirect('/admin')
    
    try:
        new_price = float(new_price)
        if new_price < 0:
            flash('El precio no puede ser negativo', 'error')
            return redirect('/admin')
        
        conn = get_db_connection()
        package = conn.execute('SELECT nombre FROM precios_freefire_id WHERE id = ?', (package_id,)).fetchone()
        conn.close()
        
        if not package:
            flash('Paquete no encontrado', 'error')
            return redirect('/admin')
        
        update_freefire_id_price(int(package_id), new_price)
        flash(f'Precio de Free Fire ID actualizado exitosamente para {package["nombre"]}: ${new_price:.2f}', 'success')
        
    except ValueError:
        flash('Precio inválido. Debe ser un número válido.', 'error')
    except Exception as e:
        flash(f'Error al actualizar precio: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/update_freefire_id_name', methods=['POST'])
def admin_update_freefire_id_name():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    package_id = request.form.get('package_id')
    new_name = request.form.get('new_name')
    
    if not package_id or not new_name:
        flash('Datos inválidos para actualizar nombre', 'error')
        return redirect('/admin')
    
    try:
        new_name = new_name.strip()
        if len(new_name) < 1:
            flash('El nombre no puede estar vacío', 'error')
            return redirect('/admin')
        
        if len(new_name) > 50:
            flash('El nombre no puede exceder 50 caracteres', 'error')
            return redirect('/admin')
        
        conn = get_db_connection()
        package = conn.execute('SELECT nombre FROM precios_freefire_id WHERE id = ?', (package_id,)).fetchone()
        conn.close()
        
        if not package:
            flash('Paquete no encontrado', 'error')
            return redirect('/admin')
        
        old_name = package['nombre']
        
        update_freefire_id_name(int(package_id), new_name)
        flash(f'Nombre de Free Fire ID actualizado exitosamente: "{old_name}" → "{new_name}"', 'success')
        
    except Exception as e:
        flash(f'Error al actualizar nombre: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/freefire_id_pin_log')
def admin_freefire_id_pin_log():
    """Vista admin para ver PINes gastados recientemente en recargas FreeFire ID"""
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    conn = get_db_connection()
    transactions = conn.execute('''
        SELECT fi.id, fi.usuario_id, fi.player_id, fi.pin_codigo, fi.paquete_id,
               fi.numero_control, fi.transaccion_id, fi.monto, fi.estado, fi.fecha,
               fi.fecha_procesado, fi.notas,
               u.nombre || ' ' || u.apellido as usuario_nombre, u.correo,
               p.nombre as paquete_nombre
        FROM transacciones_freefire_id fi
        JOIN usuarios u ON fi.usuario_id = u.id
        LEFT JOIN precios_freefire_id p ON fi.paquete_id = p.id
        ORDER BY fi.fecha DESC
        LIMIT 100
    ''').fetchall()
    conn.close()
    
    return render_template_string(PIN_LOG_TEMPLATE, transactions=transactions)

@app.route('/admin/freefire_id_audit')
def admin_freefire_id_audit():
    """Vista admin para auditar transacciones inconsistentes de Free Fire ID"""
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    inconsistent_transactions = audit_freefire_id_inconsistent_transactions()
    
    return render_template_string(AUDIT_TEMPLATE, transactions=inconsistent_transactions)

AUDIT_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auditoría FreeFire ID - Transacciones Inconsistentes</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a1a; color: #e0e0e0; padding: 20px; }
        h1 { color: #ff6b6b; margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 20px; font-size: 14px; }
        .back-link { color: #00ff88; text-decoration: none; margin-bottom: 20px; display: inline-block; }
        .back-link:hover { text-decoration: underline; }
        .alert { 
            background: #ff6b6b20; 
            border: 1px solid #ff6b6b; 
            border-radius: 8px; 
            padding: 15px; 
            margin-bottom: 20px; 
        }
        .alert p { margin: 5px 0; font-size: 14px; }
        .alert strong { color: #ff6b6b; }
        table { width: 100%; border-collapse: collapse; background: #1a1a2e; border-radius: 8px; overflow: hidden; }
        th { background: #16213e; color: #00ff88; padding: 10px 8px; text-align: left; font-size: 12px; position: sticky; top: 0; }
        td { padding: 8px; border-bottom: 1px solid #222; font-size: 13px; }
        tr:hover { background: #16213e; }
        .severity-HIGH { border-left: 4px solid #ff4444; }
        .severity-MEDIUM { border-left: 4px solid #ffaa00; }
        .pin-code { font-family: monospace; font-size: 12px; color: #ffdd57; cursor: pointer; }
        .status-refunded { color: #ff4444; font-weight: bold; }
        .status-ok { color: #00ff88; font-weight: bold; }
        .btn { 
            background: #ff6b6b; 
            color: white; 
            border: none; 
            padding: 4px 8px; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 11px; 
            margin: 0 2px;
        }
        .btn:hover { background: #ff5252; }
        .btn-success { background: #00ff88; }
        .btn-success:hover { background: #00dd77; }
        .empty-state { text-align: center; padding: 40px; color: #888; }
    </style>
</head>
<body>
    <a href="/admin" class="back-link">← Volver al Admin</a>
    <h1>🔍 Auditoría FreeFire ID</h1>
    <p class="subtitle">Transacciones inconsistentes (posible doble deducción o reembolso incorrecto)</p>
    
    <div class="alert">
        <p><strong>⚠️ ADVERTENCIA:</strong> Estas transacciones pueden indicar que un cliente recibió diamantes pero también le devolvieron el saldo.</p>
        <p><strong>Acción recomendada:</strong> Verificar manualmente con el cliente y corregir si es necesario.</p>
    </div>
    
    {% if transactions %}
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Usuario</th>
                <th>Player ID</th>
                <th>PIN</th>
                <th>Monto</th>
                <th>Fecha</th>
                <th>Estado</th>
                <th>Notas</th>
                <th>Acciones</th>
            </tr>
        </thead>
        <tbody>
            {% for trans in transactions %}
            <tr class="severity-{{ trans.severity }}">
                <td>{{ trans.id }}</td>
                <td>Usuario {{ trans.usuario_id }}</td>
                <td>{{ trans.player_id }}</td>
                <td><span class="pin-code">{{ trans.pin_codigo[:12] }}...</span></td>
                <td>${{ "%.2f"|format(trans.monto) }}</td>
                <td>{{ trans.fecha|format_date }}</td>
                <td>
                    {% if trans.was_refunded %}
                    <span class="status-refunded">REEMBOLSADO</span>
                    {% else %}
                    <span class="status-ok">OK</span>
                    {% endif %}
                </td>
                <td>{{ trans.notas[:50] }}{% if trans.notas|length > 50 %}...{% endif %}</td>
                <td>
                    <button class="btn btn-success" onclick="fixTransaction({{ trans.id }})">Corregir</button>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <div class="empty-state">
        <p>✅ No se encontraron transacciones inconsistentes en las últimas 24 horas.</p>
    </div>
    {% endif %}
    
    <script>
        function fixTransaction(transactionId) {
            if (confirm('¿Corregir esta transacción? Esto la marcará como aprobada y registrará la transacción general.')) {
                fetch('/admin/fix_freefire_id_transaction', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({transaction_id: transactionId})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Transacción corregida exitosamente');
                        location.reload();
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Error de conexión: ' + error);
                });
            }
        }
    </script>
</body>
</html>
'''

PIN_LOG_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log de PINes FreeFire ID</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a1a; color: #e0e0e0; padding: 20px; }
        h1 { color: #00ff88; margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 20px; font-size: 14px; }
        .back-link { color: #00ff88; text-decoration: none; margin-bottom: 20px; display: inline-block; }
        .back-link:hover { text-decoration: underline; }
        .stats { display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
        .stat-box { background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 12px 20px; }
        .stat-box .num { font-size: 24px; font-weight: bold; }
        .stat-box .label { font-size: 12px; color: #888; }
        .stat-box.success .num { color: #00ff88; }
        .stat-box.fail .num { color: #ff4444; }
        .stat-box.pending .num { color: #ffaa00; }
        table { width: 100%; border-collapse: collapse; background: #1a1a2e; border-radius: 8px; overflow: hidden; }
        th { background: #16213e; color: #00ff88; padding: 10px 8px; text-align: left; font-size: 12px; position: sticky; top: 0; }
        td { padding: 8px; border-bottom: 1px solid #222; font-size: 13px; }
        tr:hover { background: #16213e; }
        .pin-code { font-family: monospace; font-size: 12px; color: #ffdd57; cursor: pointer; }
        .pin-code:hover { color: #fff; }
        .estado-aprobado { color: #00ff88; font-weight: bold; }
        .estado-rechazado { color: #ff4444; font-weight: bold; }
        .estado-pendiente { color: #ffaa00; font-weight: bold; }
        .player-id { font-family: monospace; color: #88ccff; }
        .search-box { margin-bottom: 15px; }
        .search-box input { background: #1a1a2e; border: 1px solid #333; color: #fff; padding: 8px 15px;
                           border-radius: 6px; width: 300px; font-size: 14px; }
        .search-box input:focus { outline: none; border-color: #00ff88; }
        .notas { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; color: #999; }
        .copy-toast { position: fixed; bottom: 20px; right: 20px; background: #00ff88; color: #000;
                     padding: 10px 20px; border-radius: 6px; display: none; font-weight: bold; z-index: 999; }
    </style>
</head>
<body>
    <a href="/admin" class="back-link">&#8592; Volver al Panel Admin</a>
    <h1>Log de PINes FreeFire ID</h1>
    <p class="subtitle">Ultimas 100 transacciones con PIN completo, player ID y estado</p>

    <div class="stats">
        <div class="stat-box success">
            <div class="num">{{ transactions|selectattr('estado', 'equalto', 'aprobado')|list|length }}</div>
            <div class="label">Aprobadas</div>
        </div>
        <div class="stat-box fail">
            <div class="num">{{ transactions|selectattr('estado', 'equalto', 'rechazado')|list|length }}</div>
            <div class="label">Rechazadas</div>
        </div>
        <div class="stat-box pending">
            <div class="num">{{ transactions|selectattr('estado', 'equalto', 'pendiente')|list|length }}</div>
            <div class="label">Pendientes</div>
        </div>
    </div>

    <div class="search-box">
        <input type="text" id="searchInput" placeholder="Buscar por PIN, Player ID, usuario..." onkeyup="filterTable()">
    </div>

    <table id="pinTable">
        <thead>
            <tr>
                <th>#</th>
                <th>Fecha</th>
                <th>Usuario</th>
                <th>Player ID</th>
                <th>PIN Completo</th>
                <th>Paquete</th>
                <th>Monto</th>
                <th>Estado</th>
                <th>Notas</th>
            </tr>
        </thead>
        <tbody>
            {% for t in transactions %}
            <tr>
                <td>{{ t.id }}</td>
                <td>{{ t.fecha|format_date('%Y-%m-%d %H:%M') if t.fecha else '-' }}</td>
                <td>{{ t.usuario_nombre }}<br><small style="color:#666">{{ t.correo }}</small></td>
                <td class="player-id">{{ t.player_id }}</td>
                <td class="pin-code" onclick="copyPin(this)" title="Click para copiar">{{ t.pin_codigo or 'N/A' }}</td>
                <td>{{ t.paquete_nombre or '-' }}</td>
                <td>${{ '%.2f'|format(t.monto|abs) }}</td>
                <td class="estado-{{ t.estado }}">{{ t.estado|upper }}</td>
                <td class="notas" title="{{ t.notas or '' }}">{{ t.notas or '-' }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <div class="copy-toast" id="copyToast">PIN copiado al portapapeles</div>

    <script>
        function copyPin(el) {
            var pin = el.textContent.trim();
            if (pin && pin !== 'N/A') {
                navigator.clipboard.writeText(pin);
                var toast = document.getElementById('copyToast');
                toast.style.display = 'block';
                setTimeout(function() { toast.style.display = 'none'; }, 2000);
            }
        }
        function filterTable() {
            var q = document.getElementById('searchInput').value.toLowerCase();
            var rows = document.querySelectorAll('#pinTable tbody tr');
            rows.forEach(function(row) {
                row.style.display = row.textContent.toLowerCase().indexOf(q) >= 0 ? '' : 'none';
            });
        }
    </script>
</body>
</html>
'''

@app.route('/admin/approve_freefire_id/<int:transaction_id>', methods=['POST'])
def approve_freefire_id_transaction(transaction_id):
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        conn = get_db_connection()
        fi_transaction = conn.execute('''
            SELECT fi.*, u.nombre, u.apellido, p.nombre as paquete_nombre, p.precio
            FROM transacciones_freefire_id fi
            JOIN usuarios u ON fi.usuario_id = u.id
            JOIN precios_freefire_id p ON fi.paquete_id = p.id
            WHERE fi.id = ?
        ''', (transaction_id,)).fetchone()
        
        if not fi_transaction:
            conn.close()
            flash('Transaccion no encontrada', 'error')
            return redirect('/')
        
        # Verificar si auto_redeem está habilitado
        redeemer_config = get_redeemer_config_from_db(get_db_connection)
        auto_redeem = redeemer_config.get('auto_redeem', 'false').lower() in ('true', '1', 'yes')
        
        redeem_result = None
        pin_usado = None
        
        if auto_redeem:
            # === REDENCIÓN AUTOMÁTICA ===
            # 1. Obtener un pin de Free Fire Global del mismo paquete
            paquete_id = fi_transaction['paquete_id']
            pin_disponible = get_available_pin_freefire_global(paquete_id)
            
            if not pin_disponible:
                conn.close()
                flash(f'No hay pines de Free Fire Global disponibles para el paquete {fi_transaction["paquete_nombre"]}. Agrega pines al inventario primero.', 'error')
                return redirect('/')
            
            pin_codigo = pin_disponible['pin_codigo']
            player_id = fi_transaction['player_id']
            
            # 2. Ejecutar la redención automática en redeempins.com
            try:
                redeem_result = redeem_pin_vps(pin_codigo, player_id, redeemer_config, request_id=fi_transaction.get('transaccion_id'))
            except Exception as e:
                # Si falla la redención, devolver el pin al inventario
                try:
                    conn2 = get_db_connection()
                    conn2.execute('''
                        INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado)
                        VALUES (?, ?, FALSE)
                    ''', (paquete_id, pin_codigo))
                    conn2.commit()
                    conn2.close()
                except:
                    pass
                conn.close()
                flash(f'Error al redimir pin automaticamente: {str(e)}. Pin devuelto al inventario.', 'error')
                return redirect('/')
            
            if not redeem_result.success:
                # Si la redención falló, devolver el pin al inventario y NO aprobar la transacción
                try:
                    conn2 = get_db_connection()
                    conn2.execute('''
                        INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado)
                        VALUES (?, ?, FALSE)
                    ''', (paquete_id, pin_codigo))
                    conn2.commit()
                    conn2.close()
                    logger.warning(f"[FreeFire ID] Reintento fallido - PIN {pin_codigo[:8]}... devuelto al stock, transacción {transaction_id} mantiene estado pendiente")
                except:
                    pass
                conn.close()
                flash(f'Redención automática fallida: {redeem_result.message}. PIN devuelto al inventario. La transacción permanece pendiente para revisión manual.', 'error')
                return redirect('/')
            
            pin_usado = pin_codigo
        
        # === APROBAR TRANSACCIÓN ===
        admin_user_id = session.get('user_db_id')
        
        pin_info = f"ID: {fi_transaction['player_id']} - Usuario: {fi_transaction['nombre']} {fi_transaction['apellido']}"
        if pin_usado:
            pin_info += f" - PIN: {pin_usado[:8]}..."
        if redeem_result and redeem_result.success:
            pin_info += " [AUTO-REDIMIDO]"
        
        conn.execute('''
            INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            admin_user_id,
            fi_transaction['numero_control'],
            pin_info,
            fi_transaction['transaccion_id'],
            fi_transaction['paquete_nombre'],
            fi_transaction['monto']
        ))
        # Registrar en historial permanente
        _fi_precio = abs(fi_transaction['monto'])
        _fi_saldo_row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (fi_transaction['usuario_id'],)).fetchone()
        _fi_saldo = _fi_saldo_row['saldo'] if _fi_saldo_row else 0
        registrar_historial_compra(conn, fi_transaction['usuario_id'], _fi_precio, fi_transaction['paquete_nombre'], f"ID: {fi_transaction['player_id']}", 'compra', None, _fi_saldo + _fi_precio, _fi_saldo)
        
        # Persistir profit (legacy)
        try:
            admin_ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
            admin_ids = [int(x.strip()) for x in admin_ids_env.split(',') if x.strip().isdigit()]
            is_admin_target = fi_transaction['usuario_id'] in admin_ids
            record_profit_for_transaction(conn, fi_transaction['usuario_id'], is_admin_target, 'freefire_id', fi_transaction['paquete_id'], 1, fi_transaction['precio'], fi_transaction['transaccion_id'])
        except Exception:
            pass
        
        conn.execute('''
            DELETE FROM transacciones 
            WHERE usuario_id = ? AND id NOT IN (
                SELECT id FROM (SELECT id FROM transacciones 
                WHERE usuario_id = ? 
                ORDER BY fecha DESC 
                LIMIT 100) AS keep_ids
            )
        ''', (fi_transaction['usuario_id'], fi_transaction['usuario_id']))
        
        conn.commit()
        
        # Registrar venta en estadísticas semanales (solo para usuarios normales)
        admin_ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
        admin_ids = [int(x.strip()) for x in admin_ids_env.split(',') if x.strip().isdigit()]
        is_admin_user = fi_transaction['usuario_id'] in admin_ids
        
        if not is_admin_user:
            register_weekly_sale(
                'freefire_id', 
                fi_transaction['paquete_id'], 
                fi_transaction['paquete_nombre'], 
                fi_transaction['precio'], 
                1
            )
        
        # Crear notificación personalizada para el usuario
        titulo = "Free Fire ID - Recarga realizada"
        mensaje = f"Free Fire ID: Recarga realizada con exito. {fi_transaction['paquete_nombre']} por ${fi_transaction['precio']:.2f}. ID: {fi_transaction['player_id']}"
        if redeem_result and redeem_result.success:
            mensaje += " (Automatica)"
        try:
            conn.execute('''
                INSERT INTO notificaciones_personalizadas (usuario_id, titulo, mensaje, tipo, tag)
                VALUES (?, ?, ?, ?, ?)
            ''', (fi_transaction['usuario_id'], titulo, mensaje, 'success', 'freefire_id_reload'))
            conn.commit()
        except Exception:
            conn.execute('''
                INSERT INTO notificaciones_personalizadas (usuario_id, titulo, mensaje, tipo)
                VALUES (?, ?, ?, ?)
            ''', (fi_transaction['usuario_id'], titulo, mensaje, 'success'))
            conn.commit()
        
        conn.close()
        
        update_freefire_id_transaction_status(transaction_id, 'aprobado', session.get('user_db_id'))
        
        if redeem_result and redeem_result.success:
            flash(f'Transaccion aprobada y pin redimido automaticamente para jugador {fi_transaction["player_id"]}', 'success')
        else:
            flash('Transaccion aprobada exitosamente', 'success')
    except Exception as e:
        flash(f'Error al aprobar transaccion: {str(e)}', 'error')
    
    return redirect('/')

@app.route('/admin/fix_freefire_id_transaction', methods=['POST'])
def fix_freefire_id_transaction():
    """Corrige una transacción inconsistente de Free Fire ID"""
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Acceso denegado'})
    
    try:
        data = request.get_json()
        transaction_id = data.get('transaction_id')
        
        if not transaction_id:
            return jsonify({'success': False, 'error': 'ID de transacción requerido'})
        
        # Obtener detalles de la transacción
        conn = get_db_connection()
        trans = conn.execute('''
            SELECT t.*, p.nombre as paquete_nombre
            FROM transacciones_freefire_id t
            LEFT JOIN precios_freefire_id p ON t.paquete_id = p.id
            WHERE t.id = ?
        ''', (transaction_id,)).fetchone()
        
        if not trans:
            conn.close()
            return jsonify({'success': False, 'error': 'Transacción no encontrada'})
        
        # Actualizar estado a aprobado
        update_freefire_id_transaction_status(transaction_id, 'aprobado', session.get('user_db_id'), 
            'CORRECCIÓN MANUAL: Transacción inconsistente corregida por admin')
        
        # Registrar en transacciones generales para mantener consistencia
        pin_info = f"ID: {trans['player_id']} - Jugador: (corregido)"
        conn.execute('''
            INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (trans['usuario_id'], trans['numero_control'], pin_info, 
              trans['transaccion_id'], trans['paquete_nombre'] or 'FF ID', trans['monto']))
        
        # Registrar en historial permanente
        _corr_precio = abs(trans['monto'])
        _corr_saldo_row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (trans['usuario_id'],)).fetchone()
        _corr_saldo = _corr_saldo_row['saldo'] if _corr_saldo_row else 0
        registrar_historial_compra(conn, trans['usuario_id'], _corr_precio, trans['paquete_nombre'] or 'FF ID', pin_info, 'compra', None, _corr_saldo + _corr_precio, _corr_saldo)
        
        # Registrar profit
        try:
            admin_ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
            admin_ids = [int(x.strip()) for x in admin_ids_env.split(',') if x.strip().isdigit()]
            is_admin_target = trans['usuario_id'] in admin_ids
            record_profit_for_transaction(conn, trans['usuario_id'], is_admin_target, 'freefire_id', 
                trans['paquete_id'], 1, abs(trans['monto']), trans['transaccion_id'])
        except Exception:
            pass
        
        conn.commit()
        conn.close()
        
        logger.info(f"[FreeFire ID] Transacción {transaction_id} corregida manualmente por admin {session.get('user_db_id')}")
        
        return jsonify({'success': True, 'message': 'Transacción corregida exitosamente'})
        
    except Exception as e:
        logger.error(f"[FreeFire ID] Error corrigiendo transacción: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/reject_freefire_id/<int:transaction_id>', methods=['POST'])
def reject_freefire_id_transaction(transaction_id):
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        conn = get_db_connection()
        transaction = conn.execute('''
            SELECT usuario_id, monto FROM transacciones_freefire_id 
            WHERE id = ?
        ''', (transaction_id,)).fetchone()
        
        if transaction:
            conn.execute('UPDATE usuarios SET saldo = saldo + ? WHERE id = ?', 
                        (abs(transaction['monto']), transaction['usuario_id']))
            conn.commit()
        conn.close()
        
        update_freefire_id_transaction_status(transaction_id, 'rechazado', session.get('user_db_id'))
        flash('Transacción rechazada y saldo devuelto al usuario', 'success')
    except Exception as e:
        flash(f'Error al rechazar transacción: {str(e)}', 'error')
    
    return redirect('/')

@app.route('/admin/redeemer_config', methods=['GET', 'POST'])
def admin_redeemer_config():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    if request.method == 'POST':
        campos = ['nombre_completo', 'fecha_nacimiento', 'nacionalidad', 'url_base', 'headless', 'timeout_ms', 'auto_redeem']
        conn = get_db_connection()
        for campo in campos:
            valor = request.form.get(campo, '').strip()
            if valor:
                conn.execute('''
                    INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion)
                    VALUES (%s, %s, NOW()) ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion
                ''', (campo, valor))
        conn.commit()
        conn.close()
        flash('Configuracion del redeemer actualizada correctamente', 'success')
        return redirect('/admin')
    
    # GET: devolver config actual como JSON
    config = get_redeemer_config_from_db(get_db_connection)
    return jsonify(config)

@app.route('/admin/test_redeem', methods=['POST'])
def admin_test_redeem():
    """Prueba manual de redención de pin - solo para testing"""
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    pin_code = request.form.get('test_pin_code', '').strip()
    player_id = request.form.get('test_player_id', '').strip()
    
    if not pin_code or not player_id:
        flash('Debes ingresar un PIN y un Player ID para la prueba', 'error')
        return redirect('/admin')
    
    config = get_redeemer_config_from_db(get_db_connection)
    # Para pruebas, mostrar el navegador
    config['headless'] = False
    
    try:
        result = redeem_pin_vps(pin_code, player_id, config)
        if result.success:
            flash(f'Prueba exitosa: {result.message}', 'success')
        else:
            flash(f'Prueba fallida: {result.message}', 'error')
    except Exception as e:
        flash(f'Error en prueba: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/api/notifications/freefire_id_reload', methods=['GET'])
def api_freefire_id_reload_notifications():
    if 'usuario' not in session or session.get('is_admin'):
        return jsonify({'notifications': []})
    
    user_id = session.get('user_db_id')
    if not user_id:
        return jsonify({'notifications': []})
    
    try:
        conn = get_db_connection()
        notifications = conn.execute('''
            SELECT id, titulo, mensaje, tipo, fecha 
            FROM notificaciones_personalizadas 
            WHERE usuario_id = ? AND visto = FALSE AND tag = 'freefire_id_reload'
            ORDER BY fecha DESC
            LIMIT 5
        ''', (user_id,)).fetchall()
        conn.close()
        
        result = [{'id': n['id'], 'titulo': n['titulo'], 'mensaje': n['mensaje'], 'tipo': n['tipo']} for n in notifications]
        return jsonify({'notifications': result})
    except Exception:
        return jsonify({'notifications': []})

# Rutas de administración para API externa
@app.route('/admin/test_external_api', methods=['POST'])
def admin_test_external_api():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        pin_manager = create_pin_manager(DATABASE)
        result = pin_manager.test_external_api()
        
        if result.get('status') == 'success':
            flash(f'✅ API Externa: {result.get("message")}', 'success')
        else:
            flash(f'❌ API Externa: {result.get("message")}', 'error')
    except Exception as e:
        flash(f'Error al probar API externa: {str(e)}', 'error')
    
    return redirect('/admin')


@app.route('/admin/toggle_pin_source', methods=['POST'])
def admin_toggle_pin_source():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    monto_id = request.form.get('monto_id')
    fuente = request.form.get('fuente')
    
    if not monto_id or not fuente:
        flash('Datos inválidos para cambiar fuente', 'error')
        return redirect('/admin')
    
    try:
        monto_id = int(monto_id)
        if monto_id < 1 or monto_id > 9:
            flash('Monto ID debe estar entre 1 y 9', 'error')
            return redirect('/admin')
        
        if fuente not in ['local', 'api_externa']:
            flash('Fuente inválida. Debe ser "local" o "api_externa"', 'error')
            return redirect('/admin')
        
        # Actualizar configuración
        update_pin_source_config(monto_id, fuente)
        
        # Obtener información del paquete
        packages_info = get_package_info_with_prices()
        package_info = packages_info.get(monto_id, {})
        paquete_nombre = package_info.get('nombre', f'Paquete {monto_id}')
        
        fuente_texto = 'Stock Local' if fuente == 'local' else 'API Externa'
        flash(f'✅ Configuración actualizada: {paquete_nombre} → {fuente_texto}', 'success')
        
    except ValueError:
        flash('Monto ID debe ser un número válido', 'error')
    except Exception as e:
        flash(f'Error al actualizar configuración: {str(e)}', 'error')
    
    return redirect('/admin')

# ──────────────────────────────────────────────────────────────
# Rutas para Mapeo de Revendedores (sync catálogo + mappings)
# ──────────────────────────────────────────────────────────────

@app.route('/admin/revendedores/sync', methods=['POST'])
def admin_revendedores_sync():
    """Sincroniza catálogo de productos del revendedor externo."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403

    import urllib.request, urllib.error
    base_url = os.environ.get('REVENDEDORES_BASE_URL', '').strip().rstrip('/')
    api_key = os.environ.get('REVENDEDORES_API_KEY', '').strip()

    if not base_url or not api_key:
        return jsonify({'error': 'Configurar REVENDEDORES_BASE_URL y REVENDEDORES_API_KEY en variables de entorno'}), 400

    try:
        url = f"{base_url}/api/v1/products"
        req = urllib.request.Request(url, headers={
            'X-API-Key': api_key,
            'Accept': 'application/json',
            'User-Agent': '3sRecargas-Admin/1.0'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as he:
        return jsonify({'error': f'Error HTTP {he.code} del servidor revendedor'}), 502
    except Exception as e:
        return jsonify({'error': f'Error conectando al revendedor: {str(e)}'}), 502

    # El API devuelve {games: [{game_id, name, packages: [{package_id, name, price}]}]}
    products = data.get('games', data.get('products', data.get('data', [])))
    if not isinstance(products, list):
        return jsonify({'error': 'Respuesta inesperada del revendedor'}), 502

    conn = get_db_connection()
    inserted = 0
    updated = 0
    try:
        for prod in products:
            prod_id = str(prod.get('game_id', prod.get('id', prod.get('product_id', ''))))
            prod_name = prod.get('name', prod.get('product_name', ''))
            packages = prod.get('packages', prod.get('items', []))
            if not isinstance(packages, list):
                packages = []
            for pkg in packages:
                pkg_id = str(pkg.get('package_id', pkg.get('id', '')))
                pkg_name = pkg.get('name', pkg.get('package_name', ''))
                raw = json.dumps(pkg)

                existing = conn.execute(
                    'SELECT id FROM rev_catalog_items WHERE remote_product_id = ? AND remote_package_id = ?',
                    (prod_id, pkg_id)
                ).fetchone()

                if existing:
                    conn.execute('''
                        UPDATE rev_catalog_items
                        SET remote_product_name = ?, remote_package_name = ?, raw_json = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (prod_name, pkg_name, raw, existing['id']))
                    updated += 1
                else:
                    conn.execute('''
                        INSERT INTO rev_catalog_items
                        (remote_product_id, remote_product_name, remote_package_id, remote_package_name, raw_json)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (prod_id, prod_name, pkg_id, pkg_name, raw))
                    inserted += 1

        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

    return jsonify({'ok': True, 'inserted': inserted, 'updated': updated,
                    'total': inserted + updated})


@app.route('/admin/revendedores/mapping-data')
def admin_revendedores_mapping_data():
    """Devuelve juegos, paquetes, catálogo remoto y mappings actuales."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403

    from dynamic_games import get_all_dynamic_games as _dg_all, get_dynamic_packages as _dg_pkgs

    games = _dg_all()
    games_out = []
    for g in games:
        pkgs = _dg_pkgs(g['id'])
        games_out.append({
            'id': g['id'],
            'nombre': g['nombre'],
            'slug': g['slug'],
            'activo': g.get('activo', False),
            'packages': [{'id': p['id'], 'nombre': p['nombre'], 'precio': p['precio']} for p in pkgs]
        })

    conn = get_db_connection()
    catalog_rows = conn.execute(
        'SELECT id, remote_product_id, remote_product_name, remote_package_id, remote_package_name FROM rev_catalog_items WHERE active = TRUE ORDER BY remote_product_name, remote_package_name'
    ).fetchall()

    mappings_rows = conn.execute(
        'SELECT id, juego_id, paquete_id, remote_product_id, remote_package_id, remote_label, auto_enabled, active FROM rev_item_mappings'
    ).fetchall()
    conn.close()

    catalog = [dict(r) for r in catalog_rows]
    mappings = [dict(r) for r in mappings_rows]

    return jsonify({'games': games_out, 'catalog': catalog, 'mappings': mappings})


@app.route('/admin/revendedores/mappings/bulk', methods=['POST'])
def admin_revendedores_bulk_save():
    """Guarda mappings en lote. Espera JSON: {mappings: [{juego_id, paquete_id, remote_product_id, remote_package_id, auto_enabled}]}."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403

    data = request.get_json(force=True, silent=True) or {}
    items = data.get('mappings', [])
    if not isinstance(items, list):
        return jsonify({'error': 'Se espera una lista de mappings'}), 400

    conn = get_db_connection()
    saved = 0
    try:
        for m in items:
            juego_id = m.get('juego_id')
            paquete_id = m.get('paquete_id')
            r_prod = str(m.get('remote_product_id', ''))
            r_pkg = str(m.get('remote_package_id', ''))
            auto_en = bool(m.get('auto_enabled'))

            if not juego_id or not paquete_id:
                continue

            # Si remote_product_id/remote_package_id vacíos → eliminar mapping
            if not r_prod or not r_pkg:
                conn.execute('DELETE FROM rev_item_mappings WHERE juego_id = ? AND paquete_id = ?',
                             (juego_id, paquete_id))
                saved += 1
                continue

            # Buscar label del catálogo
            cat = conn.execute(
                'SELECT remote_product_name, remote_package_name FROM rev_catalog_items WHERE remote_product_id = ? AND remote_package_id = ?',
                (r_prod, r_pkg)
            ).fetchone()
            label = f"{cat['remote_product_name']} – {cat['remote_package_name']}" if cat else ''

            existing = conn.execute(
                'SELECT id FROM rev_item_mappings WHERE juego_id = ? AND paquete_id = ?',
                (juego_id, paquete_id)
            ).fetchone()

            if existing:
                conn.execute('''
                    UPDATE rev_item_mappings
                    SET remote_product_id = ?, remote_package_id = ?, remote_label = ?,
                        auto_enabled = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (r_prod, r_pkg, label, auto_en, existing['id']))
            else:
                conn.execute('''
                    INSERT INTO rev_item_mappings
                    (juego_id, paquete_id, remote_product_id, remote_package_id, remote_label, auto_enabled)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (juego_id, paquete_id, r_prod, r_pkg, label, auto_en))
            saved += 1

        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

    return jsonify({'ok': True, 'saved': saved})

# Rutas para sistema de noticias
@app.route('/noticias')
def noticias():
    if 'usuario' not in session:
        return redirect('/auth')
    
    user_id = session.get('user_db_id')
    is_admin = session.get('is_admin', False)
    
    # Para admin, usar ID 0 y permitir acceso
    if is_admin:
        user_id = 0
    elif not user_id:
        flash('Error al acceder a las noticias', 'error')
        return redirect('/')
    
    # Obtener noticias para mostrar
    noticias_list = get_user_news(user_id)
    
    return render_template('noticias.html', 
                         noticias=noticias_list,
                         user_id=session.get('id', '00000'),
                         is_admin=is_admin)

@app.route('/notificaciones')
def notificaciones():
    """Ruta unificada para ver noticias y notificaciones personalizadas"""
    if 'usuario' not in session:
        return redirect('/auth')
    
    user_id = session.get('user_db_id')
    is_admin = session.get('is_admin', False)
    
    # Para admin, usar ID 0 y permitir acceso
    if is_admin:
        user_id = 0
    elif not user_id:
        flash('Error al acceder a las notificaciones', 'error')
        return redirect('/')
    
    # Para usuarios normales, obtener y procesar notificaciones personalizadas
    notificaciones_personalizadas = []
    if not is_admin:
        # Obtener notificaciones personalizadas antes de marcarlas como leídas
        notificaciones_personalizadas = get_user_personal_notifications(user_id)
        
        # Marcar notificaciones personalizadas como leídas (las elimina)
        if notificaciones_personalizadas:
            mark_personal_notifications_as_read(user_id)
        
    # Obtener noticias para mostrar
    noticias_list = get_user_news(user_id)
    
    return render_template('noticias.html', 
                         noticias=noticias_list,
                         notificaciones_personalizadas=notificaciones_personalizadas,
                         user_id=session.get('id', '00000'),
                         is_admin=is_admin)

@app.route('/admin/create_news', methods=['POST'])
def admin_create_news():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    titulo = request.form.get('titulo')
    contenido = request.form.get('contenido')
    importante = request.form.get('importante') == '1'
    
    if not titulo or not contenido:
        flash('Por favor complete todos los campos obligatorios', 'error')
        return redirect('/admin')
    
    if len(titulo) > 200:
        flash('El título no puede exceder 200 caracteres', 'error')
        return redirect('/admin')
    
    if len(contenido) > 2000:
        flash('El contenido no puede exceder 2000 caracteres', 'error')
        return redirect('/admin')
    
    # Manejar subida de imagen
    imagen_url = None
    imagen_file = request.files.get('imagen')
    if imagen_file and imagen_file.filename:
        import uuid
        ext = imagen_file.filename.rsplit('.', 1)[-1].lower()
        if ext in ('png', 'jpg', 'jpeg', 'gif', 'webp'):
            filename = f"news_{uuid.uuid4().hex}.{ext}"
            upload_dir = os.path.join(os.path.dirname(__file__), 'static', 'news_images')
            os.makedirs(upload_dir, exist_ok=True)
            imagen_file.save(os.path.join(upload_dir, filename))
            imagen_url = f"/static/news_images/{filename}"
        else:
            flash('Formato de imagen no válido. Use PNG, JPG, GIF o WEBP.', 'error')
            return redirect('/admin')
    
    try:
        news_id = create_news(titulo, contenido, importante, imagen_url)
        tipo_noticia = "importante" if importante else "normal"
        flash(f'Noticia {tipo_noticia} creada exitosamente (ID: {news_id})', 'success')
    except Exception as e:
        flash(f'Error al crear la noticia: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/delete_news', methods=['POST'])
def admin_delete_news():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    news_id = request.form.get('news_id')
    
    if not news_id:
        flash('ID de noticia inválido', 'error')
        return redirect('/admin')
    
    try:
        delete_news(int(news_id))
        flash('Noticia eliminada exitosamente', 'success')
    except Exception as e:
        flash(f'Error al eliminar la noticia: {str(e)}', 'error')
    
    return redirect('/admin')

# ============= RUTAS PARA GESTIÓN DE RENTABILIDAD =============

@app.route('/admin/update_purchase_price', methods=['POST'])
def admin_update_purchase_price():
    """Actualiza el precio de compra de un paquete - Compatible con Render"""
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    juego = request.form.get('juego')
    paquete_id = request.form.get('paquete_id')
    nuevo_precio = request.form.get('nuevo_precio')
    
    if not all([juego, paquete_id, nuevo_precio]):
        flash('Datos inválidos para actualizar precio de compra', 'error')
        return redirect('/admin')
    
    try:
        nuevo_precio = float(nuevo_precio)
        paquete_id = int(paquete_id)
        
        if nuevo_precio < 0:
            flash('El precio de compra no puede ser negativo', 'error')
            return redirect('/admin')
        
        # Validar juego
        if juego not in ['freefire_latam', 'freefire_global', 'bloodstriker']:
            flash('Tipo de juego inválido', 'error')
            return redirect('/admin')
        
        # Obtener nombre del paquete para el mensaje
        try:
            if juego == 'freefire_latam':
                packages_info = get_package_info_with_prices()
            elif juego == 'freefire_global':
                packages_info = get_freefire_global_prices()
            else:  # bloodstriker
                packages_info = get_bloodstriker_prices()
            
            package_info = packages_info.get(paquete_id, {})
            paquete_nombre = package_info.get('nombre', f'Paquete {paquete_id}')
        except Exception as e:
            print(f"Error obteniendo información del paquete: {e}")
            paquete_nombre = f'Paquete {paquete_id}'
        
        # Actualizar precio de compra usando función compatible con Render
        success = update_purchase_price(juego, paquete_id, nuevo_precio)
        
        if success:
            juego_display = {
                'freefire_latam': 'Free Fire LATAM',
                'freefire_global': 'Free Fire',
                'bloodstriker': 'Blood Striker'
            }.get(juego, juego)
            
            flash(f'Precio de compra actualizado para {juego_display} - {paquete_nombre}: ${nuevo_precio:.2f}', 'success')
        else:
            flash('Error al actualizar precio de compra en la base de datos', 'error')
        
    except ValueError:
        flash('Precio inválido. Debe ser un número válido.', 'error')
    except Exception as e:
        print(f"Error en admin_update_purchase_price: {e}")
        flash(f'Error al actualizar precio de compra: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/profitability')
def admin_profitability():
    """Muestra el análisis de rentabilidad de todos los productos"""
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        profit_analysis = get_profit_analysis()
        # Calcular profit diario del mes actual
        from datetime import datetime
        import calendar
        from admin_stats import compute_legacy_profit_by_day
        conn = get_db_connection()
        now = datetime.now()
        year = now.year
        month = now.month
        first_day = f"{year}-{month:02d}-01"
        last_day = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
        # Usar compute_legacy_profit_by_day para obtener lista de dicts
        daily_list = compute_legacy_profit_by_day(conn, first_day, last_day)
        # Convertir a dict: {día:int: profit:float}
        daily_profit = {int(item['day'][-2:]): item['profit'] for item in daily_list if 'day' in item and item['day'][-2:].isdigit()}
        conn.close()
        # Calcular total_profit para el mes actual
        total_profit = sum(daily_profit.values())
        return render_template('admin_profitability.html', profit_analysis=profit_analysis, daily_profit=daily_profit, total_profit=total_profit)
    except Exception as e:
        flash(f'Error al obtener análisis de rentabilidad: {str(e)}', 'error')
        return redirect('/admin')

@app.route('/admin/weekly_sales')
def admin_weekly_sales():
    """Muestra las estadísticas de ventas semanales"""
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        weekly_stats = get_weekly_sales_stats()
        return render_template('admin_weekly_sales.html', **weekly_stats)
    except Exception as e:
        flash(f'Error al obtener estadísticas semanales: {str(e)}', 'error')
        return redirect('/admin')

@app.route('/admin/clean_weekly_sales', methods=['POST'])
def admin_clean_weekly_sales():
    """Limpia manualmente las ventas semanales antiguas"""
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        deleted_count = clean_old_weekly_sales()
        if deleted_count > 0:
            flash(f'Se eliminaron {deleted_count} registros de ventas antiguas', 'success')
        else:
            flash('No se encontraron registros antiguos para eliminar', 'success')
    except Exception as e:
        flash(f'Error al limpiar ventas antiguas: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/reset_all_weekly_sales', methods=['POST'])
def admin_reset_all_weekly_sales():
    """Resetea TODAS las estadísticas de ventas semanales (elimina todos los registros)"""
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    
    try:
        deleted_count = reset_all_weekly_sales()
        if deleted_count > 0:
            flash(f'Se resetearon todas las estadísticas: {deleted_count} registros eliminados', 'success')
        else:
            flash('No había estadísticas para resetear', 'success')
    except Exception as e:
        flash(f'Error al resetear estadísticas: {str(e)}', 'error')
    
    return redirect('/admin')

@app.route('/admin/simple_stats')
def admin_simple_stats():
    """Obtiene estadísticas simples de ventas para la pestaña de estadísticas"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    
    try:
        conn = get_db_connection()
        
        # Estadísticas por juego
        stats = {}
        
        # Free Fire LATAM
        ff_latam = conn.execute('''
            SELECT SUM(cantidad_vendida) as total_units, SUM(ganancia_total) as total_profit
            FROM ventas_semanales 
            WHERE juego = 'freefire_latam'
        ''').fetchone()
        
        stats['freefire_latam'] = {
            'units': ff_latam['total_units'] or 0,
            'profit': ff_latam['total_profit'] or 0.0
        }
        
        # Free Fire Global
        ff_global = conn.execute('''
            SELECT SUM(cantidad_vendida) as total_units, SUM(ganancia_total) as total_profit
            FROM ventas_semanales 
            WHERE juego = 'freefire_global'
        ''').fetchone()
        
        stats['freefire_global'] = {
            'units': ff_global['total_units'] or 0,
            'profit': ff_global['total_profit'] or 0.0
        }
        
        # Blood Striker
        bs = conn.execute('''
            SELECT SUM(cantidad_vendida) as total_units, SUM(ganancia_total) as total_profit
            FROM ventas_semanales 
            WHERE juego = 'bloodstriker'
        ''').fetchone()
        
        stats['bloodstriker'] = {
            'units': bs['total_units'] or 0,
            'profit': bs['total_profit'] or 0.0
        }
        
        conn.close()
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/get_purchase_price/<juego>/<int:paquete_id>')
def admin_get_purchase_price(juego, paquete_id):
    """Obtiene el precio de compra actual para un juego y paquete específico"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Acceso denegado'}), 403
    
    try:
        precio_compra = get_purchase_price(juego, paquete_id)
        return jsonify({'precio_compra': precio_compra})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============= FUNCIONES PARA GESTIÓN DE RENTABILIDAD =============

def get_purchase_prices():
    """Obtiene todos los precios de compra por juego y paquete"""
    conn = get_db_connection()
    prices = conn.execute('''
        SELECT * FROM precios_compra 
        WHERE activo = 1 
        ORDER BY juego, paquete_id
    ''').fetchall()
    conn.close()
    return prices

def get_purchase_price(juego, paquete_id):
    """Obtiene el precio de compra para un juego y paquete específico - Compatible con Render"""
    conn = None
    try:
        conn = get_db_connection_optimized()
        
        # Usar parámetros seguros y validados
        query = '''
            SELECT precio_compra FROM precios_compra 
            WHERE juego = ? AND paquete_id = ? AND activo = 1
        '''
        
        result = conn.execute(query, (str(juego), int(paquete_id))).fetchone()
        
        if result:
            return float(result['precio_compra'])
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error en get_purchase_price: {e}")
        return 0.0
    finally:
        if conn:
            return_db_connection(conn)

def update_purchase_price(juego, paquete_id, nuevo_precio):
    """Actualiza el precio de compra para un juego y paquete específico - Compatible con Render"""
    conn = None
    try:
        conn = get_db_connection_optimized()
        
        query = '''
            INSERT INTO precios_compra (juego, paquete_id, precio_compra, fecha_actualizacion, activo)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, TRUE)
            ON CONFLICT (juego, paquete_id) DO UPDATE SET precio_compra = EXCLUDED.precio_compra, fecha_actualizacion = EXCLUDED.fecha_actualizacion, activo = EXCLUDED.activo
        '''
        
        conn.execute(query, (str(juego), int(paquete_id), float(nuevo_precio)))
        conn.commit()
        
        return True
        
    except Exception as e:
        print(f"Error en update_purchase_price: {e}")
        if conn:
            try:
                conn.execute('ROLLBACK')
            except:
                pass
        return False
    finally:
        if conn:
            return_db_connection(conn)

def get_profit_analysis():
    """Obtiene análisis de rentabilidad por juego y paquete"""
    conn = get_db_connection()
    
    # Análisis para Free Fire LATAM
    freefire_latam_analysis = []
    freefire_latam_prices = get_package_info_with_prices()
    for paquete_id, info in freefire_latam_prices.items():
        precio_compra = get_purchase_price('freefire_latam', paquete_id)
        precio_venta = info['precio']
        ganancia = precio_venta - precio_compra
        margen = (ganancia / precio_venta * 100) if precio_venta > 0 else 0
        
        freefire_latam_analysis.append({
            'juego': 'Free Fire LATAM',
            'paquete_id': paquete_id,
            'nombre': info['nombre'],
            'precio_compra': precio_compra,
            'precio_venta': precio_venta,
            'ganancia': ganancia,
            'margen_porcentaje': margen
        })
    
    # Análisis para Free Fire Global
    freefire_global_analysis = []
    freefire_global_prices = get_freefire_global_prices()
    for paquete_id, info in freefire_global_prices.items():
        precio_compra = get_purchase_price('freefire_global', paquete_id)
        precio_venta = info['precio']
        ganancia = precio_venta - precio_compra
        margen = (ganancia / precio_venta * 100) if precio_venta > 0 else 0
        
        freefire_global_analysis.append({
            'juego': 'Free Fire',
            'paquete_id': paquete_id,
            'nombre': info['nombre'],
            'precio_compra': precio_compra,
            'precio_venta': precio_venta,
            'ganancia': ganancia,
            'margen_porcentaje': margen
        })
    
    # Análisis para Blood Striker
    bloodstriker_analysis = []
    bloodstriker_prices = get_bloodstriker_prices()
    for paquete_id, info in bloodstriker_prices.items():
        precio_compra = get_purchase_price('bloodstriker', paquete_id)
        precio_venta = info['precio']
        ganancia = precio_venta - precio_compra
        margen = (ganancia / precio_venta * 100) if precio_venta > 0 else 0
        
        bloodstriker_analysis.append({
            'juego': 'Blood Striker',
            'paquete_id': paquete_id,
            'nombre': info['nombre'],
            'precio_compra': precio_compra,
            'precio_venta': precio_venta,
            'ganancia': ganancia,
            'margen_porcentaje': margen
        })
    
    # Análisis para Free Fire ID
    freefire_id_analysis = []
    try:
        freefire_id_prices = get_freefire_id_prices()
        for paquete_id, info in freefire_id_prices.items():
            precio_compra = get_purchase_price('freefire_id', paquete_id)
            precio_venta = info['precio']
            ganancia = precio_venta - precio_compra
            margen = (ganancia / precio_venta * 100) if precio_venta > 0 else 0
            
            freefire_id_analysis.append({
                'juego': 'Free Fire ID',
                'paquete_id': paquete_id,
                'nombre': info['nombre'],
                'precio_compra': precio_compra,
                'precio_venta': precio_venta,
                'ganancia': ganancia,
                'margen_porcentaje': margen
            })
    except Exception:
        pass
    
    conn.close()
    return freefire_latam_analysis + freefire_global_analysis + bloodstriker_analysis + freefire_id_analysis

def register_weekly_sale(juego, paquete_id, paquete_nombre, precio_venta, cantidad=1):
    """Registra una venta en las estadísticas diarias (corregido para resetear a las 12 AM)"""
    from datetime import datetime
    import pytz
    
    conn = get_db_connection()
    
    # Obtener precio de compra
    precio_compra = get_purchase_price(juego, paquete_id)
    ganancia_unitaria = precio_venta - precio_compra
    ganancia_total = ganancia_unitaria * cantidad
    
    # Usar zona horaria de Venezuela para calcular el día correcto
    venezuela_tz = pytz.timezone('America/Caracas')
    now_venezuela = datetime.now(venezuela_tz)
    
    # Calcular día del año (formato: YYYY-MM-DD) - resetea a las 12:00 AM
    dia_year = now_venezuela.strftime('%Y-%m-%d')
    
    # Verificar si ya existe un registro para este día y paquete
    existing = conn.execute('''
        SELECT id, cantidad_vendida, ganancia_total FROM ventas_semanales 
        WHERE juego = ? AND paquete_id = ? AND semana_year = ?
    ''', (juego, paquete_id, dia_year)).fetchone()
    
    if existing:
        # Actualizar registro existente
        nueva_cantidad = existing['cantidad_vendida'] + cantidad
        nueva_ganancia_total = existing['ganancia_total'] + ganancia_total
        
        conn.execute('''
            UPDATE ventas_semanales 
            SET cantidad_vendida = ?, ganancia_total = ?, fecha_venta = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (nueva_cantidad, nueva_ganancia_total, existing['id']))
    else:
        # Crear nuevo registro
        conn.execute('''
            INSERT INTO ventas_semanales 
            (juego, paquete_id, paquete_nombre, precio_venta, precio_compra, 
             ganancia_unitaria, cantidad_vendida, ganancia_total, semana_year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (juego, paquete_id, paquete_nombre, precio_venta, precio_compra, 
              ganancia_unitaria, cantidad, ganancia_total, dia_year))
    
    conn.commit()
    conn.close()

def get_weekly_sales_stats():
    """Obtiene estadísticas de ventas del día actual (corregido para usar días)"""
    from datetime import datetime
    import pytz
    
    # Usar zona horaria de Venezuela para calcular el día correcto
    venezuela_tz = pytz.timezone('America/Caracas')
    now_venezuela = datetime.now(venezuela_tz)
    
    # Calcular día actual (formato: YYYY-MM-DD) - resetea a las 12:00 AM
    dia_actual = now_venezuela.strftime('%Y-%m-%d')
    
    conn = get_db_connection()
    
    # Estadísticas por juego
    stats_by_game = conn.execute('''
        SELECT juego, 
               SUM(cantidad_vendida) as total_unidades,
               SUM(ganancia_total) as ganancia_total_juego,
               COUNT(DISTINCT paquete_id) as paquetes_diferentes
        FROM ventas_semanales 
        WHERE semana_year = ?
        GROUP BY juego
        ORDER BY ganancia_total_juego DESC
    ''', (dia_actual,)).fetchall()
    
    # Estadísticas por paquete
    stats_by_package = conn.execute('''
        SELECT juego, paquete_nombre, precio_venta, precio_compra,
               cantidad_vendida, ganancia_unitaria, ganancia_total
        FROM ventas_semanales 
        WHERE semana_year = ?
        ORDER BY ganancia_total DESC
    ''', (dia_actual,)).fetchall()
    
    # Totales generales
    totals = conn.execute('''
        SELECT SUM(cantidad_vendida) as total_unidades_vendidas,
               SUM(ganancia_total) as ganancia_total_semana,
               SUM(precio_venta * cantidad_vendida) as ingresos_totales,
               SUM(precio_compra * cantidad_vendida) as costos_totales
        FROM ventas_semanales 
        WHERE semana_year = ?
    ''', (dia_actual,)).fetchone()
    
    conn.close()
    
    return {
        'semana_actual': dia_actual,
        'stats_by_game': stats_by_game,
        'stats_by_package': stats_by_package,
        'totals': totals
    }

def clean_old_weekly_sales():
    """Limpia las ventas semanales antiguas (mantiene solo las últimas 4 semanas)"""
    from datetime import datetime, timedelta
    
    conn = get_db_connection()
    
    try:
        # Calcular fecha límite (4 semanas atrás)
        fecha_limite = datetime.now() - timedelta(weeks=4)
        year_limite, week_limite, _ = fecha_limite.isocalendar()
        semana_limite = f"{year_limite}-{week_limite:02d}"
        
        # Obtener todas las semanas existentes y filtrar las que son más antiguas
        all_weeks = conn.execute('''
            SELECT DISTINCT semana_year FROM ventas_semanales
        ''').fetchall()
        
        weeks_to_delete = []
        for week_row in all_weeks:
            week_str = week_row['semana_year']
            try:
                # Parsear la semana (formato YYYY-WW)
                year_str, week_str_num = week_str.split('-')
                year = int(year_str)
                week = int(week_str_num)
                
                # Comparar con la fecha límite
                if year < year_limite or (year == year_limite and week < week_limite):
                    weeks_to_delete.append(week_row['semana_year'])
            except (ValueError, IndexError):
                # Si hay un formato inválido, eliminar ese registro también
                weeks_to_delete.append(week_row['semana_year'])
        
        # Eliminar registros antiguos
        deleted_count = 0
        for week_to_delete in weeks_to_delete:
            count = conn.execute('''
                DELETE FROM ventas_semanales 
                WHERE semana_year = ?
            ''', (week_to_delete,)).rowcount
            deleted_count += count
        
        conn.commit()
        return deleted_count
        
    except Exception as e:
        conn.rollback()
        print(f"Error en clean_old_weekly_sales: {str(e)}")
        return 0
    finally:
        conn.close()

def clean_old_transactions():
    """Limpia transacciones antiguas manteniendo solo las del último mes (mejorado)"""
    from datetime import datetime, timedelta
    
    # Verificar si ya se ejecutó la limpieza hoy
    last_cleanup_file = 'last_cleanup.txt'
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        if os.path.exists(last_cleanup_file):
            with open(last_cleanup_file, 'r') as f:
                last_cleanup_date = f.read().strip()
            
            if last_cleanup_date == today:
                # Ya se ejecutó la limpieza hoy, no hacer nada
                return 0
    except:
        pass  # Si hay error leyendo el archivo, continuar con la limpieza
    
    conn = get_db_connection()
    
    try:
        # Calcular fecha límite (1 MES atrás en lugar de 1 semana)
        fecha_limite = datetime.now() - timedelta(days=30)  # 30 días = 1 mes
        fecha_limite_str = fecha_limite.strftime('%Y-%m-%d %H:%M:%S')
        
        # Eliminar transacciones normales más antiguas de 1 mes
        deleted_normal = conn.execute('''
            DELETE FROM transacciones 
            WHERE fecha < ?
        ''', (fecha_limite_str,)).rowcount
        
        # Eliminar transacciones de Blood Striker más antiguas de 1 mes (excepto pendientes)
        deleted_bs = conn.execute('''
            DELETE FROM transacciones_bloodstriker 
            WHERE fecha < ? AND estado != 'pendiente'
        ''', (fecha_limite_str,)).rowcount
        
        # Eliminar historial_compras más antiguo de 3 días (mismo rango que la visualización en Costo)
        fecha_limite_hist = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')
        deleted_hist = conn.execute('''
            DELETE FROM historial_compras WHERE fecha < ?
        ''', (fecha_limite_hist,)).rowcount
        
        conn.commit()
        
        total_deleted = deleted_normal + deleted_bs + deleted_hist
        if total_deleted > 0:
            print(f"🧹 Limpieza automática diaria: {total_deleted} registros antiguos eliminados ({deleted_normal} transacciones, {deleted_bs} Blood Striker, {deleted_hist} historial_compras)")
        
        # Guardar fecha de última limpieza
        try:
            with open(last_cleanup_file, 'w') as f:
                f.write(today)
        except:
            pass  # Si no se puede escribir el archivo, no es crítico
        
        return total_deleted
        
    except Exception as e:
        conn.rollback()
        print(f"Error en clean_old_transactions: {str(e)}")
        return 0
    finally:
        conn.close()

def reset_all_weekly_sales():
    """Resetea TODAS las estadísticas de ventas semanales (elimina todos los registros)"""
    conn = get_db_connection()
    
    try:
        # Contar registros antes de eliminar
        total_count = conn.execute('SELECT COUNT(*) FROM ventas_semanales').fetchone()[0]
        
        # Eliminar todos los registros
        conn.execute('DELETE FROM ventas_semanales')
        conn.commit()
        
        return total_count
        
    except Exception as e:
        conn.rollback()
        print(f"Error en reset_all_weekly_sales: {str(e)}")
        return 0
    finally:
        conn.close()

# Funciones para Free Fire Global (nuevo juego)
def add_pin_freefire_global(monto_id, pin_codigo):
    """Añade un pin de Free Fire Global al stock"""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO pines_freefire_global (monto_id, pin_codigo, batch_id)
        VALUES (?, ?, NULL)
    ''', (monto_id, pin_codigo))
    conn.commit()
    conn.close()

def add_pins_batch_freefire_global(monto_id, pins_list):
    """Añade múltiples pines de Free Fire Global al stock en lote"""
    conn = get_db_connection()
    try:
        batch_id = _generate_batch_id()
        for pin_codigo in pins_list:
            pin_codigo = pin_codigo.strip()
            if pin_codigo:  # Solo agregar si el pin no está vacío
                conn.execute('''
                    INSERT INTO pines_freefire_global (monto_id, pin_codigo, batch_id)
                    VALUES (?, ?, ?)
                ''', (monto_id, pin_codigo, batch_id))
        conn.commit()
        return len([p for p in pins_list if p.strip()])  # Retornar cantidad agregada
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_pin_stock_freefire_global():
    """Obtiene el stock de pines de Free Fire Global por monto_id"""
    conn = get_db_connection()
    stock = {}
    for i in range(1, 7):  # monto_id del 1 al 6 para Free Fire Global
        count = conn.execute('''
            SELECT COUNT(*) FROM pines_freefire_global 
            WHERE monto_id = ? AND usado = FALSE
        ''', (i,)).fetchone()[0]
        stock[i] = count
    conn.close()
    return stock

def get_available_pin_freefire_global(monto_id):
    """Obtiene un pin disponible de Free Fire Global para el monto especificado y lo elimina atómicamente"""
    conn = get_db_connection()
    try:
        # Atómico: DELETE + RETURNING en una sola query para evitar race conditions
        # Si 2 requests llegan al mismo tiempo, solo 1 logrará borrar el PIN
        pin = conn.execute('''
            DELETE FROM pines_freefire_global 
            WHERE id = (
                SELECT id FROM pines_freefire_global 
                WHERE monto_id = ? AND usado = FALSE 
                LIMIT 1
            )
            RETURNING *
        ''', (monto_id,)).fetchone()
        conn.commit()
        return pin
    except Exception:
        # Fallback para SQLite antiguo sin RETURNING (< 3.35)
        pin = conn.execute('''
            SELECT * FROM pines_freefire_global 
            WHERE monto_id = ? AND usado = FALSE 
            LIMIT 1
        ''', (monto_id,)).fetchone()
        if pin:
            deleted = conn.execute('''
                DELETE FROM pines_freefire_global 
                WHERE id = ? AND usado = FALSE
            ''', (pin['id'],))
            if deleted.rowcount == 0:
                # Otro request ya lo tomó
                conn.close()
                return get_available_pin_freefire_global(monto_id)  # Reintentar
            conn.commit()
        return pin
    finally:
        conn.close()

def get_freefire_global_prices():
    """Obtiene información de paquetes de Free Fire Global con precios dinámicos"""
    conn = get_db_connection()
    packages = conn.execute('''
        SELECT id, nombre, precio, descripcion 
        FROM precios_freefire_global 
        WHERE activo = TRUE 
        ORDER BY id
    ''').fetchall()
    conn.close()
    
    # Convertir a diccionario para fácil acceso
    package_dict = {}
    for package in packages:
        package_dict[package['id']] = {
            'nombre': package['nombre'],
            'precio': package['precio'],
            'descripcion': package['descripcion']
        }
    
    return package_dict

def get_freefire_global_price_by_id(monto_id):
    """Obtiene el precio de un paquete específico de Free Fire Global"""
    conn = get_db_connection()
    price = conn.execute('''
        SELECT precio FROM precios_freefire_global 
        WHERE id = ? AND activo = TRUE
    ''', (monto_id,)).fetchone()
    conn.close()
    return price['precio'] if price else 0

def update_freefire_global_price(package_id, new_price):
    """Actualiza el precio de un paquete de Free Fire Global"""
    conn = get_db_connection_optimized()
    try:
        conn.execute('''
            UPDATE precios_freefire_global 
            SET precio = ?, fecha_actualizacion = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_price, package_id))
        conn.commit()
        # Limpiar cache después de actualizar precios
        clear_price_cache()
    finally:
        return_db_connection(conn)

def update_freefire_global_name(package_id, new_name):
    """Actualiza el nombre de un paquete de Free Fire Global"""
    conn = get_db_connection_optimized()
    try:
        conn.execute('''
            UPDATE precios_freefire_global 
            SET nombre = ?, fecha_actualizacion = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (new_name, package_id))
        conn.commit()
        # Limpiar cache después de actualizar nombres
        clear_price_cache()
    finally:
        return_db_connection(conn)

def get_all_freefire_global_prices():
    """Obtiene todos los precios de paquetes de Free Fire Global"""
    conn = get_db_connection()
    prices = conn.execute('''
        SELECT * FROM precios_freefire_global 
        ORDER BY id
    ''').fetchall()
    conn.close()
    return prices


def get_freefire_global_price_by_id_any(monto_id):
    """Obtiene el precio de un paquete específico de Free Fire Global (incluye inactivos)"""
    conn = get_db_connection()
    price = conn.execute('''
        SELECT precio FROM precios_freefire_global
        WHERE id = ?
    ''', (monto_id,)).fetchone()
    conn.close()
    return price['precio'] if price else 0

# Rutas para Free Fire Global (nuevo juego)
@app.route('/juego/freefire')
def freefire():
    if 'usuario' not in session:
        return redirect('/auth')

    is_admin = session.get('is_admin', False)
    if not is_admin:
        ga = get_games_active()
        if not ga.get('freefire_global', False):
            flash('Este juego está desactivado temporalmente.', 'error')
            return redirect('/')
    
    # Actualizar saldo desde la base de datos
    user_id = session.get('user_db_id')
    if user_id:
        conn = get_db_connection()
        user = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            session['saldo'] = user['saldo']
        conn.close()
    
    # Obtener precios dinámicos de Free Fire Global
    if is_admin:
        prices = {}
        for row in get_all_freefire_global_prices():
            try:
                prices[int(row['id'])] = {
                    'nombre': row['nombre'],
                    'precio': row['precio'],
                    'descripcion': row.get('descripcion') if hasattr(row, 'get') else row['descripcion']
                }
            except Exception:
                continue
    else:
        prices = get_freefire_global_prices()
    # Obtener stock disponible por paquete (Free Fire Global)
    stock_freefire_global = get_pin_stock_freefire_global_optimized()
    
    # Verificar si hay una compra exitosa para mostrar (solo una vez)
    compra_exitosa = False
    compra_data = {}
    
    # Solo mostrar compra exitosa si viene del redirect POST y hay datos en sesión
    if request.args.get('compra') == 'exitosa' and 'compra_freefire_global_exitosa' in session:
        compra_exitosa = True
        compra_data = session.pop('compra_freefire_global_exitosa')  # Remover después de usar
    
    return render_template('freefire.html', 
                         user_id=session.get('id', '00000'),
                         balance=session.get('saldo', 0),
                         prices=prices,
                         stock_freefire_global=stock_freefire_global,
                         compra_exitosa=compra_exitosa,
                         is_admin=is_admin,
                         games_active=get_games_active(),
                         **compra_data)

@app.route('/validar/freefire', methods=['POST'])
def validar_freefire():
    if 'usuario' not in session:
        return redirect('/auth')

    is_admin = session.get('is_admin', False)
    if not is_admin:
        ga = get_games_active()
        if not ga.get('freefire_global', False):
            flash('Este juego está desactivado temporalmente.', 'error')
            return redirect('/')
    
    monto_id = request.form.get('monto')
    cantidad = request.form.get('cantidad', '1')
    
    if not monto_id:
        flash('Por favor selecciona un paquete', 'error')
        return redirect('/juego/freefire')
    
    try:
        monto_id = int(monto_id)
        cantidad = int(cantidad)
        
        # Validar cantidad (entre 1 y 20)
        if cantidad < 1 or cantidad > 20:
            flash('La cantidad debe estar entre 1 y 20 pines', 'error')
            return redirect('/juego/freefire')
    except ValueError:
        flash('Datos inválidos', 'error')
        return redirect('/juego/freefire')
    
    user_id = session.get('user_db_id')
    is_admin = session.get('is_admin', False)
    
    # Obtener precio dinámico de la base de datos
    if is_admin:
        precio_unitario = get_freefire_global_price_by_id_any(monto_id)
    else:
        precio_unitario = get_freefire_global_price_by_id(monto_id)
    precio_total = precio_unitario * cantidad
    
    # Obtener información del paquete usando cache
    packages_info = get_freefire_global_prices_cached()
    package_info = packages_info.get(monto_id, {})
    
    paquete_nombre = f"{package_info.get('nombre', 'Paquete')} x{cantidad}" if cantidad > 1 else package_info.get('nombre', 'Paquete')
    
    if precio_unitario == 0:
        flash('Paquete no encontrado o inactivo', 'error')
        return redirect('/juego/freefire')
    
    saldo_actual = session.get('saldo', 0)
    
    # Solo verificar saldo para usuarios normales, admin puede comprar sin saldo
    if not is_admin and saldo_actual < precio_total:
        flash(f'Saldo insuficiente. Necesitas ${precio_total:.2f} pero tienes ${saldo_actual:.2f}', 'error')
        return redirect('/juego/freefire')
    
    # Verificar stock local disponible para la cantidad solicitada
    conn = get_db_connection()
    stock_disponible = conn.execute('''
        SELECT COUNT(*) FROM pines_freefire_global 
        WHERE monto_id = ? AND usado = FALSE
    ''', (monto_id,)).fetchone()[0]
    conn.close()
    
    if stock_disponible < cantidad:
        flash(f'Stock insuficiente. Solo hay {stock_disponible} pines disponibles para este paquete.', 'error')
        return redirect('/juego/freefire')
    
    # Obtener los pines necesarios
    pines_obtenidos = []
    for i in range(cantidad):
        pin_disponible = get_available_pin_freefire_global(monto_id)
        if pin_disponible:
            pines_obtenidos.append(pin_disponible['pin_codigo'])
        else:
            # Si no se pueden obtener todos los pines, devolver los ya obtenidos al stock
            if pines_obtenidos:
                logger.warning(f"[FreeFire Global] Solo se obtuvieron {len(pines_obtenidos)}/{cantidad} PINs, devolviendo al stock")
                try:
                    conn_return = get_db_connection()
                    for pin_codigo in pines_obtenidos:
                        conn_return.execute('''
                            INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado)
                            VALUES (?, ?, FALSE)
                        ''', (monto_id, pin_codigo))
                    conn_return.commit()
                    conn_return.close()
                except Exception:
                    pass
            flash('Stock insuficiente para completar la cantidad solicitada. Intente con una cantidad menor.', 'error')
            return redirect('/juego/freefire')
    
    # Generar datos de la transacción
    import random
    import string
    numero_control = ''.join(random.choices(string.digits, k=10))
    transaccion_id = 'FFG-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    
    # Procesar la transacción
    conn = get_db_connection()
    try:
        # Solo actualizar saldo si no es admin
        if not is_admin:
            conn.execute('UPDATE usuarios SET saldo = saldo - ? WHERE id = ?', (precio_total, user_id))
        
        # Registrar la transacción
        pines_texto = '\n'.join(pines_obtenidos)
        
        # Para admin, registrar con monto negativo pero agregar etiqueta [ADMIN]
        if is_admin:
            pines_texto = f"[ADMIN - PRUEBA/GESTIÓN]\n{pines_texto}"
            monto_transaccion = -precio_total  # Registrar monto real para mostrar en historial
        else:
            monto_transaccion = -precio_total
        
        conn.execute('''
            INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, numero_control, pines_texto, transaccion_id, paquete_nombre, monto_transaccion))
        
        # Registrar en historial permanente
        _g_saldo_row = conn.execute('SELECT saldo FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        _g_saldo = _g_saldo_row['saldo'] if _g_saldo_row else 0
        registrar_historial_compra(conn, user_id, abs(monto_transaccion), paquete_nombre, pines_texto, 'compra', None, _g_saldo + abs(monto_transaccion), _g_saldo)
        
        # Actualizar gastos mensuales persistentes (para top clientes)
        if not is_admin:
            try:
                update_monthly_spending(conn, user_id, precio_total)
            except Exception:
                pass
        
        # Persistir profit (legacy)
        try:
            record_profit_for_transaction(conn, user_id, is_admin, 'freefire_global', monto_id, cantidad, precio_unitario, transaccion_id)
        except Exception:
            pass
        
        # Limitar transacciones (100 para admin, 30 para usuarios normales)
        limit = 100 if is_admin else 30
        conn.execute('''
            DELETE FROM transacciones 
            WHERE usuario_id = ? AND id NOT IN (
                SELECT id FROM (SELECT id FROM transacciones 
                WHERE usuario_id = ? 
                ORDER BY fecha DESC 
                LIMIT ?) AS keep_ids
            )
        ''', (user_id, user_id, limit))
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        # CRÍTICO: Devolver los PINs al stock si la transacción falló
        logger.error(f"[FreeFire Global] Error en transacción, devolviendo {len(pines_obtenidos)} PINs al stock: {str(e)}")
        try:
            conn_return = get_db_connection()
            for pin_codigo in pines_obtenidos:
                conn_return.execute('''
                    INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado)
                    VALUES (?, ?, FALSE)
                ''', (monto_id, pin_codigo))
            conn_return.commit()
            conn_return.close()
            logger.info(f"[FreeFire Global] {len(pines_obtenidos)} PINs devueltos al stock exitosamente")
        except Exception as return_error:
            logger.error(f"[FreeFire Global] Error devolviendo PINs al stock: {str(return_error)}")
        
        flash('Error al procesar la transacción. Los PINs han sido devueltos al stock. Intente nuevamente.', 'error')
        return redirect('/juego/freefire')
    finally:
        conn.close()
    
    # Actualizar saldo en sesión solo si no es admin
    if not is_admin:
        session['saldo'] = saldo_actual - precio_total
    
    # Registrar venta en estadísticas semanales (solo para usuarios normales)
    if not is_admin:
        register_weekly_sale('freefire_global', monto_id, package_info.get('nombre', 'Paquete'), precio_unitario, cantidad)
    
    # Guardar datos de la compra en la sesión para mostrar después del redirect
    if cantidad == 1:
        # Para un solo pin
        session['compra_freefire_global_exitosa'] = {
            'paquete_nombre': paquete_nombre,
            'monto_compra': precio_total,
            'numero_control': numero_control,
            'pin': pines_obtenidos[0],
            'transaccion_id': transaccion_id
        }
    else:
        # Para múltiples pines
        session['compra_freefire_global_exitosa'] = {
            'paquete_nombre': paquete_nombre,
            'monto_compra': precio_total,
            'numero_control': numero_control,
            'pines_list': pines_obtenidos,
            'transaccion_id': transaccion_id,
            'cantidad_comprada': cantidad
        }
    
    # Redirect para evitar reenvío del formulario (patrón POST-Redirect-GET)
    return redirect('/juego/freefire?compra=exitosa')

@app.route('/dashboard')
def dashboard():
    """Dashboard con filtros de fecha y estadísticas - Accesible para usuarios y admin"""
    if 'usuario' not in session:
        return redirect('/auth')
    
    is_admin = session.get('is_admin', False)
    
    user_id = session.get('user_db_id')
    
    # Para admin, mostrar estadísticas globales
    if is_admin:
        user_id = None  # Admin ve todas las transacciones
    elif not user_id:
        flash('Error al acceder al dashboard', 'error')
        return redirect('/')
    
    # Obtener parámetros de filtro de fecha
    fecha_inicio = request.args.get('inicio', '')
    fecha_fin = request.args.get('fin', '')
    preset = request.args.get('preset', 'hoy')  # Por defecto "hoy"
    
    # Manejar presets de fecha (usar zona horaria de Venezuela)
    from datetime import datetime, timedelta
    import pytz
    venezuela_tz = pytz.timezone('America/Caracas')
    today = datetime.now(venezuela_tz)
    
    if preset == 'hoy' or (not preset and not fecha_inicio and not fecha_fin):
        # Por defecto siempre mostrar "hoy"
        fecha_inicio = today.strftime('%Y-%m-%d')
        fecha_fin = today.strftime('%Y-%m-%d')
        preset = 'hoy'  # Asegurar que el preset esté marcado como activo
    elif preset == 'ayer':
        yesterday = today - timedelta(days=1)
        fecha_inicio = yesterday.strftime('%Y-%m-%d')
        fecha_fin = yesterday.strftime('%Y-%m-%d')
    elif preset == 'antes_ayer':
        day_before_yesterday = today - timedelta(days=2)
        fecha_inicio = day_before_yesterday.strftime('%Y-%m-%d')
        fecha_fin = day_before_yesterday.strftime('%Y-%m-%d')
    elif not fecha_inicio or not fecha_fin:
        # Si no se proporcionan fechas válidas, usar "hoy" por defecto
        fecha_inicio = today.strftime('%Y-%m-%d')
        fecha_fin = today.strftime('%Y-%m-%d')
        preset = 'hoy'
    
    # Actualizar saldo desde la base de datos y obtener transacciones
    conn = get_db_connection()
    
    if is_admin:
        # Admin ve estadísticas globales
        user = None
        
        # Obtener todas las transacciones filtradas por fecha
        transacciones_filtradas = conn.execute('''
            SELECT t.*, u.nombre, u.apellido
            FROM transacciones t
            JOIN usuarios u ON t.usuario_id = u.id
            WHERE DATE(t.fecha) BETWEEN ? AND ?
            ORDER BY t.fecha DESC
        ''', (fecha_inicio, fecha_fin)).fetchall()
        
        # Obtener todas las transacciones de Blood Striker filtradas por fecha
        transacciones_bs = conn.execute('''
            SELECT bs.*, u.nombre, u.apellido, p.nombre as paquete_nombre
            FROM transacciones_bloodstriker bs
            JOIN usuarios u ON bs.usuario_id = u.id
            JOIN precios_bloodstriker p ON bs.paquete_id = p.id
            WHERE DATE(bs.fecha) BETWEEN ? AND ? AND bs.estado = 'aprobado'
            ORDER BY bs.fecha DESC
        ''', (fecha_inicio, fecha_fin)).fetchall()
        
        # Obtener los 2 usuarios con más compras del mes actual (no del período seleccionado)
        from datetime import datetime
        current_month = datetime.now().strftime('%Y-%m')
        
        top_users = conn.execute('''
            SELECT u.nombre, u.apellido, u.correo, COUNT(*) as total_compras, SUM(ABS(t.monto)) as monto_total
            FROM transacciones t
            JOIN usuarios u ON t.usuario_id = u.id
            WHERE strftime('%Y-%m', t.fecha) = ?
            GROUP BY u.id, u.nombre, u.apellido, u.correo
            ORDER BY total_compras DESC, monto_total DESC
            LIMIT 2
        ''', (current_month,)).fetchall()
        
    else:
        # Usuario normal ve solo sus datos
        user = conn.execute('SELECT * FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            session['saldo'] = user['saldo']
        
        # Obtener transacciones del usuario filtradas por fecha
        transacciones_filtradas = conn.execute('''
            SELECT t.*, u.nombre, u.apellido
            FROM transacciones t
            JOIN usuarios u ON t.usuario_id = u.id
            WHERE t.usuario_id = ? AND DATE(t.fecha) BETWEEN ? AND ?
            ORDER BY t.fecha DESC
        ''', (user_id, fecha_inicio, fecha_fin)).fetchall()
        
        # Obtener transacciones de Blood Striker del usuario filtradas por fecha
        transacciones_bs = conn.execute('''
            SELECT bs.*, u.nombre, u.apellido, p.nombre as paquete_nombre
            FROM transacciones_bloodstriker bs
            JOIN usuarios u ON bs.usuario_id = u.id
            JOIN precios_bloodstriker p ON bs.paquete_id = p.id
            WHERE bs.usuario_id = ? AND DATE(bs.fecha) BETWEEN ? AND ? AND bs.estado = 'aprobado'
            ORDER BY bs.fecha DESC
        ''', (user_id, fecha_inicio, fecha_fin)).fetchall()
        
        top_users = []  # Los usuarios normales no ven top users
    
    conn.close()
    
    # Procesar transacciones normales
    transacciones_procesadas = []
    monto_total = 0
    
    # Obtener información de paquetes para mostrar nombres correctos
    packages_info = get_package_info_with_prices()
    bloodstriker_packages_info = get_bloodstriker_prices()
    freefire_global_packages_info = get_freefire_global_prices()
    
    for transaction in transacciones_filtradas:
        transaction_dict = dict(transaction)
        monto = abs(transaction['monto'])
        monto_total += monto
        
        # Si ya tiene paquete_nombre (persistido), usarlo directamente
        if transaction_dict.get('paquete_nombre'):
            transaction_dict['paquete'] = transaction_dict['paquete_nombre']
            # Convertir fecha y monto para consistencia
            transaction_dict['fecha'] = convert_to_venezuela_time(transaction_dict['fecha'])
            transaction_dict['monto'] = monto
            transacciones_procesadas.append(transaction_dict)
            continue
        
        # Resolver nombre del paquete solo con datos del Admin.
        # Intentar por PIN (exacto) como en el historial principal; sin fallback por monto para Free Fire.
        paquete_encontrado = False
        try:
            raw_pin = transaction_dict.get('pin') or ''
            pins_list = [p.strip() for p in (raw_pin.replace('\r','').split('\n') if '\n' in raw_pin else [raw_pin]) if p.strip()]
            pin_sample = pins_list[0] if pins_list else None
            if pin_sample:
                c2 = get_db_connection()
                try:
                    row_latam = c2.execute('SELECT monto_id FROM pines_freefire WHERE pin_codigo = ? LIMIT 1', (pin_sample,)).fetchone()
                    row_global = None if row_latam else c2.execute('SELECT monto_id FROM pines_freefire_global WHERE pin_codigo = ? LIMIT 1', (pin_sample,)).fetchone()
                finally:
                    c2.close()
                if row_latam:
                    mid = int(row_latam['monto_id'])
                    nombre = packages_info.get(mid, {}).get('nombre')
                    if nombre:
                        transaction_dict['paquete'] = nombre
                        paquete_encontrado = True
                elif row_global:
                    mid = int(row_global['monto_id'])
                    nombre = freefire_global_packages_info.get(mid, {}).get('nombre')
                    if nombre:
                        transaction_dict['paquete'] = nombre
                        paquete_encontrado = True
        except Exception:
            paquete_encontrado = False or paquete_encontrado
        
        # Blood Striker: mantener por lista de admin
        if not paquete_encontrado:
            tolerance = 0.05
            for package_id, package_info in bloodstriker_packages_info.items():
                if abs(monto - package_info['precio']) <= tolerance:
                    transaction_dict['paquete'] = package_info['nombre']
                    paquete_encontrado = True
                    break
        
        if not paquete_encontrado:
            transaction_dict['paquete'] = 'Paquete'
        
        # Convertir fecha a zona horaria de Venezuela
        transaction_dict['fecha'] = convert_to_venezuela_time(transaction_dict['fecha'])
        transaction_dict['monto'] = monto
        
        transacciones_procesadas.append(transaction_dict)
    
    # Procesar transacciones de Blood Striker aprobadas
    for bs_transaction in transacciones_bs:
        transaction_dict = {
            'fecha': convert_to_venezuela_time(bs_transaction['fecha']),
            'monto': abs(bs_transaction['monto']),
            'paquete': bs_transaction['paquete_nombre'],
            'numero_control': bs_transaction['numero_control'],
            'transaccion_id': bs_transaction['transaccion_id'],
            'pin': f"ID: {bs_transaction['player_id']}",
            'nombre': bs_transaction['nombre'],
            'apellido': bs_transaction['apellido'],
            'is_bloodstriker': True
        }
        monto_total += transaction_dict['monto']
        transacciones_procesadas.append(transaction_dict)
    
    def _dashboard_tx_key(tx):
        transaccion_id = str(tx.get('transaccion_id') or '').strip()
        if transaccion_id:
            return f"tx:{transaccion_id}"
        return "|".join([
            str(tx.get('numero_control') or '').strip(),
            str(tx.get('fecha') or '').strip(),
            str(tx.get('monto') or '').strip(),
            str(tx.get('paquete') or '').strip(),
        ])

    def _dashboard_tx_priority(tx):
        priority = 0
        if tx.get('duracion_segundos') not in (None, ''):
            priority += 2
        if tx.get('pin'):
            priority += 1
        if tx.get('paquete'):
            priority += 1
        if tx.get('is_bloodstriker'):
            priority -= 1
        return priority

    deduped_transactions = {}
    for transaction in transacciones_procesadas:
        tx_key = _dashboard_tx_key(transaction)
        existing_tx = deduped_transactions.get(tx_key)
        if existing_tx is None or _dashboard_tx_priority(transaction) > _dashboard_tx_priority(existing_tx):
            deduped_transactions[tx_key] = transaction

    transacciones_procesadas = sorted(deduped_transactions.values(), key=lambda x: x['fecha'], reverse=True)
    monto_total = round(sum(float(tx.get('monto') or 0) for tx in transacciones_procesadas), 2)
    total_transacciones = len(transacciones_procesadas)

    # Calcular días analizados
    try:
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
        dias_analizados = (fecha_fin_dt - fecha_inicio_dt).days + 1
    except:
        dias_analizados = 1

    # Cargar nombres de juegos dinámicos para clasificación
    try:
        from dynamic_games import get_all_dynamic_games
        _dyn_game_names = [g['nombre'] for g in get_all_dynamic_games()]
    except Exception:
        _dyn_game_names = []

    def extract_transaction_quantity(transaction):
        raw_name = str(transaction.get('paquete') or transaction.get('paquete_nombre') or '').strip()
        match = re.search(r'\sx(\d+)\s*$', raw_name, re.IGNORECASE)
        if match:
            try:
                return max(1, int(match.group(1)))
            except Exception:
                pass

        raw_pin = str(transaction.get('pin') or '')
        pin_lines = [
            line.strip() for line in raw_pin.replace('\r', '').split('\n')
            if line.strip() and not line.strip().startswith('[') and not line.strip().startswith('ID:')
        ]
        if len(pin_lines) > 1:
            return len(pin_lines)
        return 1

    def infer_item_from_package_name(nombre_paquete, is_bloodstriker=False):
        raw_name = str(nombre_paquete or '').strip()
        lower_name = raw_name.lower()
        if is_bloodstriker or '🪙' in raw_name or 'blood' in lower_name:
            return 'Blood Striker'
        for dg_name in _dyn_game_names:
            if raw_name.startswith(dg_name + ' - ') or raw_name == dg_name:
                return dg_name
        if '💎' in raw_name or 'free fire' in lower_name or 'ff id' in lower_name or 'tarjeta' in lower_name:
            if any(x in raw_name for x in ['110 💎', '341 💎', '572 💎', '1.166 💎', '2.376 💎', '6.138 💎']) or 'tarjeta' in lower_name:
                return 'Free Fire LATAM'
            return 'Free Fire'
        return 'Otros'

    def extract_base_package_amount(nombre_paquete):
        raw_name = str(nombre_paquete or '').strip()
        if not raw_name:
            return None

        amount_match = re.search(r'(\d{1,3}(?:[\.,]\d{3})+|\d+)', raw_name)
        if not amount_match:
            return None

        raw_amount = amount_match.group(1)
        normalized_amount = re.sub(r'[^\d]', '', raw_amount)
        if not normalized_amount:
            return None
        try:
            return int(normalized_amount)
        except Exception:
            return None

    def normalize_package_display_name(nombre_paquete, item_name):
        raw_name = str(nombre_paquete or '').strip()
        raw_name = re.sub(r'\sx\d+\s*$', '', raw_name, flags=re.IGNORECASE)
        raw_name = re.sub(r'\s*\([^)]*\)\s*$', '', raw_name)
        raw_name = raw_name.replace('"', '').replace("'", '').replace('“', '').replace('”', '').strip()
        if ' - ' in raw_name:
            _, raw_name = raw_name.split(' - ', 1)
            raw_name = raw_name.strip()

        lower_name = raw_name.lower()
        if item_name in ('Free Fire', 'Free Fire LATAM', 'Blood Striker'):
            if any(word in lower_name for word in ['tarjeta', 'pase', 'elite', 'cofre']):
                return raw_name
            base_amount = extract_base_package_amount(raw_name)
            if base_amount is not None:
                return str(base_amount)
        return raw_name or 'Paquete'

    def package_order_key(package_name):
        normalized = str(package_name or '').strip().lower()
        base_amount = extract_base_package_amount(normalized)
        if base_amount is not None:
            return (0, base_amount, normalized)
        exact_number = re.fullmatch(r'(\d+)', normalized)
        if exact_number:
            return (0, int(exact_number.group(1)), normalized)
        any_number = re.search(r'(\d+)', normalized)
        if any_number:
            return (1, int(any_number.group(1)), normalized)
        return (2, 0, normalized)

    def game_order_key(item_name):
        name = (item_name or '').strip().lower()
        if name == 'free fire':
            return 0
        if name == 'free fire latam':
            return 1
        if name == 'blood striker':
            return 2
        return 3

    for transaction in transacciones_procesadas:
        tx_quantity = extract_transaction_quantity(transaction)
        tx_item = infer_item_from_package_name(transaction.get('paquete', ''), transaction.get('is_bloodstriker', False))
        tx_package = normalize_package_display_name(transaction.get('paquete', 'Desconocido'), tx_item)
        transaction['dashboard_quantity'] = tx_quantity
        transaction['dashboard_item'] = tx_item
        transaction['dashboard_package'] = tx_package

    stats_por_juego = {}
    for transaction in transacciones_procesadas:
        juego = transaction.get('dashboard_item') or 'Otros'
        if juego not in stats_por_juego:
            stats_por_juego[juego] = {'cantidad': 0, 'monto': 0}
        stats_por_juego[juego]['cantidad'] += int(transaction.get('dashboard_quantity') or 1)
        stats_por_juego[juego]['monto'] += float(transaction.get('monto') or 0)

    # Serie temporal por día (para gráfico) y gasto del día seleccionado
    from collections import OrderedDict
    try:
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
    except:
        from datetime import datetime as _dt
        fecha_inicio_dt = _dt.now()
        fecha_fin_dt = _dt.now()

    dias = []
    current = fecha_inicio_dt
    while current <= fecha_fin_dt:
        dias.append(current.strftime('%Y-%m-%d'))
        from datetime import timedelta as _td
        current += _td(days=1)
    if len(dias) < 7:
        from datetime import timedelta as _td
        faltan = 7 - len(dias)
        prepend = []
        cur = fecha_inicio_dt - _td(days=1)
        for _ in range(faltan):
            prepend.append(cur.strftime('%Y-%m-%d'))
            cur -= _td(days=1)
        dias = list(reversed(prepend)) + dias

    serie_map = OrderedDict((d, 0.0) for d in dias)
    compras_por_dia_paquete = {d: {} for d in dias}
    for transaction in transacciones_procesadas:
        fecha_str = str(transaction['fecha']).split(' ')[0]
        if fecha_str not in serie_map:
            continue

        serie_map[fecha_str] += float(transaction.get('monto') or 0)
        item_name = transaction.get('dashboard_item') or 'Otros'
        package_name = transaction.get('dashboard_package') or 'Paquete'
        aggregate_key = f"{item_name}||{package_name}"
        if aggregate_key not in compras_por_dia_paquete[fecha_str]:
            compras_por_dia_paquete[fecha_str][aggregate_key] = {
                'categoria': 'Juegos',
                'item': item_name,
                'paquete': package_name,
                'cantidad': 0,
            }
        compras_por_dia_paquete[fecha_str][aggregate_key]['cantidad'] += int(transaction.get('dashboard_quantity') or 1)

    series_labels = list(serie_map.keys())
    series_values = [round(v, 2) for v in serie_map.values()]

    gasto_dia = 0.0
    if fecha_fin in serie_map:
        gasto_dia = round(serie_map[fecha_fin], 2)

    compras_paquete_counter = compras_por_dia_paquete.get(fecha_fin, {}) or {}
    compras_paquete = sorted(
        compras_paquete_counter.values(),
        key=lambda row: (game_order_key(row.get('item')), str(row.get('item', '')).lower(), package_order_key(row.get('paquete')))
    )

    compras_por_dia_detalle = {d: [] for d in dias}
    for day in dias:
        rows = list((compras_por_dia_paquete.get(day, {}) or {}).values())
        rows.sort(key=lambda row: (game_order_key(row.get('item')), str(row.get('item', '')).lower(), package_order_key(row.get('paquete'))))
        compras_por_dia_detalle[day] = rows

    # Valores por defecto (si no existen tablas de stock/solicitudes)
    items_stock = 0
    items_solicitud = 0

    # Obtener contador de notificaciones de cartera para usuarios normales
    wallet_notification_count = 0
    if not is_admin and user_id:
        wallet_notification_count = get_unread_wallet_credits_count(user_id)
    
    # Obtener contador de notificaciones de noticias
    news_notification_count = 0
    if user_id:
        news_notification_count = get_unread_news_count(user_id)
    
    # Contador total de usuarios (para estadísticas de admin)
    total_users = 0
    try:
        c_count = get_db_connection()
        total_users = c_count.execute('SELECT COUNT(*) FROM usuarios').fetchone()[0]
    finally:
        try:
            c_count.close()
        except Exception:
            pass
    
    return render_template('dashboard.html', 
                         user=user,
                         transacciones=transacciones_procesadas,
                         monto_total=monto_total,
                         total_transacciones=total_transacciones,
                         stats_por_juego=stats_por_juego,
                         dias_analizados=dias_analizados,
                         inicio=fecha_inicio,
                         fin=fecha_fin,
                         user_id=session.get('id', '00000'),
                         balance=session.get('saldo', 0),
                         is_admin=is_admin,
                         top_users=top_users,
                         wallet_notification_count=wallet_notification_count,
                         news_notification_count=news_notification_count,
                         series_labels=series_labels,
                         series_values=series_values,
                         gasto_dia=gasto_dia,
                         items_stock=items_stock,
                         items_solicitud=items_solicitud,
                         compras_paquete=compras_paquete,
                         compras_paquete_map=compras_paquete_counter,
                         compras_por_dia_paquete=compras_por_dia_paquete,
                         compras_por_dia_detalle=compras_por_dia_detalle,
                         games_active=get_games_active(),
                         total_users=total_users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/auth')

# ============= API SIMPLE DE CONEXIÓN =============

@app.route('/api.php', methods=['GET'])
def api_simple_endpoint():
    """
    API Simple de Conexión para Revendedores51
    
    Formato: /api.php?action=recarga&usuario=email&clave=password&tipo=recargaPinFreefire&monto=1&numero=1
    
    Parámetros:
    - action: Siempre debe ser "recarga"
    - usuario: Email del usuario
    - clave: Contraseña del usuario
    - tipo: Tipo de recarga (recargaPinFreefire)
    - monto: ID del paquete (1-9)
    - numero: Cantidad de PINs (por defecto 1, máximo 10)
    """
    
    try:
        # Obtener parámetros
        action = request.args.get('action', '').lower()
        usuario = request.args.get('usuario', '')
        clave = request.args.get('clave', '')
        tipo = request.args.get('tipo', '').lower()
        monto = request.args.get('monto', '1')
        numero = request.args.get('numero', '1')
        
        # Validar parámetros básicos
        if not all([action, usuario, clave, tipo]):
            return jsonify({
                'status': 'error',
                'code': '400',
                'message': 'Parámetros requeridos: action, usuario, clave, tipo'
            }), 400
        
        # Validar action
        if action != 'recarga':
            return jsonify({
                'status': 'error',
                'code': '400',
                'message': 'Action debe ser "recarga"'
            }), 400
        
        # Validar tipo
        if tipo != 'recargapinfreefire':
            return jsonify({
                'status': 'error',
                'code': '400',
                'message': 'Tipo debe ser "recargaPinFreefire"'
            }), 400
        
        # Validar y convertir monto (package_id)
        try:
            package_id = int(monto)
            if package_id < 1 or package_id > 9:
                return jsonify({
                    'status': 'error',
                    'code': '400',
                    'message': 'Monto debe estar entre 1 y 9'
                }), 400
        except ValueError:
            return jsonify({
                'status': 'error',
                'code': '400',
                'message': 'Monto debe ser un número válido'
            }), 400
        
        # Validar y convertir numero (quantity)
        try:
            quantity = int(numero) if numero else 1
            if quantity < 1 or quantity > 10:
                return jsonify({
                    'status': 'error',
                    'code': '400',
                    'message': 'Numero debe estar entre 1 y 10'
                }), 400
        except ValueError:
            return jsonify({
                'status': 'error',
                'code': '400',
                'message': 'Numero debe ser un número válido'
            }), 400
        
        # Autenticar usuario
        user = get_user_by_email(usuario)
        
        if not user or not verify_password(clave, user['contraseña']):
            return jsonify({
                'status': 'error',
                'code': '401',
                'message': 'Credenciales incorrectas'
            }), 401
        
        # Obtener información del paquete
        packages_info = get_package_info_with_prices()
        package_info = packages_info.get(package_id)
        
        if not package_info:
            return jsonify({
                'status': 'error',
                'code': '404',
                'message': 'Paquete no encontrado'
            }), 404
        
        precio_unitario = package_info['precio']
        precio_total = precio_unitario * quantity
        saldo_actual = user['saldo']
        
        # Verificar saldo suficiente
        if saldo_actual < precio_total:
            return jsonify({
                'status': 'error',
                'code': '402',
                'message': f'Saldo insuficiente. Necesitas ${precio_total:.2f} pero tienes ${saldo_actual:.2f}'
            }), 402
        
        # Usar pin manager para obtener PINs
        pin_manager = create_pin_manager(DATABASE)
        
        if quantity == 1:
            # Para un solo PIN
            result = pin_manager.request_pin(package_id)
            
            if result.get('status') != 'success':
                return jsonify({
                    'status': 'error',
                    'code': '503',
                    'message': f'Sin stock disponible para este paquete'
                }), 503
            
            pin_code = result.get('pin_code')
            pins_list = [pin_code]
        else:
            # Para múltiples PINs
            result = pin_manager.request_multiple_pins(package_id, quantity)
            
            if result.get('status') not in ['success', 'partial_success']:
                return jsonify({
                    'status': 'error',
                    'code': '503',
                    'message': f'Error al obtener PINs: {result.get("message", "Sin stock disponible")}'
                }), 503
            
            pines_data = result.get('pins', [])
            pins_list = [pin['pin_code'] for pin in pines_data]
            
            if len(pins_list) < quantity:
                # Ajustar cantidad y precio si no se obtuvieron todos los PINs
                quantity = len(pins_list)
                precio_total = precio_unitario * quantity
        
        # Descontar saldo
        conn = get_db_connection()
        nuevo_saldo = saldo_actual - precio_total
        conn.execute('UPDATE usuarios SET saldo = ? WHERE id = ?', (nuevo_saldo, user['id']))
        
        # Crear registro de transacción
        pins_texto = '\n'.join(pins_list)
        
        # Generar datos de la transacción
        numero_control = ''.join(random.choices(string.digits, k=10))
        transaccion_id = 'API-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        
        # Calcular nombre del paquete tal como en Admin y con cantidad
        paquete_nombre = f"{package_info['nombre']} x{quantity}" if quantity > 1 else package_info['nombre']
        
        conn.execute('''
            INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user['id'], numero_control, pins_texto, transaccion_id, paquete_nombre, -precio_total))
        
        # Registrar en historial permanente
        registrar_historial_compra(conn, user['id'], precio_total, paquete_nombre, pins_texto, 'compra', None, saldo_actual, nuevo_saldo)
        
        # Persistir profit (legacy) también para compras vía API
        try:
            admin_ids_env = os.environ.get('ADMIN_USER_IDS', '').strip()
            admin_emails_env = os.environ.get('ADMIN_EMAILS', '').strip()
            single_admin_email = os.environ.get('ADMIN_EMAIL', '').strip()
            admin_ids = [int(x.strip()) for x in admin_ids_env.split(',') if x.strip().isdigit()]
            admin_emails = [x.strip() for x in admin_emails_env.split(',') if x.strip()]
            if single_admin_email and single_admin_email not in admin_emails:
                admin_emails.append(single_admin_email)
            is_admin_user = (user['id'] in admin_ids) or (user['correo'] in admin_emails)
            record_profit_for_transaction(conn, user['id'], is_admin_user, 'freefire_latam', package_id, quantity, precio_unitario, transaccion_id)
        except Exception:
            pass
        
        # Limitar transacciones (100 para admin, 30 para usuarios normales)
        limit = 100 if is_admin_user else 30
        conn.execute('''
            DELETE FROM transacciones 
            WHERE usuario_id = ? AND id NOT IN (
                SELECT id FROM (SELECT id FROM transacciones 
                WHERE usuario_id = ? 
                ORDER BY fecha DESC 
                LIMIT ?) AS keep_ids
            )
        ''', (user['id'], user['id'], limit))
        
        conn.commit()
        conn.close()
        
        # Preparar respuesta exitosa
        response_data = {
            'status': 'success',
            'code': '200',
            'message': f'{"PIN obtenido" if quantity == 1 else f"{quantity} PINs obtenidos"} exitosamente',
            'data': {
                'usuario': f"{user['nombre']} {user['apellido']}",
                'email': user['correo'],
                'paquete': package_info['nombre'],
                'precio_unitario': float(precio_unitario),
                'cantidad': quantity,
                'precio_total': float(precio_total),
                'saldo_anterior': float(saldo_actual),
                'saldo_nuevo': float(nuevo_saldo),
                'numero_control': numero_control,
                'transaccion_id': transaccion_id,
                'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Agregar PIN(s) a la respuesta
        if quantity == 1:
            response_data['data']['pin'] = pins_list[0]
        else:
            response_data['data']['pines'] = pins_list
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'code': '500',
            'message': f'Error interno del servidor: {str(e)}'
        }), 500

@app.route('/api.php', methods=['POST'])
def api_simple_endpoint_post():
    """Endpoint POST para la API simple (redirige al GET)"""
    return jsonify({
        'status': 'error',
        'code': '405',
        'message': 'Usar método GET con parámetros en la URL'
    }), 405


# ============= API v1: RECARGA AUTOMÁTICA DESDE WEB A =============

@app.route('/api/v1/ejecutar-recarga', methods=['POST'])
def api_v1_ejecutar_recarga():
    """
    Endpoint seguro para ejecutar una recarga de Free Fire desde Web A (Inefable Store).

    Headers requeridos:
        Authorization: Bearer <REVENDEDORES_API_TOKEN>

    Body JSON:
        {
            "player_id": "123456789",   # ID del jugador en Free Fire
            "package_id": 1,            # ID del paquete (1-9 según tabla de precios)
            "order_id": 42              # ID del pedido en Web A (para trazabilidad)
        }

    Respuesta exitosa:
        { "ok": true, "pin": "...", "package": "...", "order_id": 42 }

    Respuesta de error:
        { "ok": false, "error": "..." }
    """
    # --- Autenticación por Bearer Token ---
    expected_token = os.environ.get('REVENDEDORES_API_TOKEN', '').strip()
    if not expected_token:
        return jsonify({'ok': False, 'error': 'API no configurada (token vacío)'}), 503

    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({'ok': False, 'error': 'Authorization header requerido (Bearer token)'}), 401

    provided_token = auth_header[len('Bearer '):].strip()
    # Comparación segura contra timing attacks
    if not hmac_module.compare_digest(provided_token, expected_token):
        return jsonify({'ok': False, 'error': 'Token inválido'}), 401

    # --- Validar body ---
    data = request.get_json(silent=True) or {}
    player_id = str(data.get('player_id') or '').strip()
    order_id = data.get('order_id')

    try:
        package_id = int(data.get('package_id') or 0)
    except (ValueError, TypeError):
        return jsonify({'ok': False, 'error': 'package_id debe ser un número entero'}), 400

    if not player_id:
        return jsonify({'ok': False, 'error': 'player_id es requerido'}), 400

    if package_id < 1 or package_id > 9:
        return jsonify({'ok': False, 'error': 'package_id debe estar entre 1 y 9'}), 400

    # --- Obtener información del paquete ---
    # WEBB_FF_TIPO: 'freefire_id' (default), 'freefire_global', 'latam'
    ff_tipo = os.environ.get('WEBB_FF_TIPO', 'freefire_id').strip().lower()
    try:
        conn_tmp = get_db_connection()
        if ff_tipo == 'freefire_global':
            tabla_precios = 'precios_freefire_global'
        elif ff_tipo == 'latam':
            tabla_precios = 'precios_paquetes'
        else:
            tabla_precios = 'precios_freefire_id'
        row = conn_tmp.execute(
            f'SELECT id, nombre, precio FROM {tabla_precios} WHERE id = ? AND activo = TRUE',
            (package_id,)
        ).fetchone()
        conn_tmp.close()
        if not row:
            return jsonify({'ok': False, 'error': f'Paquete {package_id} no encontrado o inactivo en {tabla_precios}'}), 404
        package_info = {'nombre': row['nombre'], 'precio': row['precio']}
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Error obteniendo paquetes: {str(e)}'}), 500

    # --- Obtener PIN del stock ---
    # freefire_id y latam usan pines_freefire; freefire_global usa pines_freefire_global
    try:
        pin_manager = create_pin_manager(DATABASE)
        if ff_tipo == 'freefire_global':
            result = pin_manager.request_pin_global(package_id)
        else:
            result = pin_manager.request_pin(package_id)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Error en pin_manager: {str(e)}'}), 500

    if result.get('status') != 'success':
        return jsonify({
            'ok': False,
            'error': f'Sin stock disponible para el paquete {package_id} ({package_info.get("nombre", "")})'
        }), 503

    pin_code = result.get('pin_code', '')
    if not pin_code:
        return jsonify({'ok': False, 'error': 'Pin obtenido vacío, contacte al administrador'}), 500

    # --- Registrar la transacción en la base de datos de Web B ---
    transaccion_id = f'INEFABLE-{order_id or "X"}-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    try:
        conn = get_db_connection()
        # Usar el usuario_id del primer admin para que aparezca en el historial del dashboard
        admin_row = conn.execute("SELECT id FROM usuarios ORDER BY id ASC LIMIT 1").fetchone()
        admin_uid = int(admin_row[0]) if admin_row else 1
        # numero_control incluye player_id para trazabilidad en el historial
        numero_control = f'INEFABLE-{player_id}'
        paquete_nombre = package_info.get('nombre', f'Paquete {package_id}')
        conn.execute(
            '''INSERT INTO transacciones (usuario_id, numero_control, pin, transaccion_id, paquete_nombre, monto)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (admin_uid, numero_control, pin_code, transaccion_id, paquete_nombre, -float(package_info.get('precio', 0)))
        )
        # Registrar en historial permanente (antes de cerrar conn)
        _api_precio = float(package_info.get('precio', 0))
        registrar_historial_compra(conn, admin_uid, _api_precio, paquete_nombre, f"ID: {player_id}", 'compra', None, 0, 0)
        conn.commit()
        conn.close()
    except Exception as e:
        # No bloquear la respuesta si falla el registro
        logger.error(f'[api/v1/ejecutar-recarga] Error registrando transacción: {e}')

    logger.info(f'[api/v1/ejecutar-recarga] Recarga exitosa - order_id={order_id} player_id={player_id} package={package_id} pin={pin_code[:8]}...')

    return jsonify({
        'ok': True,
        'pin': pin_code,
        'package': package_info.get('nombre', ''),
        'package_id': package_id,
        'player_id': player_id,
        'order_id': order_id,
        'transaccion_id': transaccion_id,
    })

# ==================== ENDPOINTS PARA PESTAÑA COSTO ====================

# Identificar admin por correo
ADMIN_CORREO = os.environ.get('ADMIN_EMAIL', 'admin@inefable.com')

def _get_admin_user_id():
    conn = get_db_connection()
    row = conn.execute('SELECT id FROM usuarios WHERE correo = ?', (ADMIN_CORREO,)).fetchone()
    conn.close()
    return row['id'] if row else None

@app.route('/admin/costos/admin-summary')
def admin_costos_summary():
    """Resumen de gastos del admin: hoy, semanal, mensual"""
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Acceso denegado'})
    try:
        from datetime import datetime, timedelta, timezone
        admin_id = _get_admin_user_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Admin no encontrado'})

        utc_now = datetime.now(timezone.utc)
        local_now = utc_now - timedelta(hours=4)
        today_str = local_now.strftime('%Y-%m-%d')
        # Inicio de semana (lunes)
        weekday = local_now.weekday()  # 0=lunes
        week_start = (local_now - timedelta(days=weekday)).strftime('%Y-%m-%d')
        # Inicio de mes
        month_start = local_now.replace(day=1).strftime('%Y-%m-%d')

        conn = get_db_connection()

        # Gasto hoy (desde historial permanente)
        r_hoy = conn.execute(
            "SELECT COALESCE(SUM(monto), 0) as total, COUNT(*) as cnt FROM historial_compras WHERE usuario_id = ? AND DATE(fecha, '-4 hours') = ?",
            (admin_id, today_str)
        ).fetchone()

        # Gasto semanal
        r_sem = conn.execute(
            "SELECT COALESCE(SUM(monto), 0) as total, COUNT(*) as cnt FROM historial_compras WHERE usuario_id = ? AND DATE(fecha, '-4 hours') >= ?",
            (admin_id, week_start)
        ).fetchone()

        # Gasto mensual
        r_mes = conn.execute(
            "SELECT COALESCE(SUM(monto), 0) as total, COUNT(*) as cnt FROM historial_compras WHERE usuario_id = ? AND DATE(fecha, '-4 hours') >= ?",
            (admin_id, month_start)
        ).fetchone()

        conn.close()

        return jsonify({
            'success': True,
            'gasto_hoy': r_hoy['total'],
            'compras_hoy': r_hoy['cnt'],
            'gasto_semanal': r_sem['total'],
            'compras_semanal': r_sem['cnt'],
            'gasto_mensual': r_mes['total'],
            'compras_mensual': r_mes['cnt']
        })
    except Exception as e:
        print(f"Error en admin_costos_summary: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/costos/<day>')
def admin_get_costos_day(day):
    """Obtener compras de usuarios para un día específico (today, yesterday, daybefore o YYYY-MM-DD)"""
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Acceso denegado'})
    
    try:
        from datetime import datetime, timedelta, timezone
        # Las fechas en BD son UTC (CURRENT_TIMESTAMP). Ajustar a UTC-4 para Venezuela.
        utc_now = datetime.now(timezone.utc)
        local_now = utc_now - timedelta(hours=4)
        today_str = local_now.strftime('%Y-%m-%d')

        if day == 'today':
            target_str = today_str
        elif day == 'yesterday':
            target_str = (local_now - timedelta(days=1)).strftime('%Y-%m-%d')
        elif day == 'daybefore':
            target_str = (local_now - timedelta(days=2)).strftime('%Y-%m-%d')
        else:
            try:
                datetime.strptime(day, '%Y-%m-%d')
                target_str = day
            except ValueError:
                return jsonify({'success': False, 'error': 'Formato inválido'})

        conn = get_db_connection()

        # Filtro admin_only: si viene ?admin_only=1, solo mostrar compras del admin
        admin_only = request.args.get('admin_only') == '1'
        admin_id = _get_admin_user_id() if admin_only else None
        user_filter_h = f'AND h.usuario_id = {admin_id}' if admin_id else ''
        user_filter_cb = f'AND cb.usuario_id = {admin_id}' if admin_id else ''
        user_filter_fi = f'AND fi.usuario_id = {admin_id}' if admin_id else ''
        user_filter_rb = f'AND rb.usuario_id = {admin_id}' if admin_id else ''

        # 1. Compras desde historial permanente (no se borra con transacciones)
        q_compras = f'''
            SELECT 
                h.id, h.usuario_id, h.monto, h.fecha,
                h.paquete_nombre, h.pin, h.duracion_segundos,
                u.nombre as usuario_nombre, u.apellido as usuario_apellido, 
                u.correo as usuario_correo, u.telefono as usuario_telefono,
                u.saldo as saldo_actual,
                h.tipo_evento,
                h.saldo_antes as h_saldo_antes, h.saldo_despues as h_saldo_despues
            FROM historial_compras h
            JOIN usuarios u ON h.usuario_id = u.id
            WHERE DATE(h.fecha, '-4 hours') = ? {user_filter_h}
        '''

        # 2. Créditos añadidos (creditos_billetera)
        q_creditos = f'''
            SELECT 
                cb.id, cb.usuario_id, cb.monto, cb.fecha,
                'Crédito añadido' as paquete_nombre, '' as pin, NULL as duracion_segundos,
                u.nombre as usuario_nombre, u.apellido as usuario_apellido,
                u.correo as usuario_correo, u.telefono as usuario_telefono,
                u.saldo as saldo_actual,
                'credito' as tipo_evento,
                NULL as h_saldo_antes, NULL as h_saldo_despues
            FROM creditos_billetera cb
            JOIN usuarios u ON cb.usuario_id = u.id
            WHERE DATE(cb.fecha, '-4 hours') = ? {user_filter_cb}
        '''

        # 3. Recargas FF ID fallidas/rechazadas (reembolso)
        q_fallidas = f'''
            SELECT 
                fi.id, fi.usuario_id, fi.monto, fi.fecha,
                'FF ID Fallida (reembolso)' as paquete_nombre, 
                fi.player_id as pin, NULL as duracion_segundos,
                u.nombre as usuario_nombre, u.apellido as usuario_apellido,
                u.correo as usuario_correo, u.telefono as usuario_telefono,
                u.saldo as saldo_actual,
                'reembolso' as tipo_evento,
                NULL as h_saldo_antes, NULL as h_saldo_despues
            FROM transacciones_freefire_id fi
            JOIN usuarios u ON fi.usuario_id = u.id
            WHERE DATE(fi.fecha, '-4 hours') = ? AND fi.estado = 'rechazado' {user_filter_fi}
        '''

        # 4. Recargas Binance completadas
        q_binance = f'''
            SELECT 
                rb.id, rb.usuario_id, (rb.monto_solicitado + rb.bonus) as monto, 
                COALESCE(rb.fecha_completada, rb.fecha_creacion) as fecha,
                'Recarga Binance' as paquete_nombre, 
                rb.codigo_referencia as pin, NULL as duracion_segundos,
                u.nombre as usuario_nombre, u.apellido as usuario_apellido,
                u.correo as usuario_correo, u.telefono as usuario_telefono,
                u.saldo as saldo_actual,
                'binance' as tipo_evento,
                NULL as h_saldo_antes, NULL as h_saldo_despues
            FROM recargas_binance rb
            JOIN usuarios u ON rb.usuario_id = u.id
            WHERE DATE(COALESCE(rb.fecha_completada, rb.fecha_creacion), '-4 hours') = ? AND rb.estado = 'completada' {user_filter_rb}
        '''

        # UNION ALL de las 4 fuentes, ordenado por fecha desc
        query = f'''
            SELECT * FROM (
                {q_compras}
                UNION ALL
                {q_creditos}
                UNION ALL
                {q_fallidas}
                UNION ALL
                {q_binance}
            ) ORDER BY fecha DESC
        '''

        transactions = conn.execute(query, (target_str, target_str, target_str, target_str)).fetchall()
        
        purchases = []
        for trans in transactions:
            fecha_str = str(trans['fecha'])
            tipo = trans['tipo_evento']
            monto_raw = trans['monto']
            
            if tipo == 'credito':
                saldo_anterior_row = conn.execute(
                    'SELECT saldo_anterior FROM creditos_billetera WHERE id = ?', (trans['id'],)
                ).fetchone()
                saldo_antes = (saldo_anterior_row['saldo_anterior'] or 0) if saldo_anterior_row else 0
                saldo_despues = round(saldo_antes + monto_raw, 2)
                monto_display = monto_raw
                tipo_compra = '💳 Crédito añadido'
            elif tipo == 'binance':
                saldo_despues = trans['saldo_actual'] or 0
                saldo_antes = round(saldo_despues - monto_raw, 2)
                monto_display = monto_raw
                tipo_compra = '🪙 Recarga Binance'
            elif tipo == 'reembolso':
                saldo_posterior = conn.execute(
                    'SELECT COALESCE(SUM(monto), 0) FROM transacciones WHERE usuario_id = ? AND fecha > ?',
                    (trans['usuario_id'], fecha_str)
                ).fetchone()[0] or 0
                saldo_actual = trans['saldo_actual'] or 0
                saldo_despues = saldo_actual - saldo_posterior
                saldo_antes = saldo_despues  # net zero
                monto_display = abs(monto_raw)
                tipo_compra = '⚠️ FF ID Fallida (reembolso)'
            else:
                # Compra desde historial_compras: usar saldo guardado directamente
                saldo_antes = trans['h_saldo_antes'] or 0
                saldo_despues = trans['h_saldo_despues'] or 0
                monto_display = abs(monto_raw)
                tipo_compra = 'Recarga por ID' if 'ID:' in (trans['pin'] or '') else 'PIN'
            
            duracion = trans['duracion_segundos']
            purchases.append({
                'id': trans['id'],
                'usuario_id': trans['usuario_id'],
                'usuario_nombre': trans['usuario_nombre'],
                'usuario_apellido': trans['usuario_apellido'],
                'usuario_correo': trans['usuario_correo'],
                'usuario_telefono': trans['usuario_telefono'] or '',
                'monto': round(monto_display, 2),
                'fecha': fecha_str,
                'paquete_nombre': trans['paquete_nombre'] or 'Compra',
                'tipo_compra': tipo_compra,
                'tipo_evento': tipo,
                'saldo_antes': round(saldo_antes, 2),
                'saldo_despues': round(saldo_despues, 2),
                'pin_codigo': trans['pin'],
                'player_id': None,
                'duracion_segundos': duracion
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'purchases': purchases,
            'date': target_str
        })
        
    except Exception as e:
        print(f"Error en admin_get_costos_day: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/costos/<day>/user/<int:user_id>')
def admin_get_user_costos_detail(day, user_id):
    """Obtener historial de compras de un usuario para un día específico"""
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Acceso denegado'})
    
    try:
        from datetime import datetime, timedelta, timezone
        utc_now = datetime.now(timezone.utc)
        local_now = utc_now - timedelta(hours=4)
        today_str = local_now.strftime('%Y-%m-%d')

        if day == 'today':
            target_str = today_str
        elif day == 'yesterday':
            target_str = (local_now - timedelta(days=1)).strftime('%Y-%m-%d')
        elif day == 'daybefore':
            target_str = (local_now - timedelta(days=2)).strftime('%Y-%m-%d')
        else:
            try:
                datetime.strptime(day, '%Y-%m-%d')
                target_str = day
            except ValueError:
                return jsonify({'success': False, 'error': 'Formato inválido'})
        
        conn = get_db_connection()
        
        user = conn.execute('SELECT * FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if not user:
            conn.close()
            return jsonify({'success': False, 'error': 'Usuario no encontrado'})
        
        # Compras desde historial permanente
        q1 = '''
            SELECT h.id, h.monto, h.fecha, h.paquete_nombre, h.pin, h.tipo_evento, h.duracion_segundos,
                h.saldo_antes as h_saldo_antes, h.saldo_despues as h_saldo_despues
            FROM historial_compras h
            WHERE h.usuario_id = ? AND DATE(h.fecha, '-4 hours') = ?
        '''
        # Créditos
        q2 = '''
            SELECT cb.id, cb.monto, cb.fecha, 'Crédito añadido' as paquete_nombre, '' as pin, 'credito' as tipo_evento, NULL as duracion_segundos,
                NULL as h_saldo_antes, NULL as h_saldo_despues
            FROM creditos_billetera cb
            WHERE cb.usuario_id = ? AND DATE(cb.fecha, '-4 hours') = ?
        '''
        # Reembolsos FF ID
        q3 = '''
            SELECT fi.id, fi.monto, fi.fecha, 'FF ID Fallida (reembolso)' as paquete_nombre, fi.player_id as pin, 'reembolso' as tipo_evento, NULL as duracion_segundos,
                NULL as h_saldo_antes, NULL as h_saldo_despues
            FROM transacciones_freefire_id fi
            WHERE fi.usuario_id = ? AND DATE(fi.fecha, '-4 hours') = ? AND fi.estado = 'rechazado'
        '''
        # Recargas Binance completadas
        q4 = '''
            SELECT rb.id, (rb.monto_solicitado + rb.bonus) as monto, COALESCE(rb.fecha_completada, rb.fecha_creacion) as fecha,
                'Recarga Binance' as paquete_nombre, rb.codigo_referencia as pin, 'binance' as tipo_evento, NULL as duracion_segundos,
                NULL as h_saldo_antes, NULL as h_saldo_despues
            FROM recargas_binance rb
            WHERE rb.usuario_id = ? AND DATE(COALESCE(rb.fecha_completada, rb.fecha_creacion), '-4 hours') = ? AND rb.estado = 'completada'
        '''
        query = f'SELECT * FROM ({q1} UNION ALL {q2} UNION ALL {q3} UNION ALL {q4}) ORDER BY fecha DESC'
        
        transactions = conn.execute(query, (user_id, target_str, user_id, target_str, user_id, target_str, user_id, target_str)).fetchall()
        
        saldo_actual = user['saldo'] or 0
        
        purchases = []
        for trans in transactions:
            fecha_str = str(trans['fecha'])
            tipo = trans['tipo_evento']
            monto_raw = trans['monto']
            
            if tipo == 'credito':
                saldo_ant_row = conn.execute(
                    'SELECT saldo_anterior FROM creditos_billetera WHERE id = ?', (trans['id'],)
                ).fetchone()
                saldo_antes = (saldo_ant_row['saldo_anterior'] or 0) if saldo_ant_row else 0
                saldo_despues = round(saldo_antes + monto_raw, 2)
                monto_display = monto_raw
            elif tipo == 'binance':
                saldo_despues = saldo_actual
                saldo_antes = round(saldo_despues - monto_raw, 2)
                monto_display = monto_raw
            elif tipo == 'reembolso':
                saldo_posterior = conn.execute(
                    'SELECT COALESCE(SUM(monto), 0) FROM transacciones WHERE usuario_id = ? AND fecha > ?',
                    (user_id, fecha_str)
                ).fetchone()[0] or 0
                saldo_despues = saldo_actual - saldo_posterior
                saldo_antes = saldo_despues  # net zero
                monto_display = abs(monto_raw)
            else:
                # Compra desde historial_compras: usar saldo guardado
                saldo_antes = trans['h_saldo_antes'] or 0
                saldo_despues = trans['h_saldo_despues'] or 0
                monto_display = abs(monto_raw)
            
            purchases.append({
                'id': trans['id'],
                'monto': round(monto_display, 2),
                'fecha_hora': fecha_str,
                'tipo': tipo,
                'paquete_nombre': trans['paquete_nombre'] or 'Compra',
                'saldo_antes': round(saldo_antes, 2),
                'saldo_despues': round(saldo_despues, 2),
                'pin_codigo': trans['pin'],
                'player_id': None,
                'duracion_segundos': trans['duracion_segundos']
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'usuario': {
                'id': user['id'],
                'nombre': user['nombre'],
                'apellido': user['apellido'],
                'correo': user['correo']
            },
            'purchases': purchases
        })
        
    except Exception as e:
        print(f"Error en admin_get_user_costos_detail: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ---------------------------------------------------------------------------
# API catálogo activo para Inefable Store (sync de paquetes)
# ---------------------------------------------------------------------------

@app.route('/api/catalog/active', methods=['GET'])
def api_catalog_active():
    """Devuelve todos los paquetes activos (Free Fire ID + juegos dinámicos modo ID)
    para que Inefable Store pueda sincronizar su catálogo de mapeo."""
    env_key = os.environ.get('WEBB_API_KEY', '').strip()
    req_key = (request.args.get('api_key') or request.headers.get('X-API-Key') or '').strip()
    if not env_key or req_key != env_key:
        return jsonify({'ok': False, 'error': 'API key inválida'}), 401

    items = []

    # 1. Free Fire ID packages
    try:
        conn = get_db_connection()
        ff_rows = conn.execute(
            'SELECT id, nombre, precio FROM precios_freefire_id WHERE activo = TRUE ORDER BY id'
        ).fetchall()
        conn.close()
        for r in ff_rows:
            items.append({
                'package_id': r['id'],
                'name': r['nombre'],
                'price': float(r['precio']) if r['precio'] else 0,
                'product_id': None,
                'product_name': 'Free Fire ID',
                'is_id_game': True,
                'active': True,
            })
    except Exception as e:
        logger.warning(f'[API Catalog] Error leyendo precios_freefire_id: {e}')

    # 2. Blood Strike packages
    try:
        conn = get_db_connection()
        bs_rows = conn.execute(
            'SELECT id, nombre, precio, gamepoint_package_id FROM precios_bloodstriker WHERE activo = TRUE ORDER BY id'
        ).fetchall()
        conn.close()
        for r in bs_rows:
            items.append({
                'package_id': r['id'],
                'name': r['nombre'],
                'price': float(r['precio']) if r['precio'] else 0,
                'product_id': -155,
                'product_name': 'Blood Strike',
                'game_id': -155,
                'game_type': 'bloodstriker',
                'is_id_game': True,
                'active': True,
            })
    except Exception as e:
        logger.warning(f'[API Catalog] Error leyendo precios_bloodstriker: {e}')

    # 3. Dynamic games (modo ID) packages
    try:
        from dynamic_games import get_all_dynamic_games, get_dynamic_packages
        dyn_games = get_all_dynamic_games(only_active=True)
        for game in dyn_games:
            if (game.get('modo') or 'id') != 'id':
                continue
            pkgs = get_dynamic_packages(game['id'], only_active=True)
            for pkg in pkgs:
                items.append({
                    'package_id': pkg['id'],
                    'name': pkg['nombre'],
                    'price': float(pkg.get('precio') or pkg.get('precio_venta') or 0),
                    'product_id': game['id'],
                    'product_name': game['nombre'],
                    'game_id': game['id'],
                    'is_id_game': True,
                    'active': True,
                })
    except Exception as e:
        logger.warning(f'[API Catalog] Error leyendo juegos dinámicos: {e}')

    return jsonify({'ok': True, 'items': items, 'total': len(items)})


# ---------------------------------------------------------------------------
# API endpoint para Inefable Store → recarga juegos dinámicos (GamePoint)
# ---------------------------------------------------------------------------

@app.route('/api/recharge/dynamic', methods=['POST'])
def api_recharge_dynamic():
    """Endpoint unificado para Inefable Store. Maneja recargas de juegos dinámicos
    vía GamePoint API, y delega Free Fire ID al endpoint existente."""
    env_key = os.environ.get('WEBB_API_KEY', '').strip()
    req_key = (request.form.get('api_key') or '').strip()
    if not env_key or req_key != env_key:
        return jsonify({'ok': False, 'error': 'API key inválida'}), 401

    player_id  = (request.form.get('player_id') or '').strip()
    pkg_id_str = (request.form.get('package_id') or '').strip()
    if not player_id or not pkg_id_str:
        return jsonify({'ok': False, 'error': 'player_id y package_id son requeridos'}), 400

    try:
        package_id = int(pkg_id_str)
    except ValueError:
        return jsonify({'ok': False, 'error': 'package_id debe ser un número'}), 400

    player_id2 = (request.form.get('player_id2') or '').strip()
    product_id_str = (request.form.get('product_id') or '').strip()
    product_id_hint = None
    try:
        if product_id_str:
            product_id_hint = int(product_id_str)
    except ValueError:
        pass

    from dynamic_games import get_dynamic_package_by_id, get_dynamic_game_by_id

    # --- Routing: product_id_hint desambigua cuando package_id colisiona ---
    #   product_id == -155  → Blood Strike (precios_bloodstriker)
    #   product_id > 0      → Juego dinámico con juego_id == product_id
    #   product_id == None  → Busca en orden: dinámicos → Blood Strike → Free Fire ID
    dyn_pkg = None
    if product_id_hint != -155:
        dyn_pkg = get_dynamic_package_by_id(package_id)
        if dyn_pkg and dyn_pkg.get('gamepoint_package_id'):
            if product_id_hint and dyn_pkg.get('juego_id') != product_id_hint:
                dyn_pkg = None
        elif dyn_pkg and not dyn_pkg.get('gamepoint_package_id'):
            dyn_pkg = None

    if dyn_pkg:
        game = get_dynamic_game_by_id(dyn_pkg['juego_id'])
        if not game:
            return jsonify({'ok': False, 'error': 'Juego dinámico no encontrado'}), 404
        if not game.get('activo'):
            return jsonify({'ok': False, 'error': 'Juego dinámico desactivado'}), 400
        if not dyn_pkg.get('activo'):
            return jsonify({'ok': False, 'error': 'Paquete dinámico desactivado'}), 400

        import time as _t
        _start = _t.time()

        try:
            # 1. Get GamePoint token
            gc_token, gc_err = _gameclub_get_token()
            if not gc_token:
                err = (gc_err or {}).get('message', 'No se pudo obtener token de GamePoint')
                logger.error(f'[API DynRecharge] Token error: {err}')
                return jsonify({'ok': False, 'error': f'Error proveedor: {err}'}), 502

            # 2. Validate order
            input_fields = {'input1': str(player_id)}
            if player_id2:
                input_fields['input2'] = str(player_id2)

            validate_data = _gameclub_order_validate(gc_token, game['gamepoint_product_id'], input_fields)
            validate_code = (validate_data or {}).get('code')
            if validate_code != 200 or not (validate_data or {}).get('validation_token'):
                err_msg = (validate_data or {}).get('message', 'Error validando orden')
                logger.error(f'[API DynRecharge] Validate failed: code={validate_code} msg={err_msg}')
                return jsonify({'ok': False, 'error': f'Validación falló: {err_msg}'}), 422

            validation_token = validate_data['validation_token']

            # 3. Create order
            merchant_code = f"API-DG{game['id']}-" + secrets.token_hex(6).upper()
            gp_package_id = dyn_pkg['gamepoint_package_id']
            create_data = _gameclub_order_create(gc_token, validation_token, gp_package_id, merchant_code)
            create_code = (create_data or {}).get('code')
            reference_no = (create_data or {}).get('referenceno', '')

            _dur = round(_t.time() - _start, 1)

            if create_code in (100, 101):
                # 4. Inquiry para obtener ingamename
                ingame_name = ''
                try:
                    if reference_no:
                        for _attempt in range(3):
                            if _attempt > 0:
                                _t.sleep(1.5)
                            inq_data = _gameclub_order_inquiry(gc_token, reference_no)
                            ingame_name = (inq_data or {}).get('ingamename') or ''
                            if ingame_name:
                                break
                except Exception as e:
                    logger.warning(f'[API DynRecharge] Inquiry error: {e}')

                # Log
                try:
                    _lc = get_db_connection()
                    _lc.execute(
                        'INSERT INTO api_recharges_log (player_id, package_id, success, player_name, error_msg, duration_seconds, game_name, package_name) VALUES (?,?,?,?,?,?,?,?)',
                        (player_id, package_id, 1, ingame_name, '', _dur, game['nombre'], dyn_pkg['nombre'])
                    )
                    _lc.commit()
                    _lc.close()
                except Exception:
                    pass

                logger.info(f'[API DynRecharge] OK game={game["nombre"]} player={player_id} pkg={package_id} gp_pkg={gp_package_id} dur={_dur}s')
                return jsonify({
                    'ok': True,
                    'player_name': ingame_name,
                    'duration': _dur,
                    'reference_no': reference_no,
                    'game': game['nombre'],
                })
            else:
                err_msg = (create_data or {}).get('message', 'Error creando orden en GamePoint')
                logger.error(f'[API DynRecharge] Create failed: code={create_code} msg={err_msg}')

                try:
                    _lc = get_db_connection()
                    _lc.execute(
                        'INSERT INTO api_recharges_log (player_id, package_id, success, player_name, error_msg, duration_seconds, game_name, package_name) VALUES (?,?,?,?,?,?,?,?)',
                        (player_id, package_id, 0, '', err_msg, _dur, game['nombre'], dyn_pkg['nombre'])
                    )
                    _lc.commit()
                    _lc.close()
                except Exception:
                    pass

                return jsonify({'ok': False, 'error': err_msg}), 422

        except Exception as e:
            logger.error(f'[API DynRecharge] Error general: {e}')
            return jsonify({'ok': False, 'error': f'Error interno: {str(e)}'}), 500

    # --- Intentar como Blood Strike ---
    try:
        conn = get_db_connection()
        bs_pkg = conn.execute(
            'SELECT id, nombre, precio, gamepoint_package_id FROM precios_bloodstriker WHERE id = ? AND activo = TRUE',
            (package_id,)
        ).fetchone()
        conn.close()
    except Exception:
        bs_pkg = None

    if bs_pkg and bs_pkg['gamepoint_package_id']:
        import time as _t
        _start = _t.time()
        bloodstrike_product_id = int(os.environ.get('BLOODSTRIKE_PRODUCT_ID', '155'))

        try:
            gc_token, gc_err = _gameclub_get_token()
            if not gc_token:
                err = (gc_err or {}).get('message', 'No se pudo obtener token de GamePoint')
                logger.error(f'[API BSRecharge] Token error: {err}')
                return jsonify({'ok': False, 'error': f'Error proveedor: {err}'}), 502

            input_fields = {'input1': str(player_id)}
            validate_data = _gameclub_order_validate(gc_token, bloodstrike_product_id, input_fields)
            validate_code = (validate_data or {}).get('code')
            if validate_code != 200 or not (validate_data or {}).get('validation_token'):
                err_msg = (validate_data or {}).get('message', 'Error validando orden')
                logger.error(f'[API BSRecharge] Validate failed: code={validate_code} msg={err_msg}')
                return jsonify({'ok': False, 'error': f'Validación falló: {err_msg}'}), 422

            validation_token = validate_data['validation_token']
            merchant_code = 'API-BS-' + secrets.token_hex(6).upper()
            gp_package_id = bs_pkg['gamepoint_package_id']
            create_data = _gameclub_order_create(gc_token, validation_token, gp_package_id, merchant_code)
            create_code = (create_data or {}).get('code')
            reference_no = (create_data or {}).get('referenceno', '')
            _dur = round(_t.time() - _start, 1)

            if create_code in (100, 101):
                ingame_name = ''
                try:
                    if reference_no:
                        for _attempt in range(3):
                            if _attempt > 0:
                                _t.sleep(1.5)
                            inq_data = _gameclub_order_inquiry(gc_token, reference_no)
                            ingame_name = (inq_data or {}).get('ingamename') or ''
                            if ingame_name:
                                break
                except Exception as e:
                    logger.warning(f'[API BSRecharge] Inquiry error: {e}')

                try:
                    _lc = get_db_connection()
                    _lc.execute(
                        'INSERT INTO api_recharges_log (player_id, package_id, success, player_name, error_msg, duration_seconds, game_name, package_name) VALUES (?,?,?,?,?,?,?,?)',
                        (player_id, package_id, 1, ingame_name, '', _dur, 'Blood Strike', bs_pkg['nombre'])
                    )
                    _lc.commit()
                    _lc.close()
                except Exception:
                    pass

                logger.info(f'[API BSRecharge] OK player={player_id} pkg={package_id} gp_pkg={gp_package_id} dur={_dur}s')
                return jsonify({
                    'ok': True,
                    'player_name': ingame_name,
                    'duration': _dur,
                    'reference_no': reference_no,
                    'game': 'Blood Strike',
                })
            else:
                err_msg = (create_data or {}).get('message', 'Error creando orden en GamePoint')
                logger.error(f'[API BSRecharge] Create failed: code={create_code} msg={err_msg}')
                try:
                    _lc = get_db_connection()
                    _lc.execute(
                        'INSERT INTO api_recharges_log (player_id, package_id, success, player_name, error_msg, duration_seconds, game_name, package_name) VALUES (?,?,?,?,?,?,?,?)',
                        (player_id, package_id, 0, '', err_msg, _dur, 'Blood Strike', bs_pkg['nombre'])
                    )
                    _lc.commit()
                    _lc.close()
                except Exception:
                    pass
                return jsonify({'ok': False, 'error': err_msg}), 422

        except Exception as e:
            logger.error(f'[API BSRecharge] Error general: {e}')
            return jsonify({'ok': False, 'error': f'Error interno: {str(e)}'}), 500

    # --- Fallback: intentar como Free Fire ID ---
    precio = get_freefire_id_price_by_id(package_id)
    if precio > 0:
        return api_recharge_freefire_id()

    return jsonify({'ok': False, 'error': f'Paquete {package_id} no encontrado en juegos dinámicos, Blood Strike ni Free Fire ID'}), 404


# ---------------------------------------------------------------------------
# API endpoint para Inefable Store → recarga Free Fire ID sin sesión
# ---------------------------------------------------------------------------

@app.route('/api/recharge/freefire_id', methods=['POST'])
def api_recharge_freefire_id():
    """Endpoint dedicado para Inefable Store. Solo requiere WEBB_API_KEY."""
    env_key = os.environ.get('WEBB_API_KEY', '').strip()
    req_key = (request.form.get('api_key') or '').strip()
    if not env_key or req_key != env_key:
        return jsonify({'ok': False, 'error': 'API key inválida'}), 401

    player_id  = (request.form.get('player_id') or '').strip()
    pkg_id_str = (request.form.get('package_id') or '').strip()
    if not player_id or not pkg_id_str:
        return jsonify({'ok': False, 'error': 'player_id y package_id son requeridos'}), 400

    try:
        package_id = int(pkg_id_str)
    except ValueError:
        return jsonify({'ok': False, 'error': 'package_id debe ser un número'}), 400

    precio = get_freefire_id_price_by_id(package_id)
    if precio == 0:
        return jsonify({'ok': False, 'error': 'Paquete no encontrado o inactivo'}), 400

    pin_disponible = get_available_pin_freefire_global(package_id)
    if not pin_disponible:
        return jsonify({'ok': False, 'error': f'Sin stock para paquete {package_id}'}), 409

    pin_codigo = pin_disponible['pin_codigo']

    redeemer_config = get_redeemer_config_from_db(get_db_connection)
    import time as _t
    _start = _t.time()
    try:
        redeem_result = redeem_pin_vps(pin_codigo, player_id, redeemer_config)
    except Exception as e:
        logger.error(f'[API FF-ID] Error redención: {e}')
        return jsonify({'ok': False, 'error': f'Error interno: {e}'}), 500
    _dur = round(_t.time() - _start, 1)

    # Resolver nombre del paquete FF
    _ff_pkg_name = ''
    try:
        _cn = get_db_connection()
        _rn = _cn.execute('SELECT nombre FROM precios_freefire_id WHERE id = ?', (package_id,)).fetchone()
        _cn.close()
        _ff_pkg_name = _rn['nombre'] if _rn else ''
    except Exception:
        pass

    def _log_api_recharge(success, player_name='', error_msg=''):
        try:
            _lc = get_db_connection()
            _lc.execute(
                'INSERT INTO api_recharges_log (player_id, package_id, success, player_name, error_msg, duration_seconds, game_name, package_name) VALUES (?,?,?,?,?,?,?,?)',
                (player_id, package_id, 1 if success else 0, player_name, error_msg, _dur, 'Free Fire ID', _ff_pkg_name)
            )
            _lc.commit()
            _lc.close()
        except Exception as _le:
            logger.warning(f'[API FF-ID] No se pudo guardar log: {_le}')

    if redeem_result and redeem_result.success:
        pname = redeem_result.player_name or ''
        _log_api_recharge(True, player_name=pname)
        logger.info(f'[API FF-ID] Recarga exitosa player={player_id} pkg={package_id} dur={_dur}s')
        return jsonify({'ok': True, 'player_name': pname, 'duration': _dur})
    else:
        # Devolver el PIN al stock
        try:
            _c = get_db_connection()
            _c.execute('INSERT OR IGNORE INTO pines_freefire_global (monto_id, pin_codigo) VALUES (?,?)', (package_id, pin_codigo))
            _c.commit()
            _c.close()
        except Exception:
            pass
        err_msg = (redeem_result.message if redeem_result else None) or 'Redención fallida'
        _log_api_recharge(False, error_msg=err_msg)
        logger.warning(f'[API FF-ID] Redención fallida player={player_id} pkg={package_id}: {err_msg}')
        return jsonify({'ok': False, 'error': err_msg}), 422


# ---------------------------------------------------------------------------
# Daily Backup — clientes + pines Free Fire Global
# ---------------------------------------------------------------------------

def _build_backup_zip():
    """Genera un ZIP en memoria con clientes.csv y pines por paquete."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        conn = get_db_connection()

        # --- clientes.csv ---
        users = conn.execute(
            'SELECT correo, nombre, apellido, saldo FROM usuarios ORDER BY id'
        ).fetchall()
        csv_buf = io.StringIO()
        w = csv.writer(csv_buf)
        w.writerow(['correo', 'nombre', 'apellido', 'saldo'])
        for u in users:
            w.writerow([u['correo'], u['nombre'], u['apellido'], u['saldo']])
        zf.writestr('clientes.csv', csv_buf.getvalue())

        # --- pines_freefire_global_<monto_id>.csv (solo no usados) ---
        monto_ids = conn.execute(
            'SELECT DISTINCT monto_id FROM pines_freefire_global WHERE usado = FALSE ORDER BY monto_id'
        ).fetchall()
        for row in monto_ids:
            mid = row['monto_id']
            pins = conn.execute(
                'SELECT pin_codigo, batch_id FROM pines_freefire_global WHERE monto_id = ? AND usado = FALSE',
                (mid,)
            ).fetchall()
            pin_buf = io.StringIO()
            pw = csv.writer(pin_buf)
            pw.writerow(['monto_id', 'pin_codigo', 'batch_id'])
            for p in pins:
                pw.writerow([mid, p['pin_codigo'], p['batch_id'] or ''])
            zf.writestr(f'pines_freefire_global_{mid}.csv', pin_buf.getvalue())

        conn.close()
    buf.seek(0)
    return buf


def _send_daily_backup():
    """Envía el backup por correo al MAIL_USERNAME configurado."""
    dest = app.config.get('MAIL_USERNAME')
    if not dest:
        logger.warning('[Backup] MAIL_USERNAME no configurado, omitiendo envío.')
        return
    try:
        with app.app_context():
            fecha = datetime.now(pytz.timezone('America/Caracas')).strftime('%Y-%m-%d')
            zip_buf = _build_backup_zip()
            msg = Message(
                subject=f'[Inefable Store] Backup diario {fecha}',
                recipients=[dest],
                body=(
                    f'Backup automático generado el {fecha}.\n\n'
                    'Adjunto:\n'
                    '  • clientes.csv — correo, nombre, apellido y saldo de todos los clientes\n'
                    '  • pines_freefire_global_X.csv — pines no usados por paquete\n\n'
                    'Para restaurar pines, usa el botón "Importar Backup" en el panel de Pines del admin.'
                )
            )
            msg.attach(
                filename=f'backup_{fecha}.zip',
                content_type='application/zip',
                data=zip_buf.read()
            )
            mail.send(msg)
            logger.info(f'[Backup] Backup diario enviado a {dest}')
    except Exception as e:
        logger.error(f'[Backup] Error enviando backup: {e}')


def _backup_scheduler_thread():
    """Espera hasta medianoche y envía backup cada 24 h."""
    logger.info('[Backup] Thread de backup diario iniciado.')
    while True:
        now = datetime.now(pytz.timezone('America/Caracas'))
        # Próxima medianoche
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sleep_secs = (next_midnight - now).total_seconds()
        logger.info(f'[Backup] Próximo backup en {sleep_secs/3600:.1f} h')
        time_module.sleep(sleep_secs)
        _send_daily_backup()


# Iniciar thread de backup (evita doble arranque en modo debug)
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
    _bt = threading.Thread(target=_backup_scheduler_thread, daemon=True)
    _bt.start()


@app.route('/admin/api_recharges_log')
def admin_api_recharges_log():
    if not session.get('is_admin'):
        return jsonify({'ok': False, 'error': 'Acceso denegado'}), 403
    try:
        conn = get_db_connection()
        rows = conn.execute(
            'SELECT id, player_id, package_id, success, player_name, error_msg, duration_seconds, fecha, game_name, package_name FROM api_recharges_log ORDER BY fecha DESC LIMIT 100'
        ).fetchall()
        conn.close()
        return jsonify({'ok': True, 'logs': [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/admin/restore_backup', methods=['POST'])
def admin_restore_backup():
    if not session.get('is_admin'):
        return jsonify({'ok': False, 'error': 'Acceso denegado'}), 403

    f = request.files.get('backup_zip')
    if not f or not f.filename.endswith('.zip'):
        flash('Sube un archivo .zip de backup válido.', 'error')
        return redirect('/admin')

    try:
        raw = f.read()
        buf = io.BytesIO(raw)
        restored_pins = 0
        restored_packages = []

        with zipfile.ZipFile(buf, 'r') as zf:
            for name in zf.namelist():
                # Restaurar pines Free Fire Global
                if name.startswith('pines_freefire_global_') and name.endswith('.csv'):
                    try:
                        mid = int(name.replace('pines_freefire_global_', '').replace('.csv', ''))
                    except ValueError:
                        continue
                    content = zf.read(name).decode('utf-8', errors='ignore')
                    reader = csv.DictReader(io.StringIO(content))
                    pins_list = [row['pin_codigo'].strip() for row in reader
                                 if row.get('pin_codigo', '').strip()]
                    if pins_list:
                        conn = get_db_connection()
                        batch_id = _generate_batch_id()
                        added = 0
                        for pin in pins_list:
                            try:
                                conn.execute(
                                    'INSERT OR IGNORE INTO pines_freefire_global (monto_id, pin_codigo, batch_id) VALUES (?,?,?)',
                                    (mid, pin, batch_id)
                                )
                                added += conn.execute(
                                    'SELECT changes()'
                                ).fetchone()[0]
                            except Exception:
                                pass
                        conn.commit()
                        conn.close()
                        restored_pins += added
                        if added:
                            restored_packages.append(f'monto #{mid}: {added} pines')

        if restored_pins:
            flash(f'Backup restaurado: {restored_pins} pines importados ({", ".join(restored_packages)})', 'success')
        else:
            flash('No se encontraron pines nuevos en el backup (puede que ya estén en stock).', 'warning')

    except zipfile.BadZipFile:
        flash('El archivo no es un ZIP válido.', 'error')
    except Exception as e:
        logger.error(f'[Restore] Error restaurando backup: {e}')
        flash(f'Error al restaurar backup: {e}', 'error')

    return redirect('/admin')


@app.route('/mockup/gameclub-catalogo')
def gameclub_catalogo_mockup():
    return render_template('gameclub_catalogo_mockup.html')


if __name__ == '__main__':
    app.run(debug=True)

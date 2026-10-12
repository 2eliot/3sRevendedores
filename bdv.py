"""
BDV (Banco de Venezuela) dentro de Referencias: la conexión de cada usuario con su banco.

El usuario guarda en su panel su usuario, su contraseña y su número de cuenta del BDV. Un LECTOR EXTERNO
(lector_externo/, en otro servidor) los pide por /api/lector-bdv/*, entra al banco, lee sus movimientos y manda a la
web los pagos recibidos, que se guardan en pagos_banco, en su cuenta BDV. La web no abre ningún navegador.

Seguridad (cifrado de llave pública, PyNaCl SealedBox):
  - La web solo tiene la llave PÚBLICA del lector: con ella cierra usuario, contraseña y número completo, pero no
    puede abrirlos. Aunque se filtre la base de datos o el servidor de la web, las claves del banco siguen ilegibles.
  - La llave PRIVADA solo existe en el servidor del lector, que descifra en memoria al momento de entrar al banco.
  - La pública sale de BDV_CLAVE_PUBLICA (si está en el .env) o del lector aprobado por el admin en el panel.
    Sin llave pública la pestaña BDV no deja guardar datos.
  - Nadie puede volver a ver los datos: ni el usuario, ni el admin, ni la API. Solo se cambian o se borran.

Interruptor «Leer movimientos» (lector_encendido): apagado, el lector cierra la sesión en el banco con «Salir» para
que el dueño pueda entrar desde su teléfono; encendido, vuelve a entrar.
"""
import base64
import hashlib
import logging
import os
import re
import secrets

logger = logging.getLogger(__name__)

# Estados del lector de cada cuenta (cuentas_banco.lector_estado)
ESTADOS = {
    'pendiente': 'Conectando…',
    'conectando': 'Conectando…',
    'conectado': 'Conectado',
    'banco_no_responde': 'El banco no responde, reintentando',
    'rechazado': 'El banco rechazó la entrada, se reintenta en 5 min',
    'detenido': 'Datos rechazados',
    'pausado': 'Pausado',
    'en_espera': 'En espera de un lugar',
    'sin_conexion': 'Sin conectar',
}
MAX_CUENTAS_POR_USUARIO = 3
CONFIG_PUBLICA = 'bdv_lector_publica'   # llave pública del lector aprobado en el panel (no es secreta)

_aviso_llave = False


class SinLlave(Exception):
    """No hay llave pública válida del lector: no se puede cifrar."""


def publica_valida(texto):
    """La llave pública en bytes (32) si el texto es una llave válida en base64; None si no."""
    try:
        crudo = base64.b64decode(str(texto or '').strip(), validate=True)
        return crudo if len(crudo) == 32 else None
    except Exception:
        return None


def huella(publica_b64):
    """Código corto para reconocer una llave pública a simple vista (el lector muestra el mismo): AB12-CD34."""
    h = hashlib.sha256(str(publica_b64 or '').strip().encode('ascii', 'ignore')).hexdigest()[:8].upper()
    return f'{h[:4]}-{h[4:]}'


def _publica_aprobada():
    """La pública del lector aprobado en el panel (configuracion_redeemer), o None."""
    try:
        from pg_compat import get_db_connection
        conn = get_db_connection()
        try:
            fila = conn.execute('SELECT valor FROM configuracion_redeemer WHERE clave = ?', (CONFIG_PUBLICA,)).fetchone()
            return fila['valor'] if fila and fila['valor'] else None
        finally:
            conn.close()
    except Exception:
        return None


def llave_publica():
    """Texto base64 de la llave pública con la que se cifra: BDV_CLAVE_PUBLICA o la del lector aprobado."""
    global _aviso_llave
    env = os.environ.get('BDV_CLAVE_PUBLICA', '').strip()
    if env:
        if publica_valida(env):
            return env
        if not _aviso_llave:
            logger.error('[BDV] BDV_CLAVE_PUBLICA no es una llave pública válida')
            _aviso_llave = True
        return None
    aprobada = _publica_aprobada()
    return aprobada if publica_valida(aprobada) else None


def cifrado_disponible():
    return llave_publica() is not None


def cifrar(texto):
    """Cierra el texto con la llave pública del lector. La web NO tiene cómo abrirlo."""
    from nacl.public import PublicKey, SealedBox
    publica = llave_publica()
    if publica is None:
        raise SinLlave('No hay llave pública del lector')
    caja = SealedBox(PublicKey(base64.b64decode(publica)))
    return base64.b64encode(caja.encrypt(str(texto).encode('utf-8'))).decode('ascii')


def nuevo_cambio():
    """Marca de «datos cambiados»: el lector la compara y, si cambió, vuelve a entrar con los datos nuevos."""
    return secrets.token_hex(8)


def _cuenta_bdv(conn, cuenta_id, uid):
    sql = "SELECT * FROM cuentas_banco WHERE id = ? AND banco = 'BDV'"
    params = [cuenta_id]
    if uid is not None:
        sql += ' AND usuario_id = ?'
        params.append(uid)
    return conn.execute(sql, tuple(params)).fetchone()


def guardar_conexion(conn, uid, usuario, clave, numero_cuenta, alias='', consentimiento=False):
    """Guarda (o reemplaza) la conexión del usuario con una cuenta del BDV. Devuelve (status, datos).

    Si ya tenía esa cuenta (mismos últimos 4 dígitos) se actualiza, así sus pagos siguen en la misma cuenta.
    """
    from antiduplic import ahora_local, ahora_minuto, aplicar_corte
    if not cifrado_disponible():
        return 503, {'mensaje': 'El BDV todavía no está disponible: falta conectar el lector del banco. '
                                'Avísale al administrador.'}
    usuario = str(usuario or '').strip()
    clave = str(clave or '')
    numero = re.sub(r'\D', '', str(numero_cuenta or ''))
    if not consentimiento:
        return 400, {'mensaje': 'Marca la casilla de autorización para conectar tu cuenta'}
    if not usuario or len(usuario) > 60:
        return 400, {'mensaje': 'Escribe tu usuario del BDV'}
    if not clave or len(clave) > 100:
        return 400, {'mensaje': 'Escribe tu contraseña del BDV'}
    if len(numero) != 20 or not numero.startswith('0102'):
        return 400, {'mensaje': 'El número de cuenta del BDV tiene 20 dígitos y empieza por 0102'}
    ultimos = numero[-4:]
    alias = str(alias or '').strip()[:60] or f'BDV ··{ultimos}'
    ahora = ahora_local()
    datos = (cifrar(usuario), cifrar(clave), cifrar(numero))
    existente = conn.execute("SELECT id, bdv_clave_cifrada, usar_desde FROM cuentas_banco WHERE usuario_id = ? AND banco = 'BDV' "
                             'AND ultimos_digitos = ? ORDER BY id LIMIT 1', (uid, ultimos)).fetchone()
    if existente:
        cuenta_id = int(existente['id'])
        # Si estaba desconectada, el corte pasa a ahora (lo de mientras tanto pudo usarse por fuera); si solo cambia
        # sus datos, el corte se queda como estaba
        if not existente['bdv_clave_cifrada'] or not existente['usar_desde']:
            conn.execute('UPDATE cuentas_banco SET usar_desde = ? WHERE id = ?', (ahora_minuto(), cuenta_id))
            aplicar_corte(conn, cuenta_id)
        conn.execute(
            'UPDATE cuentas_banco SET alias = ?, activo = TRUE, bdv_usuario_cifrado = ?, bdv_clave_cifrada = ?, '
            "numero_cuenta_cifrado = ?, lector_estado = 'pendiente', lector_mensaje = ?, lector_fallos_login = 0, "
            'lector_encendido = TRUE, lector_cambio = ?, consentimiento_en = ? WHERE id = ? AND usuario_id = ?',
            (alias, *datos, 'Datos guardados: conectando con el banco…', nuevo_cambio(), ahora, cuenta_id, uid))
    else:
        conectadas = conn.execute("SELECT COUNT(*) AS n FROM cuentas_banco WHERE usuario_id = ? AND banco = 'BDV' "
                                  'AND bdv_clave_cifrada IS NOT NULL', (uid,)).fetchone()['n']
        if int(conectadas or 0) >= MAX_CUENTAS_POR_USUARIO:
            return 409, {'mensaje': f'Máximo {MAX_CUENTAS_POR_USUARIO} cuentas del BDV conectadas'}
        cur = conn.execute(
            'INSERT INTO cuentas_banco (usuario_id, banco, alias, ultimos_digitos, activo, creado_en, bdv_usuario_cifrado, '
            'bdv_clave_cifrada, numero_cuenta_cifrado, lector_estado, lector_mensaje, lector_fallos_login, lector_encendido, '
            "lector_cambio, consentimiento_en, usar_desde) VALUES (?, 'BDV', ?, ?, TRUE, ?, ?, ?, ?, 'pendiente', ?, 0, TRUE, ?, ?, ?) "
            'RETURNING id',
            (uid, alias, ultimos, ahora, *datos, 'Datos guardados: conectando con el banco…', nuevo_cambio(), ahora, ahora_minuto()))
        cuenta_id = int(cur.fetchone()[0])
    conn.commit()
    logger.info(f'[BDV] Usuario {uid}: conexión guardada en la cuenta {cuenta_id}')  # sin datos de acceso
    return 200, {'mensaje': 'Datos guardados. La web se conectará a tu BDV en menos de un minuto.', 'cuenta_id': cuenta_id}


def cambiar_lectura(conn, uid, cuenta_id, encender):
    """Interruptor «Leer movimientos». uid=None → el admin (pausa cualquier conexión)."""
    cuenta = _cuenta_bdv(conn, cuenta_id, uid)
    if not cuenta:
        return 404, {'mensaje': 'Cuenta no encontrada'}
    if not cuenta['bdv_clave_cifrada']:
        return 409, {'mensaje': 'Esta cuenta no tiene conexión: guarda tus datos primero'}
    if encender:
        conn.execute("UPDATE cuentas_banco SET lector_encendido = TRUE, lector_estado = 'pendiente', "
                     "lector_mensaje = 'Volviendo a entrar al banco…', lector_cambio = ? WHERE id = ?", (nuevo_cambio(), cuenta['id']))
        mensaje = 'Lectura encendida: la web vuelve a entrar a tu BDV'
    else:
        conn.execute("UPDATE cuentas_banco SET lector_encendido = FALSE, lector_mensaje = 'Cerrando la sesión en el banco…', "
                     'lector_cambio = ? WHERE id = ?', (nuevo_cambio(), cuenta['id']))
        mensaje = 'Pausando: la web cierra la sesión en el banco en unos segundos'
    conn.commit()
    return 200, {'mensaje': mensaje}


def reanudar(conn, uid, cuenta_id):
    """Después de «Datos rechazados»: vuelve a intentar con los mismos datos (los intentos empiezan de cero)."""
    cuenta = _cuenta_bdv(conn, cuenta_id, uid)
    if not cuenta or not cuenta['bdv_clave_cifrada']:
        return 404, {'mensaje': 'Cuenta no encontrada'}
    conn.execute("UPDATE cuentas_banco SET lector_fallos_login = 0, lector_estado = 'pendiente', lector_encendido = TRUE, "
                 "lector_mensaje = 'Reintentando con tus datos…', lector_cambio = ? WHERE id = ?", (nuevo_cambio(), cuenta['id']))
    conn.commit()
    return 200, {'mensaje': 'Reintentando: la web vuelve a entrar con tus datos'}


def borrar_conexion(conn, uid, cuenta_id):
    """Borra los datos cifrados (el lector cierra la sesión del banco). Los pagos ya guardados se quedan."""
    cuenta = _cuenta_bdv(conn, cuenta_id, uid)
    if not cuenta:
        return 404, {'mensaje': 'Cuenta no encontrada'}
    conn.execute("UPDATE cuentas_banco SET bdv_usuario_cifrado = NULL, bdv_clave_cifrada = NULL, numero_cuenta_cifrado = NULL, "
                 "activo = FALSE, lector_estado = 'sin_conexion', lector_mensaje = 'Conexión borrada', lector_encendido = FALSE, "
                 'lector_cambio = ? WHERE id = ?', (nuevo_cambio(), cuenta['id']))
    conn.commit()
    logger.info(f'[BDV] Cuenta {cuenta["id"]}: conexión borrada')
    return 200, {'mensaje': 'Conexión borrada. Tus pagos ya guardados se quedan.'}

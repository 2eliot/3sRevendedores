"""
BDV (Banco de Venezuela) dentro de Referencias: la conexión de cada usuario con su banco.

El usuario guarda en su panel su usuario, su contraseña y su número de cuenta del BDV; la web (lector_bdv.py, un
proceso aparte) entra con ellos, lee sus movimientos y guarda los pagos recibidos en pagos_banco, en su cuenta BDV.

Seguridad:
  - Usuario, contraseña y número completo se guardan cifrados (Fernet) con la llave BDV_LLAVE_CIFRADO, que solo
    vive en el servidor. Sin esa llave no se puede guardar nada y el lector no arranca.
  - Nadie los puede volver a ver: ni el usuario, ni el admin, ni la API. Solo se cambian o se borran.
  - Solo se descifran dentro del lector, en memoria, al entrar al banco. Nunca van a registros ni a mensajes.

Interruptor «Leer movimientos» (lector_encendido): apagado, el lector cierra la sesión en el banco con «Salir» para
que el dueño pueda entrar desde su teléfono; encendido, vuelve a entrar.

Generar una llave:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
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
MSG_DETENIDO = 'El banco rechazó tus datos: revísalos y guárdalos de nuevo'
MSG_PAUSADO = 'En pausa — la sesión del banco está cerrada ✅. Ya puedes entrar al BDV desde tu teléfono o tu computadora'
MSG_PAUSA_SIN_CONFIRMAR = 'No se pudo confirmar el cierre de sesión en el banco. Si no te deja entrar, espera unos minutos'

_aviso_llave = False


class SinLlave(Exception):
    """Falta BDV_LLAVE_CIFRADO (o no es válida): no se puede cifrar ni descifrar."""


def _fernet():
    global _aviso_llave
    llave = os.environ.get('BDV_LLAVE_CIFRADO', '').strip()
    if not llave:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(llave.encode())
    except Exception:
        if not _aviso_llave:  # nunca se registra la llave
            logger.error('[BDV] BDV_LLAVE_CIFRADO no es una llave válida (debe ser una llave Fernet)')
            _aviso_llave = True
        return None


def cifrado_disponible():
    return _fernet() is not None


def cifrar(texto):
    f = _fernet()
    if f is None:
        raise SinLlave('Falta BDV_LLAVE_CIFRADO')
    return f.encrypt(str(texto).encode('utf-8')).decode('ascii')


def descifrar(token):
    """Solo para el lector, en memoria. Lanza SinLlave si no hay llave o si no corresponde."""
    f = _fernet()
    if f is None:
        raise SinLlave('Falta BDV_LLAVE_CIFRADO')
    try:
        return f.decrypt(str(token).encode('ascii')).decode('utf-8')
    except Exception:
        raise SinLlave('Los datos no se pueden descifrar con la llave actual')


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
    from antiduplic import ahora_local
    if not cifrado_disponible():
        return 503, {'mensaje': 'El BDV todavía no está disponible en este servidor (falta la llave de cifrado). '
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
    existente = conn.execute("SELECT id FROM cuentas_banco WHERE usuario_id = ? AND banco = 'BDV' AND ultimos_digitos = ? "
                             'ORDER BY id LIMIT 1', (uid, ultimos)).fetchone()
    if existente:
        cuenta_id = int(existente['id'])
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
            "lector_cambio, consentimiento_en) VALUES (?, 'BDV', ?, ?, TRUE, ?, ?, ?, ?, 'pendiente', ?, 0, TRUE, ?, ?) RETURNING id",
            (uid, alias, ultimos, ahora, *datos, 'Datos guardados: conectando con el banco…', nuevo_cambio(), ahora))
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
                     "lector_mensaje = 'Volviendo a entrar al banco…' WHERE id = ?", (cuenta['id'],))
        mensaje = 'Lectura encendida: la web vuelve a entrar a tu BDV'
    else:
        conn.execute("UPDATE cuentas_banco SET lector_encendido = FALSE, lector_mensaje = 'Cerrando la sesión en el banco…' "
                     'WHERE id = ?', (cuenta['id'],))
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

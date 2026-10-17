"""
Compra por el Mapeo con Inefable (camino directo, sin respaldo): si el pedido se envió pero no llegó la
respuesta (tiempo agotado leyendo), NO se devuelve el saldo enseguida: queda en proceso y el reconciliador
consulta /api/v1/order-status. Si ni siquiera se pudo conectar, sí se devuelve al momento.

    python -m unittest test_reseller_sin_respuesta -v
"""
import io
import json
import os
import socket
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'sin_respuesta.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['REVENDEDORES_BASE_URL'] = 'https://inefable.invalid'
os.environ['REVENDEDORES_API_KEY'] = 'clave-de-prueba'

import app as webapp  # noqa: E402
import dynamic_games as dg  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if (sql.lstrip().upper().startswith('SELECT') or 'RETURNING' in sql.upper()) else None
        conn.commit()
        return rows
    finally:
        conn.close()


class Respuesta(io.BytesIO):
    """Lo que devuelve urlopen: se usa con «with» y se lee una vez."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def responde(data):
    return Respuesta(json.dumps(data).encode())


class ResellerSinRespuestaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webapp.app.config['TESTING'] = True
        cls.uid = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                    "VALUES ('Cliente', 'C', '0', 'cliente-sr@test.local', 'x', 0) RETURNING id")[0][0]
        cls.jid = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo) "
                    "VALUES ('Free Fire ID', 'ff-sr', 1, 'id', TRUE) RETURNING id")[0][0]
        cls.pid = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) "
                    "VALUES (?, '110 💎', 1.0, TRUE, 0) RETURNING id", (cls.jid,))[0][0]

    def setUp(self):
        q('UPDATE usuarios SET saldo = 10 WHERE id = ?', (self.uid,))
        q('DELETE FROM transacciones_dinamicas')

    def saldo(self):
        return float(q('SELECT saldo FROM usuarios WHERE id = ?', (self.uid,))[0]['saldo'])

    def compra(self, error_de_red=None, respuesta=None):
        """Compra por el camino directo con Inefable simulado. Devuelve la fila de la transacción."""
        game = {'id': self.jid, 'nombre': 'Free Fire ID', 'slug': 'ff-sr'}
        pkg = {'id': self.pid, 'nombre': '110 💎', 'precio': 1.0}
        efecto = {'side_effect': error_de_red} if error_de_red else {'return_value': responde(respuesta)}
        with webapp.app.test_request_context('/juego/d/ff-sr'), mock.patch('urllib.request.urlopen', **efecto):
            from flask import session
            session.update(usuario='cliente-sr@test.local', user_db_id=self.uid)
            dg._purchase_via_reseller(game, pkg, {'remote_product_id': '20', 'remote_package_id': '7'}, 'ff-sr',
                                      self.uid, False, 1.0, self.pid, '123456789', '', '', '/juego/d/ff-sr', time.time())
        return q('SELECT * FROM transacciones_dinamicas ORDER BY id DESC LIMIT 1')[0]

    def reconciliar(self, estado_inefable):
        orden = {'ok': True, 'found': True, 'status': estado_inefable,
                 'order': {'reference_no': 'REF-1', 'player_name': 'Jugador', 'error': 'Recarga fallida'}}
        with mock.patch('urllib.request.urlopen', return_value=responde(orden)), mock.patch.object(dg.time_module, 'sleep'):
            dg.poll_pending_reseller_transactions()

    def test_tiempo_agotado_leyendo_queda_en_proceso_sin_devolver(self):
        tx = self.compra(error_de_red=TimeoutError('The read operation timed out'))
        self.assertEqual(tx['estado'], 'procesando')
        self.assertTrue(tx['notas'].startswith('reseller_pending'), tx['notas'])
        self.assertEqual(self.saldo(), 9.0)   # cobrado, todavía sin devolver

    def test_conexion_cortada_tras_enviar_tambien_queda_en_proceso(self):
        tx = self.compra(error_de_red=ConnectionResetError('Connection reset by peer'))
        self.assertEqual(tx['estado'], 'procesando')
        self.assertEqual(self.saldo(), 9.0)

    def test_si_inefable_la_hizo_el_reconciliador_la_aprueba_sin_devolver(self):
        self.compra(error_de_red=TimeoutError('The read operation timed out'))
        self.reconciliar('completada')
        tx = q('SELECT * FROM transacciones_dinamicas ORDER BY id DESC LIMIT 1')[0]
        self.assertEqual(tx['estado'], 'aprobado')
        self.assertEqual(self.saldo(), 9.0)

    def test_si_inefable_fallo_el_reconciliador_devuelve_el_saldo(self):
        self.compra(error_de_red=TimeoutError('The read operation timed out'))
        self.reconciliar('fallida')
        tx = q('SELECT * FROM transacciones_dinamicas ORDER BY id DESC LIMIT 1')[0]
        self.assertEqual(tx['estado'], 'rechazado')
        self.assertEqual(self.saldo(), 10.0)

    def test_sin_conectar_no_llego_y_se_devuelve_al_momento(self):
        tx = self.compra(error_de_red=urllib.error.URLError(ConnectionRefusedError('Connection refused')))
        self.assertEqual(tx['estado'], 'rechazado')
        self.assertEqual(self.saldo(), 10.0)
        tx = self.compra(error_de_red=urllib.error.URLError(socket.timeout('timed out')))   # tiempo agotado al conectar
        self.assertEqual(tx['estado'], 'rechazado')
        self.assertEqual(self.saldo(), 10.0)

    def test_inefable_responde_que_fallo_se_devuelve_al_momento(self):
        tx = self.compra(respuesta={'ok': False, 'error': 'El proveedor no responde en este momento.'})
        self.assertEqual(tx['estado'], 'rechazado')
        self.assertEqual(self.saldo(), 10.0)

    def test_recarga_correcta(self):
        tx = self.compra(respuesta={'ok': True, 'reference_no': 'REF-9', 'player_name': 'Jugador'})
        self.assertEqual(tx['estado'], 'aprobado')
        self.assertEqual(self.saldo(), 9.0)


if __name__ == '__main__':
    unittest.main()

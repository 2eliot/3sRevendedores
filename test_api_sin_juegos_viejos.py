"""
Ninguna API ofrece Blood Strike ni el Free Fire ID viejo, y ninguna usa los PINs del Bot de Free Fire
(ni las rutas viejas que usaba Inefable Store con WEBB_API_KEY).

    python -m unittest test_api_sin_juegos_viejos -v
"""
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'viejos.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['WEBB_API_KEY'] = 'clave-webb-prueba'

import app as webapp  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

CLAVE = {'api_key': 'clave-webb-prueba'}


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if (sql.lstrip().upper().startswith('SELECT') or 'RETURNING' in sql.upper()) else None
        conn.commit()
        return rows
    finally:
        conn.close()


class ApiSinJuegosViejosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webapp.app.config['TESTING'] = True
        # Aunque alguien volviera a activar sus paquetes, no deben salir por ninguna API
        q('UPDATE precios_bloodstriker SET activo = TRUE')
        q('UPDATE precios_freefire_id SET activo = TRUE')
        if not q('SELECT COUNT(*) AS n FROM precios_bloodstriker')[0]['n']:
            q("INSERT INTO precios_bloodstriker (id, nombre, precio, descripcion, activo) VALUES (1, '100+10 BS', 1.0, 'x', TRUE)")
        if not q('SELECT COUNT(*) AS n FROM precios_freefire_id')[0]['n']:
            q("INSERT INTO precios_freefire_id (id, nombre, precio, descripcion, activo) VALUES (1, '100+10 FF', 1.0, 'x', TRUE)")
        q("INSERT INTO pines_freefire_global (monto_id, pin_codigo) VALUES (1, 'PIN-DEL-BOT-1')")
        jid = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo) VALUES ('Juego creado', 'juego-creado', 1, 'id', TRUE) "
                'RETURNING id')[0][0]
        q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) VALUES (?, '50 💎', 0.5, TRUE, 0)", (jid,))

    def setUp(self):
        self.c = webapp.app.test_client()

    def pines_del_bot_sin_usar(self):
        return q('SELECT COUNT(*) AS n FROM pines_freefire_global WHERE usado = FALSE')[0]['n']

    def test_el_catalogo_viejo_solo_trae_los_juegos_creados(self):
        d = self.c.get('/api/catalog/active', query_string=CLAVE).get_json()
        nombres = {i['product_name'] for i in d['items']}
        self.assertEqual(nombres, {'Juego creado'})
        self.assertEqual(self.c.get('/api/catalog/active').status_code, 401)

    def test_ninguna_ruta_vieja_recarga_blood_strike_ni_usa_pines_del_bot(self):
        antes = self.pines_del_bot_sin_usar()
        r = self.c.post('/api/recharge/freefire_id', data=dict(CLAVE, player_id='123456789', package_id='1'))
        self.assertEqual(r.status_code, 404)   # la ruta ya no existe
        for extra in ({'product_id': '-155'}, {}):   # Blood Strike o el Free Fire ID viejo por su número de paquete
            r = self.c.post('/api/recharge/dynamic', data=dict(CLAVE, player_id='123456789', package_id='1', **extra))
            self.assertEqual(r.status_code, 404, r.get_json())
            self.assertNotIn('Blood Strike', r.get_json()['error'])
        self.assertEqual(self.pines_del_bot_sin_usar(), antes)   # ningún PIN del bot se tocó

    def test_la_api_de_recargas_tampoco_los_ofrece(self):
        import api_panel
        nombres = {j['nombre'] for j in api_panel.catalogo()}
        self.assertFalse(nombres & {'Blood Strike', 'Free Fire ID', 'Pines Free Fire'})


if __name__ == '__main__':
    unittest.main()

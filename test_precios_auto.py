"""
Precio automático (precios_auto.py): en «Free fire ID» el precio de cada paquete es su costo en el proveedor principal del
Mapeo, redondeado hacia arriba al centavo, y se pone al día cada 20 minutos. Base SQLite temporal y Central One simulado.

    python -m unittest test_precios_auto -v
"""
import json
import os
import tempfile
import time
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'precios_auto.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['CENTRALONE_API_KEY'] = 'co_live_test'
os.environ.pop('REVENDEDORES_BASE_URL', None)

import app as webapp  # noqa: E402,F401
import central_one  # noqa: E402
import dynamic_games  # noqa: E402
import precios_auto  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

D110, D341, TARJETA = 'uuid-110', 'uuid-341', 'uuid-tarjeta'


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if (sql.lstrip().upper().startswith('SELECT') or 'RETURNING' in sql.upper()) else None
        conn.commit()
        return rows
    finally:
        conn.close()


class FakeCentralOne:
    def __init__(self):
        self.precios = {D110: '0.6545', D341: '1.9634', TARJETA: '0.4794'}
        self.caido = False

    def __call__(self, metodo, path, body=None, idem=None, sin_llave=False):
        if self.caido:
            return 0, {}
        if path == '/api/v1/catalog':
            return 200, {'items': [{'product_id': u, 'name': u, 'reseller_price': p, 'status': 'active',
                                    'product_family_id': 'free-fire', 'product_family_name': 'Free Fire'}
                                   for u, p in self.precios.items()]}
        return 404, {}


class PreciosAutoTest(unittest.TestCase):
    def setUp(self):
        for t in ('rev_item_mapping_steps', 'rev_item_mappings', 'rev_catalog_items', 'paquetes_dinamicos', 'juegos_dinamicos'):
            q(f'DELETE FROM {t}')
        q("DELETE FROM configuracion_redeemer WHERE clave = ?", (precios_auto.CLAVE_ULTIMA,))
        self.co = FakeCentralOne()
        self._orig = central_one._pedir
        central_one._pedir = self.co
        for uuid, precio in self.co.precios.items():
            q("INSERT INTO rev_catalog_items (remote_product_id, remote_product_name, remote_package_id, remote_package_name, raw_json) "
              "VALUES ('co:free-fire', 'Free Fire', ?, ?, ?)", (uuid, uuid, json.dumps({'price': float(precio)})))
        self.ff = self.juego('Free fire ID')
        self.p110 = self.paquete(self.ff, '110💎', 0.66, [(D110, 1)])
        self.p220 = self.paquete(self.ff, '220💎', 1.32, [(D110, 2)])
        self.p341 = self.paquete(self.ff, '341💎', 1.97, [(D341, 1)])
        self.tarjeta = self.paquete(self.ff, 'Tarjeta Básica', 0.40, [(TARJETA, 1)])

    def tearDown(self):
        central_one._pedir = self._orig

    # ---------- ayudas ----------
    def juego(self, nombre, usuario_id=None):
        return q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo, usuario_id) VALUES (?, ?, 0, 'id', TRUE, ?) "
                 "RETURNING id", (nombre, nombre.lower().replace(' ', '-') + str(time.time_ns()), usuario_id))[0][0]

    def paquete(self, jid, nombre, precio, pasos):
        pid = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) VALUES (?, ?, ?, TRUE, 0) RETURNING id",
                (jid, nombre, precio))[0][0]
        if pasos:
            q("INSERT INTO rev_item_mappings (juego_id, paquete_id, remote_product_id, remote_package_id, remote_label, auto_enabled, active) "
              "VALUES (?, ?, 'co:free-fire', ?, 'x', TRUE, TRUE)", (jid, pid, pasos[0][0]))
            for i, (uuid, cantidad) in enumerate(pasos):
                q("INSERT INTO rev_item_mapping_steps (juego_id, paquete_id, orden, remote_product_id, remote_package_id, remote_label, cantidad) "
                  "VALUES (?, ?, ?, 'co:free-fire', ?, 'x', ?)", (jid, pid, i, uuid, cantidad))
        return pid

    def precio(self, pid):
        return float(q('SELECT precio FROM paquetes_dinamicos WHERE id = ?', (pid,))[0]['precio'])

    # ---------- precio = costo ----------
    def test_precio_igual_al_costo_redondeado_hacia_arriba(self):
        r = precios_auto.actualizar()
        self.assertEqual(r['fallos'], {})
        self.assertEqual((self.precio(self.p110), self.precio(self.p220), self.precio(self.p341)), (0.66, 1.31, 1.97))
        self.assertEqual(self.precio(self.tarjeta), 0.48)   # se vendía por debajo del costo
        self.assertEqual(sorted(c['nombre'] for c in r['cambiados']), ['220💎', 'Tarjeta Básica'])
        self.assertEqual(r['iguales'], 2)

    def test_si_el_proveedor_cambia_el_precio_cambia(self):
        precios_auto.actualizar()
        self.co.precios[D110] = '0.7001'
        r = precios_auto.actualizar()
        self.assertEqual((self.precio(self.p110), self.precio(self.p220)), (0.71, 1.41))
        self.assertEqual(json.loads(q("SELECT raw_json FROM rev_catalog_items WHERE remote_package_id = ?", (D110,))[0]['raw_json'])['price'], 0.7001)
        self.assertEqual(len(r['cambiados']), 2)
        self.co.precios[D110] = '0.6000'   # y si baja, baja
        precios_auto.actualizar()
        self.assertEqual(self.precio(self.p110), 0.60)

    # ---------- nunca a ciegas ----------
    def test_sin_mapeo_o_sin_precio_no_se_toca(self):
        sin_mapeo = self.paquete(self.ff, 'Pase raro', 0.80, [])
        sin_precio = self.paquete(self.ff, 'Nuevo', 1.00, [('uuid-que-no-existe', 1)])
        r = precios_auto.actualizar()
        self.assertEqual((self.precio(sin_mapeo), self.precio(sin_precio)), (0.80, 1.00))
        motivos = {s['nombre']: s['motivo'] for s in r['saltados']}
        self.assertIn('sin Mapeo', motivos['Pase raro'])
        self.assertIn('sin precio', motivos['Nuevo'])

    def test_salto_enorme_se_deja_para_revisar(self):
        self.co.precios[D341] = '19.6340'   # 10 veces más: catálogo mal leído
        r = precios_auto.actualizar()
        self.assertEqual(self.precio(self.p341), 1.97)
        self.assertIn('revisar', next(s for s in r['saltados'] if s['id'] == self.p341)['motivo'])

    def test_proveedor_caido_sigue_con_el_ultimo_precio(self):
        self.co.caido = True
        r = precios_auto.actualizar()
        self.assertIn('centralone', r['fallos'])
        self.assertEqual(self.precio(self.p220), 1.31)   # con el precio guardado de su catálogo

    def test_otros_juegos_y_juegos_propios_no_se_tocan(self):
        otro = self.paquete(self.juego('Mobile Legends'), '86💎', 5.00, [(D110, 1)])
        propio = self.paquete(self.juego('Free fire ID', usuario_id=99), '110💎', 3.00, [(D110, 1)])
        precios_auto.actualizar()
        self.assertEqual((self.precio(otro), self.precio(propio)), (5.00, 3.00))

    # ---------- cada 20 minutos, un solo worker ----------
    def test_turno_cada_20_minutos(self):
        self.assertIsNotNone(precios_auto.tick())
        self.assertIsNone(precios_auto.tick())   # el otro worker (o el siguiente minuto) no repite
        hace = f'{int(time.time() - precios_auto.INTERVALO_S - 60):012d}'
        q('UPDATE configuracion_redeemer SET valor = ? WHERE clave = ?', (hace, precios_auto.CLAVE_ULTIMA))
        self.assertIsNotNone(precios_auto.tick())

    def test_la_sincronizacion_vieja_se_salta_free_fire_id(self):
        juegos = [r['game'] for r in dynamic_games.sync_all_dynamic_games_prices()]
        self.assertNotIn('Free fire ID', juegos)


if __name__ == '__main__':
    unittest.main()

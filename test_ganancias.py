"""
Ganancias del Dashboard (ganancias.py): gasto (lo que se pagó en la web), venta (el precio de venta del usuario, en $ o Bs
con su tasa) y ganancia, por paquete y en total. Cada venta usa los precios y la tasa que había cuando se hizo.

    python -m unittest test_ganancias -v
"""
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'ganancias.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30

import app as webapp  # noqa: E402
import ganancias as gn  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

DIA = '2026-11-02'   # día de Venezuela: de 2026-11-02 04:00 a 2026-11-03 04:00 UTC


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if (sql.lstrip().upper().startswith('SELECT') or 'RETURNING' in sql.upper()) else None
        conn.commit()
        return rows
    finally:
        conn.close()


class GananciasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        gn.init_tablas()
        cls.A = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) VALUES ('Ana', 'A', '0', 'ana-g@test.local', 'x', 0) RETURNING id")[0][0]
        cls.B = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) VALUES ('Beto', 'B', '0', 'beto-g@test.local', 'x', 0) RETURNING id")[0][0]
        cls.ff = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo) VALUES ('Free fire ID', 'ff-gan', 0, 'id', TRUE) RETURNING id")[0][0]
        cls.p110 = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) VALUES (?, '110💎', 0.66, TRUE, 0) RETURNING id", (cls.ff,))[0][0]
        cls.tarjeta = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) VALUES (?, 'Tarjeta Semanal', 2.13, TRUE, 1) RETURNING id", (cls.ff,))[0][0]
        cls.propio = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo, usuario_id) VALUES ('Mi ML', 'u-ml-gan', 0, 'id', TRUE, ?) RETURNING id", (cls.A,))[0][0]
        cls.ml = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) VALUES (?, '86💎', 3.00, TRUE, 0) RETURNING id", (cls.propio,))[0][0]

    def setUp(self):
        for t in ('transacciones_dinamicas', 'ganancias_precios', 'ganancias_tasas'):
            q(f'DELETE FROM {t}')
        self.c = self.app.test_client()
        self._ahora = gn._ahora

    def tearDown(self):
        gn._ahora = self._ahora

    # ---------- ayudas ----------
    def venta(self, uid, pid, monto, fecha, estado='aprobado'):
        jid = self.propio if pid == self.ml else self.ff
        q("INSERT INTO transacciones_dinamicas (juego_id, usuario_id, paquete_id, numero_control, transaccion_id, monto, estado, fecha) "
          "VALUES (?, ?, ?, 'NC', 'TX', ?, ?, ?)", (jid, uid, pid, monto, estado, fecha))

    def a_las(self, utc):
        gn._ahora = lambda: utc

    def precio(self, uid, pid, monto, moneda='USD', tipo='venta', utc='2026-11-02 05:00:00'):
        self.a_las(utc)
        gn.guardar_precio(uid, pid, tipo, moneda, monto)

    def tasa(self, uid, tasa, utc='2026-11-02 05:00:00'):
        self.a_las(utc)
        gn.guardar_tasa(uid, tasa)

    def calcular(self, uid=None, inicio=DIA, fin=DIA, **kw):
        conn = get_db_connection()
        try:
            return gn.calcular(conn, inicio, fin, uid, **kw)
        finally:
            conn.close()

    def fila(self, res, pid):
        return next(p for p in res['paquetes'] if p['paquete_id'] == pid)

    # ---------- gasto, venta y ganancia ----------
    def test_gasto_es_lo_pagado_y_la_ganancia_venta_menos_gasto(self):
        self.precio(self.A, self.p110, 1.00)
        self.venta(self.A, self.p110, 0.66, '2026-11-02 12:00:00')
        self.venta(self.A, self.p110, 0.65, '2026-11-02 13:00:00')   # el proveedor bajó y la web cobró menos
        self.venta(self.A, self.p110, 0.66, '2026-11-02 14:00:00', estado='rechazado')   # fallida: no cuenta
        r = self.calcular(self.A)
        f = self.fila(r, self.p110)
        self.assertEqual((f['vendidos'], f['gasto'], f['venta'], f['ganancia']), (2, 1.31, 2.0, 0.69))
        self.assertEqual((r['total']['vendidos'], r['total']['ganancia']), (2, 0.69))

    def test_un_cambio_de_precio_vale_desde_ese_momento(self):
        self.precio(self.A, self.p110, 1.00, utc='2026-11-02 05:00:00')
        self.venta(self.A, self.p110, 0.66, '2026-11-02 12:00:00')
        self.precio(self.A, self.p110, 1.20, utc='2026-11-02 15:00:00')   # a media tarde sube su precio
        self.venta(self.A, self.p110, 0.66, '2026-11-02 16:00:00')
        r = self.calcular(self.A, con_detalle=True)
        self.assertEqual([v['venta'] for v in r['ventas']], [1.0, 1.2])
        self.assertEqual(self.fila(r, self.p110)['venta'], 2.2)
        # el día anterior no cambia
        self.venta(self.A, self.p110, 0.66, '2026-11-01 12:00:00')
        self.assertEqual(self.calcular(self.A, inicio='2026-11-01', fin='2026-11-01')['total']['venta'], 1.0)

    def test_el_primer_precio_vale_para_lo_vendido_antes(self):
        self.venta(self.A, self.p110, 0.66, '2026-11-02 08:00:00')
        self.precio(self.A, self.p110, 1.00, utc='2026-11-02 20:00:00')   # lo pone por primera vez en la noche
        self.assertEqual(self.calcular(self.A)['total']['venta'], 1.0)

    def test_corregir_la_moneda_enseguida_no_deja_el_precio_viejo(self):
        """Lo que pasó en producción: escribió 760 con la moneda en $ y un segundo después la pasó a Bs."""
        self.tasa(self.A, 1010, utc='2026-11-02 13:00:00')
        self.venta(self.A, self.p110, 0.66, '2026-11-02 12:00:00')   # vendido antes de poner precios
        self.precio(self.A, self.p110, 760, moneda='USD', utc='2026-11-02 13:13:29')
        self.precio(self.A, self.p110, 760, moneda='VES', utc='2026-11-02 13:13:30')
        self.assertEqual([(r['moneda'], r['monto']) for r in q('SELECT moneda, monto FROM ganancias_precios')], [('VES', 760.0)])
        f = self.fila(self.calcular(self.A), self.p110)
        self.assertEqual((f['venta_bs'], round(f['venta'], 4)), (760.0, round(760 / 1010, 4)))

    def test_con_ventas_en_medio_el_cambio_se_suma_al_historial(self):
        self.precio(self.A, self.p110, 1.00, utc='2026-11-02 05:00:00')
        self.venta(self.A, self.p110, 0.66, '2026-11-02 06:00:00')
        self.precio(self.A, self.p110, 1.50, utc='2026-11-02 07:00:00')
        self.assertEqual(len(q('SELECT id FROM ganancias_precios')), 2)
        self.venta(self.A, self.tarjeta, 2.13, '2026-11-02 08:00:00')   # otra venta, de otro paquete: no cuenta para el 110
        self.precio(self.A, self.p110, 1.40, utc='2026-11-02 09:00:00')   # nadie compró 110 desde las 07:00: corrige
        self.assertEqual([r['monto'] for r in q('SELECT monto FROM ganancias_precios ORDER BY id')], [1.0, 1.4])

    def test_corregir_la_tasa_sin_ventas_en_medio(self):
        self.tasa(self.A, 100, utc='2026-11-02 05:00:00')
        self.tasa(self.A, 1010, utc='2026-11-02 05:01:00')   # se equivocó y la corrigió
        self.assertEqual([r['tasa'] for r in q('SELECT tasa FROM ganancias_tasas')], [1010.0])
        self.venta(self.A, self.p110, 0.66, '2026-11-02 06:00:00')
        self.tasa(self.A, 1020, utc='2026-11-02 07:00:00')   # con una venta en medio: queda el historial
        self.assertEqual(len(q('SELECT id FROM ganancias_tasas')), 2)

    def test_precio_en_bolivares_con_su_tasa(self):
        self.tasa(self.A, 200, utc='2026-11-02 05:00:00')
        self.precio(self.A, self.tarjeta, 600, moneda='VES')
        self.venta(self.A, self.tarjeta, 2.13, '2026-11-02 12:00:00')
        self.tasa(self.A, 300, utc='2026-11-02 15:00:00')   # sube la tasa: los 600 Bs ahora valen $2
        self.venta(self.A, self.tarjeta, 2.13, '2026-11-02 16:00:00')
        r = self.calcular(self.A, con_detalle=True)
        self.assertEqual([(v['venta'], v['venta_bs'], v['tasa']) for v in r['ventas']], [(3.0, 600.0, 200.0), (2.0, 600.0, 300.0)])
        f = self.fila(r, self.tarjeta)
        self.assertEqual((f['venta'], f['venta_bs'], f['ganancia']), (5.0, 1200.0, 0.74))
        self.assertAlmostEqual(f['ganancia_bs'], 0.87 * 200 + (-0.13) * 300, places=2)

    def test_en_dolares_muestra_tambien_los_bolivares(self):
        self.tasa(self.A, 250)
        self.precio(self.A, self.p110, 1.00)
        self.venta(self.A, self.p110, 0.66, '2026-11-02 12:00:00')
        f = self.fila(self.calcular(self.A), self.p110)
        self.assertEqual((f['venta_bs'], f['ganancia_bs']), (250.0, 85.0))

    def test_sin_precio_o_en_bs_sin_tasa_no_suman(self):
        self.venta(self.A, self.p110, 0.66, '2026-11-02 12:00:00')
        self.precio(self.A, self.tarjeta, 600, moneda='VES')   # en Bs pero sin tasa
        self.venta(self.A, self.tarjeta, 2.13, '2026-11-02 12:00:00')
        r = self.calcular(self.A)
        self.assertEqual((r['total']['sin_precio'], r['total']['venta'], r['total']['gasto']), (2, 0.0, 2.79))

    def test_con_su_proveedor_el_gasto_es_su_costo(self):
        self.venta(self.A, self.ml, 0, '2026-11-02 12:00:00')   # la web no cobra nada
        self.precio(self.A, self.ml, 2.50)
        self.assertEqual(self.fila(self.calcular(self.A), self.ml)['gasto'], 0.0)   # sin «Mi costo»
        self.precio(self.A, self.ml, 1.80, tipo='costo')
        self.venta(self.A, self.ml, 0, '2026-11-02 13:00:00')
        f = self.fila(self.calcular(self.A), self.ml)
        self.assertTrue(f['propio'])
        self.assertEqual((f['gasto'], f['venta'], f['ganancia']), (3.6, 5.0, 1.4))

    def test_dia_de_venezuela(self):
        self.precio(self.A, self.p110, 1.00)
        self.venta(self.A, self.p110, 0.66, '2026-11-02 03:59:00')   # 23:59 del 1 en Venezuela
        self.venta(self.A, self.p110, 0.66, '2026-11-03 03:59:00')   # 23:59 del 2
        self.assertEqual(self.calcular(self.A)['total']['vendidos'], 1)
        self.assertEqual(self.calcular(self.A, inicio='2026-11-01', fin='2026-11-02')['total']['vendidos'], 2)

    # ---------- cada usuario lo suyo; el admin, el global ----------
    def test_cada_usuario_con_sus_precios_y_el_admin_global(self):
        self.precio(self.A, self.p110, 1.00)
        self.precio(self.B, self.p110, 0.80)
        self.venta(self.A, self.p110, 0.66, '2026-11-02 12:00:00')
        self.venta(self.B, self.p110, 0.66, '2026-11-02 12:00:00')
        self.assertEqual(self.calcular(self.A)['total']['venta'], 1.0)
        self.assertEqual(self.calcular(self.B)['total']['venta'], 0.8)
        todos = self.calcular(None)
        self.assertEqual((todos['total']['vendidos'], todos['total']['venta'], todos['total']['ganancia']), (2, 1.8, 0.48))
        # por la web: un usuario no puede ver a otro ni el global
        with self.c.session_transaction() as s:
            s.update(usuario='beto-g@test.local', user_db_id=self.B, is_admin=False)
        for quien in ('todos', str(self.A)):
            d = self.c.get(f'/ganancias/datos?inicio={DIA}&fin={DIA}&usuario={quien}').get_json()
            self.assertEqual((d['total']['vendidos'], d['total']['venta'], d['puede_editar']), (1, 0.8, True))
        with self.c.session_transaction() as s:
            s.update(usuario='admin@test.local', user_db_id=self.A, is_admin=True)
        d = self.c.get(f'/ganancias/datos?inicio={DIA}&fin={DIA}&usuario=todos').get_json()
        self.assertEqual((d['total']['vendidos'], d['puede_editar']), (2, False))
        self.assertEqual(sorted(u['id'] for u in d['usuarios']), sorted([self.A, self.B]))
        d = self.c.get(f'/ganancias/datos?inicio={DIA}&fin={DIA}&usuario={self.B}&paquete={self.p110}').get_json()
        self.assertEqual([v['usuario'] for v in d['ventas']], ['Beto B'])

    # ---------- precios por la web ----------
    def test_guardar_precios_y_tasa(self):
        with self.c.session_transaction() as s:
            s.update(usuario='ana-g@test.local', user_db_id=self.A, is_admin=False)
        self.assertTrue(self.c.post('/ganancias/precio', json={'paquete_id': self.p110, 'tipo': 'venta', 'moneda': 'VES', 'monto': '250,5'}).get_json()['ok'])
        self.c.post('/ganancias/precio', json={'paquete_id': self.p110, 'tipo': 'venta', 'moneda': 'VES', 'monto': 250.5})   # igual: no repite
        self.assertEqual(len(q('SELECT id FROM ganancias_precios')), 1)
        self.assertEqual(self.c.post('/ganancias/tasa', json={'tasa': 0}).status_code, 400)
        self.assertTrue(self.c.post('/ganancias/tasa', json={'tasa': '245,50'}).get_json()['ok'])
        d = self.c.get('/ganancias/precios').get_json()
        self.assertEqual(d['tasa'], 245.5)
        ff = next(j for j in d['juegos'] if j['id'] == self.ff)
        self.assertEqual(next(p for p in ff['paquetes'] if p['id'] == self.p110)['venta'], {'moneda': 'VES', 'monto': 250.5})
        self.assertTrue(next(j for j in d['juegos'] if j['id'] == self.propio)['propio'])
        # el costo solo en los paquetes con su proveedor; un paquete de otro usuario no existe
        self.assertEqual(self.c.post('/ganancias/precio', json={'paquete_id': self.p110, 'tipo': 'costo', 'monto': 1}).status_code, 400)
        self.assertTrue(self.c.post('/ganancias/precio', json={'paquete_id': self.ml, 'tipo': 'costo', 'monto': 1.8}).get_json()['ok'])
        with self.c.session_transaction() as s:
            s.update(usuario='beto-g@test.local', user_db_id=self.B, is_admin=False)
        self.assertEqual(self.c.post('/ganancias/precio', json={'paquete_id': self.ml, 'tipo': 'venta', 'monto': 1}).status_code, 400)
        self.assertEqual(self.c.get('/ganancias/precios').get_json()['tasa'], None)   # la tasa es de cada uno

    def test_el_dashboard_trae_la_seccion(self):
        with self.c.session_transaction() as s:
            s.update(usuario='ana-g@test.local', user_db_id=self.A, is_admin=False)
        html = self.c.get('/dashboard').get_data(as_text=True)
        self.assertIn('id="ganancias"', html)
        self.assertIn('/ganancias/datos', html)
        self.assertEqual(self.c.get('/ganancias/datos').status_code, 200)
        self.c = self.app.test_client()
        self.assertEqual(self.c.get('/ganancias/datos').status_code, 401)


if __name__ == '__main__':
    unittest.main()

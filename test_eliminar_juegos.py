"""
Eliminar juegos y paquetes del admin: se borran con su Mapeo; los que tienen ventas o PINs no se borran (historial).
Las pantallas piden con fetch (Accept: application/json) y siempre reciben JSON, nunca una redirección.

    python -m unittest test_eliminar_juegos -v
"""
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'juegos.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30

import app as webapp  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

JSON = {'Accept': 'application/json'}   # como lo piden los botones (sin cuerpo)


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if (sql.lstrip().upper().startswith('SELECT') or 'RETURNING' in sql.upper()) else None
        conn.commit()
        return rows
    finally:
        conn.close()


def contar(tabla, columna, valor):
    return q(f'SELECT COUNT(*) AS n FROM {tabla} WHERE {columna} = ?', (valor,))[0]['n']


class EliminarJuegosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webapp.app.config['TESTING'] = True
        cls.uid = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                    "VALUES ('Admin', 'A', '0', 'admin-juegos@test.local', 'x', 0) RETURNING id")[0][0]

    def setUp(self):
        self.c = webapp.app.test_client()
        with self.c.session_transaction() as s:
            s.update(usuario='admin@test.local', user_db_id=self.uid, is_admin=True)

    def juego(self, nombre):
        jid = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo) VALUES (?, ?, 1, 'id', TRUE) "
                'RETURNING id', (nombre, nombre.lower().replace(' ', '-')))[0][0]
        pid = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) VALUES (?, '100 💎', 1.0, TRUE, 0) "
                'RETURNING id', (jid,))[0][0]
        q("INSERT INTO rev_item_mappings (juego_id, paquete_id, remote_product_id, remote_package_id, auto_enabled, active) "
          "VALUES (?, ?, '1', '2', TRUE, TRUE)", (jid, pid))
        q("INSERT INTO rev_item_mapping_steps (juego_id, paquete_id, orden, remote_product_id, remote_package_id) "
          "VALUES (?, ?, 0, '1', '2')", (jid, pid))
        q("INSERT INTO rev_item_mapping_respaldo (juego_id, paquete_id, orden, remote_product_id, remote_package_id) "
          "VALUES (?, ?, 0, '9', '8')", (jid, pid))
        return jid, pid

    def venta(self, jid, pid):
        q("INSERT INTO transacciones_dinamicas (juego_id, usuario_id, paquete_id, numero_control, transaccion_id, monto, estado) "
          "VALUES (?, ?, ?, 'NC1', 'DG1', 1.0, 'aprobado')", (jid, self.uid, pid))

    def test_juego_con_mapeo_y_sin_ventas_se_borra_con_su_mapeo(self):
        jid, pid = self.juego('free fire prueba')
        r = self.c.post(f'/admin/dynamic-games/{jid}/delete', headers=JSON)
        self.assertEqual((r.status_code, r.get_json()), (200, {'success': True}))   # JSON, no una redirección
        self.assertEqual(contar('juegos_dinamicos', 'id', jid), 0)
        self.assertEqual(contar('paquetes_dinamicos', 'juego_id', jid), 0)
        for tabla in ('rev_item_mappings', 'rev_item_mapping_steps', 'rev_item_mapping_respaldo'):
            self.assertEqual(contar(tabla, 'juego_id', jid), 0, tabla)

    def test_juego_con_ventas_no_se_borra(self):
        jid, pid = self.juego('Free fire ID')
        self.venta(jid, pid)
        r = self.c.post(f'/admin/dynamic-games/{jid}/delete', headers=JSON)
        self.assertEqual(r.status_code, 409)
        self.assertIn('1 venta', r.get_json()['error'])
        self.assertIn('Desactívalo', r.get_json()['error'])
        self.assertEqual(contar('juegos_dinamicos', 'id', jid), 1)            # nada se borró
        self.assertEqual(contar('transacciones_dinamicas', 'juego_id', jid), 1)
        self.assertEqual(contar('rev_item_mappings', 'juego_id', jid), 1)

    def test_juego_con_pines_no_se_borra(self):
        jid, pid = self.juego('Pines prueba')
        q("INSERT INTO pines_dinamicos_stock (juego_id, paquete_id, pin_codigo) VALUES (?, ?, 'PIN-1')", (jid, pid))
        r = self.c.post(f'/admin/dynamic-games/{jid}/delete', headers=JSON)
        self.assertEqual(r.status_code, 409)
        self.assertIn('1 PIN cargado', r.get_json()['error'])
        self.assertEqual(contar('pines_dinamicos_stock', 'juego_id', jid), 1)

    def test_paquete_con_mapeo_se_borra_y_con_ventas_no(self):
        jid, pid = self.juego('Paquetes prueba')
        otro = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) VALUES (?, '200 💎', 2.0, TRUE, 1) "
                 'RETURNING id', (jid,))[0][0]
        self.venta(jid, otro)
        r = self.c.post(f'/admin/dynamic-games/packages/{pid}/delete', headers=JSON)
        self.assertEqual((r.status_code, r.get_json()), (200, {'success': True}))
        self.assertEqual(contar('paquetes_dinamicos', 'id', pid), 0)
        self.assertEqual(contar('rev_item_mappings', 'paquete_id', pid), 0)
        r = self.c.post(f'/admin/dynamic-games/packages/{otro}/delete', headers=JSON)
        self.assertEqual(r.status_code, 409)
        self.assertEqual(contar('paquetes_dinamicos', 'id', otro), 1)

    def test_formulario_sin_json_sigue_redirigiendo_y_solo_admin(self):
        jid, _ = self.juego('Formulario')
        self.assertEqual(self.c.post(f'/admin/dynamic-games/{jid}/delete').status_code, 302)
        self.assertEqual(contar('juegos_dinamicos', 'id', jid), 0)
        jid, _ = self.juego('Sin permiso')
        with self.c.session_transaction() as s:
            s['is_admin'] = False
        self.assertEqual(self.c.post(f'/admin/dynamic-games/{jid}/delete', headers=JSON).status_code, 403)
        self.assertEqual(contar('juegos_dinamicos', 'id', jid), 1)

    def test_las_pantallas_ya_no_llaman_a_alertesc(self):
        for plantilla in ('admin.html', 'admin_dynamic_games.html'):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', plantilla), encoding='utf-8') as f:
                self.assertNotIn('alertesc(', f.read(), plantilla)


if __name__ == '__main__':
    unittest.main()

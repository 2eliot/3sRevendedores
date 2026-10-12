"""
Panel admin: los PINs del Bot de Free Fire solo están en su apartado (cargar, importar CSV y restaurar backup), el
backup nunca vuelve a cargar un PIN que ya existe, y los paquetes de un juego creado se pueden eliminar desde Precios.

    python -m unittest test_admin_pines_paquetes -v
"""
import io
import json
import os
import re
import tempfile
import unittest
import zipfile

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'admin.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30

import app as webapp  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

RAIZ = os.path.dirname(os.path.abspath(__file__))


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if (sql.lstrip().upper().startswith('SELECT') or 'RETURNING' in sql.upper()) else None
        conn.commit()
        return rows
    finally:
        conn.close()


def plantilla(nombre):
    with open(os.path.join(RAIZ, 'templates', nombre), encoding='utf-8') as f:
        return f.read()


class AdminPinesPaquetesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webapp.app.config['TESTING'] = True
        cls.uid = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                    "VALUES ('Admin', 'A', '0', 'admin-pines@test.local', 'x', 0) RETURNING id")[0][0]

    def setUp(self):
        q('DELETE FROM pines_freefire_global')
        self.c = webapp.app.test_client()
        with self.c.session_transaction() as s:
            s.update(usuario='admin@test.local', user_db_id=self.uid, is_admin=True)

    def pines(self):
        return sorted(r['pin_codigo'] for r in q('SELECT pin_codigo FROM pines_freefire_global'))

    # ---------- los PINs del bot, solo en su apartado ----------
    def test_gestion_de_pines_ya_no_muestra_los_del_bot(self):
        html = plantilla('admin.html')
        selector = html[html.index('id="pin_game_selector"'):html.index('</select>', html.index('id="pin_game_selector"'))]
        self.assertNotIn('freefire_global', selector)
        self.assertNotIn('Bot de Free Fire (PINs)', html)
        self.assertNotIn('Restaurar Backup de Pines', html)
        self.assertIn('href="/admin/bot-freefire"', html)

    def test_el_apartado_del_bot_importa_csv_y_restaura_backup(self):
        html = self.c.get('/admin/bot-freefire').get_data(as_text=True)
        self.assertIn('action="/admin/import_pins_csv"', html)
        self.assertIn('action="/admin/restore_backup"', html)
        self.assertEqual(html.count('name="volver" value="/admin/bot-freefire"'), 2)

    def test_importar_csv_vuelve_al_apartado_del_bot(self):
        csv = io.BytesIO(b'Id,Serial,Clave\n1,S1,ABCD1234EFGH\n2,S2,IJKL5678MNOP\n')
        r = self.c.post('/admin/import_pins_csv', data={'game_type': 'freefire_global', 'batch_monto_id': '1',
                                                        'volver': '/admin/bot-freefire', 'csv_file': (csv, 'pines-prueba-1.csv')},
                        content_type='multipart/form-data')
        self.assertEqual(r.status_code, 302)
        self.assertTrue(r.headers['Location'].endswith('/admin/bot-freefire'))
        self.assertEqual(self.pines(), ['ABCD1234EFGH', 'IJKL5678MNOP'])
        self.assertIn('Se importaron 2 pines', self.c.get('/admin/bot-freefire').get_data(as_text=True))   # el mensaje se ve allí
        # nunca a una dirección de fuera
        r = self.c.post('/admin/import_pins_csv', data={'volver': 'https://otro-sitio.com/'}, content_type='multipart/form-data')
        self.assertTrue(r.headers['Location'].endswith('/admin'))

    def test_backup_solo_carga_pines_que_no_existen(self):
        q("INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado) VALUES (1, 'PIN-USADO', TRUE)")
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, 'w') as z:
            z.writestr('pines_freefire_global_1.csv', 'pin_codigo\nPIN-USADO\nPIN-NUEVO\nPIN-NUEVO\n')
        zbuf.seek(0)
        r = self.c.post('/admin/restore_backup', data={'volver': '/admin/bot-freefire', 'backup_zip': (zbuf, 'backup_2026-10-12.zip')},
                        content_type='multipart/form-data')
        self.assertEqual(r.status_code, 302)
        self.assertTrue(r.headers['Location'].endswith('/admin/bot-freefire'))
        self.assertEqual(self.pines(), ['PIN-NUEVO', 'PIN-USADO'])   # el usado no vuelve como nuevo; el repetido entra una vez
        self.assertEqual(q("SELECT usado FROM pines_freefire_global WHERE pin_codigo = 'PIN-USADO'")[0]['usado'], 1)

    # ---------- eliminar paquetes desde Precios ----------
    def test_precios_tiene_eliminar_y_activar_paquetes(self):
        html = plantilla('admin.html')
        for texto in ('data-del-pkg', 'data-act-pkg', "/admin/dynamic-games/packages/${id}/delete", "/packages/${id}/update"):
            self.assertIn(texto, html)
        self.assertIn('"activo":', html)

    def test_los_datos_de_los_juegos_son_json_valido_aunque_el_nombre_tenga_comillas(self):
        """El bloque dyn-games-data que usa Precios se arma con el nombre en JSON (tojson)."""
        jid = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo) VALUES ('Juego \"X\"', 'juego-x', 1, 'id', TRUE) "
                'RETURNING id')[0][0]
        q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo, orden) VALUES (?, 'Pase \"VIP\"', 2.5, FALSE, 0)", (jid,))
        from flask import render_template_string
        bloque = plantilla('admin.html')
        bloque = bloque[bloque.index('<script type="application/json" id="dyn-games-data">'):]
        bloque = bloque[:bloque.index('</script>') + len('</script>')]
        from dynamic_games import get_all_dynamic_games, get_dynamic_packages
        juegos = get_all_dynamic_games()
        for g in juegos:
            g['_packages'] = get_dynamic_packages(g['id'])
            g.update(_edit_dual_id=False, _edit_server_enabled=False, _edit_server_options='', _edit_categoria='giftcard')
        with webapp.app.test_request_context('/admin'):
            html = render_template_string(bloque, dyn_games=juegos)
        datos = json.loads(re.sub(r'^<script[^>]*>|</script>$', '', html.strip()))
        paquete = datos['dyn_juego-x']['packages'][0]
        self.assertEqual((paquete['nombre'], paquete['activo']), ('Pase "VIP"', False))


if __name__ == '__main__':
    unittest.main()

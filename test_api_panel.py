"""
Pruebas del apartado API (cuentas, permisos, verificar ID e idempotencia de recargas)
sobre una base SQLite temporal.

    python -m unittest test_api_panel -v
"""
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'api_panel_test.db')
os.environ.pop('DATABASE_URL', None)

from flask import Flask  # noqa: E402

import api_panel  # noqa: E402
import api_whitelabel  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402


def crear_app():
    app = Flask(__name__, template_folder='templates')
    app.secret_key = 'test'
    app.jinja_env.globals['brand_css_url'] = lambda: ''
    app.register_blueprint(api_panel.bp)
    app.register_blueprint(api_whitelabel.bp)
    return app


def base_inicial():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS usuarios (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, apellido TEXT, '
                 'telefono TEXT, correo TEXT, contraseña TEXT, saldo REAL DEFAULT 0)')
    conn.execute('CREATE TABLE IF NOT EXISTS configuracion_redeemer (clave TEXT PRIMARY KEY, valor TEXT NOT NULL, '
                 'fecha_actualizacion TEXT)')
    api_whitelabel.init_whitelabel_tables(conn)
    conn.commit()
    conn.close()
    api_panel.init_permisos()


def crear_cuenta(clave, **perms):
    conn = get_db_connection()
    cols = ['perm_recargas', 'perm_verificar_id', 'perm_verificar_pago']
    vals = [perms.get(c[5:], False) for c in cols]
    cur = conn.execute(f"INSERT INTO webservice_accounts (nombre, api_key, usuario_id, activo, {', '.join(cols)}) "
                       "VALUES ('CRM', ?, 1, TRUE, ?, ?, ?) RETURNING id", (clave, *vals))
    aid = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return aid


def saldo():
    conn = get_db_connection()
    v = conn.execute('SELECT saldo FROM usuarios WHERE id = 1').fetchone()['saldo']
    conn.close()
    return v


class ApiPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_inicial()
        cls.app = crear_app()

    def setUp(self):
        conn = get_db_connection()
        conn.execute('DELETE FROM api_orders')
        conn.execute('DELETE FROM webservice_accounts')
        conn.execute('DELETE FROM usuarios')
        conn.execute("INSERT INTO usuarios (id, nombre, apellido, correo, saldo) VALUES (1, 'CRM', 'Bot', 'crm@x.com', 50)")
        conn.commit()
        conn.close()
        api_panel._rate.clear()
        self.c = self.app.test_client()

    # --- Permisos en la API ----------------------------------------------------
    def test_recarga_sin_permiso_403(self):
        crear_cuenta('wsk_a', recargas=False, verificar_id=True)
        r = self.c.post('/api/v1/recharge', headers={'X-API-Key': 'wsk_a'},
                        json={'product_id': -1, 'package_id': 1, 'player_id': '123'})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(saldo(), 50)

    def test_recarga_con_external_order_id_repetida_no_cobra(self):
        aid = crear_cuenta('wsk_b', recargas=True)
        conn = get_db_connection()
        conn.execute("INSERT INTO api_orders (account_id, usuario_id, game_type, package_id, player_id, precio, estado, "
                     "external_order_id, player_name) VALUES (?, 1, 'freefire_id', 1, '123', 5, 'completada', 'ORD-1', 'Pro')",
                     (aid,))
        conn.commit()
        conn.close()
        r = self.c.post('/api/v1/recharge', headers={'X-API-Key': 'wsk_b'},
                        json={'product_id': -1, 'package_id': 1, 'player_id': '123', 'external_order_id': 'ORD-1'})
        d = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(d['duplicada'])
        self.assertEqual(d['player_name'], 'Pro')
        self.assertEqual(saldo(), 50)

    def test_verify_id_permisos_y_respuesta(self):
        crear_cuenta('wsk_sin', verificar_id=False)
        crear_cuenta('wsk_con', verificar_id=True)
        body = {'product_id': 5, 'player_id': '123456789'}
        self.assertEqual(self.c.post('/api/v1/verify-id', json=body).status_code, 401)
        self.assertEqual(self.c.post('/api/v1/verify-id', headers={'X-API-Key': 'wsk_sin'}, json=body).status_code, 403)

        import dynamic_games
        import id_verify
        orig = (api_panel._verify_cfg_para, id_verify.call_verify_api, dynamic_games.get_dynamic_game_by_id)
        api_panel._verify_cfg_para = lambda g: {'enabled': True, 'url': 'x', 'name_path': 'n'}
        id_verify.call_verify_api = lambda cfg, a, b='', c='': (True, 'ProPlayer99')
        dynamic_games.get_dynamic_game_by_id = lambda i: {'id': 5, 'activo': True, 'modo': 'id', 'slug': 'ff'}
        try:
            d = self.c.post('/api/v1/verify-id', headers={'X-API-Key': 'wsk_con'}, json=body).get_json()
        finally:
            api_panel._verify_cfg_para, id_verify.call_verify_api, dynamic_games.get_dynamic_game_by_id = orig
        self.assertEqual(d, {'ok': True, 'player_id': '123456789', 'player_name': 'ProPlayer99'})

    def test_verify_id_juego_sin_verificacion(self):
        crear_cuenta('wsk_con', verificar_id=True)
        r = self.c.post('/api/v1/verify-id', headers={'X-API-Key': 'wsk_con'}, json={'product_id': -1, 'player_id': '1'})
        self.assertEqual(r.status_code, 404)

    def test_cuenta_desactivada_401(self):
        aid = crear_cuenta('wsk_off', verificar_id=True)
        conn = get_db_connection()
        conn.execute('UPDATE webservice_accounts SET activo = FALSE WHERE id = ?', (aid,))
        conn.commit()
        conn.close()
        r = self.c.post('/api/v1/verify-id', headers={'X-API-Key': 'wsk_off'}, json={'product_id': -1, 'player_id': '1'})
        self.assertEqual(r.status_code, 401)

    # --- Admin -----------------------------------------------------------------
    def admin(self):
        with self.c.session_transaction() as s:
            s['is_admin'] = True

    def test_admin_crea_edita_regenera_y_elimina(self):
        self.admin()
        d = self.c.post('/admin/api/cuentas', json={'nombre': 'CRM', 'usuario_id': 1,
                                                     'permisos': {'recargas': True, 'verificar_pago': True}}).get_json()
        self.assertTrue(d['ok'], d)
        cta = d['cuenta']
        self.assertTrue(cta['api_key'].startswith('wsk_'))
        self.assertEqual(cta['permisos'], {'recargas': True, 'verificar_id': False, 'verificar_pago': True})

        d = self.c.post(f"/admin/api/cuentas/{cta['id']}", json={'permisos': {'verificar_id': True, 'recargas': False}}).get_json()
        self.assertEqual(d['cuenta']['permisos'], {'recargas': False, 'verificar_id': True, 'verificar_pago': True})

        d = self.c.post(f"/admin/api/cuentas/{cta['id']}/regenerar", json={}).get_json()
        self.assertNotEqual(d['cuenta']['api_key'], cta['api_key'])

        d = self.c.post(f"/admin/api/cuentas/{cta['id']}/eliminar", json={}).get_json()
        self.assertTrue(d['ok'])
        self.assertEqual(self.c.get('/admin/api/cuentas').get_json()['cuentas'], [])

    def test_admin_no_borra_cuenta_con_recargas(self):
        self.admin()
        aid = crear_cuenta('wsk_h', recargas=True)
        conn = get_db_connection()
        conn.execute("INSERT INTO api_orders (account_id, usuario_id, game_type, package_id, player_id, precio, estado) "
                     "VALUES (?, 1, 'freefire_id', 1, '1', 1, 'completada')", (aid,))
        conn.commit()
        conn.close()
        d = self.c.post(f'/admin/api/cuentas/{aid}/eliminar', json={}).get_json()
        self.assertTrue(d.get('desactivada'))
        self.assertFalse(self.c.get('/admin/api/cuentas').get_json()['cuentas'][0]['activo'])

    def test_admin_valida_usuario(self):
        self.admin()
        r = self.c.post('/admin/api/cuentas', json={'nombre': 'X', 'usuario_id': 999})
        self.assertEqual(r.status_code, 404)

    def test_solo_admin(self):
        self.assertEqual(self.c.get('/admin/api/cuentas').status_code, 403)
        self.assertEqual(self.c.post('/admin/api/cuentas', json={}).status_code, 403)
        self.assertEqual(self.c.get('/admin/api', follow_redirects=False).status_code, 302)

    def test_docs_se_muestra(self):
        self.admin()
        orig = api_panel.catalogo
        api_panel.catalogo = lambda: [{'product_id': 12, 'nombre': 'Free fire ID', 'slug': 'free-fire-id', 'modo': 'id', 'icono': '',
                                       'player_id2': None, 'servidor': None, 'verifica_id': True,
                                       'paquetes': [{'package_id': 1, 'nombre': '100 Diamantes', 'precio': 0.86,
                                                     'recargable': True}]}]
        try:
            html = self.c.get('/admin/api/docs').get_data(as_text=True)
        finally:
            api_panel.catalogo = orig
        self.assertIn('100 Diamantes', html)
        self.assertIn('/api/v1/recharge', html)


if __name__ == '__main__':
    unittest.main()

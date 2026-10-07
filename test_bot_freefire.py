"""
Pruebas del Bot de Free Fire, de la API solo con juegos mapeados y del retiro de los juegos
de ejemplo del código. Arranca la web completa sobre una base SQLite temporal y simula el
proveedor (Mapeo) y el VPS del bot.

    python -m unittest test_bot_freefire -v
"""
import json
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'bot_test.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['REVENDEDORES_BASE_URL'] = 'http://proveedor.test'
os.environ['REVENDEDORES_API_KEY'] = 'clave-proveedor'

import app as webapp  # noqa: E402
import bot_freefire  # noqa: E402
import dynamic_games  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402
from pin_redeemer import PinRedeemResult  # noqa: E402


def q(sql, params=(), one=False):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        if sql.lstrip().upper().startswith(('SELECT', 'INSERT', 'WITH')) and 'RETURNING' in sql.upper() or sql.lstrip().upper().startswith('SELECT'):
            rows = cur.fetchall()
            conn.commit()
            return rows[0] if (one and rows) else (None if one else rows)
        conn.commit()
    finally:
        conn.close()


class Proveedor:
    """Simula /api/v1/recharge del proveedor."""
    def __init__(self):
        self.llamadas, self.respuesta = [], {'ok': True, 'reference_no': 'REF-P', 'player_name': 'Prov'}

    def __call__(self, path, payload=None, query=None):
        self.llamadas.append(payload)
        return dict(self.respuesta)


class VPS:
    """Simula el canje de PINs del bot."""
    def __init__(self):
        self.canjeados, self.respuestas = [], []

    def __call__(self, pin, player_id, ref):
        self.canjeados.append(pin)
        ok, msg = self.respuestas.pop(0) if self.respuestas else (True, 'Recarga completada (VPS)')
        return PinRedeemResult(ok, msg, pin, player_id, player_name='BotPlayer' if ok else '')


class BotFreeFireTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        cls.juego = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo) "
                      "VALUES ('Free fire ID', 'free-fire-id', 0, 'id', TRUE) RETURNING id", one=True)[0]
        cls.p110 = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo) VALUES (?, '110 💎', 1.0, TRUE) RETURNING id",
                     (cls.juego,), one=True)[0]
        cls.p220 = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo) VALUES (?, '220 💎', 2.0, TRUE) RETURNING id",
                     (cls.juego,), one=True)[0]
        cls.sinmapa = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo) VALUES (?, '341 💎', 3.0, TRUE) RETURNING id",
                        (cls.juego,), one=True)[0]
        for pid, rp in ((cls.p110, '11'), (cls.p220, '22')):
            q("INSERT INTO rev_item_mappings (juego_id, paquete_id, remote_product_id, remote_package_id, auto_enabled, active) "
              "VALUES (?, ?, '7', ?, TRUE, TRUE)", (cls.juego, pid, rp))
        cls.uid = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                    "VALUES ('CRM', 'Bot', '0', 'crm@test.local', 'x', 100) RETURNING id", one=True)[0]
        import api_panel
        api_panel.init_permisos()
        q("INSERT INTO webservice_accounts (nombre, api_key, usuario_id, activo, perm_recargas, perm_verificar_id) "
          "VALUES ('CRM', 'wsk_test', ?, TRUE, TRUE, TRUE)", (cls.uid,))

    def setUp(self):
        self.prov, self.vps = Proveedor(), VPS()
        self._orig = (dynamic_games._reseller_call, bot_freefire._canjear)
        dynamic_games._reseller_call = self.prov
        bot_freefire._canjear = self.vps
        q('DELETE FROM pines_freefire_global')
        q('DELETE FROM api_orders')
        q("DELETE FROM configuracion_redeemer WHERE clave IN ('bot_ff_config', 'bot_ff_log', 'bot_ff_dudosos')")
        q('UPDATE usuarios SET saldo = 100 WHERE id = ?', (self.uid,))
        self.c = self.app.test_client()

    def tearDown(self):
        dynamic_games._reseller_call, bot_freefire._canjear = self._orig

    # ---------- ayudas ----------
    def bot(self, activo=True, cantidad220=2):
        bot_freefire._guardar(bot_freefire.CONFIG_KEY, {'activo': activo, 'juego_id': self.juego, 'paquetes': {
            str(self.p110): {'monto_id': 1, 'cantidad': 1}, str(self.p220): {'monto_id': 1, 'cantidad': cantidad220}}})

    def pines(self, n):
        for i in range(n):
            q('INSERT INTO pines_freefire_global (monto_id, pin_codigo, usado) VALUES (1, ?, FALSE)', (f'PIN{i:012d}',))

    def stock(self):
        return q('SELECT COUNT(*) AS n FROM pines_freefire_global WHERE usado = FALSE', one=True)['n']

    def saldo(self):
        return float(q('SELECT saldo FROM usuarios WHERE id = ?', (self.uid,), one=True)['saldo'])

    def recargar(self, pkg, ext=None):
        body = {'product_id': self.juego, 'package_id': pkg, 'player_id': '1271181990'}
        if ext:
            body['external_order_id'] = ext
        return self.c.post('/api/v1/recharge', headers={'X-API-Key': 'wsk_test'}, json=body)

    # ---------- API: solo juegos creados y mapeados ----------
    def test_catalogo_solo_paquetes_mapeados_sin_juegos_de_ejemplo(self):
        d = self.c.get('/api/v1/products', headers={'X-API-Key': 'wsk_test'}).get_json()
        juegos = {g['game_id']: g for g in d['games']}
        self.assertNotIn(-1, juegos)
        self.assertNotIn(-155, juegos)
        self.assertEqual(sorted(p['name'] for p in juegos[self.juego]['packages']), ['110 💎', '220 💎'])
        self.assertEqual({p['package_id']: p['price'] for p in juegos[self.juego]['packages']}[self.p110], 1.0)

    def test_api_rechaza_paquete_sin_mapeo_y_juegos_de_ejemplo(self):
        self.assertEqual(self.recargar(self.sinmapa).status_code, 404)
        r = self.c.post('/api/v1/recharge', headers={'X-API-Key': 'wsk_test'},
                        json={'product_id': -1, 'package_id': 1, 'player_id': '123'})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(self.saldo(), 100)

    def test_api_recarga_por_proveedor(self):
        r = self.recargar(self.p110, 'ORD-1')
        d = r.get_json()
        self.assertEqual((r.status_code, d['status'], d['reference_no']), (200, 'completada', 'REF-P'))
        self.assertEqual(self.saldo(), 99)
        self.assertEqual(len(self.prov.llamadas), 1)
        h = q("SELECT pin FROM transacciones WHERE usuario_id = ? ORDER BY id DESC LIMIT 1", (self.uid,), one=True)
        self.assertIn('[API: CRM]', h['pin'])
        # repetir la misma orden no cobra ni recarga otra vez
        d2 = self.recargar(self.p110, 'ORD-1').get_json()
        self.assertTrue(d2['duplicada'])
        self.assertEqual((self.saldo(), len(self.prov.llamadas)), (99, 1))

    def test_api_fallo_del_proveedor_devuelve_saldo(self):
        self.prov.respuesta = {'ok': False, 'error': 'ID inválido'}
        r = self.recargar(self.p110)
        self.assertEqual((r.status_code, r.get_json()['status']), (422, 'fallida'))
        self.assertEqual(self.saldo(), 100)

    def test_api_en_proceso_y_se_sincroniza(self):
        self.prov.respuesta = {'ok': False, 'pending': True, 'status': 'procesando'}
        r = self.recargar(self.p110, 'ORD-P')
        self.assertEqual((r.status_code, r.get_json()['status']), (202, 'procesando'))
        orden = q("SELECT reference_no FROM api_orders WHERE external_order_id = 'ORD-P'", one=True)
        # el reconciliador cierra la compra
        tx = q('SELECT * FROM transacciones_dinamicas WHERE transaccion_id = ?', (orden['reference_no'],), one=True)
        state = json.loads(tx['notas'][len(dynamic_games.MULTI_PREFIX):])
        state['units'][0].update(st='ok', ref='REF-TARDE', name='Tarde')
        dynamic_games._multi_finalize({'id': tx['id'], 'usuario_id': self.uid, 'monto': tx['monto'], 'numero_control': tx['numero_control'],
                                       'transaccion_id': tx['transaccion_id'], 'player_id': tx['player_id'], 'player_id2': '',
                                       'paquete_id': self.p110, 'juego_nombre': 'Free fire ID', 'slug': 'free-fire-id',
                                       'paquete_nombre': '110 💎'}, state, True)
        d = self.c.get('/api/v1/order-status?external_order_id=ORD-P', headers={'X-API-Key': 'wsk_test'}).get_json()
        self.assertEqual((d['status'], d['order']['player_name']), ('completada', 'Tarde'))

    # ---------- Bot de Free Fire ----------
    def test_bot_apagado_va_directo_al_proveedor(self):
        self.bot(activo=False)
        self.pines(3)
        self.recargar(self.p110)
        self.assertEqual((len(self.prov.llamadas), len(self.vps.canjeados), self.stock()), (1, 0, 3))

    def test_bot_encendido_usa_pines_antes_que_el_proveedor(self):
        self.bot()
        self.pines(3)
        d = self.recargar(self.p220).get_json()
        self.assertEqual((d['status'], d['player_name']), ('completada', 'BotPlayer'))
        self.assertEqual((len(self.vps.canjeados), len(self.prov.llamadas), self.stock()), (2, 0, 1))
        self.assertEqual(self.saldo(), 98)

    def test_bot_sin_pines_suficientes_usa_el_proveedor(self):
        self.bot()
        self.pines(1)
        self.recargar(self.p220)
        self.assertEqual((len(self.vps.canjeados), len(self.prov.llamadas), self.stock()), (0, 1, 1))

    def test_bot_falla_con_pin_intacto_lo_devuelve_y_sigue_con_proveedor(self):
        self.bot()
        self.pines(1)
        self.vps.respuestas = [(False, 'No se pudo conectar al VPS. Verifica que esté encendido.')]
        d = self.recargar(self.p110).get_json()
        self.assertEqual((d['status'], d['reference_no']), ('completada', 'REF-P'))
        self.assertEqual((self.stock(), len(self.prov.llamadas)), (1, 1))

    def test_bot_sin_respuesta_aparta_el_pin(self):
        self.bot()
        self.pines(1)
        self.vps.respuestas = [(False, 'El VPS no respondió en 60s. Reintenta.')]
        self.recargar(self.p110)
        self.assertEqual((self.stock(), len(self.prov.llamadas)), (0, 1))
        self.assertEqual(len(bot_freefire._leer(bot_freefire.DUDOSOS_KEY, [])), 1)

    def test_bot_parcial_devuelve_la_parte_no_entregada(self):
        self.bot()
        self.pines(2)
        self.vps.respuestas = [(True, 'ok'), (False, 'Player ID inválido (debe ser numérico)')]
        d = self.recargar(self.p220).get_json()
        self.assertTrue(d['parcial'])
        self.assertEqual((d['cobrado'], d['devuelto']), (1.0, 1.0))
        self.assertEqual((self.saldo(), self.stock(), len(self.prov.llamadas)), (99, 1, 0))

    def test_compra_web_usa_el_bot(self):
        self.bot()
        self.pines(1)
        with self.c.session_transaction() as s:
            s.update(usuario='crm@test.local', user_db_id=self.uid, is_admin=False); s['dg_nonce_free-fire-id'] = 'N1'
        r = self.c.post('/validar/dinamico/free-fire-id', data={'dg_form_nonce': 'N1', 'monto': self.p110, 'player_id': '1271181990'})
        self.assertIn('compra=exitosa', r.headers['Location'])
        self.assertEqual((len(self.vps.canjeados), len(self.prov.llamadas), self.saldo()), (1, 0, 99))

    def test_admin_bot_requiere_admin_y_guarda(self):
        self.assertEqual(self.c.get('/admin/bot-freefire/datos').status_code, 403)
        with self.c.session_transaction() as s:
            s['is_admin'] = True
        d = self.c.post('/admin/bot-freefire/config', json={'activo': True, 'juego_id': self.juego,
                                                           'paquetes': {str(self.p110): {'monto_id': 1, 'cantidad': 1},
                                                                        str(self.p220): {'monto_id': 999}}}).get_json()
        self.assertEqual(d['config']['paquetes'], {str(self.p110): {'monto_id': 1, 'cantidad': 1}})
        self.assertEqual(self.c.get('/admin/bot-freefire').status_code, 200)

    # ---------- Juegos de ejemplo retirados ----------
    def test_sin_paquetes_de_ejemplo_y_rutas_retiradas(self):
        self.assertEqual(q('SELECT COUNT(*) AS n FROM precios_bloodstriker WHERE activo = TRUE', one=True)['n'], 0)
        self.assertEqual(q('SELECT COUNT(*) AS n FROM precios_freefire_id WHERE activo = TRUE', one=True)['n'], 0)
        with self.c.session_transaction() as s:
            s.update(usuario='crm@test.local', user_db_id=self.uid, is_admin=False)
        for ruta in ('/juego/freefire', '/juego/freefire_id', '/juego/bloodstriker'):
            r = self.c.get(ruta)
            self.assertEqual((r.status_code, r.headers['Location']), (302, '/'), ruta)
        self.assertNotIn('href="/juego/freefire"', self.c.get('/').get_data(as_text=True))
        with self.c.session_transaction() as s:
            s['is_admin'] = True
        self.assertEqual(self.c.get('/juego/freefire_id').headers['Location'], '/admin/bot-freefire')


if __name__ == '__main__':
    unittest.main()

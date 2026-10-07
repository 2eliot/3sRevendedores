"""
Pruebas del proveedor Central One y del respaldo automático entre proveedores, sobre la web completa
con una base SQLite temporal y una API de Central One simulada (no se hace ningún pedido real).

    python -m unittest test_central_one -v
"""
import json
import os
import tempfile
import unittest
import uuid

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'co_test.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['REVENDEDORES_BASE_URL'] = 'http://inefable.test'
os.environ['REVENDEDORES_API_KEY'] = 'clave-inefable'
os.environ['CENTRALONE_API_KEY'] = 'co_live_test'

import app as webapp  # noqa: E402
import central_one  # noqa: E402
import dynamic_games  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

central_one.ESPERA_ENTREGA_S = 0.05
central_one.INTERVALO_S = 0.01

ML_UUID = '22222222-3333-4444-5555-666666666666'
FF_UUID = '33333333-4444-5555-6666-777777777777'
GC_UUID = '11111111-2222-3333-4444-555555555555'
CATALOGO = [
    {'product_id': ML_UUID, 'name': 'Mobile Legends 100 Diamonds', 'reseller_price': '1.9013', 'status': 'active',
     'product_family_id': 'mobile-legends', 'product_family_name': 'Mobile Legends', 'requires_target': True,
     'target_fields': ['game_user_id', 'game_zone_id'], 'target_schema': [], 'in_stock': True},
    {'product_id': FF_UUID, 'name': 'Free Fire 110 Diamonds', 'reseller_price': '0.9500', 'status': 'active',
     'product_family_id': 'free-fire', 'product_family_name': 'Free Fire', 'requires_target': True,
     'target_fields': ['player_id', 'server'],
     'target_schema': [{'key': 'player_id', 'label': 'Player ID', 'type': 'text', 'options': None},
                       {'key': 'server', 'label': 'Server', 'type': 'select',
                        'options': [{'value': 'os_usa', 'label': 'America'}, {'value': 'os_euro', 'label': 'Europe'}]}],
     'in_stock': True},
    {'product_id': GC_UUID, 'name': 'Steam Gift Card 10 USD', 'reseller_price': '9.9900', 'status': 'active',
     'product_family_id': 'steam', 'product_family_name': 'Steam', 'requires_target': False, 'target_fields': [],
     'target_schema': [], 'in_stock': True},
]


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
    """API de Central One simulada."""
    def __init__(self):
        self.catalogo = [dict(i) for i in CATALOGO]
        self.pedidos, self.por_llave, self.posts = {}, {}, []
        self.modo = 'completar'          # completar | lento | sin_stock | caido
        self.consultas_hasta_completar = 0

    def __call__(self, metodo, path, body=None, idem=None, sin_llave=False):
        if path == '/api/v1/catalog':
            return 200, {'items': self.catalogo}
        if path == '/api/v1/balance':
            return 200, {'available_balance': '470.0000', 'held_balance': '0', 'total_balance': '470.0000', 'currency': 'USD'}
        if metodo == 'POST' and path == '/api/v1/orders':
            self.posts.append({'body': body, 'idem': idem})
            if self.modo == 'caido':
                return 0, {}
            if self.modo == 'sin_stock':
                return 409, {'error': {'code': 'insufficient_stock'}}
            if idem in self.por_llave:
                return 201, {'order': self.pedidos[self.por_llave[idem]]['resumen']}
            oid = str(uuid.uuid4())
            item = body['items'][0]
            prod = next(p for p in self.catalogo if p['product_id'] == item['catalog_item_id'])
            codigo = prod['product_id'] == GC_UUID
            self.pedidos[oid] = {'resumen': {'id': oid, 'reference_code': 'CO-ORD-' + oid[:6], 'status': 'confirmed'},
                                 'consultas': 0, 'codigo': codigo, 'target': item.get('target_payload')}
            self.por_llave[idem] = oid
            return 201, {'order': self.pedidos[oid]['resumen']}
        if metodo == 'GET' and path.startswith('/api/v1/orders/'):
            oid = path.split('/')[4]
            p = self.pedidos.get(oid)
            if not p:
                return 404, {'error': {'code': 'not_found'}}
            if path.endswith('/codes'):
                return 200, {'order': {'items': [{'status': 'completed', 'codes': ['A995E04D-6980 - 80980592']}]}}
            p['consultas'] += 1
            listo = self.modo != 'lento' or p['consultas'] > self.consultas_hasta_completar
            est = 'completed' if listo else 'processing'
            return 200, {'order': dict(p['resumen'], status=est, items=[
                {'status': est, 'delivered_count': 1 if (listo and p['codigo']) else 0}])}
        return 404, {'error': {'code': 'not_found'}}


class CentralOneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        cls.juego = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo, campos_config) "
                      "VALUES ('Mobile Legends', 'mobile-legends', 0, 'id', TRUE, ?) RETURNING id",
                      (json.dumps({'campo_id2': {'enabled': True, 'label': 'Zone ID'}}),))[0][0]
        cls.ff = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo, campos_config) "
                   "VALUES ('Free Fire Global', 'ff-global', 0, 'id', TRUE, ?) RETURNING id",
                   (json.dumps({'servidor': {'enabled': True, 'label': 'Servidor', 'opciones': ['America', 'Europe']}}),))[0][0]
        cls.gc = q("INSERT INTO juegos_dinamicos (nombre, slug, gamepoint_product_id, modo, activo) "
                   "VALUES ('Steam', 'steam', 0, 'pin', TRUE) RETURNING id")[0][0]
        cls.p_ml = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo) VALUES (?, '100 💎', 3.0, TRUE) RETURNING id", (cls.juego,))[0][0]
        cls.p_ff = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo) VALUES (?, '110 💎', 1.5, TRUE) RETURNING id", (cls.ff,))[0][0]
        cls.p_gc = q("INSERT INTO paquetes_dinamicos (juego_id, nombre, precio, activo) VALUES (?, '10 USD', 12.0, TRUE) RETURNING id", (cls.gc,))[0][0]
        cls.uid = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                    "VALUES ('Cli', 'Ente', '0', 'cli@test.local', 'x', 100) RETURNING id")[0][0]
        import api_panel
        api_panel.init_permisos()
        q("INSERT INTO webservice_accounts (nombre, api_key, usuario_id, activo, perm_recargas) VALUES ('CRM', 'wsk_co', ?, TRUE, TRUE)", (cls.uid,))

    def setUp(self):
        self.co = FakeCentralOne()
        self.ine_llamadas = []
        self.ine_respuesta = {'ok': True, 'reference_no': 'INE-1', 'player_name': 'Ine'}
        self._orig = (central_one._pedir, dynamic_games._reseller_call)
        central_one._pedir = self.co

        def fake_ine(path, payload=None, query=None):
            self.ine_llamadas.append(payload)
            return dict(self.ine_respuesta)
        dynamic_games._reseller_call = fake_ine
        for t in ('rev_item_mapping_steps', 'rev_item_mappings', 'rev_item_mapping_respaldo', 'api_orders'):
            q(f'DELETE FROM {t}')
        q("DELETE FROM configuracion_redeemer WHERE clave = 'proveedores_respaldo'")
        q('UPDATE usuarios SET saldo = 100 WHERE id = ?', (self.uid,))
        central_one.sincronizar_catalogo(get_db_connection)
        q("INSERT INTO rev_catalog_items (remote_product_id, remote_product_name, remote_package_id, remote_package_name, raw_json) "
          "VALUES ('7', 'ML Inefable', '70', '100 Diamantes', '{\"price\": 2.0}') ON CONFLICT DO NOTHING")
        self.c = self.app.test_client()
        with self.c.session_transaction() as s:
            s['is_admin'] = True

    def tearDown(self):
        central_one._pedir, dynamic_games._reseller_call = self._orig

    # ---------- ayudas ----------
    def mapear(self, juego, pkg, items, respaldo=()):
        r = self.c.post('/admin/revendedores/mapping-steps', json={
            'juego_id': juego, 'paquete_id': pkg, 'auto_enabled': True,
            'items': [{'remote_product_id': a, 'remote_package_id': b, 'cantidad': 1} for a, b in items],
            'respaldo': [{'remote_product_id': a, 'remote_package_id': b, 'cantidad': 1} for a, b in respaldo]})
        self.assertTrue(r.get_json().get('ok'), r.get_json())

    def saldo(self):
        return float(q('SELECT saldo FROM usuarios WHERE id = ?', (self.uid,))[0]['saldo'])

    def recargar(self, juego, pkg, player_id='123456789', player_id2='', ext=None):
        body = {'product_id': juego, 'package_id': pkg, 'player_id': player_id, 'player_id2': player_id2}
        if ext:
            body['external_order_id'] = ext
        return self.c.post('/api/v1/recharge', headers={'X-API-Key': 'wsk_co'}, json=body)

    def comprar_web(self, slug, pkg, **form):
        with self.c.session_transaction() as s:
            s.update(usuario='cli@test.local', user_db_id=self.uid, is_admin=False)
            s[f'dg_nonce_{slug}'] = 'N'
        return self.c.post(f'/validar/dinamico/{slug}', data=dict({'dg_form_nonce': 'N', 'monto': pkg}, **form))

    # ---------- datos que pide cada producto ----------
    def test_armar_target(self):
        item = central_one.item_de_catalogo(get_db_connection(), ML_UUID)
        self.assertEqual(central_one.armar_target(item, '123', '2001'), ({'game_user_id': '123', 'game_zone_id': '2001'}, None))
        _, err = central_one.armar_target(item, '123', '')
        self.assertIn('game_zone_id', err)
        ff = central_one.item_de_catalogo(get_db_connection(), FF_UUID)
        self.assertEqual(central_one.armar_target(ff, '9', servidor='america')[0], {'player_id': '9', 'server': 'os_usa'})
        self.assertIn('no es válido', central_one.armar_target(ff, '9', servidor='Asia')[1])
        self.assertEqual(central_one.armar_target({'requires_target': False}, '9'), (None, None))

    # ---------- catálogo ----------
    def test_sincronizar_catalogo(self):
        rows = q("SELECT remote_product_id, remote_package_name, raw_json FROM rev_catalog_items WHERE remote_product_id LIKE 'co:%' AND active = TRUE")
        self.assertEqual(len(rows), 3)
        ml = next(r for r in rows if r['remote_product_id'] == 'co:mobile-legends')
        self.assertAlmostEqual(json.loads(ml['raw_json'])['price'], 1.9013)
        self.co.catalogo = self.co.catalogo[:2]
        r = central_one.sincronizar_catalogo(get_db_connection)
        self.assertEqual(r['retirados'], 1)
        d = self.c.get('/admin/revendedores/mapping-data').get_json()
        provs = {c['remote_product_id']: c['prov'] for c in d['catalog']}
        self.assertEqual((provs['co:mobile-legends'], provs['7']), ('centralone', 'inefable'))

    # ---------- recargas ----------
    def test_recarga_api_con_central_one(self):
        self.mapear(self.juego, self.p_ml, [('co:mobile-legends', ML_UUID)])
        d = self.recargar(self.juego, self.p_ml, player_id2='2001', ext='CRM-77').get_json()
        self.assertEqual(d['status'], 'completada', d)
        self.assertEqual(self.saldo(), 97)
        post = self.co.posts[0]
        self.assertEqual(post['body']['items'][0]['target_payload'], {'game_user_id': '123456789', 'game_zone_id': '2001'})
        self.assertTrue(post['idem'].startswith('DG'))
        self.assertEqual(self.ine_llamadas, [])

    def test_compra_web_con_servidor_de_central_one(self):
        self.mapear(self.ff, self.p_ff, [('co:free-fire', FF_UUID)])
        r = self.comprar_web('ff-global', self.p_ff, player_id='999', servidor='Europe')
        self.assertIn('compra=exitosa', r.headers['Location'])
        self.assertEqual(self.co.posts[0]['body']['items'][0]['target_payload'], {'player_id': '999', 'server': 'os_euro'})
        self.assertEqual(self.saldo(), 98.5)

    def test_codigo_de_gift_card(self):
        self.mapear(self.gc, self.p_gc, [('co:steam', GC_UUID)])
        r = self.comprar_web('steam', self.p_gc)
        self.assertIn('compra=exitosa', r.headers['Location'])
        tx = q("SELECT pin_entregado, gamepoint_referenceno FROM transacciones_dinamicas ORDER BY id DESC LIMIT 1")[0]
        self.assertEqual(tx['pin_entregado'], 'A995E04D-6980')
        self.assertIn('80980592', tx['gamepoint_referenceno'])

    def test_datos_que_faltan_no_se_cobra_nada_en_central_one(self):
        self.mapear(self.juego, self.p_ml, [('co:mobile-legends', ML_UUID)])
        d = self.recargar(self.juego, self.p_ml, player_id2='').get_json()
        self.assertEqual(d['status'], 'fallida')
        self.assertIn('game_zone_id', d['error'])
        self.assertEqual((self.co.posts, self.saldo()), ([], 100))

    def test_entrega_lenta_queda_en_proceso_y_el_reconciliador_la_cierra(self):
        self.mapear(self.juego, self.p_ml, [('co:mobile-legends', ML_UUID)])
        self.co.modo, self.co.consultas_hasta_completar = 'lento', 50
        r = self.recargar(self.juego, self.p_ml, player_id2='2001', ext='CRM-L')
        self.assertEqual((r.status_code, r.get_json()['status']), (202, 'procesando'))
        self.co.consultas_hasta_completar = 0
        self._forzar_reconciliador()
        dynamic_games.poll_pending_reseller_multi()
        d = self.c.get('/api/v1/order-status?external_order_id=CRM-L', headers={'X-API-Key': 'wsk_co'}).get_json()
        self.assertEqual(d['status'], 'completada')
        self.assertEqual(self.saldo(), 97)

    def test_sin_respuesta_se_reintenta_con_la_misma_llave_sin_comprar_dos_veces(self):
        self.mapear(self.juego, self.p_ml, [('co:mobile-legends', ML_UUID)])
        self.co.modo = 'caido'
        r = self.recargar(self.juego, self.p_ml, player_id2='2001', ext='CRM-C')
        self.assertEqual(r.get_json()['status'], 'procesando')
        self.co.modo = 'completar'
        self._forzar_reconciliador()
        dynamic_games.poll_pending_reseller_multi()
        llaves = [p['idem'] for p in self.co.posts]
        self.assertEqual(len(set(llaves)), 1)        # siempre la misma Idempotency-Key
        self.assertEqual(len(self.co.pedidos), 1)    # un solo pedido real
        d = self.c.get('/api/v1/order-status?external_order_id=CRM-C', headers={'X-API-Key': 'wsk_co'}).get_json()
        self.assertEqual(d['status'], 'completada')

    def _forzar_reconciliador(self):
        # Simula que pasó el margen de 10 minutos de una compra en curso
        for r in q("SELECT id, notas FROM transacciones_dinamicas WHERE estado = 'procesando'"):
            st = json.loads(r['notas'][len(dynamic_games.MULTI_PREFIX):])
            st['ts'] = 0
            q('UPDATE transacciones_dinamicas SET notas = ? WHERE id = ?', (dynamic_games.MULTI_PREFIX + json.dumps(st), r['id']))

    # ---------- respaldo automático ----------
    def test_sin_stock_sin_respaldo_devuelve_el_saldo(self):
        self.mapear(self.juego, self.p_ml, [('co:mobile-legends', ML_UUID)], respaldo=[('7', '70')])
        self.co.modo = 'sin_stock'
        d = self.recargar(self.juego, self.p_ml, player_id2='2001').get_json()
        self.assertEqual(d['status'], 'fallida')
        self.assertIn('stock', d['error'])
        self.assertEqual((self.saldo(), self.ine_llamadas), (100, []))

    def test_respaldo_encendido_usa_el_otro_proveedor(self):
        self.mapear(self.juego, self.p_ml, [('co:mobile-legends', ML_UUID)], respaldo=[('7', '70')])
        self.assertTrue(self.c.post('/admin/proveedores/config', json={'respaldo': True}).get_json()['respaldo'])
        self.co.modo = 'sin_stock'
        d = self.recargar(self.juego, self.p_ml, player_id2='2001').get_json()
        self.assertEqual((d['status'], d['reference_no']), ('completada', 'INE-1'))
        self.assertEqual(len(self.ine_llamadas), 1)
        self.assertEqual(self.saldo(), 97)

    def test_respaldo_de_inefable_a_central_one_en_la_web(self):
        self.mapear(self.juego, self.p_ml, [('7', '70')], respaldo=[('co:mobile-legends', ML_UUID)])
        self.c.post('/admin/proveedores/config', json={'respaldo': True})
        self.ine_respuesta = {'ok': False, 'error': 'Proveedor en mantenimiento'}
        r = self.comprar_web('mobile-legends', self.p_ml, player_id='123', player_id2='2001')
        self.assertIn('compra=exitosa', r.headers['Location'])
        self.assertEqual((len(self.ine_llamadas), len(self.co.posts), self.saldo()), (1, 1, 97))

    def test_respaldo_no_actua_si_el_principal_quedo_en_proceso(self):
        self.mapear(self.juego, self.p_ml, [('7', '70')], respaldo=[('co:mobile-legends', ML_UUID)])
        self.c.post('/admin/proveedores/config', json={'respaldo': True})
        self.ine_respuesta = {'ok': False, 'pending': True, 'status': 'procesando'}
        self.recargar(self.juego, self.p_ml, player_id2='2001')
        self.assertEqual(self.co.posts, [])

    def test_solo_inefable_sin_respaldo_sigue_el_camino_de_siempre(self):
        self.mapear(self.juego, self.p_ml, [('7', '70')])
        before = len(q("SELECT id FROM transacciones_dinamicas"))
        self.comprar_web('mobile-legends', self.p_ml, player_id='123', player_id2='2001')
        tx = q("SELECT notas FROM transacciones_dinamicas ORDER BY id DESC LIMIT 1")[0]
        self.assertEqual(len(q("SELECT id FROM transacciones_dinamicas")), before + 1)
        self.assertFalse(str(tx['notas'] or '').startswith(dynamic_games.MULTI_PREFIX))  # flujo antiguo de un ítem

    # ---------- admin ----------
    def test_mapeo_guarda_respaldo_y_apartado_proveedores(self):
        self.mapear(self.juego, self.p_ml, [('7', '70')], respaldo=[('co:mobile-legends', ML_UUID)])
        d = self.c.get('/admin/revendedores/mapping-data').get_json()
        self.assertEqual(d['respaldo'][f'{self.juego}:{self.p_ml}'][0]['remote_package_id'], ML_UUID)
        p = self.c.get('/admin/proveedores/datos?saldos=1').get_json()
        co = next(x for x in p['proveedores'] if x['clave'] == 'centralone')
        self.assertEqual((co['configurado'], co['saldo']['disponible']), (True, '470.0000'))
        self.assertEqual(self.c.get('/admin/proveedores').status_code, 200)
        with self.c.session_transaction() as s:
            s['is_admin'] = False
        self.assertEqual(self.c.get('/admin/proveedores/datos').status_code, 403)


if __name__ == '__main__':
    unittest.main()

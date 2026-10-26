"""
Proveedores propios de cada usuario (proveedores_usuario): cada uno conecta su Central One con su llave, crea sus
juegos y paquetes, y solo él los ve y los usa (en la web y en su API). Se recargan con SU llave y no tocan su saldo
de la web. Base SQLite temporal y Central One simulado (no se hace ningún pedido real).

    python -m unittest test_proveedores_usuario -v
"""
import json
import os
import tempfile
import unittest
import uuid

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'prov_usuario.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['REVENDEDORES_BASE_URL'] = 'http://inefable.test'
os.environ['REVENDEDORES_API_KEY'] = 'clave-inefable'
os.environ['CENTRALONE_API_KEY'] = 'co_live_DE_LA_WEB_0000'

import app as webapp  # noqa: E402
from api_whitelabel import hash_api_key  # noqa: E402
import central_one  # noqa: E402
import dynamic_games  # noqa: E402
import proveedores_usuario as pu  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

central_one.ESPERA_ENTREGA_S = 0.05
central_one.INTERVALO_S = 0.01

LLAVE_A = 'co_live_usuarioA_1234567890'
LLAVE_B = 'co_live_usuarioB_0987654321'
LLAVE_WEB = os.environ['CENTRALONE_API_KEY']
ML_UUID = '22222222-3333-4444-5555-666666666666'
CATALOGO = [
    {'product_id': ML_UUID, 'name': 'Mobile Legends 100 Diamonds', 'reseller_price': '1.9013', 'status': 'active',
     'product_family_id': 'mobile-legends', 'product_family_name': 'Mobile Legends', 'requires_target': True,
     'target_fields': ['game_user_id', 'game_zone_id'], 'target_schema': [], 'in_stock': True},
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
    """Central One simulado: responde según la llave con la que se llama (la del usuario o la de la web)."""
    def __init__(self):
        self.saldos = {LLAVE_A: '25.5000', LLAVE_B: '3.0000', LLAVE_WEB: '999.0000'}
        self.pedidos, self.por_idem, self.llamadas = {}, {}, []
        self.modo = 'completar'          # completar | lento | falla

    def __call__(self, metodo, path, body=None, idem=None, sin_llave=False):
        llave = central_one._llave()
        self.llamadas.append((metodo, path, llave))
        if llave not in self.saldos:
            return 401, {'error': {'code': 'invalid_api_key'}}
        if path == '/api/v1/balance':
            return 200, {'available_balance': self.saldos[llave], 'held_balance': '0', 'total_balance': self.saldos[llave], 'currency': 'USD'}
        if path == '/api/v1/catalog':
            return 200, {'items': [dict(i) for i in CATALOGO]}
        if metodo == 'POST' and path == '/api/v1/orders':
            if idem in self.por_idem:
                return 201, {'order': self.pedidos[self.por_idem[idem]]}
            oid = str(uuid.uuid4())
            self.pedidos[oid] = {'id': oid, 'reference_code': 'CO-' + oid[:6], 'status': 'confirmed',
                                 'target': body['items'][0].get('target_payload')}
            self.por_idem[idem] = oid
            return 201, {'order': self.pedidos[oid]}
        if metodo == 'GET' and path.startswith('/api/v1/orders/'):
            p = self.pedidos.get(path.split('/')[4])
            if not p:
                return 404, {'error': {'code': 'not_found'}}
            est = {'completar': 'completed', 'lento': 'processing', 'falla': 'failed'}[self.modo]
            return 200, {'order': dict(p, status=est, items=[{'status': est, 'delivered_count': 0}])}
        return 404, {'error': {'code': 'not_found'}}

    def con_llave(self, llave, metodo=None):
        return [c for c in self.llamadas if c[2] == llave and (metodo is None or c[0] == metodo)]


class ProveedoresUsuarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        pu.init_tablas()
        cls.a = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                  "VALUES ('Ana', 'A', '0', 'a@test.local', 'x', 0) RETURNING id")[0][0]
        cls.b = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                  "VALUES ('Beto', 'B', '0', 'b@test.local', 'x', 0) RETURNING id")[0][0]
        import api_panel
        api_panel.init_permisos()
        for clave, uid in (('wsk_a', cls.a), ('wsk_b', cls.b)):
            q("INSERT INTO webservice_accounts (nombre, api_key, usuario_id, activo, perm_recargas) VALUES ('CRM', ?, ?, TRUE, TRUE)",
              (hash_api_key(clave), uid))

    def setUp(self):
        self.co = FakeCentralOne()
        self._orig = central_one._pedir
        central_one._pedir = self.co
        for t in ('rev_item_mapping_steps', 'rev_item_mappings', 'api_orders', 'transacciones_dinamicas', 'profit_ledger',
                  'catalogo_proveedor_usuario', 'proveedores_usuario'):
            q(f'DELETE FROM {t}')
        q('DELETE FROM paquetes_dinamicos WHERE juego_id IN (SELECT id FROM juegos_dinamicos WHERE usuario_id IS NOT NULL)')
        q('DELETE FROM juegos_dinamicos WHERE usuario_id IS NOT NULL')
        q('UPDATE usuarios SET saldo = 0 WHERE id IN (?, ?)', (self.a, self.b))   # sin saldo en la web: no hace falta
        self.c = self.app.test_client()
        # Ana conecta su proveedor y crea su juego con un paquete, todo desde «Mis proveedores»
        self.como(self.a)
        d = self.post('/mis-proveedores/conexiones', {'nombre': 'Mi Central', 'llave': LLAVE_A})
        self.assertTrue(d['ok'], d)
        self.pid = d['id']
        self.assertEqual(d['catalogo']['nuevos'], 1)
        d = self.post('/mis-proveedores/juegos', {'nombre': 'Mobile Legends', 'icono': '⚔️', 'zona': True, 'zona_label': 'Zone ID'})
        self.assertTrue(d['ok'], d)
        self.jid = d['id']
        d = self.post(f'/mis-proveedores/juegos/{self.jid}/paquetes',
                      {'nombre': '100 💎', 'precio': 2.5, 'conexion_id': self.pid, 'package_id': ML_UUID})
        self.assertTrue(d['ok'], d)
        self.pkid = d['id']
        self.slug = q('SELECT slug FROM juegos_dinamicos WHERE id = ?', (self.jid,))[0]['slug']

    def tearDown(self):
        central_one._pedir = self._orig

    # ---------- ayudas ----------
    def como(self, uid):
        correo = 'a@test.local' if uid == self.a else 'b@test.local'
        with self.c.session_transaction() as s:
            s.clear()
            s.update(usuario=correo, user_db_id=uid, is_admin=False)

    def post(self, url, body):
        return self.c.post(url, json=body).get_json()

    def saldo(self, uid):
        return float(q('SELECT saldo FROM usuarios WHERE id = ?', (uid,))[0]['saldo'])

    def recargar_api(self, clave, ext):
        return self.c.post('/api/v1/recharge', headers={'X-API-Key': clave}, json={
            'product_id': self.jid, 'package_id': self.pkid, 'player_id': '123456', 'player_id2': '2001', 'external_order_id': ext})

    def comprar_web(self, uid):
        self.como(uid)
        with self.c.session_transaction() as s:
            s[f'dg_nonce_{self.slug}'] = 'N'
        return self.c.post(f'/validar/dinamico/{self.slug}',
                           data={'dg_form_nonce': 'N', 'monto': self.pkid, 'player_id': '123456', 'player_id2': '2001'})

    def forzar_reconciliador(self):
        for r in q("SELECT id, notas FROM transacciones_dinamicas WHERE estado = 'procesando'"):
            st = json.loads(r['notas'][len(dynamic_games.MULTI_PREFIX):])
            st['ts'] = 0
            q('UPDATE transacciones_dinamicas SET notas = ? WHERE id = ?', (dynamic_games.MULTI_PREFIX + json.dumps(st), r['id']))

    # ---------- la llave ----------
    def test_lo_que_no_es_una_llave_no_sale_de_la_web(self):
        # p. ej. la contraseña de la web que el navegador pegó solo en el campo
        self.co.llamadas.clear()
        r = self.c.post('/mis-proveedores/conexiones', json={'nombre': 'a@test.local', 'llave': 'MiClaveDeLaWeb2026'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('no es una llave', r.get_json()['error'])
        self.assertEqual(self.co.llamadas, [])
        self.assertEqual(len(pu.conexiones(self.a)), 1)
        self.assertNotIn('type="password"', self.c.get('/mis-proveedores').get_data(as_text=True))

    def test_llave_cifrada_y_nunca_se_muestra(self):
        fila = q('SELECT llave_cifrada, prefijo FROM proveedores_usuario WHERE id = ?', (self.pid,))[0]
        self.assertNotIn(LLAVE_A, fila['llave_cifrada'])
        self.assertEqual(pu.descifrar(fila['llave_cifrada']), LLAVE_A)
        self.assertEqual(fila['prefijo'], LLAVE_A[:12])
        datos = self.c.get('/mis-proveedores/datos').get_data(as_text=True)
        self.assertNotIn(LLAVE_A, datos)
        self.assertNotIn('llave_cifrada', datos)

    def test_llave_que_el_proveedor_no_acepta_no_se_guarda(self):
        r = self.c.post('/mis-proveedores/conexiones', json={'nombre': 'Mala', 'llave': 'co_live_mala_1234567890'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('no aceptó', r.get_json()['error'])
        self.assertEqual(len(pu.conexiones(self.a)), 1)

    def test_saldo_de_su_proveedor_con_su_llave(self):
        d = self.c.get(f'/mis-proveedores/conexiones/{self.pid}/saldo').get_json()
        self.assertEqual((d['saldo']['disponible'], d['saldo']['moneda']), ('25.5000', 'USD'))
        self.assertTrue(self.co.con_llave(LLAVE_A, 'GET'))
        self.assertEqual(self.co.con_llave(LLAVE_WEB), [])
        self.como(self.b)   # otro usuario no ve el saldo de Ana
        r = self.c.get(f'/mis-proveedores/conexiones/{self.pid}/saldo')
        self.assertEqual(r.status_code, 400)
        self.assertNotIn('25.5', r.get_data(as_text=True))

    # ---------- solo los ve y los usa su dueño ----------
    def test_el_juego_propio_no_sale_para_nadie_mas(self):
        self.assertNotIn(self.jid, [g['id'] for g in dynamic_games.get_all_dynamic_games()])
        self.assertNotIn(self.jid, [g['id'] for g in dynamic_games.juegos_para(self.b)])
        self.assertNotIn(self.jid, [g['id'] for g in dynamic_games.juegos_para(None)])
        mio = [g for g in dynamic_games.juegos_para(self.a) if g['id'] == self.jid]
        self.assertTrue(mio and mio[0]['propio'])
        self.como(self.b)
        r = self.c.get(f'/juego/d/{self.slug}')
        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.c.get('/mis-proveedores/datos').get_json()['juegos'], [])

    def test_api_productos_solo_con_la_clave_del_dueno(self):
        juegos_a = self.c.get('/api/v1/products', headers={'X-API-Key': 'wsk_a'}).get_json()['games']
        mio = [g for g in juegos_a if g['game_id'] == self.jid]
        self.assertEqual(len(mio), 1)
        self.assertTrue(mio[0]['own_provider'])
        self.assertFalse(mio[0]['verify_id'])
        self.assertEqual([(p['package_id'], p['price']) for p in mio[0]['packages']], [(self.pkid, 2.5)])
        juegos_b = self.c.get('/api/v1/products', headers={'X-API-Key': 'wsk_b'}).get_json()['games']
        self.assertNotIn(self.jid, [g['game_id'] for g in juegos_b])

    def test_otro_usuario_no_puede_recargarlo_ni_tocarlo(self):
        r = self.recargar_api('wsk_b', 'B-1')
        self.assertEqual(r.status_code, 404)
        r = self.comprar_web(self.b)
        self.assertEqual(r.status_code, 302)
        self.assertEqual(q('SELECT COUNT(*) AS n FROM transacciones_dinamicas')[0]['n'], 0)
        self.assertEqual(self.co.con_llave(LLAVE_A, 'POST'), [])
        # ni agregarle paquetes, ni editarlo, ni usar la conexión de Ana en un juego suyo
        self.como(self.b)
        self.assertFalse(self.post(f'/mis-proveedores/juegos/{self.jid}/paquetes',
                                   {'nombre': 'X', 'precio': 1, 'conexion_id': self.pid, 'package_id': ML_UUID})['ok'])
        self.assertFalse(self.post(f'/mis-proveedores/juegos/{self.jid}', {'accion': 'eliminar'})['ok'])
        self.assertFalse(self.post(f'/mis-proveedores/paquetes/{self.pkid}', {'accion': 'desactivar'})['ok'])
        self.assertFalse(self.post(f'/mis-proveedores/conexiones/{self.pid}/quitar', {})['ok'])
        jb = self.post('/mis-proveedores/juegos', {'nombre': 'De Beto'})['id']
        d = self.post(f'/mis-proveedores/juegos/{jb}/paquetes', {'nombre': 'X', 'precio': 1, 'conexion_id': self.pid, 'package_id': ML_UUID})
        self.assertEqual(d['error'], 'Ese proveedor no existe.')
        self.assertEqual(len(pu.conexiones(self.a)), 1)

    # ---------- compras: con su llave y sin tocar su saldo ----------
    def test_recarga_por_api_con_su_llave_sin_tocar_su_saldo(self):
        r = self.recargar_api('wsk_a', 'A-1')
        d = r.get_json()
        self.assertEqual((r.status_code, d['status']), (200, 'completada'), d)
        self.assertEqual(d['remaining_balance'], 0.0)
        self.assertEqual(self.saldo(self.a), 0.0)
        posts = self.co.con_llave(LLAVE_A, 'POST')
        self.assertEqual(len(posts), 1)
        self.assertEqual(self.co.con_llave(LLAVE_WEB), [])
        pedido = list(self.co.pedidos.values())[0]
        self.assertEqual(pedido['target'], {'game_user_id': '123456', 'game_zone_id': '2001'})
        tx = q('SELECT * FROM transacciones_dinamicas WHERE usuario_id = ?', (self.a,))[0]
        self.assertEqual((tx['estado'], float(tx['monto'])), ('aprobado', 0.0))
        self.assertEqual(float(q('SELECT precio FROM api_orders')[0]['precio']), 0.0)
        self.assertEqual(q('SELECT COUNT(*) AS n FROM profit_ledger')[0]['n'], 0)
        # la misma external_order_id no recarga dos veces
        self.assertTrue(self.recargar_api('wsk_a', 'A-1').get_json().get('duplicada'))
        self.assertEqual(len(self.co.con_llave(LLAVE_A, 'POST')), 1)

    def test_compra_en_la_web_con_su_proveedor(self):
        r = self.comprar_web(self.a)
        self.assertEqual(r.status_code, 302)
        tx = q('SELECT * FROM transacciones_dinamicas WHERE usuario_id = ?', (self.a,))[0]
        self.assertEqual((tx['estado'], float(tx['monto'])), ('aprobado', 0.0))
        self.assertEqual(self.saldo(self.a), 0.0)
        self.assertEqual(len(self.co.con_llave(LLAVE_A, 'POST')), 1)
        self.assertEqual(self.co.con_llave(LLAVE_WEB), [])

    def test_si_falla_no_se_devuelve_nada_que_no_se_cobro(self):
        self.co.modo = 'falla'
        d = self.recargar_api('wsk_a', 'A-F').get_json()
        self.assertEqual(d['status'], 'fallida')
        self.assertEqual(self.saldo(self.a), 0.0)

    def test_el_reconciliador_consulta_con_la_llave_del_dueno(self):
        self.co.modo = 'lento'
        r = self.recargar_api('wsk_a', 'A-L')
        self.assertEqual((r.status_code, r.get_json()['status']), (202, 'procesando'))
        self.co.modo = 'completar'
        self.co.llamadas.clear()
        self.forzar_reconciliador()
        dynamic_games.poll_pending_reseller_multi()
        d = self.c.get('/api/v1/order-status?external_order_id=A-L', headers={'X-API-Key': 'wsk_a'}).get_json()
        self.assertEqual(d['status'], 'completada')
        self.assertTrue(self.co.con_llave(LLAVE_A))
        self.assertEqual(self.co.con_llave(LLAVE_WEB), [])
        self.assertEqual(self.saldo(self.a), 0.0)
        self.assertEqual(q('SELECT COUNT(*) AS n FROM profit_ledger')[0]['n'], 0)

    # ---------- quitar el proveedor y borrar ----------
    def test_sin_su_proveedor_no_se_recarga_con_el_de_la_web(self):
        self.assertTrue(self.post(f'/mis-proveedores/conexiones/{self.pid}/quitar', {})['ok'])
        self.co.llamadas.clear()
        r = self.recargar_api('wsk_a', 'A-Q')
        self.assertIn(r.status_code, (404, 503))
        self.assertEqual(self.co.llamadas, [])
        self.assertEqual(q('SELECT COUNT(*) AS n FROM catalogo_proveedor_usuario')[0]['n'], 0)

    def test_con_compras_solo_se_desactiva(self):
        self.recargar_api('wsk_a', 'A-H')
        self.como(self.a)
        d = self.post(f'/mis-proveedores/paquetes/{self.pkid}', {'accion': 'eliminar'})
        self.assertIn('desactívalo', d['error'])
        self.assertFalse(self.post(f'/mis-proveedores/juegos/{self.jid}', {'accion': 'eliminar'})['ok'])
        self.assertTrue(self.post(f'/mis-proveedores/paquetes/{self.pkid}', {'accion': 'desactivar'})['ok'])
        juegos = self.c.get('/mis-proveedores/datos').get_json()['juegos']
        self.assertFalse(juegos[0]['paquetes'][0]['activo'])

    def test_sin_compras_se_borra_con_su_mapeo(self):
        self.assertTrue(self.post(f'/mis-proveedores/juegos/{self.jid}', {'accion': 'eliminar'})['ok'])
        self.assertEqual(q('SELECT COUNT(*) AS n FROM rev_item_mapping_steps WHERE juego_id = ?', (self.jid,))[0]['n'], 0)
        self.assertEqual(q('SELECT COUNT(*) AS n FROM juegos_dinamicos WHERE id = ?', (self.jid,))[0]['n'], 0)

    def test_la_pagina_carga(self):
        r = self.c.get('/mis-proveedores')
        self.assertEqual(r.status_code, 200)
        self.assertIn('Mis proveedores', r.get_data(as_text=True))
        self.c = self.app.test_client()   # sin sesión, al login
        self.assertEqual(self.c.get('/mis-proveedores').status_code, 302)
        self.assertEqual(self.c.get('/mis-proveedores/datos').status_code, 401)


if __name__ == '__main__':
    unittest.main()

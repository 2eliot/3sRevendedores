"""
Referencias por usuario: aislamiento, claves de lector, claves de API separadas y permisos.
Arranca la web completa sobre una base SQLite temporal.

    python -m unittest test_referencias -v
"""
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'referencias.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['PAGOS_BANCO_TOKEN'] = 'token-antiguo'

import app as webapp  # noqa: E402
import antiduplic as ad  # noqa: E402
import api_panel  # noqa: E402
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


def pago(ref, monto, fecha='2026-10-05', hora='08:14:52'):
    return {'referencia': ref, 'fecha': fecha, 'hora': hora, 'monto': monto, 'tipo': 'NC'}


class ReferenciasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        cls.A = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                  "VALUES ('Ana', 'A', '0', 'ana@test.local', 'x', 0) RETURNING id")[0][0]
        cls.B = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                  "VALUES ('Beto', 'B', '0', 'beto@test.local', 'x', 0) RETURNING id")[0][0]
        os.environ['ANTIDUPLIC_USUARIO_DUENO'] = str(cls.A)
        ad.init_tablas()

    def setUp(self):
        for t in ('usos_referencia', 'revisiones_pago', 'pagos_banco', 'claves_lector', 'cuentas_banco', 'webservice_accounts'):
            q(f'DELETE FROM {t}')
        api_panel._limites.clear()
        self.c = self.app.test_client()
        self.cuenta_b = self.crear_cuenta(self.B, 'Mi Bancamiga')
        self.lector_b = self.crear_lector(self.B, self.cuenta_b)

    # ---------- ayudas ----------
    def crear_cuenta(self, uid, alias):
        return q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) VALUES (?, 'Bancamiga', ?, TRUE, 'x') "
                 "RETURNING id", (uid, alias))[0][0]

    def crear_lector(self, uid, cuenta):
        clave, h, pref = ad.nueva_clave_lector()
        q("INSERT INTO claves_lector (usuario_id, cuenta_banco_id, nombre, clave_hash, prefijo_visible, activo, creado_en) "
          "VALUES (?, ?, 'PC', ?, ?, TRUE, 'x')", (uid, cuenta, h, pref))
        return clave

    def clave_api(self, uid, tipo='referencias', **perms):
        conn = get_db_connection()
        try:
            aid = api_panel.crear_cuenta(conn, f'Clave {uid}', uid, tipo, perms)
            conn.commit()
            return conn.execute('SELECT api_key FROM webservice_accounts WHERE id = ?', (aid,)).fetchone()['api_key']
        finally:
            conn.close()

    def enviar(self, pagos, clave):
        return self.c.post('/api/pagos-banco', json=pagos, headers={'Authorization': f'Bearer {clave}'})

    def verificar(self, clave, ref, monto, orden):
        return self.c.post('/api/verificar-pago', headers={'X-API-Key': clave},
                           json={'referencia': ref, 'monto': monto, 'orden_id': orden}).get_json()

    # ---------- envío de movimientos ----------
    def test_token_antiguo_deja_los_pagos_al_dueno(self):
        r = self.enviar([pago('7568126123', 100)], 'token-antiguo').get_json()
        self.assertEqual(r['insertados'], 1)
        fila = q('SELECT usuario_id, cuenta_banco_id FROM pagos_banco')[0]
        self.assertEqual(fila['usuario_id'], self.A)
        cuenta = q('SELECT usuario_id, alias, ultimo_envio FROM cuentas_banco WHERE id = ?', (fila['cuenta_banco_id'],))[0]
        self.assertEqual((cuenta['usuario_id'], cuenta['alias']), (self.A, 'Bancamiga principal'))
        self.assertTrue(cuenta['ultimo_envio'])

    def test_clave_de_lector_valida_desactivada_y_regenerada(self):
        self.assertEqual(self.enviar([pago('111222333', 10)], self.lector_b).status_code, 200)
        fila = q('SELECT usuario_id, cuenta_banco_id FROM pagos_banco')[0]
        self.assertEqual((fila['usuario_id'], fila['cuenta_banco_id']), (self.B, self.cuenta_b))
        self.assertTrue(q('SELECT ultimo_uso FROM claves_lector')[0]['ultimo_uso'])
        q('UPDATE claves_lector SET activo = FALSE')
        self.assertEqual(self.enviar([pago('111222333', 10)], self.lector_b).status_code, 401)
        q('UPDATE claves_lector SET activo = TRUE')
        nueva, h, pref = ad.nueva_clave_lector()
        q('UPDATE claves_lector SET clave_hash = ?, prefijo_visible = ?', (h, pref))  # regenerar
        self.assertEqual(self.enviar([pago('111222333', 10)], self.lector_b).status_code, 401)
        self.assertEqual(self.enviar([pago('111222333', 10)], nueva).status_code, 200)
        self.assertEqual(self.enviar([], 'lec_inventada').status_code, 401)

    def test_misma_referencia_en_dos_usuarios(self):
        self.enviar([pago('7568126123', 100)], 'token-antiguo')
        self.enviar([pago('7568126123', 100)], self.lector_b)
        self.assertEqual(len(q('SELECT id FROM pagos_banco')), 2)
        ka, kb = self.clave_api(self.A, verificar_pago=True), self.clave_api(self.B, verificar_pago=True)
        self.assertEqual(self.verificar(ka, '7568126123', 100, 'O-A')['codigo'], 'aprobado')
        self.assertEqual(self.verificar(ka, '7568126123', 100, 'O-A2')['codigo'], 'usado')
        self.assertEqual(self.verificar(kb, '7568126123', 100, 'O-B')['codigo'], 'aprobado')

    def test_mismo_orden_id_en_dos_usuarios_no_choca(self):
        self.enviar([pago('5555666677', 20)], 'token-antiguo')
        self.enviar([pago('8888999911', 20)], self.lector_b)
        ka, kb = self.clave_api(self.A, verificar_pago=True), self.clave_api(self.B, verificar_pago=True)
        self.assertEqual(self.verificar(ka, '5555666677', 20, 'ORD-1')['codigo'], 'aprobado')
        self.assertEqual(self.verificar(kb, '8888999911', 20, 'ORD-1')['codigo'], 'aprobado')
        self.assertEqual(self.verificar(kb, '8888999911', 20, 'ORD-1')['codigo'], 'ya_aprobado')

    # ---------- aislamiento por API ----------
    def test_un_usuario_no_ve_ni_toca_lo_de_otro(self):
        self.enviar([pago('7000123456', 50, hora='08:00:00'), pago('8000123456', 50, hora='08:00:00')], 'token-antiguo')
        ad.verificar_y_usar_pago('123456', 50, hora='08:00', origen='crm', orden_id='O-A', usuario_id=self.A)
        pago_a = q('SELECT id FROM pagos_banco WHERE usuario_id = ? ORDER BY id', (self.A,))[0]['id']
        rev_a = q('SELECT id FROM revisiones_pago WHERE usuario_id = ?', (self.A,))[0]['id']
        kb = self.clave_api(self.B, verificar_pago=True, ver_pagos=True, gestionar_pagos=True)
        H = {'X-API-Key': kb}
        self.assertEqual(self.c.get('/api/v1/referencias/pagos', headers=H).get_json()['pagos'], [])
        self.assertEqual(self.c.get(f'/api/v1/referencias/pagos/{pago_a}', headers=H).status_code, 404)
        self.assertEqual(self.c.post('/api/v1/referencias/asignar', headers=H, json={'pago_id': pago_a, 'orden_id': 'X'}).status_code, 404)
        self.assertEqual(self.c.post(f'/api/v1/referencias/revisiones/{rev_a}/resolver', headers=H, json={'pago_id': pago_a}).status_code, 404)
        self.assertEqual(self.c.post(f'/api/v1/referencias/revisiones/{rev_a}/descartar', headers=H).status_code, 404)
        self.assertEqual(self.c.get('/api/v1/referencias/revisiones', headers=H).get_json()['revisiones'], [])
        self.assertEqual(self.verificar(kb, '7000123456', 50, 'O-B')['codigo'], 'no_encontrado')
        self.assertEqual(q("SELECT COUNT(*) AS n FROM revisiones_pago WHERE estado = 'pendiente'")[0]['n'], 1)

    def test_api_completa_del_usuario(self):
        self.enviar([pago('1000555111', 30, hora='10:00:00'), pago('2000555111', 30, hora='10:00:00'),
                     pago('3333444455', 12.5, hora='11:00:00')], self.lector_b)
        kb = self.clave_api(self.B, verificar_pago=True, ver_pagos=True, gestionar_pagos=True)
        H = {'X-API-Key': kb}
        res = self.c.get('/api/v1/referencias/resumen?fecha=2026-10-05', headers=H).get_json()['resumen']
        self.assertEqual((res['pagos_hoy'], res['total_hoy'], res['disponibles_hoy']), (3, 72.5, 3))
        self.assertEqual(res['cuentas'][0]['alias'], 'Mi Bancamiga')
        d = self.c.get('/api/v1/referencias/pagos?por_pagina=2&pagina=1', headers=H).get_json()
        self.assertEqual((d['total'], len(d['pagos'])), (3, 2))
        self.assertEqual(self.c.get('/api/v1/referencias/pagos?por_pagina=201', headers=H).status_code, 400)
        self.assertEqual(self.c.get('/api/v1/referencias/cuentas', headers=H).get_json()['cuentas'][0]['id'], self.cuenta_b)
        # Empate → revisión → resolver
        r = self.c.post('/api/verificar-pago', headers=H, json={'referencia': '555111', 'monto': 30, 'hora': '10:00', 'orden_id': 'ORD-R'}).get_json()
        self.assertEqual(r['codigo'], 'revision')
        rev = self.c.get('/api/v1/referencias/revisiones', headers=H).get_json()['revisiones'][0]
        elegido = rev['candidatos'][0]['id']
        r = self.c.post(f"/api/v1/referencias/revisiones/{rev['id']}/resolver", headers=H, json={'pago_id': elegido})
        self.assertEqual((r.status_code, r.get_json()['ok']), (200, True))
        uso = self.c.get(f'/api/v1/referencias/pagos/{elegido}', headers=H).get_json()['pago']['uso']
        self.assertEqual((uso['orden_id'], uso['origen']), ('ORD-R', 'manual'))
        self.assertTrue(uso['hecho_por'].startswith('api:'))
        # Asignar a mano y conflictos
        libre = q("SELECT id FROM pagos_banco WHERE referencia = '3333444455'")[0]['id']
        self.assertEqual(self.c.post('/api/v1/referencias/asignar', headers=H, json={'pago_id': libre, 'orden_id': 'ORD-R'}).status_code, 409)
        self.assertEqual(self.c.post('/api/v1/referencias/asignar', headers=H, json={'pago_id': libre, 'orden_id': 'ORD-M'}).status_code, 200)
        self.assertEqual(self.c.post('/api/v1/referencias/asignar', headers=H, json={'pago_id': libre, 'orden_id': 'ORD-N'}).status_code, 409)
        self.assertEqual(self.c.post('/api/v1/referencias/asignar', headers=H, json={'pago_id': libre}).status_code, 400)

    # ---------- aislamiento en la pantalla del usuario ----------
    def entrar(self, uid, correo):
        with self.c.session_transaction() as s:
            s.update(usuario=correo, user_db_id=uid, is_admin=False)

    def test_pantalla_aislada_entre_usuarios(self):
        self.enviar([pago('7000123456', 50, hora='08:00:00'), pago('8000123456', 50, hora='08:00:00')], 'token-antiguo')
        ad.verificar_y_usar_pago('123456', 50, hora='08:00', origen='crm', orden_id='O-A', usuario_id=self.A)
        pago_a = q('SELECT id FROM pagos_banco WHERE usuario_id = ? ORDER BY id', (self.A,))[0]['id']
        rev_a = q('SELECT id FROM revisiones_pago WHERE usuario_id = ?', (self.A,))[0]['id']
        cuenta_a = q('SELECT id FROM cuentas_banco WHERE usuario_id = ?', (self.A,))[0]['id']
        lector_a = self.crear_lector(self.A, cuenta_a)
        lector_a_id = q('SELECT id FROM claves_lector WHERE usuario_id = ?', (self.A,))[0]['id']
        self.clave_api(self.A, verificar_pago=True)
        api_a = q("SELECT id FROM webservice_accounts WHERE usuario_id = ?", (self.A,))[0]['id']
        self.entrar(self.B, 'beto@test.local')
        d = self.c.get('/referencias/datos').get_json()
        self.assertEqual((d['pagos'], d['revisiones']), ([], []))
        self.assertEqual(self.c.post('/referencias/asignar', json={'pago_id': pago_a, 'orden_id': 'X'}).status_code, 404)
        self.assertEqual(self.c.post('/referencias/asignar', json={'pago_id': pago_a, 'revision_id': rev_a}).status_code, 404)
        self.assertEqual(self.c.post('/referencias/descartar', json={'revision_id': rev_a}).status_code, 404)
        self.assertEqual(self.c.post(f'/referencias/cuentas/{cuenta_a}', json={'activo': False}).status_code, 404)
        self.assertEqual(self.c.post(f'/referencias/claves-lector/{lector_a_id}/regenerar', json={}).status_code, 404)
        self.assertEqual(self.c.post('/referencias/claves-lector', json={'cuenta_id': cuenta_a}).status_code, 404)
        self.assertEqual(self.c.post(f'/referencias/claves-api/{api_a}/eliminar', json={}).status_code, 404)
        claves = self.c.get('/referencias/claves').get_json()
        self.assertEqual([k['cuenta_id'] for k in claves['lector']], [self.cuenta_b])
        self.assertEqual(claves['api'], [])
        self.assertEqual(self.enviar([pago('9999888877', 5)], lector_a).status_code, 200)  # la de A sigue intacta
        self.assertEqual(q("SELECT activo FROM cuentas_banco WHERE id = ?", (cuenta_a,))[0]['activo'], 1)

    def test_pantalla_del_usuario_crea_cuenta_y_claves(self):
        self.entrar(self.B, 'beto@test.local')
        self.assertEqual(self.c.get('/referencias').status_code, 200)
        self.assertEqual(self.c.post('/referencias/cuentas', json={'alias': 'X', 'ultimos_digitos': '12345678'}).status_code, 400)
        d = self.c.post('/referencias/cuentas', json={'alias': 'Segunda', 'ultimos_digitos': '4021'}).get_json()
        nueva = [c for c in d['cuentas'] if c['alias'] == 'Segunda'][0]
        d = self.c.post('/referencias/claves-lector', json={'nombre': 'PC', 'cuenta_id': nueva['id']}).get_json()
        self.assertTrue(d['clave'].startswith('lec_'))
        self.assertNotIn(d['clave'], str(d['lector']))  # la clave completa no se vuelve a mostrar
        self.assertEqual(self.enviar([pago('1212121212', 9)], d['clave']).status_code, 200)
        self.assertEqual(q("SELECT cuenta_banco_id FROM pagos_banco WHERE referencia = '1212121212'")[0]['cuenta_banco_id'], nueva['id'])
        d = self.c.post('/referencias/claves-api', json={'nombre': 'CRM', 'permisos': {'verificar_pago': True, 'recargas': True}}).get_json()
        fila = q("SELECT tipo, perm_recargas, perm_verificar_pago FROM webservice_accounts WHERE usuario_id = ?", (self.B,))[0]
        self.assertEqual((fila['tipo'], bool(fila['perm_recargas']), bool(fila['perm_verificar_pago'])), ('referencias', False, True))
        self.assertEqual(self.c.post('/referencias/cuentas', data='alias=x').status_code, 415)  # solo JSON

    # ---------- vista general del admin ----------
    def test_admin_ve_todos_filtra_y_resuelve(self):
        self.enviar([pago('7000123456', 50, hora='08:00:00'), pago('8000123456', 50, hora='08:00:00')], self.lector_b)
        self.enviar([pago('1111222233', 10)], 'token-antiguo')
        ad.verificar_y_usar_pago('123456', 50, hora='08:00', origen='crm', orden_id='O-B', usuario_id=self.B)
        with self.c.session_transaction() as s:
            s.update(usuario='admin@test.local', user_db_id=self.A, is_admin=True)
        todos = self.c.get('/admin/antiduplic/datos').get_json()
        self.assertEqual({p['usuario_id'] for p in todos['pagos']}, {self.A, self.B})
        solo_b = self.c.get(f'/admin/antiduplic/datos?usuario_id={self.B}').get_json()
        self.assertEqual({p['usuario_id'] for p in solo_b['pagos']}, {self.B})
        rev = todos['revisiones'][0]
        self.assertEqual(rev['usuario_id'], self.B)
        r = self.c.post('/admin/antiduplic/asignar', json={'pago_id': rev['candidatos'][1]['id'], 'revision_id': rev['id']}).get_json()
        self.assertTrue(r['ok'], r)
        uso = q('SELECT usuario_id, orden_id, hecho_por FROM usos_referencia')[0]
        self.assertEqual((uso['usuario_id'], uso['orden_id']), (self.B, 'O-B'))
        self.assertTrue(uso['hecho_por'].startswith('admin:'))
        self.assertEqual(self.c.get('/admin/antiduplic').status_code, 200)
        d = self.c.post('/admin/api/cuentas', json={'nombre': 'CRM', 'usuario_id': self.B, 'tipo': 'referencias',
                                                     'permisos': {'verificar_pago': True, 'recargas': True}}).get_json()
        self.assertEqual((d['cuenta']['tipo'], d['cuenta']['permisos']),
                         ('referencias', {'verificar_pago': True, 'ver_pagos': False, 'gestionar_pagos': False}))
        self.assertEqual(self.c.post('/admin/api/cuentas', json={'nombre': 'X', 'usuario_id': self.B, 'tipo': 'otro'}).status_code, 400)

    # ---------- permisos y tipos de clave ----------
    def test_permisos(self):
        solo_verificar = {'X-API-Key': self.clave_api(self.B, verificar_pago=True)}
        solo_ver = {'X-API-Key': self.clave_api(self.B, ver_pagos=True)}
        self.assertEqual(self.c.get('/api/v1/referencias/resumen', headers=solo_verificar).status_code, 403)
        self.assertEqual(self.c.get('/api/v1/referencias/resumen', headers=solo_ver).status_code, 200)
        self.assertEqual(self.c.post('/api/v1/referencias/asignar', headers=solo_ver, json={'pago_id': 1, 'orden_id': 'X'}).status_code, 403)
        self.assertEqual(self.c.post('/api/verificar-pago', headers=solo_ver, json={'referencia': '123456', 'monto': 1, 'orden_id': 'X'}).status_code, 403)
        self.assertEqual(self.c.get('/api/v1/referencias/resumen').status_code, 401)

    def test_claves_de_recargas_y_de_referencias_estan_separadas(self):
        recargas = {'X-API-Key': self.clave_api(self.B, tipo='recargas', recargas=True, verificar_id=True,
                                                verificar_pago=True, ver_pagos=True)}
        referencias = {'X-API-Key': self.clave_api(self.B, verificar_pago=True, ver_pagos=True, recargas=True)}
        self.assertEqual(self.c.get('/api/v1/referencias/resumen', headers=recargas).status_code, 403)
        self.assertEqual(self.c.post('/api/verificar-pago', headers=recargas, json={'referencia': '123456', 'monto': 1, 'orden_id': 'X'}).status_code, 403)
        self.assertEqual(self.c.get('/api/v1/products', headers=referencias).status_code, 403)
        self.assertEqual(self.c.post('/api/v1/recharge', headers=referencias, json={'product_id': 1, 'package_id': 1, 'player_id': '1'}).status_code, 403)
        fila = q("SELECT perm_recargas FROM webservice_accounts WHERE tipo = 'referencias'")[0]
        self.assertFalse(bool(fila['perm_recargas']))  # nunca se mezcla aunque se pida

    def test_claves_anteriores_quedan_desconectadas(self):
        q("DELETE FROM configuracion_redeemer WHERE clave = 'migr_claves_separadas'")
        q("INSERT INTO webservice_accounts (nombre, api_key, usuario_id, activo, perm_recargas, perm_verificar_pago) "
          "VALUES ('CRM viejo', 'wsk_viejo', ?, TRUE, TRUE, TRUE)", (self.A,))
        api_panel._reiniciar_claves_anteriores()
        fila = q("SELECT activo, tipo FROM webservice_accounts WHERE api_key = 'wsk_viejo'")[0]
        self.assertEqual((bool(fila['activo']), fila['tipo']), (False, 'antigua'))
        self.assertEqual(self.c.post('/api/verificar-pago', headers={'X-API-Key': 'wsk_viejo'}, json={}).status_code, 401)
        self.assertEqual(self.c.get('/api/v1/products', headers={'X-API-Key': 'wsk_viejo'}).status_code, 401)
        with self.c.session_transaction() as s:
            s['is_admin'] = True
        aid = q("SELECT id FROM webservice_accounts WHERE api_key = 'wsk_viejo'")[0]['id']
        self.assertEqual(self.c.post(f'/admin/api/cuentas/{aid}', json={'activo': True}).status_code, 409)

    def test_documentacion_del_usuario_y_del_admin(self):
        self.assertEqual(self.c.get('/referencias/docs').status_code, 302)  # sin sesión
        self.entrar(self.B, 'b@test.local')
        html = self.c.get('/referencias/docs').get_data(as_text=True)
        self.assertIn('Descargar PDF', html)
        self.assertIn('/api/v1/referencias/pagos', html)
        self.assertNotIn('id="admin"', html)
        self.assertEqual(self.c.get('/admin/antiduplic/docs').status_code, 302)  # no es admin
        with self.c.session_transaction() as s:
            s['is_admin'] = True
        html = self.c.get('/admin/antiduplic/docs').get_data(as_text=True)
        self.assertIn('id="admin"', html)
        self.assertIn('Descargar PDF', self.c.get('/admin/api/docs').get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()

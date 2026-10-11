"""
BDV dentro de Referencias: cifrado de los datos de acceso, conexión de cada usuario y separación total por banco.
Arranca la web completa sobre una base SQLite temporal (sin tocar el BDV real).

    python -m unittest test_bdv -v
"""
import json
import os
import tempfile
import unittest

from cryptography.fernet import Fernet

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'bdv.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
LLAVE = Fernet.generate_key().decode()
os.environ['BDV_LLAVE_CIFRADO'] = LLAVE

import app as webapp  # noqa: E402,F401
import antiduplic as ad  # noqa: E402
import api_panel  # noqa: E402
import bdv  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

USUARIO, CLAVE, NUMERO = 'usuario.secreto', 'Clave-Muy-Secreta-123', '01020123456789012411'


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if (sql.lstrip().upper().startswith('SELECT') or 'RETURNING' in sql.upper()) else None
        conn.commit()
        return rows
    finally:
        conn.close()


def pago(ref, monto, fecha='2026-10-11', hora='09:15'):
    return {'referencia': ref, 'fecha': fecha, 'hora': hora, 'monto': monto, 'tipo': 'BDV PAGO MOVIL'}


class BdvDatosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                  "VALUES ('Ana', 'A', '0', 'ana-bdv@test.local', 'x', 0) RETURNING id")[0][0]
        cls.B = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                  "VALUES ('Beto', 'B', '0', 'beto-bdv@test.local', 'x', 0) RETURNING id")[0][0]
        ad.init_tablas()

    def setUp(self):
        os.environ['BDV_LLAVE_CIFRADO'] = LLAVE
        for t in ('usos_referencia', 'revisiones_pago', 'pagos_banco', 'claves_lector', 'cuentas_banco', 'webservice_accounts'):
            q(f'DELETE FROM {t}')
        self.conn = get_db_connection()
        self.bancamiga_a = q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) "
                             "VALUES (?, 'Bancamiga', 'Bancamiga A', TRUE, 'x') RETURNING id", (self.A,))[0][0]

    def tearDown(self):
        self.conn.close()

    def conectar(self, uid=None, numero=NUMERO, **extra):
        datos = dict(usuario=USUARIO, clave=CLAVE, numero_cuenta=numero, alias='', consentimiento=True)
        datos.update(extra)
        return bdv.guardar_conexion(self.conn, uid or self.A, **datos)

    # ---------- cifrado ----------
    def test_cifrar_y_descifrar(self):
        token = bdv.cifrar('hola')
        self.assertNotIn('hola', token)
        self.assertEqual(bdv.descifrar(token), 'hola')
        os.environ['BDV_LLAVE_CIFRADO'] = Fernet.generate_key().decode()  # otra llave no lo abre
        with self.assertRaises(bdv.SinLlave):
            bdv.descifrar(token)
        os.environ['BDV_LLAVE_CIFRADO'] = ''
        self.assertFalse(bdv.cifrado_disponible())
        with self.assertRaises(bdv.SinLlave):
            bdv.cifrar('x')

    def test_sin_llave_no_se_puede_guardar(self):
        os.environ['BDV_LLAVE_CIFRADO'] = ''
        status, datos = self.conectar()
        self.assertEqual(status, 503)
        self.assertEqual(q("SELECT COUNT(*) AS n FROM cuentas_banco WHERE banco = 'BDV'")[0]['n'], 0)

    def test_en_la_base_no_hay_nada_en_claro(self):
        status, datos = self.conectar(alias='Mi BDV')
        self.assertEqual(status, 200, datos)
        fila = q('SELECT * FROM cuentas_banco WHERE id = ?', (datos['cuenta_id'],))[0]
        crudo = ' '.join(str(v) for v in dict(fila).values())
        for secreto in (USUARIO, CLAVE, NUMERO, NUMERO[:16]):
            self.assertNotIn(secreto, crudo)
        self.assertEqual(fila['ultimos_digitos'], '2411')
        self.assertEqual(bdv.descifrar(fila['bdv_clave_cifrada']), CLAVE)
        self.assertEqual(bdv.descifrar(fila['numero_cuenta_cifrado']), NUMERO)
        self.assertEqual(fila['lector_estado'], 'pendiente')
        self.assertTrue(fila['consentimiento_en'])

    def test_la_cuenta_publica_nunca_trae_datos_de_acceso(self):
        self.conectar()
        cuentas = ad.listar_cuentas(self.conn, self.A, banco='BDV')
        self.assertEqual(len(cuentas), 1)
        texto = str(cuentas)
        for secreto in (USUARIO, CLAVE, NUMERO, 'cifrad', 'gAAAA'):
            self.assertNotIn(secreto, texto)
        self.assertTrue(cuentas[0]['conectada'])
        self.assertEqual(cuentas[0]['lector']['estado'], 'pendiente')
        self.assertEqual([c['banco'] for c in ad.listar_cuentas(self.conn, self.A, banco='Bancamiga')], ['Bancamiga'])

    def test_validaciones_y_reemplazo(self):
        self.assertEqual(self.conectar(consentimiento=False)[0], 400)
        self.assertEqual(self.conectar(numero='0102123')[0], 400)
        self.assertEqual(self.conectar(numero='01340123456789012411')[0], 400)  # no es del BDV
        _, d1 = self.conectar()
        _, d2 = self.conectar(clave='otra-clave')  # misma cuenta (mismos últimos 4): se reemplaza, no se duplica
        self.assertEqual(d1['cuenta_id'], d2['cuenta_id'])
        self.assertEqual(q("SELECT COUNT(*) AS n FROM cuentas_banco WHERE banco = 'BDV'")[0]['n'], 1)
        for i in range(2):
            self.assertEqual(self.conectar(numero=f'0102000000000000{i}{i}{i}{i}')[0], 200)
        self.assertEqual(self.conectar(numero='01029999999999999999')[0], 409)  # máximo 3 por usuario

    def test_interruptor_reanudar_y_borrar(self):
        _, d = self.conectar()
        cid = d['cuenta_id']
        self.assertEqual(bdv.cambiar_lectura(self.conn, self.B, cid, False)[0], 404)  # de otro usuario
        self.assertEqual(bdv.cambiar_lectura(self.conn, self.A, cid, False)[0], 200)
        self.assertFalse(q('SELECT lector_encendido FROM cuentas_banco WHERE id = ?', (cid,))[0]['lector_encendido'])
        self.assertEqual(bdv.cambiar_lectura(self.conn, None, cid, True)[0], 200)  # el admin también puede
        cambio = q('SELECT lector_cambio FROM cuentas_banco WHERE id = ?', (cid,))[0]['lector_cambio']
        q("UPDATE cuentas_banco SET lector_estado = 'detenido', lector_fallos_login = 2 WHERE id = ?", (cid,))
        self.assertEqual(bdv.reanudar(self.conn, self.A, cid)[0], 200)
        fila = q('SELECT * FROM cuentas_banco WHERE id = ?', (cid,))[0]
        self.assertEqual((fila['lector_estado'], fila['lector_fallos_login']), ('pendiente', 0))
        self.assertNotEqual(fila['lector_cambio'], cambio)
        q('INSERT INTO pagos_banco (usuario_id, cuenta_banco_id, referencia, ref_ultimos6, fecha, hora, monto, estado, creado_en) '
          "VALUES (?, ?, '123456789', '456789', '2026-10-11', '09:00', 10, 'disponible', 'x')", (self.A, cid))
        self.assertEqual(bdv.borrar_conexion(self.conn, self.A, cid)[0], 200)
        fila = q('SELECT * FROM cuentas_banco WHERE id = ?', (cid,))[0]
        self.assertIsNone(fila['bdv_clave_cifrada'])
        self.assertIsNone(fila['bdv_usuario_cifrado'])
        self.assertEqual(q('SELECT COUNT(*) AS n FROM pagos_banco WHERE cuenta_banco_id = ?', (cid,))[0]['n'], 1)  # se quedan

    # ---------- separación por banco ----------
    def test_cada_banco_ve_solo_sus_pagos_y_sus_ordenes(self):
        _, d = self.conectar()
        bdv_id = d['cuenta_id']
        ad.guardar_pagos([pago('9000111222333', 50)], self.A, bdv_id)
        ad.guardar_pagos([pago('7000111222333', 50, hora='10:00:00')], self.A, self.bancamiga_a)

        # cada banco encuentra solo los suyos
        self.assertEqual(ad.verificar_y_usar_pago('9000111222333', 50, origen='crm', orden_id='X1', usuario_id=self.A,
                                                  banco='Bancamiga')['codigo'], 'no_encontrado')
        self.assertEqual(ad.verificar_y_usar_pago('7000111222333', 50, origen='crm', orden_id='X1', usuario_id=self.A,
                                                  banco='BDV')['codigo'], 'no_encontrado')
        # la misma orden en los dos bancos no choca: cada banco tiene sus órdenes
        self.assertEqual(ad.verificar_y_usar_pago('9000111222333', 50, origen='crm', orden_id='ORD-1', usuario_id=self.A,
                                                  banco='BDV')['codigo'], 'aprobado')
        self.assertEqual(ad.verificar_y_usar_pago('7000111222333', 50, origen='crm', orden_id='ORD-1', usuario_id=self.A,
                                                  banco='Bancamiga')['codigo'], 'aprobado')
        res = ad.verificar_y_usar_pago('9000111222333', 50, origen='crm', orden_id='ORD-1', usuario_id=self.A, banco='BDV')
        self.assertEqual(res['codigo'], 'ya_aprobado')
        self.assertEqual(res['pago']['referencia'], '9000111222333')  # el de su banco, nunca el de Bancamiga

        pagos_bdv, _ = ad.listar_pagos(self.conn, self.A, banco='BDV')
        pagos_bm, _ = ad.listar_pagos(self.conn, self.A, banco='Bancamiga')
        self.assertEqual([p['referencia'] for p in pagos_bdv], ['9000111222333'])
        self.assertEqual([p['referencia'] for p in pagos_bm], ['7000111222333'])
        self.assertEqual(pagos_bdv[0]['banco'], 'BDV')
        self.assertIsNone(ad.detalle_pago(self.conn, self.A, pagos_bdv[0]['id'], banco='Bancamiga'))
        self.assertEqual(ad.resumen(self.conn, self.A, '2026-10-11', banco='BDV')['pagos_hoy'], 1)
        self.assertEqual(ad.resumen(self.conn, self.A, '2026-10-11', banco='Bancamiga')['pagos_hoy'], 1)

    def test_asignar_y_revisiones_por_banco(self):
        _, d = self.conectar()
        bdv_id = d['cuenta_id']
        ad.guardar_pagos([pago('5555000011112222', 20, hora='08:00'), pago('6666000011112222', 20, hora='09:00')], self.A, bdv_id)
        p_bdv = ad.listar_pagos(self.conn, self.A, banco='BDV')[0][0]['id']
        self.assertEqual(ad.asignar(self.A, p_bdv, orden_id='M-1', banco='Bancamiga')[0], 404)
        self.assertEqual(ad.asignar(self.A, p_bdv, orden_id='M-1', banco='BDV')[0], 200)
        uso = q('SELECT banco FROM usos_referencia WHERE pago_id = ?', (p_bdv,))[0]
        self.assertEqual(uso['banco'], 'BDV')
        # empate en el BDV → revisión del BDV, que Bancamiga no ve
        ad.guardar_pagos([pago('1234567890', 33, hora='11:00'), pago('99991234567890', 33, hora='11:00')], self.A, bdv_id)
        res = ad.verificar_y_usar_pago('1234567890', 33, hora='11:00', origen='crm', orden_id='R-1', usuario_id=self.A, banco='BDV')
        self.assertEqual(res['codigo'], 'revision')
        self.assertEqual(len(ad.listar_revisiones(self.conn, self.A, banco='BDV')), 1)
        self.assertEqual(ad.listar_revisiones(self.conn, self.A, banco='Bancamiga'), [])
        rev = ad.listar_revisiones(self.conn, self.A, banco='BDV')[0]
        self.assertEqual(ad.descartar(self.A, rev['id'], banco='Bancamiga')[0], 404)
        self.assertEqual(ad.descartar(self.A, rev['id'], banco='BDV')[0], 200)

    def test_bancamiga_no_cambia(self):
        # la cuenta principal y el bot nunca usan una cuenta del BDV, aunque sea la primera del usuario
        q('DELETE FROM cuentas_banco')
        _, d = self.conectar()
        principal = ad.cuenta_principal(self.conn, self.A)
        self.conn.commit()
        self.assertNotEqual(principal, d['cuenta_id'])
        self.assertEqual(q('SELECT banco FROM cuentas_banco WHERE id = ?', (principal,))[0]['banco'], 'Bancamiga')
        # una clave de lector colgada de una cuenta del BDV no sirve
        clave, h, pref = ad.nueva_clave_lector()
        q("INSERT INTO claves_lector (usuario_id, cuenta_banco_id, nombre, clave_hash, prefijo_visible, activo, creado_en) "
          "VALUES (?, ?, 'x', ?, ?, TRUE, 'x')", (self.A, d['cuenta_id'], h, pref))
        c = webapp.app.test_client()
        r = c.post('/api/pagos-banco', json=[pago('111122223333', 5)], headers={'Authorization': f'Bearer {clave}'})
        self.assertEqual(r.status_code, 401)

    def test_claves_api_por_banco(self):
        status, datos = api_panel.crear_clave_usuario(self.conn, self.A, 'referencias', 'CRM BDV', {'verificar_pago': True}, banco='BDV')
        self.assertEqual(status, 200)
        api_panel.crear_clave_usuario(self.conn, self.A, 'referencias', 'CRM BM', {'verificar_pago': True}, banco='Bancamiga')
        self.assertEqual([k['nombre'] for k in api_panel.claves_de_usuario(self.conn, self.A, 'referencias', 'BDV')['api']], ['CRM BDV'])
        self.assertEqual([k['nombre'] for k in api_panel.claves_de_usuario(self.conn, self.A, 'referencias', 'Bancamiga')['api']],
                         ['CRM BM'])
        id_bdv = api_panel.claves_de_usuario(self.conn, self.A, 'referencias', 'BDV')['api'][0]['id']
        self.assertEqual(api_panel.accion_clave_usuario(self.conn, self.A, 'referencias', id_bdv, 'desactivar', banco='Bancamiga')[0], 404)
        self.assertEqual(api_panel.cuenta_por_clave(datos['clave'])['banco'], 'BDV')


class BdvPantallaTest(unittest.TestCase):
    """La pantalla Referencias con pestañas de banco y la vista del admin."""

    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        ad.init_tablas()
        cls.A = cls.usuario('ana-pantalla@test.local')
        cls.B = cls.usuario('beto-pantalla@test.local')

    @staticmethod
    def usuario(correo):
        filas = q('SELECT id FROM usuarios WHERE correo = ?', (correo,))
        if filas:
            return filas[0]['id']
        return q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                 "VALUES ('Prueba', 'P', '0', ?, 'x', 0) RETURNING id", (correo,))[0][0]

    def setUp(self):
        os.environ['BDV_LLAVE_CIFRADO'] = LLAVE
        for t in ('usos_referencia', 'revisiones_pago', 'pagos_banco', 'claves_lector', 'cuentas_banco', 'webservice_accounts'):
            q(f'DELETE FROM {t}')
        self.c = self.app.test_client()

    def entrar(self, uid, correo, admin=False):
        with self.c.session_transaction() as s:
            s.update(usuario=correo, user_db_id=uid, is_admin=admin)

    def conectar(self, **extra):
        datos = {'usuario': USUARIO, 'clave': CLAVE, 'numero_cuenta': NUMERO, 'alias': 'Mi BDV', 'consentimiento': True}
        datos.update(extra)
        return self.c.post('/referencias/bdv/cuentas', json=datos)

    def sin_secretos(self, texto):
        for secreto in (USUARIO, CLAVE, NUMERO, NUMERO[4:16], 'gAAAA'):
            self.assertNotIn(secreto, texto)

    def test_pestanas_y_conexion_sin_secretos(self):
        self.entrar(self.A, 'ana-pantalla@test.local')
        html = self.c.get('/referencias').get_data(as_text=True)
        self.assertIn('data-banco="BDV"', html)
        self.assertIn('Leer movimientos', html)
        self.assertEqual(self.c.post('/referencias/bdv/cuentas', data='x').status_code, 415)
        self.assertEqual(self.conectar(consentimiento=False).status_code, 400)
        r = self.conectar()
        self.assertEqual(r.status_code, 200)
        self.sin_secretos(r.get_data(as_text=True))
        d = self.c.get('/referencias/bdv/cuentas').get_json()
        self.sin_secretos(json.dumps(d))
        self.assertTrue(d['disponible'])
        self.assertEqual(d['cuentas'][0]['ultimos_digitos'], '2411')
        self.assertEqual(d['cuentas'][0]['lector']['estado'], 'pendiente')
        for url in ('/referencias/datos?banco=BDV', '/referencias/claves?banco=BDV', '/referencias/cuentas'):
            self.sin_secretos(self.c.get(url).get_data(as_text=True))

    def test_sin_llave_la_pestana_avisa_y_no_guarda(self):
        os.environ['BDV_LLAVE_CIFRADO'] = ''
        self.entrar(self.A, 'ana-pantalla@test.local')
        self.assertFalse(self.c.get('/referencias/bdv/cuentas').get_json()['disponible'])
        self.assertEqual(self.conectar().status_code, 503)

    def test_cada_pestana_ve_solo_su_banco(self):
        self.entrar(self.A, 'ana-pantalla@test.local')
        cid = self.conectar().get_json()['cuentas'][0]['id']
        bm = q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) VALUES (?, 'Bancamiga', 'BM', TRUE, 'x') "
               'RETURNING id', (self.A,))[0][0]
        ad.guardar_pagos([pago('9000111222333', 50)], self.A, cid)
        ad.guardar_pagos([pago('7000111222333', 50, hora='10:00')], self.A, bm)
        bdv_d = self.c.get('/referencias/datos?banco=BDV&fecha=2026-10-11').get_json()
        bm_d = self.c.get('/referencias/datos?banco=Bancamiga&fecha=2026-10-11').get_json()
        self.assertEqual([p['referencia'] for p in bdv_d['pagos']], ['9000111222333'])
        self.assertEqual([p['referencia'] for p in bm_d['pagos']], ['7000111222333'])
        self.assertEqual([c['banco'] for c in bdv_d['resumen']['cuentas']], ['BDV'])
        pid_bm = bm_d['pagos'][0]['id']   # un pago de Bancamiga no existe en la pestaña BDV
        self.assertEqual(self.c.post('/referencias/asignar', json={'pago_id': pid_bm, 'orden_id': 'O-1', 'banco': 'BDV'}).status_code, 404)
        self.assertEqual(self.c.post('/referencias/asignar', json={'pago_id': pid_bm, 'orden_id': 'O-1', 'banco': 'Bancamiga'}).status_code, 200)
        self.assertEqual([c['banco'] for c in self.c.get('/referencias/cuentas').get_json()['cuentas']], ['Bancamiga'])
        self.assertEqual(self.c.post(f'/referencias/cuentas/{cid}', json={'activo': False}).status_code, 404)
        self.assertEqual(self.c.post('/referencias/cuentas', json={'banco': 'BDV', 'alias': 'x'}).status_code, 400)

    def test_claves_del_bdv_aparte(self):
        self.entrar(self.A, 'ana-pantalla@test.local')
        d = self.c.post('/referencias/claves-api', json={'nombre': 'CRM BDV', 'banco': 'BDV', 'permisos': {'verificar_pago': True}}).get_json()
        self.assertTrue(d['ok'])
        self.assertEqual(d['lector'], [])
        self.assertEqual([k['banco'] for k in d['api']], ['BDV'])
        self.assertEqual(self.c.get('/referencias/claves?banco=Bancamiga').get_json()['api'], [])
        kid = d['api'][0]['id']
        self.assertEqual(self.c.post(f'/referencias/claves-api/{kid}/desactivar', json={'banco': 'Bancamiga'}).status_code, 404)
        self.assertEqual(self.c.post(f'/referencias/claves-api/{kid}/desactivar', json={'banco': 'BDV'}).status_code, 200)

    def test_interruptor_reanudar_borrar_y_aislamiento(self):
        self.entrar(self.A, 'ana-pantalla@test.local')
        cid = self.conectar().get_json()['cuentas'][0]['id']
        d = self.c.post(f'/referencias/bdv/cuentas/{cid}/pausar', json={}).get_json()
        self.assertFalse(d['cuentas'][0]['lector']['encendido'])
        d = self.c.post(f'/referencias/bdv/cuentas/{cid}/leer', json={}).get_json()
        self.assertTrue(d['cuentas'][0]['lector']['encendido'])
        q("UPDATE cuentas_banco SET lector_estado = 'detenido' WHERE id = ?", (cid,))
        d = self.c.post(f'/referencias/bdv/cuentas/{cid}/reanudar', json={}).get_json()
        self.assertEqual(d['cuentas'][0]['lector']['estado'], 'pendiente')
        self.entrar(self.B, 'beto-pantalla@test.local')   # otro usuario no la ve ni la toca
        self.assertEqual(self.c.get('/referencias/bdv/cuentas').get_json()['cuentas'], [])
        for accion in ('pausar', 'leer', 'reanudar', 'borrar'):
            self.assertEqual(self.c.post(f'/referencias/bdv/cuentas/{cid}/{accion}', json={}).status_code, 404)
        self.entrar(self.A, 'ana-pantalla@test.local')
        d = self.c.post(f'/referencias/bdv/cuentas/{cid}/borrar', json={}).get_json()
        self.assertFalse(d['cuentas'][0]['conectada'])

    def test_admin_ve_conexiones_sin_datos_y_puede_pausar(self):
        self.entrar(self.A, 'ana-pantalla@test.local')
        cid = self.conectar().get_json()['cuentas'][0]['id']
        self.assertEqual(self.c.post(f'/admin/antiduplic/bdv/{cid}/pausar', json={}).status_code, 403)
        self.entrar(self.B, 'admin@test.local', admin=True)
        self.sin_secretos(self.c.get('/admin/antiduplic').get_data(as_text=True))
        d = self.c.get('/admin/antiduplic/datos?banco=BDV').get_json()
        self.sin_secretos(json.dumps(d))
        self.assertEqual([c['id'] for c in d['conexiones_bdv']], [cid])
        self.assertEqual(self.c.post(f'/admin/antiduplic/bdv/{cid}/pausar', json={}).status_code, 200)
        self.assertFalse(q('SELECT lector_encendido FROM cuentas_banco WHERE id = ?', (cid,))[0]['lector_encendido'])


if __name__ == '__main__':
    unittest.main()

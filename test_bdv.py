"""
BDV dentro de Referencias: cifrado de los datos de acceso, conexión de cada usuario y separación total por banco.
Arranca la web completa sobre una base SQLite temporal (sin tocar el BDV real).

    python -m unittest test_bdv -v
"""
import json
import os
import tempfile
import unittest

import base64

from nacl.exceptions import CryptoError
from nacl.public import PrivateKey, SealedBox

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'bdv.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
# La prueba hace de lector: solo ella tiene la llave privada. La web recibe únicamente la pública.
PRIVADA = PrivateKey.generate()
PUBLICA = base64.b64encode(bytes(PRIVADA.public_key)).decode()
os.environ['BDV_CLAVE_PUBLICA'] = PUBLICA


def abrir(cifrado, privada=PRIVADA):
    """Lo que hará el lector externo con su llave privada."""
    return SealedBox(privada).decrypt(base64.b64decode(cifrado)).decode('utf-8')


def cifrados_guardados():
    """Los textos cifrados guardados en la base: tampoco deben salir nunca en una respuesta."""
    filas = q("SELECT bdv_usuario_cifrado, bdv_clave_cifrada, numero_cuenta_cifrado FROM cuentas_banco WHERE banco = 'BDV'")
    return [v for f in filas for v in (f['bdv_usuario_cifrado'], f['bdv_clave_cifrada'], f['numero_cuenta_cifrado']) if v]

import app as webapp  # noqa: E402,F401
import antiduplic as ad  # noqa: E402
import api_panel  # noqa: E402
import bdv  # noqa: E402
import bdv_lector_api  # noqa: E402
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
        os.environ['BDV_CLAVE_PUBLICA'] = PUBLICA
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
    def test_la_web_solo_cierra_y_solo_el_lector_abre(self):
        token = bdv.cifrar('hola')
        self.assertNotIn('hola', token)
        self.assertNotEqual(token, bdv.cifrar('hola'))   # cada vez distinto
        self.assertEqual(abrir(token), 'hola')           # el lector (llave privada) sí lo abre
        with self.assertRaises(CryptoError):
            abrir(token, PrivateKey.generate())          # otra llave no
        self.assertFalse(hasattr(bdv, 'descifrar'))      # la web no tiene cómo abrir nada
        os.environ['BDV_CLAVE_PUBLICA'] = 'esto-no-es-una-llave'
        self.assertFalse(bdv.cifrado_disponible())
        os.environ['BDV_CLAVE_PUBLICA'] = ''
        self.assertFalse(bdv.cifrado_disponible())
        with self.assertRaises(bdv.SinLlave):
            bdv.cifrar('x')
        # sin la variable vale la pública del lector aprobado en el panel
        q("INSERT INTO configuracion_redeemer (clave, valor) VALUES ('bdv_lector_publica', ?)", (PUBLICA,))
        try:
            self.assertTrue(bdv.cifrado_disponible())
            self.assertEqual(abrir(bdv.cifrar('hola')), 'hola')
        finally:
            q("DELETE FROM configuracion_redeemer WHERE clave = 'bdv_lector_publica'")
        self.assertEqual(bdv.huella(PUBLICA), bdv.huella(PUBLICA + '\n'))
        self.assertRegex(bdv.huella(PUBLICA), r'^[0-9A-F]{4}-[0-9A-F]{4}$')

    def test_sin_llave_no_se_puede_guardar(self):
        os.environ['BDV_CLAVE_PUBLICA'] = ''
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
        self.assertEqual(abrir(fila['bdv_usuario_cifrado']), USUARIO)
        self.assertEqual(abrir(fila['bdv_clave_cifrada']), CLAVE)
        self.assertEqual(abrir(fila['numero_cuenta_cifrado']), NUMERO)
        self.assertEqual(fila['lector_estado'], 'pendiente')
        self.assertTrue(fila['consentimiento_en'])

    def test_la_cuenta_publica_nunca_trae_datos_de_acceso(self):
        self.conectar()
        cuentas = ad.listar_cuentas(self.conn, self.A, banco='BDV')
        self.assertEqual(len(cuentas), 1)
        texto = str(cuentas)
        for secreto in (USUARIO, CLAVE, NUMERO, 'cifrad', *cifrados_guardados()):
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
        os.environ['BDV_CLAVE_PUBLICA'] = PUBLICA
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
        for secreto in (USUARIO, CLAVE, NUMERO, NUMERO[4:16], *cifrados_guardados()):
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
        os.environ['BDV_CLAVE_PUBLICA'] = ''
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


    def test_documentacion_del_bdv_aparte(self):
        self.assertEqual(self.c.get('/referencias/docs/bdv').status_code, 302)   # sin sesión
        self.entrar(self.A, 'ana-pantalla@test.local')
        html = self.c.get('/referencias/docs/bdv').get_data(as_text=True)
        self.assertIn('Documentación del BDV', html)
        self.assertIn('referencias-bdv-documentacion.pdf', html)   # su propio PDF
        self.assertIn('http://localhost/api/verificar-pago', html)  # la URL real del sitio
        self.assertNotIn('id="admin"', html)
        self.assertIn('/referencias/docs/bdv', self.c.get('/referencias/docs').get_data(as_text=True))
        self.assertEqual(self.c.get('/admin/antiduplic/docs/bdv').status_code, 302)   # no es admin
        self.entrar(self.B, 'admin@test.local', admin=True)
        html = self.c.get('/admin/antiduplic/docs/bdv').get_data(as_text=True)
        self.assertIn('id="admin"', html)
        self.assertIn('BDV_LLAVE_CIFRADO', html)


class BdvApiTest(unittest.TestCase):
    """Los mismos endpoints de siempre: el banco lo decide la clave."""

    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        ad.init_tablas()
        cls.A = BdvPantallaTest.usuario('ana-api@test.local')
        cls.B = BdvPantallaTest.usuario('beto-api@test.local')

    def setUp(self):
        os.environ['BDV_CLAVE_PUBLICA'] = PUBLICA
        for t in ('usos_referencia', 'revisiones_pago', 'pagos_banco', 'claves_lector', 'cuentas_banco', 'webservice_accounts'):
            q(f'DELETE FROM {t}')
        api_panel._limites.clear()
        self.c = self.app.test_client()
        conn = get_db_connection()
        try:
            _, d = bdv.guardar_conexion(conn, self.A, USUARIO, CLAVE, NUMERO, 'Mi BDV', True)
            self.cuenta_bdv = d['cuenta_id']
            self.clave_bdv = api_panel.crear_cuenta(conn, 'CRM BDV', self.A, 'referencias',
                                                    {'verificar_pago': True, 'ver_pagos': True, 'gestionar_pagos': True}, banco='BDV')[1]
            self.clave_bm = api_panel.crear_cuenta(conn, 'CRM BM', self.A, 'referencias',
                                                   {'verificar_pago': True, 'ver_pagos': True, 'gestionar_pagos': True})[1]
            conn.commit()
        finally:
            conn.close()
        self.cuenta_bm = q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) VALUES (?, 'Bancamiga', 'BM', TRUE, 'x') "
                           'RETURNING id', (self.A,))[0][0]
        ad.guardar_pagos([pago('9000111222333', 50)], self.A, self.cuenta_bdv)
        ad.guardar_pagos([pago('7000111222333', 50, hora='10:00')], self.A, self.cuenta_bm)

    def get(self, url, clave):
        return self.c.get(url, headers={'X-API-Key': clave})

    def post(self, url, clave, datos):
        return self.c.post(url, headers={'X-API-Key': clave}, json=datos)

    def verificar(self, clave, ref, orden):
        return self.post('/api/verificar-pago', clave, {'referencia': ref, 'monto': 50, 'orden_id': orden}).get_json()

    def test_verificar_pago_segun_la_clave(self):
        self.assertEqual(self.verificar(self.clave_bdv, '7000111222333', 'O-1')['codigo'], 'no_encontrado')
        self.assertEqual(self.verificar(self.clave_bm, '9000111222333', 'O-1')['codigo'], 'no_encontrado')
        self.assertEqual(self.verificar(self.clave_bdv, '9000111222333', 'O-1')['codigo'], 'aprobado')
        self.assertEqual(self.verificar(self.clave_bm, '7000111222333', 'O-1')['codigo'], 'aprobado')   # misma orden, otro banco
        self.assertEqual(self.verificar(self.clave_bdv, '9000111222333', 'O-2')['codigo'], 'usado')

    def test_consultas_solo_del_banco_de_la_clave(self):
        d = self.get('/api/v1/referencias/pagos', self.clave_bdv).get_json()
        self.assertEqual((d['banco'], [p['referencia'] for p in d['pagos']]), ('BDV', ['9000111222333']))
        d = self.get('/api/v1/referencias/pagos', self.clave_bm).get_json()
        self.assertEqual((d['banco'], [p['referencia'] for p in d['pagos']]), ('Bancamiga', ['7000111222333']))
        id_bm = d['pagos'][0]['id']
        self.assertEqual(self.get(f'/api/v1/referencias/pagos/{id_bm}', self.clave_bdv).status_code, 404)   # de otro banco
        self.assertEqual(self.get(f'/api/v1/referencias/pagos/{id_bm}', self.clave_bm).status_code, 200)
        self.assertEqual(self.get('/api/v1/referencias/resumen?fecha=2026-10-11', self.clave_bdv).get_json()['resumen']['pagos_hoy'], 1)
        self.assertEqual(self.post('/api/v1/referencias/asignar', self.clave_bdv, {'pago_id': id_bm, 'orden_id': 'X'}).status_code, 404)

    def test_cuentas_del_bdv_con_estado_y_sin_datos(self):
        r = self.get('/api/v1/referencias/cuentas', self.clave_bdv)
        texto = r.get_data(as_text=True)
        for secreto in (USUARIO, CLAVE, NUMERO, NUMERO[4:16], 'cifrad', *cifrados_guardados()):
            self.assertNotIn(secreto, texto)
        cuentas = r.get_json()['cuentas']
        self.assertEqual([(c['banco'], c['ultimos_digitos']) for c in cuentas], [('BDV', '2411')])
        self.assertEqual(cuentas[0]['lector']['estado'], 'pendiente')
        self.assertEqual([c['banco'] for c in self.get('/api/v1/referencias/cuentas', self.clave_bm).get_json()['cuentas']], ['Bancamiga'])

    def test_revisiones_por_banco(self):
        ad.guardar_pagos([pago('1234567890', 33, hora='11:00'), pago('99991234567890', 33, hora='11:00')], self.A, self.cuenta_bdv)
        res = self.post('/api/verificar-pago', self.clave_bdv, {'referencia': '1234567890', 'monto': 33, 'hora': '11:00',
                                                                 'orden_id': 'R-1'}).get_json()
        self.assertEqual(res['codigo'], 'revision')
        self.assertEqual(len(self.get('/api/v1/referencias/revisiones', self.clave_bdv).get_json()['revisiones']), 1)
        self.assertEqual(self.get('/api/v1/referencias/revisiones', self.clave_bm).get_json()['revisiones'], [])
        rid = self.get('/api/v1/referencias/revisiones', self.clave_bdv).get_json()['revisiones'][0]['id']
        self.assertEqual(self.post(f'/api/v1/referencias/revisiones/{rid}/descartar', self.clave_bm, {}).status_code, 404)
        self.assertEqual(self.post(f'/api/v1/referencias/revisiones/{rid}/descartar', self.clave_bdv, {}).status_code, 200)

    def test_otro_usuario_y_bot_de_bancamiga_igual(self):
        conn = get_db_connection()
        try:
            clave_b = api_panel.crear_cuenta(conn, 'B', self.B, 'referencias', {'ver_pagos': True}, banco='BDV')[1]
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(self.get('/api/v1/referencias/pagos', clave_b).get_json()['pagos'], [])   # otro usuario, nada
        # el bot de Bancamiga sigue igual: su clave de lector escribe en Bancamiga
        clave, h, pref = ad.nueva_clave_lector()
        q("INSERT INTO claves_lector (usuario_id, cuenta_banco_id, nombre, clave_hash, prefijo_visible, activo, creado_en) "
          "VALUES (?, ?, 'PC', ?, ?, TRUE, 'x')", (self.A, self.cuenta_bm, h, pref))
        r = self.c.post('/api/pagos-banco', json=[pago('5550001112', 9)], headers={'Authorization': f'Bearer {clave}'})
        self.assertEqual(r.get_json()['insertados'], 1)
        res = self.post('/api/verificar-pago', self.clave_bm, {'referencia': '5550001112', 'monto': 9, 'orden_id': 'B-9'}).get_json()
        self.assertEqual(res['codigo'], 'aprobado')

    def test_admin_crea_claves_del_bdv(self):
        with self.c.session_transaction() as s:
            s.update(usuario='admin@test.local', user_db_id=self.B, is_admin=True)
        d = self.c.post('/admin/api/cuentas', json={'nombre': 'CRM BDV', 'usuario_id': self.A, 'tipo': 'referencias', 'banco': 'BDV',
                                                    'permisos': {'verificar_pago': True}}).get_json()
        self.assertEqual(d['cuenta']['banco'], 'BDV')
        self.assertEqual(self.verificar(d['clave'], '9000111222333', 'ADM-1')['codigo'], 'aprobado')
        d = self.c.post('/admin/api/cuentas', json={'nombre': 'Recargas', 'usuario_id': self.A, 'tipo': 'recargas'}).get_json()
        self.assertIsNone(d['cuenta']['banco'])   # las de Recargas no tienen banco


class BdvLectorApiTest(unittest.TestCase):
    """/api/lector-bdv/*: el lector externo se registra, el admin lo aprueba y habla con su clave temporal."""

    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        ad.init_tablas()
        cls.A = BdvPantallaTest.usuario('ana-lector@test.local')
        cls.B = BdvPantallaTest.usuario('beto-lector@test.local')

    def setUp(self):
        os.environ.pop('BDV_CLAVE_PUBLICA', None)   # aquí la llave sale del lector aprobado en el panel
        for t in ('usos_referencia', 'revisiones_pago', 'pagos_banco', 'claves_lector', 'cuentas_banco', 'webservice_accounts'):
            q(f'DELETE FROM {t}')
        q("DELETE FROM configuracion_redeemer WHERE clave LIKE 'bdv_lector_%'")
        bdv_lector_api._golpes.clear()
        self.c = self.app.test_client()
        self.lector = PrivateKey.generate()
        self.publica = base64.b64encode(bytes(self.lector.public_key)).decode()

    def tearDown(self):
        os.environ['BDV_CLAVE_PUBLICA'] = PUBLICA
        q("DELETE FROM configuracion_redeemer WHERE clave LIKE 'bdv_lector_%'")   # que no quede un lector aprobado

    # ---------- ayudas ----------
    def registro(self, publica=None):
        return self.c.post('/api/lector-bdv/registro', json={'publica': publica or self.publica})

    def admin(self, accion, codigo=None):
        with self.c.session_transaction() as s:
            s.update(usuario='admin@test.local', user_db_id=self.B, is_admin=True)
        r = self.c.post(f'/admin/antiduplic/lector/{accion}', json={'codigo': codigo} if codigo else {})
        with self.c.session_transaction() as s:
            s.clear()
        return r

    def aprobado(self):
        self.registro()
        self.assertEqual(self.admin('aprobar', bdv.huella(self.publica)).status_code, 200)

    def clave(self, privada=None):
        r = self.c.post('/api/lector-bdv/reto', json={'publica': self.publica}).get_json()
        return SealedBox(privada or self.lector).decrypt(base64.b64decode(r['reto'])).decode()

    def get(self, url, clave, **headers):
        return self.c.get(url, headers={'Authorization': f'Bearer {clave}', **headers})

    def post(self, url, clave, datos):
        return self.c.post(url, headers={'Authorization': f'Bearer {clave}'}, json=datos)

    def conectar(self, uid=None, numero=NUMERO):
        conn = get_db_connection()
        try:
            status, datos = bdv.guardar_conexion(conn, uid or self.A, USUARIO, CLAVE, numero, '', True)
        finally:
            conn.close()
        self.assertEqual(status, 200, datos)
        return datos['cuenta_id']

    def abrir(self, cifrado):
        return SealedBox(self.lector).decrypt(base64.b64decode(cifrado)).decode()

    # ---------- pruebas ----------
    def test_registro_y_aprobacion_con_codigo(self):
        r = self.registro()
        self.assertEqual((r.status_code, r.get_json()['estado']), (202, 'pendiente'))
        self.assertEqual(r.get_json()['codigo'], bdv.huella(self.publica))
        self.assertFalse(bdv.cifrado_disponible())   # sin lector aprobado no se guardan datos
        self.assertEqual(self.c.post('/api/lector-bdv/reto', json={'publica': self.publica}).status_code, 401)
        self.assertEqual(self.c.post('/admin/antiduplic/lector/aprobar', json={'codigo': bdv.huella(self.publica)}).status_code, 403)
        self.assertEqual(self.admin('aprobar', 'AAAA-BBBB').status_code, 409)   # otro código: no se aprueba nada
        self.assertEqual(self.admin('aprobar', bdv.huella(self.publica).lower()).status_code, 200)
        self.assertEqual(self.registro().get_json()['estado'], 'aprobado')
        self.assertTrue(bdv.cifrado_disponible())
        self.assertEqual(self.abrir(bdv.cifrar('hola')), 'hola')   # la web cierra con la llave del lector aprobado
        # otro lector que se presenta queda pendiente y avisa que reemplazaría al aprobado
        otra = base64.b64encode(bytes(PrivateKey.generate().public_key)).decode()
        self.assertEqual(self.registro(otra).get_json()['estado'], 'pendiente')
        conn = get_db_connection()
        try:
            info = bdv_lector_api.info_lector(conn)
        finally:
            conn.close()
        self.assertTrue(info['pendiente']['reemplaza'])
        self.assertEqual(info['aprobado']['codigo'], bdv.huella(self.publica))
        self.assertNotIn(self.publica, json.dumps(info))   # el panel solo ve códigos
        self.assertEqual(self.admin('rechazar').status_code, 200)

    def test_clave_temporal(self):
        self.aprobado()
        clave = self.clave()
        with self.assertRaises(CryptoError):
            self.clave(PrivateKey.generate())   # solo el lector de verdad abre el reto
        self.assertEqual(self.c.get('/api/lector-bdv/cuentas').status_code, 401)
        self.assertEqual(self.get('/api/lector-bdv/cuentas', 'inventada').status_code, 401)
        self.assertEqual(self.get('/api/lector-bdv/cuentas', clave).status_code, 200)
        conn = get_db_connection()
        try:
            self.assertTrue(bdv_lector_api.lector_vivo(conn))
            sesiones = json.loads(ad._config_get(conn, 'bdv_lector_sesiones'))
        finally:
            conn.close()
        self.assertNotIn(clave, json.dumps(sesiones))   # solo se guarda su hash
        q("UPDATE configuracion_redeemer SET valor = ? WHERE clave = 'bdv_lector_sesiones'",
          (json.dumps([dict(s, vence=1) for s in sesiones]),))
        self.assertEqual(self.get('/api/lector-bdv/cuentas', clave).status_code, 401)   # vencida
        clave = self.clave()
        self.assertEqual(self.admin('desconectar').status_code, 200)
        self.assertEqual(self.get('/api/lector-bdv/cuentas', clave).status_code, 401)
        self.assertEqual(self.c.post('/api/lector-bdv/reto', json={'publica': self.publica}).status_code, 401)
        # una clave de API de un usuario no sirve aquí
        conn = get_db_connection()
        try:
            wsk = api_panel.crear_cuenta(conn, 'CRM', self.A, 'referencias', {'ver_pagos': True}, banco='BDV')[1]
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(self.get('/api/lector-bdv/cuentas', wsk).status_code, 401)
        self.assertEqual(self.c.get('/api/lector-bdv/cuentas', headers={'X-API-Key': wsk}).status_code, 401)

    def test_cuentas_a_leer_y_a_cerrar_con_etag(self):
        self.aprobado()
        clave = self.clave()
        leer = self.conectar()
        pausada = self.conectar(numero='01020000000000001111')
        borrada = self.conectar(numero='01020000000000002222')
        detenida = self.conectar(uid=self.B)
        conn = get_db_connection()
        try:
            bdv.cambiar_lectura(conn, self.A, pausada, False)
            bdv.borrar_conexion(conn, self.A, borrada)
        finally:
            conn.close()
        q("UPDATE cuentas_banco SET lector_estado = 'detenido' WHERE id = ?", (detenida,))
        bm = q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) VALUES (?, 'Bancamiga', 'BM', TRUE, 'x') "
               'RETURNING id', (self.A,))[0][0]
        r = self.get('/api/lector-bdv/cuentas', clave)
        d = r.get_json()
        self.assertEqual([c['id'] for c in d['cuentas']], [leer])
        self.assertEqual(sorted(x['id'] for x in d['cerrar']), sorted([pausada, borrada, detenida]))
        self.assertNotIn(bm, [x['id'] for x in d['cerrar']])
        version_pausada = q('SELECT lector_cambio FROM cuentas_banco WHERE id = ?', (pausada,))[0]['lector_cambio']
        self.assertIn({'id': pausada, 'version': version_pausada, 'falta_pausa': True}, d['cerrar'])
        self.assertFalse([x for x in d['cerrar'] if x['id'] == detenida][0]['falta_pausa'])   # detenida no es pausa
        self.assertFalse([x for x in d['cerrar'] if x['id'] == borrada][0]['falta_pausa'])
        c = d['cuentas'][0]
        self.assertEqual((self.abrir(c['bdv_usuario_cifrado']), self.abrir(c['bdv_clave_cifrada']),
                          self.abrir(c['numero_cuenta_cifrado'])), (USUARIO, CLAVE, NUMERO))
        self.assertEqual((c['usuario_id'], c['ultimos_digitos'], c['estado'], c['fallos_login']), (self.A, '2411', 'pendiente', 0))
        self.assertTrue(c['version'])
        r2 = self.get('/api/lector-bdv/cuentas', clave, **{'If-None-Match': r.headers['ETag']})
        self.assertEqual((r2.status_code, r2.get_data()), (304, b''))   # nada cambió: sin cuerpo
        # el reporte del propio lector no cambia la ETag; pausar sí
        self.post('/api/lector-bdv/estado', clave, {'cuentas': [{'id': leer, 'version': c['version'], 'estado': 'conectado'}]})
        self.assertEqual(self.get('/api/lector-bdv/cuentas', clave, **{'If-None-Match': r.headers['ETag']}).status_code, 304)
        conn = get_db_connection()
        try:
            bdv.cambiar_lectura(conn, self.A, leer, False)
        finally:
            conn.close()
        d = self.get('/api/lector-bdv/cuentas', clave, **{'If-None-Match': r.headers['ETag']}).get_json()
        self.assertEqual(d['cuentas'], [])
        cerrada = [x for x in d['cerrar'] if x['id'] == leer][0]
        # con esa versión el lector puede reportar «pausado» y la web lo acepta
        self.post('/api/lector-bdv/estado', clave, {'cuentas': [{'id': leer, 'version': cerrada['version'], 'estado': 'pausado'}]})
        self.assertEqual(q('SELECT lector_estado FROM cuentas_banco WHERE id = ?', (leer,))[0]['lector_estado'], 'pausado')
        d = self.get('/api/lector-bdv/cuentas', clave).get_json()
        self.assertFalse([x for x in d['cerrar'] if x['id'] == leer][0]['falta_pausa'])   # ya confirmada

    def test_estado_validado_y_por_version(self):
        self.aprobado()
        clave = self.clave()
        cid = self.conectar()
        bm = q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) VALUES (?, 'Bancamiga', 'BM', TRUE, 'x') "
               'RETURNING id', (self.A,))[0][0]
        version = self.get('/api/lector-bdv/cuentas', clave).get_json()['cuentas'][0]['version']
        d = self.post('/api/lector-bdv/estado', clave, {
            'cuentas': [{'id': cid, 'version': version, 'estado': 'conectado', 'mensaje': 'x' * 500 + '\x00',
                         'ultima_lectura': '2026-10-12 10:00:00', 'ultimo_pago': 'no es fecha', 'fallos_login': 99},
                        {'id': cid, 'version': version, 'estado': 'hackeado'},
                        {'id': bm, 'version': '', 'estado': 'conectado'}],
            'lector': {'memoria_mb': 812.5, 'cuentas_leyendo': 3, 'max_cuentas': 30, 'en_espera': 0, 'version': '1.0'}}).get_json()
        self.assertEqual((d['actualizadas'], d['ignoradas']), (1, 2))
        fila = q('SELECT * FROM cuentas_banco WHERE id = ?', (cid,))[0]
        self.assertEqual((fila['lector_estado'], len(fila['lector_mensaje']), fila['lector_ultima_lectura'], fila['lector_fallos_login']),
                         ('conectado', 200, '2026-10-12 10:00:00', 10))
        self.assertIsNone(fila['lector_ultimo_pago'])
        self.assertNotIn('\x00', fila['lector_mensaje'])
        self.assertIsNone(q('SELECT lector_estado FROM cuentas_banco WHERE id = ?', (bm,))[0]['lector_estado'])
        # el usuario guarda datos nuevos: un reporte viejo («detenido» de la versión anterior) no lo pisa
        self.conectar()
        self.post('/api/lector-bdv/estado', clave, {'cuentas': [{'id': cid, 'version': version, 'estado': 'detenido'}]})
        self.assertEqual(q('SELECT lector_estado FROM cuentas_banco WHERE id = ?', (cid,))[0]['lector_estado'], 'pendiente')
        # un reporte sin fallos_login no los pone en cero
        version = self.get('/api/lector-bdv/cuentas', clave).get_json()['cuentas'][0]['version']
        q('UPDATE cuentas_banco SET lector_fallos_login = 1 WHERE id = ?', (cid,))
        self.post('/api/lector-bdv/estado', clave, {'cuentas': [{'id': cid, 'version': version, 'estado': 'conectando'}]})
        self.assertEqual(q('SELECT lector_fallos_login FROM cuentas_banco WHERE id = ?', (cid,))[0]['lector_fallos_login'], 1)
        conn = get_db_connection()
        try:
            estado_lector = json.loads(ad._config_get(conn, 'bdv_lector_estado'))
        finally:
            conn.close()
        self.assertEqual((estado_lector['memoria_mb'], estado_lector['cuentas_leyendo']), (812.5, 3))
        self.assertEqual(self.c.post('/api/lector-bdv/estado', headers={'Authorization': f'Bearer {clave}'}, data='x').status_code, 400)

    def test_pagos_van_al_dueno_de_la_cuenta(self):
        self.aprobado()
        clave = self.clave()
        cid = self.conectar()
        bm = q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) VALUES (?, 'Bancamiga', 'BM', TRUE, 'x') "
               'RETURNING id', (self.A,))[0][0]
        lote = {'cuenta_id': cid, 'usuario_id': self.B,   # el usuario del cuerpo se ignora
                'pagos': [dict(pago('9000111222333', 50), tipo='PAGO MOVIL')]}
        d = self.post('/api/lector-bdv/pagos', clave, lote).get_json()
        self.assertEqual((d['recibidos'], d['insertados']), (1, 1))
        fila = q("SELECT * FROM pagos_banco WHERE referencia = '9000111222333'")[0]
        self.assertEqual((fila['usuario_id'], fila['cuenta_banco_id'], fila['tipo']), (self.A, cid, 'BDV PAGO MOVIL'))
        self.assertEqual(self.post('/api/lector-bdv/pagos', clave, lote).get_json()['insertados'], 0)   # repetido
        self.assertTrue(q('SELECT lector_ultimo_pago FROM cuentas_banco WHERE id = ?', (cid,))[0]['lector_ultimo_pago'])
        self.assertEqual(self.post('/api/lector-bdv/pagos', clave, {'cuenta_id': bm, 'pagos': []}).status_code, 404)
        self.assertEqual(self.post('/api/lector-bdv/pagos', clave, {'cuenta_id': cid, 'pagos': 'x'}).status_code, 400)
        self.assertEqual(self.c.post('/api/lector-bdv/pagos', json=lote).status_code, 401)

    def test_llave_fijada_en_el_env(self):
        os.environ['BDV_CLAVE_PUBLICA'] = self.publica
        self.assertEqual(self.registro().get_json()['estado'], 'aprobado')   # no hace falta aprobar
        otra = base64.b64encode(bytes(PrivateKey.generate().public_key)).decode()
        self.assertEqual(self.registro(otra).status_code, 403)
        self.assertEqual(self.admin('desconectar').status_code, 409)
        self.assertEqual(self.get('/api/lector-bdv/cuentas', self.clave()).status_code, 200)

if __name__ == '__main__':
    unittest.main()

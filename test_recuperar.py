"""
Recuperar la contraseña por correo: enlace de un solo uso, 30 minutos, sin revelar cuentas, límites y prueba del admin.
Arranca la web completa sobre una base SQLite temporal; el envío SMTP se sustituye por una lista.

    python -m unittest test_recuperar -v
"""
import os
import re
import smtplib
import tempfile
import time
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'recuperar.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['ADMIN_EMAIL'] = 'jefe@test.local'

import app as webapp  # noqa: E402
import recuperar  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402
from werkzeug.security import check_password_hash, generate_password_hash  # noqa: E402


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if (sql.lstrip().upper().startswith('SELECT') or 'RETURNING' in sql.upper()) else None
        conn.commit()
        return rows
    finally:
        conn.close()


class RecuperarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        clave = generate_password_hash('vieja-123', method='pbkdf2:sha256')
        cls.U = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                  "VALUES ('Ana', 'A', '0', 'ana@test.local', ?, 0) RETURNING id", (clave,))[0][0]
        q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
          "VALUES ('Jefe', 'J', '0', 'jefe@test.local', ?, 0)", (clave,))
        recuperar.init_tabla()

    def setUp(self):
        q('DELETE FROM recuperar_clave')
        recuperar._ips.clear()
        recuperar.ENVIAR_EN_HILO = False
        self.enviados = []
        self.fallo = None
        self._orig = recuperar._smtp_enviar

        def falso(cfg, destino, asunto, texto, html):
            if self.fallo:
                raise self.fallo
            self.enviados.append({'cfg': cfg, 'a': destino, 'asunto': asunto, 'texto': texto, 'html': html})
        recuperar._smtp_enviar = falso
        self.app.config['MAIL_USERNAME'] = 'web@test.local'
        self.app.config['MAIL_PASSWORD'] = 'abcd efgh ijkl mnop'
        os.environ['SITE_URL'] = 'https://web.test'
        self.c = self.app.test_client()

    def tearDown(self):
        recuperar._smtp_enviar = self._orig
        os.environ.pop('SITE_URL', None)

    # ---------- ayudas ----------
    def pedir(self, correo='ana@test.local', **kw):
        return self.c.post('/recuperar', data={'correo': correo}, **kw).get_data(as_text=True)

    def token(self, i=-1):
        return re.search(r'/restablecer/(\S+)', self.enviados[i]['texto']).group(1)

    def clave_actual(self):
        return q('SELECT contraseña FROM usuarios WHERE id = ?', (self.U,))[0]['contraseña']

    # ---------- pruebas ----------
    def test_flujo_completo(self):
        html = self.pedir('ANA@test.local')
        self.assertIn('te enviamos un enlace', html)
        self.assertEqual(len(self.enviados), 1)
        correo = self.enviados[0]
        self.assertEqual(correo['a'], 'ana@test.local')
        self.assertEqual(correo['cfg']['clave'], 'abcdefghijklmnop')  # sin espacios
        self.assertTrue(correo['texto'].count('https://web.test/restablecer/'))
        tok = self.token()

        self.assertIn('Contraseña nueva', self.c.get(f'/restablecer/{tok}').get_data(as_text=True))
        r = self.c.post(f'/restablecer/{tok}', data={'clave': 'nueva-clave-1', 'repetir': 'otra-cosa-1'})
        self.assertIn('no coinciden', r.get_data(as_text=True))
        r = self.c.post(f'/restablecer/{tok}', data={'clave': 'corta', 'repetir': 'corta'})
        self.assertIn('entre 8 y 128', r.get_data(as_text=True))
        self.assertTrue(check_password_hash(self.clave_actual(), 'vieja-123'))  # todavía no cambió

        r = self.c.post(f'/restablecer/{tok}', data={'clave': 'nueva-clave-1', 'repetir': 'nueva-clave-1'})
        self.assertIn('Tu contraseña se cambió', r.get_data(as_text=True))
        self.assertTrue(check_password_hash(self.clave_actual(), 'nueva-clave-1'))
        self.assertIn('se cambió', self.enviados[-1]['asunto'])

        # el enlace ya no sirve, ni para ver ni para cambiar
        self.assertIn('no es válido', self.c.get(f'/restablecer/{tok}').get_data(as_text=True))
        self.c.post(f'/restablecer/{tok}', data={'clave': 'tercera-123', 'repetir': 'tercera-123'})
        self.assertTrue(check_password_hash(self.clave_actual(), 'nueva-clave-1'))

        # entra con la nueva
        r = self.c.post('/login', data={'correo': 'ana@test.local', 'contraseña': 'nueva-clave-1'})
        self.assertEqual(r.headers['Location'], '/')

    def test_correo_que_no_existe_da_la_misma_respuesta(self):
        a = self.pedir('nadie@test.local')
        self.assertIn('te enviamos un enlace', a)
        self.assertEqual(self.enviados, [])

    def test_admin_no_se_recupera_por_correo(self):
        self.assertIn('te enviamos un enlace', self.pedir('jefe@test.local'))
        self.assertEqual(self.enviados, [])

    def test_solo_se_guarda_el_hash(self):
        self.pedir()
        tok = self.token()
        filas = q('SELECT token_hash FROM recuperar_clave')
        self.assertEqual(len(filas), 1)
        self.assertNotEqual(filas[0]['token_hash'], tok)

    def test_enlace_vencido(self):
        self.pedir()
        q('UPDATE recuperar_clave SET expira = ?', (int(time.time()) - 1,))
        self.assertIn('no es válido', self.c.get(f'/restablecer/{self.token()}').get_data(as_text=True))

    def test_enlace_nuevo_anula_el_anterior(self):
        self.pedir()
        self.pedir()
        primero, segundo = self.token(0), self.token(1)
        self.assertIn('no es válido', self.c.get(f'/restablecer/{primero}').get_data(as_text=True))
        self.assertIn('Contraseña nueva', self.c.get(f'/restablecer/{segundo}').get_data(as_text=True))

    def test_limites_por_cuenta_y_por_ip(self):
        for _ in range(5):
            self.pedir()
        self.assertEqual(len(self.enviados), recuperar.MAX_POR_CUENTA_HORA)
        for _ in range(recuperar.MAX_POR_IP_HORA - 5):
            self.pedir('nadie@test.local')
        self.assertIn('Pediste demasiados enlaces', self.pedir('nadie@test.local'))

    def test_el_enlace_no_usa_la_cabecera_host(self):
        self.pedir(base_url='https://sitio-falso.example')
        self.assertIn('https://web.test/restablecer/', self.enviados[0]['texto'])
        self.assertNotIn('sitio-falso', self.enviados[0]['html'])

    def test_sin_site_url_solo_funciona_en_local(self):
        os.environ.pop('SITE_URL')
        self.assertIn('no está disponible', self.c.get('/recuperar', base_url='https://sitio-falso.example').get_data(as_text=True))
        self.pedir(base_url='http://127.0.0.1:5000')
        self.assertIn('http://127.0.0.1:5000/restablecer/', self.enviados[0]['texto'])

    def test_sin_correo_configurado(self):
        self.app.config['MAIL_PASSWORD'] = ''
        self.assertIn('no está disponible', self.c.get('/recuperar').get_data(as_text=True))
        self.assertNotIn('/recuperar', self.c.get('/auth').get_data(as_text=True))
        self.app.config['MAIL_PASSWORD'] = 'x'
        self.assertIn('¿Olvidaste tu contraseña?', self.c.get('/auth').get_data(as_text=True))

    def test_admin_correo_y_prueba(self):
        self.assertEqual(self.c.get('/admin/correo').status_code, 302)
        self.assertEqual(self.c.post('/admin/correo/prueba', json={'destino': 'yo@test.local'}).status_code, 403)
        with self.c.session_transaction() as s:
            s['is_admin'] = True
        html = self.c.get('/admin/correo').get_data(as_text=True)
        self.assertIn('Recuperación activa', html)
        self.assertNotIn('abcd', html)  # la contraseña nunca se muestra
        self.assertEqual(self.c.post('/admin/correo/prueba', data={'destino': 'yo@test.local'}).status_code, 415)

        d = self.c.post('/admin/correo/prueba', json={'destino': 'yo@test.local'}).get_json()
        self.assertTrue(d['ok'])
        self.assertEqual(self.enviados[-1]['a'], 'yo@test.local')

        self.fallo = smtplib.SMTPAuthenticationError(535, b'bad')
        d = self.c.post('/admin/correo/prueba', json={'destino': 'yo@test.local'}).get_json()
        self.assertIn('contraseña de aplicación', d['error'])
        self.fallo = TimeoutError()
        d = self.c.post('/admin/correo/prueba', json={'destino': 'yo@test.local'}).get_json()
        self.assertIn('bloquee', d['error'])


if __name__ == '__main__':
    unittest.main()

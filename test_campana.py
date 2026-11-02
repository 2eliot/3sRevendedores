"""
Campana de notificaciones (campana.py + static/campana.js): avisos del admin, noticias y saldo acreditado, con lo nuevo
contado; al abrirla se marca como visto. El admin ve el saldo acreditado a todos los usuarios.

    python -m unittest test_campana -v
"""
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'campana.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30

import app as webapp  # noqa: E402
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


class CampanaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        webapp.create_news_table()
        webapp.create_news_views_table()
        webapp.get_unread_wallet_credits_count(0)          # crea creditos_billetera con su columna «visto»
        webapp.get_unread_personal_notifications_count(0)  # crea notificaciones_personalizadas
        cls.A = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) VALUES ('Ana', 'A', '0', 'ana-c@test.local', 'x', 0) RETURNING id")[0][0]
        cls.B = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) VALUES ('Beto', 'B', '0', 'beto-c@test.local', 'x', 0) RETURNING id")[0][0]
        cls.ADMIN = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) VALUES ('Admin', '', '0', 'admin-c@test.local', 'x', 0) RETURNING id")[0][0]

    def setUp(self):
        for t in ('creditos_billetera', 'notificaciones_personalizadas', 'noticias_vistas', 'noticias'):
            q(f'DELETE FROM {t}')
        q("DELETE FROM configuracion_redeemer WHERE clave LIKE 'campana_%'")
        self.c = self.app.test_client()

    def como(self, uid, correo, admin=False):
        with self.c.session_transaction() as s:
            s.clear()
            s.update(usuario=correo, user_db_id=uid, is_admin=admin)

    def credito(self, uid, monto, fecha='2026-11-02 15:00:00'):
        q('INSERT INTO creditos_billetera (usuario_id, monto, fecha, visto) VALUES (?, ?, ?, FALSE)', (uid, monto, fecha))

    def estado(self):
        return self.c.get('/campana/estado').get_json()

    def test_avisos_noticias_y_saldo_y_al_abrir_se_marcan(self):
        self.credito(self.A, 10, '2026-11-02 15:00:00')
        q("INSERT INTO notificaciones_personalizadas (usuario_id, titulo, mensaje, tipo, visto, fecha) VALUES (?, 'Pago recibido', 'Tu pago <b>ya</b> se acreditó', 'success', FALSE, '2026-11-02 16:00:00')", (self.A,))
        q("INSERT INTO noticias (titulo, contenido, fecha) VALUES ('Nuevo juego', '<p>Ya está Mobile Legends</p>', '2026-11-02 14:00:00')")
        self.como(self.A, 'ana-c@test.local')
        d = self.estado()
        self.assertEqual((d['nuevos'], d['admin']), (3, False))
        self.assertEqual([i['tipo'] for i in d['items']], ['aviso', 'saldo', 'noticia'])   # lo más nuevo primero
        self.assertEqual(d['items'][1]['texto'], 'Se acreditaron $10.00 a tu saldo')
        self.assertEqual(d['items'][0]['texto'], 'Tu pago ya se acreditó')   # sin HTML
        self.assertEqual(d['items'][1]['fecha'], '02/11 11:00')   # hora de Venezuela
        self.assertEqual(self.c.post('/campana/visto', data='x').status_code, 415)
        self.assertTrue(self.c.post('/campana/visto', json={}).get_json()['ok'])
        d = self.estado()
        self.assertEqual(d['nuevos'], 0)
        self.assertEqual(len(d['items']), 3)   # siguen en la lista, ya vistos
        self.assertEqual(webapp.get_unread_wallet_credits_count(self.A), 0)   # la billetera también los da por vistos
        # y lo que llega después vuelve a contar
        self.credito(self.A, 5, '2026-11-02 17:00:00')
        self.assertEqual(self.estado()['nuevos'], 1)

    def test_cada_usuario_ve_lo_suyo(self):
        self.credito(self.A, 10)
        q("INSERT INTO notificaciones_personalizadas (usuario_id, titulo, mensaje, tipo, visto) VALUES (?, 'Solo para Ana', 'x', 'info', FALSE)", (self.A,))
        self.como(self.B, 'beto-c@test.local')
        self.assertEqual(self.estado()['items'], [])

    def test_el_admin_ve_el_saldo_acreditado_a_todos(self):
        self.credito(self.A, 10)
        self.credito(self.B, 25.5)
        self.como(self.ADMIN, 'admin-c@test.local', admin=True)
        d = self.estado()
        self.assertEqual((d['admin'], d['nuevos']), (True, 2))
        self.assertEqual(sorted(i['titulo'] for i in d['items']), ['Saldo acreditado a Ana A', 'Saldo acreditado a Beto B'])
        self.c.post('/campana/visto', json={})
        self.assertEqual(self.estado()['nuevos'], 0)
        self.assertEqual(q('SELECT visto FROM creditos_billetera WHERE usuario_id = ?', (self.A,))[0]['visto'] in (0, False), True)   # al usuario no se le marca
        self.credito(self.B, 3)
        self.assertEqual(self.estado()['nuevos'], 1)

    def test_sin_sesion(self):
        self.assertEqual(self.c.get('/campana/estado').status_code, 401)
        self.assertEqual(self.c.post('/campana/visto', json={}).status_code, 401)

    def test_la_campana_esta_en_la_barra_tambien_para_el_admin(self):
        for uid, correo, admin in ((self.A, 'ana-c@test.local', False), (self.ADMIN, 'admin-c@test.local', True)):
            self.como(uid, correo, admin)
            html = self.c.get('/dashboard').get_data(as_text=True)
            self.assertIn('class="news-icon-container"', html)
            self.assertIn('campana.js', html)


if __name__ == '__main__':
    unittest.main()

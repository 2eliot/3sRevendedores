"""
Dashboard: el «Resumen del día» debe aceptar una fecha suelta del calendario
(?fecha=YYYY-MM-DD) sin romper los botones Hoy / Ayer / Antes de ayer.

    python -m unittest test_dashboard_fecha -v
"""
import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta

import pytz

_tmp = tempfile.mkdtemp(prefix='dash-fecha-test-')
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'dash_fecha.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 40
os.environ['DEFAULT_TZ'] = 'America/Caracas'
os.environ['DEV_LOGIN'] = '1'

import app as appmod
from pg_compat import get_db_connection


class DashboardFechaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        appmod.app.config['TESTING'] = True
        cls.client = appmod.app.test_client()

        tz = pytz.timezone('America/Caracas')
        hoy = datetime.now(tz).date()
        cls.hoy = hoy.isoformat()
        cls.ayer = (hoy - timedelta(days=1)).isoformat()
        cls.antes_ayer = (hoy - timedelta(days=2)).isoformat()
        cls.otra_fecha = (hoy - timedelta(days=3)).isoformat()

        conn = get_db_connection()
        cur = conn.cursor()
        correo = f'dash-fecha-{os.getpid()}@test.local'
        cur.execute(
            'INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            ('Ana', 'Fecha', '04140000000', correo, 'x', 50.0),
        )
        cls.uid = cur.execute('SELECT id FROM usuarios WHERE correo = ?', (correo,)).fetchone()['id']
        filas = [
            (cls.hoy, 2.00, 'Free Fire - 110 💎'),
            (cls.ayer, 3.00, 'Free Fire - 341 💎'),
            (cls.antes_ayer, 7.00, 'Free Fire - 572 💎'),
            (cls.otra_fecha, 11.00, 'Free Fire - 1.166 💎'),
        ]
        for fecha, monto, paquete in filas:
            cur.execute(
                'INSERT INTO historial_compras (usuario_id, monto, fecha, paquete_nombre, pin, tipo_evento) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (cls.uid, monto, f'{fecha} 15:00:00', paquete, 'PIN-DEMO', 'compra'),
            )
        conn.commit()
        conn.close()

        with cls.client.session_transaction() as sess:
            sess['usuario'] = correo
            sess['user_db_id'] = cls.uid
            sess['id'] = str(cls.uid).zfill(5)
            sess['nombre'] = 'Ana'
            sess['apellido'] = 'Fecha'
            sess['saldo'] = 50.0
            sess['is_admin'] = False

    def _get(self, url):
        respuesta = self.client.get(url)
        self.assertEqual(respuesta.status_code, 200, url)
        return respuesta.get_data(as_text=True)

    def _fecha_calendario(self, html):
        m = re.search(r'id="dash-fecha"[^>]*value="([^"]+)"', html)
        self.assertIsNotNone(m, 'No se encontró el calendario')
        return m.group(1)

    def _preset_activo(self, html):
        m = re.search(r'class="filter-btn active"[^>]*>([^<]+)</a>', html)
        return m.group(1).strip() if m else None

    def test_por_defecto_muestra_hoy(self):
        html = self._get('/dashboard')
        self.assertEqual(self._fecha_calendario(html), self.hoy)
        self.assertEqual(self._preset_activo(html), 'Hoy')
        self.assertIn('$2.00', html)

    def test_ayer(self):
        html = self._get('/dashboard?preset=ayer')
        self.assertEqual(self._fecha_calendario(html), self.ayer)
        self.assertEqual(self._preset_activo(html), 'Ayer')
        self.assertIn('$3.00', html)

    def test_antes_de_ayer(self):
        html = self._get('/dashboard?preset=antes_ayer')
        self.assertEqual(self._fecha_calendario(html), self.antes_ayer)
        self.assertEqual(self._preset_activo(html), 'Antes de ayer')
        self.assertIn('$7.00', html)

    def test_fecha_del_calendario(self):
        html = self._get(f'/dashboard?fecha={self.otra_fecha}')
        self.assertEqual(self._fecha_calendario(html), self.otra_fecha)
        self.assertIsNone(self._preset_activo(html), 'Un día suelto no debe marcar Hoy/Ayer/Antes de ayer')
        self.assertIn('$11.00', html)
        self.assertIn('dash-cal is-active', html)

    def test_fecha_invalida_vuelve_a_hoy(self):
        html = self._get('/dashboard?fecha=no-es-fecha')
        self.assertEqual(self._fecha_calendario(html), self.hoy)
        self.assertEqual(self._preset_activo(html), 'Hoy')

    def test_fecha_futura_se_limita_a_hoy(self):
        html = self._get('/dashboard?fecha=2099-01-01')
        self.assertEqual(self._fecha_calendario(html), self.hoy)
        self.assertEqual(self._preset_activo(html), 'Hoy')

    def test_rango_viejo_sigue_funcionando(self):
        html = self._get(f'/dashboard?inicio={self.otra_fecha}&fin={self.ayer}')
        # Cuando hay rango, el resumen muestra el último día del rango.
        self.assertEqual(self._fecha_calendario(html), self.ayer)
        self.assertIn('$3.00', html)
        self.assertIsNone(self._preset_activo(html))


if __name__ == '__main__':
    unittest.main()

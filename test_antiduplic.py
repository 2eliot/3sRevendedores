"""
Pruebas de Antiduplic (referencias de Bancamiga) sobre una base SQLite temporal.

    python -m unittest test_antiduplic -v
"""
import os
import tempfile
import threading
import unittest

_tmp = tempfile.TemporaryDirectory()
os.environ['DATABASE_PATH'] = os.path.join(_tmp.name, 'antiduplic_test.db')
os.environ.pop('DATABASE_URL', None)
os.environ['PAGOS_BANCO_TOKEN'] = 'token-de-prueba'

from flask import Flask  # noqa: E402

import antiduplic as ad  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

AUTH = {'Authorization': 'Bearer token-de-prueba'}


def crear_app():
    app = Flask(__name__)
    app.secret_key = 'test'
    app.register_blueprint(ad.bp)
    return app


def base_inicial():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS configuracion_redeemer (clave TEXT PRIMARY KEY, valor TEXT NOT NULL, '
                 'fecha_actualizacion TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS usuarios (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, apellido TEXT, '
                 'telefono TEXT, correo TEXT, contraseña TEXT, saldo REAL DEFAULT 0)')
    conn.commit()
    conn.close()


def limpiar():
    conn = get_db_connection()
    for t in ('usos_referencia', 'revisiones_pago', 'pagos_banco'):
        conn.execute(f'DELETE FROM {t}')
    conn.commit()
    conn.close()


def pago(ref, monto, fecha='2026-10-05', hora='08:14:52', tipo='NC Credito Inmediato'):
    return {'referencia': ref, 'fecha': fecha, 'hora': hora, 'monto': monto, 'tipo': tipo}


class AntiduplicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_inicial()
        ad.init_tablas()
        cls.app = crear_app()

    def setUp(self):
        limpiar()
        self.c = self.app.test_client()

    def enviar(self, pagos, headers=AUTH):
        return self.c.post('/api/pagos-banco', json=pagos, headers=headers)

    # --- PASO 2: recepción ---------------------------------------------------
    def test_api_exige_token(self):
        self.assertEqual(self.enviar([pago('123456789', 10)], headers={}).status_code, 401)
        self.assertEqual(self.enviar([pago('123456789', 10)], headers={'Authorization': 'Bearer otro'}).status_code, 401)

    def test_pago_duplicado_se_ignora(self):
        lote = [pago('7568126123', 7350.00), pago('0012345678', 100.5, hora='09:00:00')]
        r1 = self.enviar(lote).get_json()
        r2 = self.enviar(lote + [pago('999888777', 20, hora='10:00:00')]).get_json()
        self.assertEqual(r1, {'recibidos': 2, 'insertados': 2})
        self.assertEqual(r2, {'recibidos': 3, 'insertados': 1})

    def test_conserva_ceros_a_la_izquierda(self):
        self.enviar([pago('0012345678', 50)])
        conn = get_db_connection()
        row = conn.execute('SELECT referencia, ref_ultimos6 FROM pagos_banco').fetchone()
        conn.close()
        self.assertEqual((row['referencia'], row['ref_ultimos6']), ('0012345678', '345678'))

    def test_limite_de_pagos_por_envio(self):
        lote = [pago(str(1000000 + i), 1) for i in range(ad.MAX_PAGOS_POR_ENVIO + 1)]
        self.assertEqual(self.enviar(lote).status_code, 413)

    # --- PASO 3: verificación ------------------------------------------------
    def test_aprueba_y_no_deja_usar_dos_veces(self):
        self.enviar([pago('7568126123', 7350.00)])
        r1 = ad.verificar_y_usar_pago('7568126123', '7.350,00', origen='revendedor', revendedor_id=1)
        r2 = ad.verificar_y_usar_pago('7568126123', 7350, origen='revendedor', revendedor_id=2)
        self.assertEqual(r1['codigo'], 'aprobado')
        self.assertEqual(r2['codigo'], 'usado')
        self.assertNotIn('revendedor', r2['mensaje'].lower())  # no dice quién la usó

    def test_dos_usos_al_mismo_tiempo(self):
        self.enviar([pago('5555666677', 200)])
        resultados, barrera = [], threading.Barrier(8)

        def intento(i):
            barrera.wait()
            resultados.append(ad.verificar_y_usar_pago('5555666677', 200, origen='revendedor', revendedor_id=i)['codigo'])

        hilos = [threading.Thread(target=intento, args=(i,)) for i in range(8)]
        [h.start() for h in hilos]
        [h.join() for h in hilos]
        self.assertEqual(resultados.count('aprobado'), 1, resultados)
        conn = get_db_connection()
        n = conn.execute('SELECT COUNT(*) AS n FROM usos_referencia').fetchone()['n']
        conn.close()
        self.assertEqual(n, 1)

    def test_referencia_larga_con_digitos_de_mas(self):
        self.enviar([pago('0026123456', 75)])
        r = ad.verificar_y_usar_pago('987026123456', 75)  # reportó dígitos de más delante
        self.assertEqual(r['codigo'], 'aprobado')

    def test_referencia_reportada_mas_corta_que_la_del_banco(self):
        self.enviar([pago('7568126123', 30)])
        self.assertEqual(ad.verificar_y_usar_pago('126123', 30)['codigo'], 'aprobado')

    def test_referencia_corta_sin_fecha(self):
        self.enviar([pago('7568126123', 30)])
        self.assertEqual(ad.verificar_y_usar_pago('6123', 30)['codigo'], 'falta_fecha')
        self.assertEqual(ad.verificar_y_usar_pago('6123', 30, fecha='2026-10-05')['codigo'], 'aprobado')

    def test_referencia_muy_corta(self):
        self.assertEqual(ad.verificar_y_usar_pago('123', 30, fecha='2026-10-05')['codigo'], 'ref_corta')
        self.assertEqual(ad.verificar_y_usar_pago('12a456', 30)['codigo'], 'ref_corta')

    def test_monto_no_coincide_y_no_encontrado(self):
        self.enviar([pago('7568126123', 30)])
        self.assertEqual(ad.verificar_y_usar_pago('7568126123', 31)['codigo'], 'monto_no_coincide')
        self.assertEqual(ad.verificar_y_usar_pago('1111222233', 30)['codigo'], 'no_encontrado')

    def test_empate_pide_hora_y_desempata(self):
        self.enviar([pago('1000123456', 50, hora='08:00:00'), pago('2000123456', 50, hora='09:30:00')])
        r = ad.verificar_y_usar_pago('123456', 50)
        self.assertEqual(r['codigo'], 'revision')
        self.assertTrue(r.get('pedir_hora'))
        r = ad.verificar_y_usar_pago('123456', 50, hora='09:30')
        self.assertEqual(r['codigo'], 'aprobado')
        self.assertEqual(r['pago']['referencia'], '2000123456')

    def test_empate_sin_desempate_va_a_revision_manual(self):
        self.enviar([pago('1000123456', 50, hora='08:00:00'), pago('2000123456', 50, hora='08:00:00')])
        r = ad.verificar_y_usar_pago('123456', 50, hora='08:00', origen='revendedor', revendedor_id=7)
        self.assertEqual(r['codigo'], 'revision')
        self.assertFalse(r['ok'])
        conn = get_db_connection()
        estados = [x['estado'] for x in conn.execute('SELECT estado FROM pagos_banco').fetchall()]
        rev = conn.execute("SELECT COUNT(*) AS n FROM revisiones_pago WHERE estado = 'pendiente'").fetchone()['n']
        conn.close()
        self.assertEqual(estados, ['revision', 'revision'])
        self.assertEqual(rev, 1)

    def test_orden_id_repetido_no_usa_otro_pago(self):
        self.enviar([pago('3333444455', 10), pago('6666777788', 10, hora='11:00:00')])
        r1 = ad.verificar_y_usar_pago('3333444455', 10, origen='crm', orden_id='ORD-1')
        r2 = ad.verificar_y_usar_pago('6666777788', 10, origen='crm', orden_id='ORD-1')
        self.assertEqual(r1['codigo'], 'aprobado')
        self.assertEqual(r2['codigo'], 'ya_aprobado')
        self.assertEqual(r2['pago']['referencia'], '3333444455')
        self.assertEqual(ad.verificar_y_usar_pago('6666777788', 10, origen='crm', orden_id='ORD-2')['codigo'], 'aprobado')

    def test_api_verificar_con_token_y_sin_sesion(self):
        self.enviar([pago('3333444455', 10)])
        self.assertEqual(self.c.post('/api/verificar-pago', json={'referencia': '3333444455', 'monto': 10}).status_code, 401)
        r = self.c.post('/api/verificar-pago', headers=AUTH,
                        json={'referencia': '3333444455', 'monto': 10, 'origen': 'crm', 'orden_id': 'X1'}).get_json()
        self.assertEqual(r['codigo'], 'aprobado')

    def test_limite_de_intentos_del_revendedor(self):
        ad._rate.clear()
        conn = get_db_connection()
        ad._config_set(conn, ad.CONFIG_REPORTAR, '1')
        conn.commit()
        conn.close()
        with self.c.session_transaction() as s:
            s['usuario'] = 'r@x.com'
            s['user_db_id'] = 99
        codigos = [self.c.post('/api/verificar-pago', json={'referencia': '123456', 'monto': 1}).status_code
                   for _ in range(ad.REPORTES_MAX + 1)]
        self.assertEqual(codigos[-1], 429)
        self.assertTrue(all(c == 200 for c in codigos[:-1]))

    # --- PASO 5: asignación manual -------------------------------------------
    def test_admin_resuelve_revision(self):
        conn = get_db_connection()
        conn.execute("INSERT INTO usuarios (id, nombre, apellido, correo) VALUES (7, 'Ana', 'Ruiz', 'a@x.com')")
        conn.commit()
        conn.close()
        self.enviar([pago('1000123456', 50, hora='08:00:00'), pago('2000123456', 50, hora='08:00:00')])
        ad.verificar_y_usar_pago('123456', 50, hora='08:00', origen='revendedor', revendedor_id=7)
        with self.c.session_transaction() as s:
            s['is_admin'] = True
        datos = self.c.get('/admin/antiduplic/datos').get_json()
        rev = datos['revisiones'][0]
        elegido = rev['candidatos'][1]['id']
        r = self.c.post('/admin/antiduplic/asignar', json={'pago_id': elegido, 'revision_id': rev['id']}).get_json()
        self.assertTrue(r['ok'], r)
        conn = get_db_connection()
        est = {x['id']: x['estado'] for x in conn.execute('SELECT id, estado FROM pagos_banco').fetchall()}
        uso = conn.execute('SELECT * FROM usos_referencia').fetchone()
        conn.execute('DELETE FROM usuarios WHERE id = 7')
        conn.commit()
        conn.close()
        self.assertEqual(est[elegido], 'usado')
        self.assertEqual(sorted(est.values()), ['disponible', 'usado'])
        self.assertEqual((uso['origen'], uso['revendedor_id']), ('manual', 7))

    def test_admin_requiere_sesion_admin(self):
        self.assertEqual(self.c.get('/admin/antiduplic/datos').status_code, 403)
        self.assertEqual(self.c.post('/admin/antiduplic/asignar', json={'pago_id': 1}).status_code, 403)


if __name__ == '__main__':
    unittest.main()

"""
Referencias: fecha de corte por cuenta («no usar referencias antes de»), desbloquear uno a uno, antigüedad máxima
opcional y eliminar referencias sin usar (por banco). Arranca la web completa sobre una base SQLite temporal.

    python -m unittest test_corte_referencias -v
"""
import base64
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta

from nacl.public import PrivateKey

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'corte.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['BDV_CLAVE_PUBLICA'] = base64.b64encode(bytes(PrivateKey.generate().public_key)).decode()

import app as webapp  # noqa: E402,F401
import antiduplic as ad  # noqa: E402
import api_panel  # noqa: E402
import bdv  # noqa: E402
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


def momento(horas_atras=0, minutos_atras=0):
    """(fecha, hora) de Venezuela hace un rato."""
    t = datetime.now(ad._tz()).replace(tzinfo=None) - timedelta(hours=horas_atras, minutes=minutos_atras)
    return t.strftime('%Y-%m-%d'), t.strftime('%H:%M:%S')


def pago(ref, monto, horas_atras=0, minutos_atras=0):
    fecha, hora = momento(horas_atras, minutos_atras)
    return {'referencia': ref, 'fecha': fecha, 'hora': hora, 'monto': monto, 'tipo': 'NC'}


class CorteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = webapp.app
        cls.app.config['TESTING'] = True
        ad.init_tablas()
        cls.A = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                  "VALUES ('Ana', 'A', '0', 'ana-corte@test.local', 'x', 0) RETURNING id")[0][0]
        cls.B = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                  "VALUES ('Beto', 'B', '0', 'beto-corte@test.local', 'x', 0) RETURNING id")[0][0]

    def setUp(self):
        for t in ('usos_referencia', 'revisiones_pago', 'pagos_banco', 'pagos_borrados', 'claves_lector', 'cuentas_banco',
                  'webservice_accounts'):
            q(f'DELETE FROM {t}')
        api_panel._limites.clear()
        self.c = self.app.test_client()
        self.entrar(self.A)

    # ---------- ayudas ----------
    def entrar(self, uid):
        with self.c.session_transaction() as s:
            s.update(usuario=f'{uid}@test.local', user_db_id=uid, is_admin=False)

    def cuenta_nueva(self, alias='Bancamiga principal'):
        """Como la crea el usuario en la pantalla: el corte queda en este momento."""
        d = self.c.post('/referencias/cuentas', json={'alias': alias, 'ultimos_digitos': '4021'}).get_json()
        return [c for c in d['cuentas'] if c['alias'] == alias][0]

    def lector(self, uid, cuenta_id):
        clave, h, pref = ad.nueva_clave_lector()
        q('INSERT INTO claves_lector (usuario_id, cuenta_banco_id, nombre, clave_hash, prefijo_visible, activo, creado_en) '
          "VALUES (?, ?, 'PC', ?, ?, TRUE, 'x')", (uid, cuenta_id, h, pref))
        return clave

    def enviar(self, clave, pagos):
        r = self.c.post('/api/pagos-banco', json=pagos, headers={'Authorization': f'Bearer {clave}'})
        self.assertEqual(r.status_code, 200)
        return r.get_json()

    def clave_api(self, uid, banco=None):
        conn = get_db_connection()
        try:
            _, clave = api_panel.crear_cuenta(conn, f'CRM {uid}', uid, 'referencias',
                                              {'verificar_pago': True, 'ver_pagos': True, 'gestionar_pagos': True}, banco=banco)
            conn.commit()
            return clave
        finally:
            conn.close()

    def verificar(self, clave, ref, monto, orden):
        return self.c.post('/api/verificar-pago', json={'referencia': ref, 'monto': monto, 'orden_id': orden, 'origen': 'crm'},
                           headers={'X-API-Key': clave}).get_json()

    def estado(self, ref):
        return q('SELECT estado FROM pagos_banco WHERE referencia = ?', (ref,))[0]['estado']

    def pago_id(self, ref):
        return q('SELECT id FROM pagos_banco WHERE referencia = ?', (ref,))[0]['id']

    def corte(self, cuenta_id, usar_desde, antiguedad=None, banco='Bancamiga'):
        return self.c.post(f'/referencias/cuentas/{cuenta_id}/corte',
                           json={'usar_desde': usar_desde, 'antiguedad_max_horas': antiguedad, 'banco': banco})

    # ---------- pruebas ----------
    def test_cuenta_nueva_usa_referencias_desde_que_se_creo(self):
        cuenta = self.cuenta_nueva()
        self.assertEqual(cuenta['usar_desde'], ad.ahora_minuto())
        clave = self.lector(self.A, cuenta['id'])
        d = self.enviar(clave, [pago('111111111', 10, horas_atras=26), pago('222222222', 20, horas_atras=3),
                                pago('333333333', 30, minutos_atras=-5)])
        self.assertEqual(d['insertados'], 3)   # se guardan todos (se ven en la lista)…
        self.assertEqual([self.estado(r) for r in ('111111111', '222222222', '333333333')], ['bloqueado', 'bloqueado', 'disponible'])
        api = self.clave_api(self.A)
        r = self.verificar(api, '222222222', 20, 'ORD-1')   # …pero los de antes del corte no se usan
        self.assertEqual((r['ok'], r['codigo'], r['corte']), (False, 'anterior_al_corte', cuenta['usar_desde']))
        self.assertEqual(self.verificar(api, '333333333', 30, 'ORD-2')['codigo'], 'aprobado')
        # resumen y filtros
        conn = get_db_connection()
        try:   # el día de ese pago (según la hora a la que corra la prueba puede ser hoy o ayer)
            res = ad.resumen(conn, self.A, fecha=momento(horas_atras=3)[0], banco='Bancamiga')
        finally:
            conn.close()
        self.assertGreaterEqual(res['bloqueados_hoy'], 1)
        self.assertIn('bloqueados_hoy', self.c.get('/referencias/datos?banco=Bancamiga').get_json()['resumen'])
        lista = self.c.get('/referencias/datos?estado=bloqueado&fecha=').get_json()['pagos']
        self.assertEqual(sorted(p['referencia'] for p in lista), ['111111111', '222222222'])
        api_lista = self.c.get('/api/v1/referencias/pagos?estado=bloqueado', headers={'X-API-Key': api}).get_json()
        self.assertEqual(api_lista['total'], 2)
        cuentas = self.c.get('/api/v1/referencias/cuentas', headers={'X-API-Key': api}).get_json()['cuentas']
        self.assertEqual((cuentas[0]['usar_desde'], cuentas[0]['antiguedad_max_horas']), (cuenta['usar_desde'], None))

    def test_desbloquear_uno_y_volver_a_bloquear(self):
        cuenta = self.cuenta_nueva()
        clave = self.lector(self.A, cuenta['id'])
        self.enviar(clave, [pago('444444444', 44, horas_atras=5)])
        pid = self.pago_id('444444444')
        # a mano tampoco se puede usar mientras esté bloqueado
        r = self.c.post('/referencias/asignar', json={'pago_id': pid, 'orden_id': 'ORD-M', 'banco': 'Bancamiga'})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self.c.post(f'/referencias/pagos/{pid}/bloquear', json={}).status_code, 409)   # no fue desbloqueado
        self.assertEqual(self.c.post(f'/referencias/pagos/{pid}/desbloquear', json={'banco': 'Bancamiga'}).status_code, 200)
        self.assertEqual(self.estado('444444444'), 'disponible')
        # un cambio de corte no vuelve a bloquear lo desbloqueado a mano
        self.assertEqual(self.corte(cuenta['id'], cuenta['usar_desde']).status_code, 200)
        self.assertEqual(self.estado('444444444'), 'disponible')
        self.assertEqual(self.c.post(f'/referencias/pagos/{pid}/bloquear', json={'banco': 'Bancamiga'}).status_code, 200)
        self.assertEqual(self.estado('444444444'), 'bloqueado')
        self.c.post(f'/referencias/pagos/{pid}/desbloquear', json={'banco': 'Bancamiga'})
        self.assertEqual(self.verificar(self.clave_api(self.A), '444444444', 44, 'ORD-3')['codigo'], 'aprobado')
        # de otro usuario o del otro banco: 404
        self.entrar(self.B)
        self.assertEqual(self.c.post(f'/referencias/pagos/{pid}/bloquear', json={}).status_code, 404)
        self.entrar(self.A)
        self.assertEqual(self.c.post(f'/referencias/pagos/{pid}/bloquear', json={'banco': 'BDV'}).status_code, 404)
        self.assertEqual(self.c.post(f'/referencias/pagos/{pid}/otra', json={}).status_code, 404)

    def test_cambiar_y_quitar_el_corte(self):
        cuenta = self.cuenta_nueva()
        clave = self.lector(self.A, cuenta['id'])
        self.enviar(clave, [pago('555555555', 5, horas_atras=48), pago('666666666', 6, horas_atras=2)])
        hace_un_dia = (datetime.now(ad._tz()) - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M')
        d = self.corte(cuenta['id'], hace_un_dia).get_json()
        self.assertTrue(d['ok'], d)
        self.assertIn('1', d['mensaje'])   # 1 pago anterior sin usar
        self.assertEqual((self.estado('555555555'), self.estado('666666666')), ('bloqueado', 'disponible'))
        self.assertEqual(self.corte(cuenta['id'], '').status_code, 200)   # sin corte: todo se puede usar
        self.assertEqual((self.estado('555555555'), self.estado('666666666')), ('disponible', 'disponible'))
        self.assertIsNone(q('SELECT usar_desde FROM cuentas_banco WHERE id = ?', (cuenta['id'],))[0]['usar_desde'])
        manana = (datetime.now(ad._tz()) + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M')
        self.assertEqual(self.corte(cuenta['id'], manana).status_code, 400)        # no puede ser futuro
        self.assertEqual(self.corte(cuenta['id'], 'ayer').status_code, 400)
        self.assertEqual(self.corte(cuenta['id'], hace_un_dia, antiguedad=0).status_code, 400)
        self.assertEqual(self.corte(cuenta['id'], hace_un_dia, antiguedad=721).status_code, 400)
        self.entrar(self.B)
        self.assertEqual(self.corte(cuenta['id'], hace_un_dia).status_code, 404)  # cuenta de otro usuario

    def test_antiguedad_maxima_opcional(self):
        cuenta = self.cuenta_nueva()
        clave = self.lector(self.A, cuenta['id'])
        self.corte(cuenta['id'], '')   # sin corte: aquí solo se prueba la antigüedad
        self.enviar(clave, [pago('777777777', 70, horas_atras=50), pago('888888888', 80, horas_atras=2)])
        api = self.clave_api(self.A)
        self.assertEqual(self.corte(cuenta['id'], '', antiguedad=24).status_code, 200)
        r = self.verificar(api, '777777777', 70, 'ORD-4')
        self.assertEqual((r['codigo'], r['antiguedad_max_horas']), ('pago_antiguo', 24))
        self.assertEqual(self.estado('777777777'), 'disponible')   # no cambia de estado: solo no se acepta solo
        self.assertEqual(self.verificar(api, '888888888', 80, 'ORD-5')['codigo'], 'aprobado')
        # a mano sí se puede
        r = self.c.post('/referencias/asignar', json={'pago_id': self.pago_id('777777777'), 'orden_id': 'ORD-4', 'banco': 'Bancamiga'})
        self.assertEqual(r.status_code, 200)
        # es opcional: sin límite vuelve a aceptarse
        self.enviar(clave, [pago('999999999', 90, horas_atras=60)])
        self.assertEqual(self.corte(cuenta['id'], '', antiguedad='').status_code, 200)
        self.assertEqual(self.verificar(api, '999999999', 90, 'ORD-6')['codigo'], 'aprobado')

    def test_eliminar_referencias_sin_usar(self):
        cuenta = self.cuenta_nueva()
        clave = self.lector(self.A, cuenta['id'])
        self.corte(cuenta['id'], '')
        self.enviar(clave, [pago('100000001', 1, horas_atras=30), pago('100000002', 2, horas_atras=29),
                            pago('100000003', 3, horas_atras=1), pago('100000004', 4, horas_atras=28)])
        api = self.clave_api(self.A)
        self.assertEqual(self.verificar(api, '100000004', 4, 'ORD-USADA')['codigo'], 'aprobado')   # usada: se queda
        # de otro usuario: no se toca
        otra = q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) VALUES (?, 'Bancamiga', 'B', TRUE, 'x') "
                 'RETURNING id', (self.B,))[0][0]
        self.enviar(self.lector(self.B, otra), [pago('100000001', 1, horas_atras=30)])
        hace_un_dia = (datetime.now(ad._tz()) - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M')
        r = self.c.post('/referencias/eliminar', json={'modo': 'antes', 'antes': hace_un_dia, 'banco': 'Bancamiga'}).get_json()
        self.assertEqual(r['cuantas'], 2)   # sin confirmar solo cuenta (la usada no entra)
        self.assertEqual(q('SELECT COUNT(*) AS n FROM pagos_banco')[0]['n'], 5)
        r = self.c.post('/referencias/eliminar', json={'modo': 'antes', 'antes': hace_un_dia, 'confirmar': True, 'banco': 'Bancamiga'}).get_json()
        self.assertEqual(r['eliminadas'], 2)
        mias = sorted(x['referencia'] for x in q('SELECT referencia FROM pagos_banco WHERE usuario_id = ?', (self.A,)))
        self.assertEqual(mias, ['100000003', '100000004'])
        self.assertEqual(q('SELECT COUNT(*) AS n FROM pagos_banco WHERE usuario_id = ?', (self.B,))[0]['n'], 1)
        # si el bot las vuelve a mandar, no vuelven; y verificarlas dice que se eliminaron
        self.assertEqual(self.enviar(clave, [pago('100000001', 1, horas_atras=30)])['insertados'], 0)
        r = self.verificar(api, '100000001', 1, 'ORD-7')
        self.assertEqual((r['ok'], r['codigo']), (False, 'eliminado'))
        # todas: también las anteriores al corte, nunca las usadas
        self.c.post('/referencias/eliminar', json={'modo': 'todas', 'confirmar': True, 'banco': 'Bancamiga'})
        self.assertEqual([x['referencia'] for x in q('SELECT referencia FROM pagos_banco WHERE usuario_id = ?', (self.A,))], ['100000004'])
        self.assertEqual(self.c.post('/referencias/eliminar', json={'modo': 'x'}).status_code, 400)
        self.assertEqual(self.c.post('/referencias/eliminar', json={'modo': 'antes', 'antes': ''}).status_code, 400)
        self.assertEqual(self.c.post('/referencias/eliminar', data='modo=todas').status_code, 415)   # solo JSON

    def test_eliminar_es_por_banco(self):
        bm = self.cuenta_nueva()
        self.corte(bm['id'], '')
        self.enviar(self.lector(self.A, bm['id']), [pago('200000001', 1, horas_atras=3)])
        conn = get_db_connection()
        try:
            _, d = bdv.guardar_conexion(conn, self.A, 'u', 'c', '01020000000000002411', '', True)
        finally:
            conn.close()
        ad.guardar_pagos([pago('200000002', 2, horas_atras=3)], self.A, d['cuenta_id'])
        self.c.post('/referencias/eliminar', json={'modo': 'todas', 'confirmar': True, 'banco': 'BDV'})
        self.assertEqual([x['referencia'] for x in q('SELECT referencia FROM pagos_banco')], ['200000001'])   # Bancamiga intacto

    def test_bdv_corte_al_conectar_y_al_reconectar(self):
        conn = get_db_connection()
        try:
            _, d = bdv.guardar_conexion(conn, self.A, 'u', 'c', '01020000000000003333', '', True)
            cid = d['cuenta_id']
            self.assertEqual(q('SELECT usar_desde FROM cuentas_banco WHERE id = ?', (cid,))[0]['usar_desde'], ad.ahora_minuto())
            ad.guardar_pagos([pago('300000001', 1, horas_atras=20), pago('300000002', 2, minutos_atras=-3)], self.A, cid)
            self.assertEqual((self.estado('300000001'), self.estado('300000002')), ('bloqueado', 'disponible'))
            # cambiar sus datos no mueve el corte
            q("UPDATE cuentas_banco SET usar_desde = '2026-01-01 00:00' WHERE id = ?", (cid,))
            bdv.guardar_conexion(conn, self.A, 'u2', 'c2', '01020000000000003333', '', True)
            self.assertEqual(q('SELECT usar_desde FROM cuentas_banco WHERE id = ?', (cid,))[0]['usar_desde'], '2026-01-01 00:00')
            # borrada y vuelta a conectar: el corte pasa a ahora (lo de mientras tanto pudo usarse por fuera)
            bdv.borrar_conexion(conn, self.A, cid)
            bdv.guardar_conexion(conn, self.A, 'u3', 'c3', '01020000000000003333', '', True)
            self.assertEqual(q('SELECT usar_desde FROM cuentas_banco WHERE id = ?', (cid,))[0]['usar_desde'], ad.ahora_minuto())
        finally:
            conn.close()

    def test_migracion_pone_el_corte_a_las_del_bdv_una_sola_vez(self):
        cid = q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en, consentimiento_en, bdv_clave_cifrada) "
                "VALUES (?, 'BDV', 'BDV', TRUE, '2026-10-12 00:30:00', '2026-10-12 00:35:21', 'x') RETURNING id", (self.A,))[0][0]
        bm = q("INSERT INTO cuentas_banco (usuario_id, banco, alias, activo, creado_en) VALUES (?, 'Bancamiga', 'BM', TRUE, "
               "'2026-10-01 10:00:00') RETURNING id", (self.A,))[0][0]
        ad.guardar_pagos([{'referencia': '400000001', 'fecha': '2026-10-11', 'hora': '09:15', 'monto': 1},
                          {'referencia': '400000002', 'fecha': '2026-10-12', 'hora': '00:36', 'monto': 2}], self.A, cid)
        q("DELETE FROM configuracion_redeemer WHERE clave = 'migr_corte_cuentas'")
        ad._migrar_cortes()
        self.assertEqual(q('SELECT usar_desde FROM cuentas_banco WHERE id = ?', (cid,))[0]['usar_desde'], '2026-10-12 00:35')
        self.assertIsNone(q('SELECT usar_desde FROM cuentas_banco WHERE id = ?', (bm,))[0]['usar_desde'])   # Bancamiga igual que antes
        self.assertEqual((self.estado('400000001'), self.estado('400000002')), ('bloqueado', 'disponible'))
        # si el usuario lo quita, la migración no lo vuelve a poner
        q('UPDATE cuentas_banco SET usar_desde = NULL WHERE id = ?', (cid,))
        ad._migrar_cortes()
        self.assertIsNone(q('SELECT usar_desde FROM cuentas_banco WHERE id = ?', (cid,))[0]['usar_desde'])

    def test_pantalla(self):
        html = self.c.get('/referencias').get_data(as_text=True)
        for texto in ('data-dias="0">Hoy', 'data-dias="1">Ayer', 'data-dias="2">Anteayer', 'id="rf-cortes"', 'id="rf-eliminar"',
                      'Antigüedad máxima (opcional)', 'value="bloqueado"'):
            self.assertIn(texto, html)
        # «Buscar» es el primer campo de los filtros
        filtros = html[html.index('id="rf-filtros"'):]
        self.assertLess(filtros.index('name="q"'), filtros.index('name="fecha"'))


if __name__ == '__main__':
    unittest.main()

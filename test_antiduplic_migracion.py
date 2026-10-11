"""
Migración de Antiduplic a datos por usuario, partiendo del esquema ANTERIOR con datos (SQLite).

    python -m unittest test_antiduplic_migracion -v
"""
import json
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'migracion.db')
os.environ.pop('DATABASE_URL', None)
os.environ.pop('ANTIDUPLIC_USUARIO_DUENO', None)
os.environ['ADMIN_EMAIL'] = 'dueno@test.local'
os.environ['PAGOS_BANCO_TOKEN'] = 'token-antiguo'

from pg_compat import get_db_connection  # noqa: E402


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if sql.lstrip().upper().startswith('SELECT') else None
        conn.commit()
        return rows
    finally:
        conn.close()


def esquema_anterior():
    """Tablas y datos tal como estaban antes del cambio (orden_id único global, índice único sin usuario)."""
    for sql in (
        'CREATE TABLE usuarios (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, apellido TEXT, telefono TEXT, '
        'correo TEXT, contraseña TEXT, saldo REAL DEFAULT 0)',
        'CREATE TABLE configuracion_redeemer (clave TEXT PRIMARY KEY, valor TEXT NOT NULL, fecha_actualizacion TEXT)',
        '''CREATE TABLE pagos_banco (id INTEGER PRIMARY KEY AUTOINCREMENT, referencia TEXT NOT NULL, ref_ultimos6 TEXT NOT NULL,
           fecha TEXT NOT NULL, hora TEXT NOT NULL, monto NUMERIC(15,2) NOT NULL, tipo TEXT,
           estado TEXT NOT NULL DEFAULT 'disponible', creado_en TEXT NOT NULL)''',
        'CREATE UNIQUE INDEX ux_pagos_banco_mov ON pagos_banco (referencia, fecha, hora, monto)',
        '''CREATE TABLE usos_referencia (id INTEGER PRIMARY KEY AUTOINCREMENT, pago_id INTEGER NOT NULL UNIQUE REFERENCES pagos_banco (id),
           origen TEXT NOT NULL, revendedor_id INTEGER, orden_id TEXT UNIQUE, referencia_reportada TEXT,
           monto_reportado NUMERIC(15,2), usado_en TEXT NOT NULL)''',
        '''CREATE TABLE revisiones_pago (id INTEGER PRIMARY KEY AUTOINCREMENT, origen TEXT NOT NULL, revendedor_id INTEGER,
           orden_id TEXT, referencia_reportada TEXT, monto_reportado NUMERIC(15,2), fecha_reportada TEXT, hora_reportada TEXT,
           candidatos TEXT NOT NULL, estado TEXT NOT NULL DEFAULT 'pendiente', creado_en TEXT NOT NULL, resuelto_en TEXT)'''):
        q(sql)
    q("INSERT INTO usuarios (id, nombre, correo) VALUES (1, 'Otro', 'otro@test.local')")
    q("INSERT INTO usuarios (id, nombre, correo) VALUES (5, 'Dueño', 'dueno@test.local')")
    q("INSERT INTO configuracion_redeemer (clave, valor) VALUES ('antiduplic_ultimo_envio', '2026-10-09 10:00:00')")
    q("INSERT INTO pagos_banco (referencia, ref_ultimos6, fecha, hora, monto, tipo, estado, creado_en) "
      "VALUES ('7568126123', '126123', '2026-10-09', '08:00:00', 100, 'NC', 'usado', '2026-10-09 08:01:00')")
    q("INSERT INTO pagos_banco (referencia, ref_ultimos6, fecha, hora, monto, tipo, estado, creado_en) "
      "VALUES ('1000555111', '555111', '2026-10-09', '09:00:00', 50, 'NC', 'revision', '2026-10-09 09:01:00')")
    q("INSERT INTO pagos_banco (referencia, ref_ultimos6, fecha, hora, monto, tipo, estado, creado_en) "
      "VALUES ('2000555111', '555111', '2026-10-09', '09:00:00', 50, 'NC', 'revision', '2026-10-09 09:01:00')")
    q("INSERT INTO usos_referencia (pago_id, origen, orden_id, referencia_reportada, monto_reportado, usado_en) "
      "VALUES (1, 'crm', 'ORD-1', '126123', 100, '2026-10-09 08:05:00')")
    q("INSERT INTO revisiones_pago (origen, orden_id, referencia_reportada, monto_reportado, candidatos, creado_en) "
      "VALUES ('crm', 'ORD-2', '555111', 50, '[2, 3]', '2026-10-09 09:05:00')")


class MigracionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        esquema_anterior()
        import antiduplic
        cls.ad = antiduplic
        antiduplic.init_tablas()

    def test_datos_pasan_al_dueno_sin_perder_nada(self):
        pagos = q('SELECT usuario_id, cuenta_banco_id, estado FROM pagos_banco ORDER BY id')
        self.assertEqual(len(pagos), 3)
        self.assertEqual({p['usuario_id'] for p in pagos}, {5})
        cuenta = q('SELECT * FROM cuentas_banco WHERE usuario_id = 5')
        self.assertEqual(len(cuenta), 1)
        self.assertEqual(cuenta[0]['ultimo_envio'], '2026-10-09 10:00:00')
        self.assertEqual({p['cuenta_banco_id'] for p in pagos}, {cuenta[0]['id']})
        self.assertEqual([p['estado'] for p in pagos], ['usado', 'revision', 'revision'])
        self.assertEqual(q('SELECT usuario_id, orden_id FROM usos_referencia')[0]['usuario_id'], 5)
        self.assertEqual(q('SELECT usuario_id FROM revisiones_pago')[0]['usuario_id'], 5)

    def test_orden_id_ahora_es_unico_por_usuario(self):
        sql = q("SELECT sql FROM sqlite_master WHERE name = 'usos_referencia'")[0]['sql']
        self.assertNotRegex(sql, r'orden_id\s+TEXT\s+UNIQUE')
        # Otro usuario puede usar el mismo orden_id; el mismo usuario no
        q("INSERT INTO pagos_banco (usuario_id, referencia, ref_ultimos6, fecha, hora, monto, estado, creado_en) "
          "VALUES (1, '7568126123', '126123', '2026-10-09', '08:00:00', 100, 'disponible', 'x')")
        nuevo = q("SELECT id FROM pagos_banco WHERE usuario_id = 1")[0]['id']
        q("INSERT INTO usos_referencia (pago_id, usuario_id, origen, orden_id, usado_en) VALUES (?, 1, 'crm', 'ORD-1', 'x')", (nuevo,))
        with self.assertRaises(Exception):
            q("INSERT INTO usos_referencia (pago_id, usuario_id, origen, orden_id, usado_en) VALUES (2, 5, 'crm', 'ORD-1', 'x')")

    def test_la_verificacion_sigue_igual_para_el_dueno(self):
        r = self.ad.verificar_y_usar_pago('126123', 100, origen='crm', orden_id='ORD-1', usuario_id=5)
        self.assertEqual(r['codigo'], 'ya_aprobado')
        r = self.ad.verificar_y_usar_pago('555111', 50, origen='crm', orden_id='ORD-3', usuario_id=5)
        self.assertEqual(r['codigo'], 'revision')
        rev = self.ad.listar_revisiones(get_db_connection(), 5)
        self.assertEqual(len(rev), 1)
        self.assertEqual(len(rev[0]['candidatos']), 2)

    def test_migracion_es_repetible(self):
        self.ad._tablas_listas = False
        self.ad.init_tablas()
        self.assertEqual(len(q('SELECT id FROM cuentas_banco')), 1)
        self.assertEqual(len(q('SELECT id FROM pagos_banco WHERE usuario_id IS NULL')), 0)


if __name__ == '__main__':
    unittest.main()

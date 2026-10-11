"""
Lector del BDV contra un BDV FALSO (nunca el banco real): entra, lee y guarda; sesión vencida; banco que no responde;
dos rechazos → detenido; interruptor (pausa con «Salir» y vuelta a entrar); borrar la conexión; tope de cuentas;
y que nada secreto llegue a los registros ni a los mensajes.

El BDV falso imita los selectores del real (Angular Material): entrada en dos pasos, table.table-saldo-cuenta,
mat-dialog-container con mat-row y su paginador, «Regresar» (mat-dialog-close) y «Salir» (aria-label).

    python -m unittest test_lector_bdv -v
"""
import asyncio
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cryptography.fernet import Fernet

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'lector_bdv.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ['BDV_LLAVE_CIFRADO'] = Fernet.generate_key().decode()

import app as webapp  # noqa: E402,F401  (crea todas las tablas)
import antiduplic as ad  # noqa: E402
import bdv  # noqa: E402
import lector_bdv as lb  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

U1, C1 = 'usuario-secreto-1', 'clave-secreta-1'
U2, C2 = 'usuario-secreto-2', 'clave-secreta-2'
CUENTA = {U1: '2411', U2: '7788'}


# ---------------------------------------------------------------------------
# BDV falso
# ---------------------------------------------------------------------------

PAGINA = r"""<!doctype html><html><head><meta charset="utf-8"><title>BDV falso</title></head><body><div id="app"></div>
<script>
let usuario = '', filas = [], pagina = 0;
const POR_PAGINA = 2, app = document.getElementById('app');
function esc(t) { return String(t).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c])); }
function pintarLogin() {
  app.innerHTML = '<form id="f1"><input formcontrolname="username"><button type="submit">Siguiente</button></form>';
  document.getElementById('f1').onsubmit = e => { e.preventDefault(); usuario = e.target.querySelector('input').value; pintarClave(); };
}
function pintarClave() {
  app.innerHTML = '<form id="f2"><input formcontrolname="password" type="password"><button type="submit">Entrar</button></form><div id="err"></div>';
  document.getElementById('f2').onsubmit = async e => {
    e.preventDefault();
    const r = await fetch('/api/login', {method: 'POST', body: JSON.stringify({usuario, clave: e.target.querySelector('input').value})});
    if (r.ok) pintarInicio(await r.json());
    else document.getElementById('err').innerHTML = '<mat-error>Usuario o contraseña inválidos</mat-error>';
  };
}
function pintarInicio(d) {
  const salir = d.estilo === 'lateral'   // como en pantallas chicas: menú lateral y confirmación
    ? '<button class="hide-on-large-only" onclick="abrirMenu()"><mat-icon>menu</mat-icon></button>' +
      '<mat-sidenav id="nav" style="display:none"><a class="menu-list-item" onclick="confirmarSalir()"><img src="/assets/icons/salir.png"> Salir</a></mat-sidenav>'
    : '<header><button aria-label="Salir" onclick="salir()">Salir</button></header>';
  app.innerHTML = salir +
    '<table class="table-saldo-cuenta"><tbody><tr><td headers="cuenta">Corriente</td><td headers="numero">0102***' + d.ultimos +
    '</td><td><mat-icon onclick="movs()">subject</mat-icon></td></tr></tbody></table><div id="dlg"></div>';
}
async function movs() {
  const r = await fetch('/api/movimientos');
  if (r.status === 401) { pintarLogin(); return; }
  filas = (await r.json()).filas; pagina = 0; pintarDialogo();
}
function pintarDialogo() {
  const desde = pagina * POR_PAGINA, ultima = desde + POR_PAGINA >= filas.length;
  document.getElementById('dlg').innerHTML = '<mat-dialog-container>' + filas.slice(desde, desde + POR_PAGINA).map(f =>
    '<mat-row><span class="mat-column-fecha">' + esc(f.fecha) + '</span> <span class="mat-column-referencia">' + esc(f.referencia) +
    '</span> <span class="mat-column-descripcion">' + esc(f.descripcion) + '</span> <span class="mat-column-indicadorCargoAbono">' +
    esc(f.tipo) + '</span> <span class="mat-column-importe">' + esc(f.importe) + '</span></mat-row>').join('') +
    '<button class="mat-paginator-navigation-next" ' + (ultima ? 'disabled' : '') + ' onclick="pagina++; pintarDialogo()">&gt;</button>' +
    '<button mat-dialog-close onclick="document.getElementById(\'dlg\').innerHTML=\'\'">Regresar</button></mat-dialog-container>';
}
function abrirMenu() { document.getElementById('nav').style.display = 'block'; }
function confirmarSalir() {
  document.getElementById('dlg').innerHTML = '<mat-dialog-container><p>¿Desea salir?</p><button onclick="salir()">Sí</button></mat-dialog-container>';
}
async function salir() { await fetch('/api/salir', {method: 'POST'}); pintarLogin(); }
(async () => { const r = await fetch('/api/estado'); const d = await r.json(); if (d.dentro) pintarInicio(d); else pintarLogin(); })();
</script></body></html>"""


class BancoFalso:
    def __init__(self):
        self.lock = threading.Lock()
        self.reiniciar()
        banco = self

        class Manejador(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _sesion(self):
                for parte in (self.headers.get('Cookie') or '').split(';'):
                    k, _, v = parte.strip().partition('=')
                    if k == 'sesion':
                        return v
                return None

            def _json(self, status, datos, cookie=None):
                cuerpo = json.dumps(datos).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                if cookie:
                    self.send_header('Set-Cookie', cookie)
                self.send_header('Content-Length', str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)

            def do_GET(self):
                with banco.lock:
                    usuario = banco.sesiones.get(self._sesion())
                if self.path == '/':
                    if banco.caido:   # la página no termina de cargar: el lector no encuentra la casilla
                        cuerpo = b'<html><body>Cargando...</body></html>'
                    else:
                        cuerpo = PAGINA.encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(cuerpo)))
                    self.end_headers()
                    self.wfile.write(cuerpo)
                elif self.path == '/api/estado':
                    self._json(200, {'dentro': bool(usuario), 'ultimos': CUENTA.get(usuario, ''), 'estilo': banco.estilo})
                elif self.path == '/api/movimientos':
                    if not usuario:
                        self._json(401, {'error': 'sesion'})
                    else:
                        with banco.lock:
                            banco.lecturas[usuario] += 1
                            filas = list(banco.movimientos.get(usuario, []))
                        self._json(200, {'filas': filas})
                else:
                    self._json(404, {})

            def do_POST(self):
                largo = int(self.headers.get('Content-Length') or 0)
                datos = json.loads(self.rfile.read(largo) or b'{}')
                if self.path == '/api/login':
                    usuario = datos.get('usuario')
                    with banco.lock:
                        banco.intentos[usuario] += 1
                        ok = banco.claves.get(usuario) == datos.get('clave') and usuario not in banco.rechazar
                        if ok:
                            token = secrets.token_hex(8)
                            banco.sesiones[token] = usuario
                            banco.entradas[usuario] += 1
                    if ok:
                        self._json(200, {'ultimos': CUENTA[usuario], 'estilo': banco.estilo}, cookie=f'sesion={token}; Path=/')
                    else:
                        self._json(401, {'error': 'Usuario o contraseña inválidos'})
                elif self.path == '/api/salir':
                    with banco.lock:
                        usuario = banco.sesiones.pop(self._sesion(), None)
                        if usuario:
                            banco.salidas[usuario] += 1
                    self._json(200, {'ok': True})
                else:
                    self._json(404, {})

        self.servidor = ThreadingHTTPServer(('127.0.0.1', 0), Manejador)
        threading.Thread(target=self.servidor.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.servidor.server_address[1]}/'

    def reiniciar(self):
        with self.lock:
            self.claves = {U1: C1, U2: C2}
            self.sesiones, self.rechazar, self.caido, self.estilo = {}, set(), False, 'arriba'
            self.intentos, self.entradas, self.salidas, self.lecturas = Counter(), Counter(), Counter(), Counter()
            self.movimientos = {U1: [], U2: []}

    def dentro(self, usuario):
        with self.lock:
            return usuario in self.sesiones.values()

    def expirar(self, usuario):
        with self.lock:
            for t in [t for t, u in self.sesiones.items() if u == usuario]:
                del self.sesiones[t]


def mov(ref, importe, tipo='CREDITO', desc='PAGO MOVIL DE CLIENTE', fecha='11-10-2026 09:15'):
    return {'fecha': fecha, 'referencia': ref, 'descripcion': desc, 'tipo': tipo, 'importe': importe}


def fila(*a, **k):
    """Una fila tal como la lee el lector de la tabla del banco."""
    d = mov(*a, **k)
    d['monto'] = d.pop('importe')
    return d


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if sql.lstrip().upper().startswith('SELECT') else None
        conn.commit()
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------

class LecturaDeLaTablaTest(unittest.TestCase):
    def test_reglas(self):
        self.assertEqual(lb.monto_numero('-1.234,56 Bs.'), -1234.56)
        self.assertEqual(lb.monto_numero('7.350,00 Bs.'), 7350.0)
        p = lb.a_pago(fila('000123456789', '1.234,56 Bs.'))
        self.assertEqual(p, {'referencia': '000123456789', 'fecha': '2026-10-11', 'hora': '09:15', 'monto': 1234.56,
                             'tipo': 'BDV PAGO MOVIL DE CLIENTE'})
        self.assertIsNone(lb.a_pago(fila('1234', '-50,00 Bs.', tipo='DEBITO', desc='PAGO A TERCERO')))
        self.assertIsNone(lb.a_pago(fila('1234', '3,21 Bs.', desc='ABONO INTERESES')))
        self.assertIsNone(lb.a_pago(fila('1234', '1,00 Bs.', desc='REINTEGRO COMISION')))
        self.assertIsNone(lb.a_pago(fila('', '10,00 Bs.')))
        self.assertEqual(len(lb.a_pago(fila('1', '10,00', desc='X' * 300))['tipo']), 120)


class LectorBdvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.banco = BancoFalso()
        lb.URL = cls.banco.url
        lb.CADA_SEG, lb.REVISAR_SEG, lb.ESCALON_SEG = 0.5, 0.3, 0.2
        lb.ESPERA_RECHAZO_SEG, lb.NO_RESPONDE_BASE_SEG, lb.NO_RESPONDE_MAX_SEG = 1.5, 1, 2
        lb.TIMEOUT_MS, lb.MEDIR_CADA_SEG = 4000, 0
        ad.init_tablas()
        conn = get_db_connection()
        cls.uid = conn.execute("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                               "VALUES ('Lea', 'L', '0', 'lector@test.local', 'x', 0) RETURNING id").fetchone()[0]
        conn.commit()
        conn.close()

    def setUp(self):
        self.banco.reiniciar()
        for t in ('usos_referencia', 'pagos_banco', 'cuentas_banco'):
            q(f'DELETE FROM {t}')
        q("DELETE FROM configuracion_redeemer WHERE clave IN ('bdv_lector_instancia', 'bdv_lector_estado')")

    def conectar(self, usuario=U1, clave=C1):
        conn = get_db_connection()
        try:
            status, datos = bdv.guardar_conexion(conn, self.uid, usuario, clave, '0102' + '0' * 12 + CUENTA[usuario], '', True)
        finally:
            conn.close()
        self.assertEqual(status, 200, datos)
        return datos['cuenta_id']

    def cuenta(self, cid):
        return q('SELECT * FROM cuentas_banco WHERE id = ?', (cid,))[0]

    def accion(self, fn, *args):
        conn = get_db_connection()
        try:
            return fn(conn, *args)
        finally:
            conn.close()

    def correr(self, escenario, max_cuentas=30):
        """Arranca el lector (con un Chromium real sin ventana) contra el BDV falso y corre el escenario."""
        async def principal():
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                gestor = lb.Gestor(browser, max_cuentas)
                tarea = asyncio.create_task(gestor.correr())

                async def hasta(condicion, segundos=25, que=''):
                    fin = time.monotonic() + segundos
                    while time.monotonic() < fin:
                        if await asyncio.to_thread(condicion):
                            return
                        await asyncio.sleep(0.2)
                    raise AssertionError(f'No pasó a tiempo: {que}')
                try:
                    await escenario(hasta)
                finally:
                    tarea.cancel()
                    await gestor.cerrar_todo()
                    await browser.close()
        asyncio.run(principal())

    def estado(self, cid):
        return self.cuenta(cid)['lector_estado']

    def pagos(self, cid):
        return [r['referencia'] for r in q('SELECT referencia FROM pagos_banco WHERE cuenta_banco_id = ? ORDER BY referencia', (cid,))]

    # ---------- pruebas ----------
    def test_entra_lee_todas_las_paginas_y_guarda_solo_los_pagos(self):
        self.banco.movimientos[U1] = [mov('000111222333', '1.234,56 Bs.'), mov('444555', '-80,00 Bs.', tipo='DEBITO'),
                                      mov('777888999', '3,21 Bs.', desc='ABONO INTERESES'), mov('5550001112', '75,00 Bs.')]
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: len(self.pagos(cid)) == 2, que='guardar los 2 pagos (en 2 páginas)')
            self.assertEqual(self.estado(cid), 'conectado')
            self.banco.movimientos[U1].append(mov('9998887776', '10,00 Bs.', fecha='11-10-2026 10:40'))
            await hasta(lambda: len(self.pagos(cid)) == 3, que='leer el pago nuevo en la vuelta siguiente')
            await hasta(lambda: self.banco.lecturas[U1] >= 4, que='seguir leyendo')
        self.correr(escenario)
        self.assertEqual(self.banco.entradas[U1], 1)   # entra una vez y se queda dentro
        fila = self.cuenta(cid)
        self.assertTrue(fila['lector_ultima_lectura'] and fila['lector_ultimo_pago'])
        pago = q("SELECT * FROM pagos_banco WHERE referencia = '000111222333'")[0]
        self.assertEqual((pago['fecha'], pago['hora'], float(pago['monto']), pago['usuario_id']), ('2026-10-11', '09:15', 1234.56, self.uid))
        self.assertTrue(pago['tipo'].startswith('BDV '))

    def test_sesion_vencida_vuelve_a_entrar(self):
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.lecturas[U1] >= 1, que='entrar')
            self.banco.expirar(U1)
            await hasta(lambda: self.banco.entradas[U1] == 2 and self.banco.dentro(U1), que='volver a entrar')
        self.correr(escenario)
        self.assertEqual(self.cuenta(cid)['lector_fallos_login'], 0)

    def test_banco_que_no_responde_no_cuenta_como_rechazo(self):
        self.banco.caido = True
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'banco_no_responde', que='marcar banco_no_responde')
            self.assertEqual(self.banco.intentos[U1], 0)
            self.banco.caido = False
            await hasta(lambda: self.estado(cid) == 'conectado', que='entrar cuando el banco vuelve')
        self.correr(escenario)
        self.assertEqual(self.cuenta(cid)['lector_fallos_login'], 0)

    def test_dos_rechazos_lo_detienen_y_no_reintenta_solo(self):
        self.banco.rechazar.add(U1)
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'rechazado', que='primer rechazo')
            await hasta(lambda: self.estado(cid) == 'detenido', que='segundo rechazo → detenido')
            self.assertEqual(self.banco.intentos[U1], 2)
            await asyncio.sleep(3)
            self.assertEqual(self.banco.intentos[U1], 2)   # no vuelve a intentar solo
            self.assertEqual(self.cuenta(cid)['lector_mensaje'], bdv.MSG_DETENIDO)
            self.banco.rechazar.clear()
            self.accion(bdv.reanudar, self.uid, cid)   # «Reanudar»
            await hasta(lambda: self.estado(cid) == 'conectado', que='entrar tras reanudar')
        self.correr(escenario)
        self.assertEqual(self.cuenta(cid)['lector_fallos_login'], 0)

    def test_interruptor_cierra_la_sesion_con_salir_y_vuelve_a_entrar(self):
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='entrar')
            inicio = time.monotonic()
            self.accion(bdv.cambiar_lectura, self.uid, cid, False)
            await hasta(lambda: self.estado(cid) == 'pausado', segundos=10, que='pausar')
            self.assertLess(time.monotonic() - inicio, 6)   # se entera en segundos
            self.assertEqual(self.banco.salidas[U1], 1)     # tocó «Salir»
            self.assertFalse(self.banco.dentro(U1))
            self.assertEqual(self.cuenta(cid)['lector_mensaje'], bdv.MSG_PAUSADO)
            lecturas = self.banco.lecturas[U1]
            await asyncio.sleep(2)
            self.assertEqual(self.banco.lecturas[U1], lecturas)   # en pausa no lee ni se reanuda solo
            self.accion(bdv.cambiar_lectura, self.uid, cid, True)
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='volver a entrar')
        self.correr(escenario)
        self.assertEqual(self.banco.entradas[U1], 2)
        self.assertEqual(self.cuenta(cid)['lector_fallos_login'], 0)   # la pausa no cuenta como fallo

    def test_salir_por_el_menu_lateral_con_confirmacion(self):
        self.banco.estilo = 'lateral'
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='entrar')
            self.accion(bdv.cambiar_lectura, self.uid, cid, False)
            await hasta(lambda: self.estado(cid) == 'pausado', segundos=15, que='pausar por el menú lateral')
        self.correr(escenario)
        self.assertEqual(self.banco.salidas[U1], 1)
        self.assertEqual(self.cuenta(cid)['lector_mensaje'], bdv.MSG_PAUSADO)

    def test_borrar_la_conexion_cierra_la_sesion(self):
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='entrar')
            self.accion(bdv.borrar_conexion, self.uid, cid)
            await hasta(lambda: not self.banco.dentro(U1), que='salir del banco')
            await asyncio.sleep(1)
        self.correr(escenario)
        self.assertEqual(self.banco.salidas[U1], 1)
        self.assertEqual(self.estado(cid), 'sin_conexion')

    def test_tope_de_cuentas_a_la_vez(self):
        c1, c2 = self.conectar(U1, C1), self.conectar(U2, C2)

        async def escenario(hasta):
            await hasta(lambda: {self.estado(c1), self.estado(c2)} == {'conectado', 'en_espera'}, que='una dentro y otra en espera')
            dentro, espera = (c1, c2) if self.estado(c1) == 'conectado' else (c2, c1)
            self.accion(bdv.cambiar_lectura, self.uid, dentro, False)   # se libera un lugar
            await hasta(lambda: self.estado(espera) == 'conectado', que='la que esperaba entra')
        self.correr(escenario, max_cuentas=1)

    def test_nada_secreto_en_registros_ni_mensajes(self):
        self.banco.rechazar.add(U1)
        cid = self.conectar()
        registros = []

        class Captura(logging.Handler):
            def emit(self, record):
                registros.append(record.getMessage())
        manejador = Captura()
        logging.getLogger().addHandler(manejador)
        logging.getLogger().setLevel(logging.INFO)
        try:
            async def escenario(hasta):
                await hasta(lambda: self.estado(cid) == 'detenido', que='detenido')
            self.correr(escenario)
        finally:
            logging.getLogger().removeHandler(manejador)
        texto = ' '.join(registros) + ' ' + json.dumps({k: str(v) for k, v in dict(self.cuenta(cid)).items()
                                                        if k.startswith('lector_')})
        self.assertTrue(registros)
        for secreto in (U1, C1, '0102' + '0' * 12):
            self.assertNotIn(secreto, texto)


if __name__ == '__main__':
    unittest.main()

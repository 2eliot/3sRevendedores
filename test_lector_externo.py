"""
Lector externo del BDV (lector_externo/): llaves, conversación con la web y lectura del banco.
Nunca usa el BDV real ni la cuenta del dueño: un BDV falso (mismos selectores que el real) y la web de verdad, en un
hilo, sobre una base SQLite temporal. El lector habla con ella por HTTP como lo hará desde su servidor.

    python -m unittest test_lector_externo -v
"""
import asyncio
import base64
import json
import logging
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nacl.public import PrivateKey

AQUI = Path(__file__).resolve().parent
LECTOR = AQUI / 'lector_externo'
sys.path.insert(0, str(LECTOR))

_tmp = tempfile.mkdtemp()
os.environ['DATABASE_PATH'] = os.path.join(_tmp, 'lector_externo.db')
os.environ.pop('DATABASE_URL', None)
os.environ['SECRET_KEY'] = 'test-' + 'x' * 30
os.environ.pop('BDV_CLAVE_PUBLICA', None)   # la llave sale del lector aprobado en el panel

import generar_llaves  # noqa: E402  (del lector externo: no importa nada de la web)
import lector_bdv as lb  # noqa: E402

import app as webapp  # noqa: E402  (la web; crea todas las tablas)
import antiduplic as ad  # noqa: E402
import bdv  # noqa: E402
import bdv_lector_api  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

U1, C1 = 'usuario-secreto-1', 'clave-secreta-1'
U2, C2 = 'usuario-secreto-2', 'clave-secreta-2'
CUENTA = {U1: '2411', U2: '7788'}


def numero(usuario):
    return '0102' + '0' * 12 + CUENTA[usuario]


def q(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall() if (sql.lstrip().upper().startswith('SELECT') or 'RETURNING' in sql.upper()) else None
        conn.commit()
        return rows
    finally:
        conn.close()


def config(clave, valor):
    conn = get_db_connection()
    try:
        ad._config_set(conn, clave, valor)
        conn.commit()
    finally:
        conn.close()


def info_lector():
    conn = get_db_connection()
    try:
        return bdv_lector_api.info_lector(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# BDV falso: imita los selectores del real (Angular Material): entrada en dos pasos, table.table-saldo-cuenta,
# mat-dialog-container con mat-row y su paginador, «Regresar» (mat-dialog-close) y «Salir» (arriba o menú lateral)
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
                    host = (self.headers.get('Host') or '').split(':')[0]
                    if banco.redirigir and host != 'localhost':   # manda a otro sitio (otro dominio)
                        self.send_response(302)
                        self.send_header('Location', f'http://localhost:{banco.servidor.server_address[1]}/')
                        self.send_header('Content-Length', '0')
                        self.end_headers()
                        return
                    if host == 'localhost':
                        with banco.lock:
                            banco.otro_sitio += 1
                    # caído: la página no termina de cargar y el lector no encuentra la casilla
                    cuerpo = b'<html><body>Cargando...</body></html>' if banco.caido else PAGINA.encode()
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
            self.redirigir, self.otro_sitio = False, 0
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


# ---------------------------------------------------------------------------
# Sin navegador
# ---------------------------------------------------------------------------

class LlavesTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.ruta = self.dir / 'privada' / 'llave_privada'

    def correr(self, *args):
        env = dict(os.environ, BDV_CLAVE_PRIVADA=str(self.ruta), PYTHONIOENCODING='utf-8')
        r = subprocess.run([sys.executable, str(LECTOR / 'generar_llaves.py'), *args], capture_output=True, text=True,
                           env=env, encoding='utf-8')
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_crea_el_par_sin_mostrar_la_privada(self):
        salida = self.correr()
        privada = self.ruta.read_text().strip()
        self.assertNotIn(privada, salida)               # la privada nunca sale en pantalla
        self.assertIn('Llave pública', salida)
        _, publica, creada = generar_llaves.cargar_o_crear(self.ruta, crear=False)
        self.assertFalse(creada)
        self.assertIn(publica, salida)
        self.assertIn(generar_llaves.huella(publica), salida)
        if os.name == 'posix':
            self.assertEqual(self.ruta.stat().st_mode & 0o777, 0o600)

    def test_no_pisa_una_llave_que_ya_existe(self):
        self.correr()
        antes = self.ruta.read_text()
        salida = self.correr()
        self.assertIn('Ya hay una llave', salida)
        self.assertEqual(self.ruta.read_text(), antes)
        self.correr('--forzar')
        self.assertNotEqual(self.ruta.read_text(), antes)

    def test_la_web_cierra_y_el_lector_abre(self):
        from nacl.public import SealedBox
        privada, publica, _ = generar_llaves.cargar_o_crear(self.ruta)
        os.environ['BDV_CLAVE_PUBLICA'] = publica
        try:
            cifrado = bdv.cifrar('Clave-Secreta-1')   # lo que hace la web al guardar la conexión
            self.assertEqual(bdv.huella(publica), generar_llaves.huella(publica))   # mismo código en panel y lector
        finally:
            os.environ.pop('BDV_CLAVE_PUBLICA', None)
        self.assertEqual(SealedBox(privada).decrypt(base64.b64decode(cifrado)).decode(), 'Clave-Secreta-1')


class PiezasTest(unittest.TestCase):
    def test_reglas_de_la_tabla(self):
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

    def test_cola_no_repite_y_sobrevive_a_un_reinicio(self):
        ruta = Path(tempfile.mkdtemp()) / 'cola.json'
        p1, p2 = lb.a_pago(fila('111', '10,00')), lb.a_pago(fila('222', '20,00'))
        cola = lb.Cola(ruta)
        self.assertEqual(cola.agregar(7, [p1, p2]), 2)
        self.assertEqual(cola.agregar(7, [p1]), 0)              # ya está en cola
        cola = lb.Cola(ruta)                                    # reinicio: la cola sigue ahí
        self.assertEqual(cola.por_enviar(), {7: [p1, p2]})
        cola.enviados_ok(7, [p1])
        self.assertEqual(cola.agregar(7, [p1, p2]), 0)          # uno enviado y otro en cola: nada nuevo
        cola = lb.Cola(ruta)
        self.assertEqual(cola.por_enviar(), {7: [p2]})
        cola.descartar(7)
        self.assertEqual(lb.Cola(ruta).por_enviar(), {})
        if os.name == 'posix':
            self.assertEqual(ruta.stat().st_mode & 0o777, 0o600)

    def test_solo_https_y_un_solo_lector(self):
        for url in ('https://mi-web.com', 'http://127.0.0.1:5000', 'http://localhost:5000'):
            self.assertTrue(lb.url_web_valida(url), url)
        for url in ('', 'http://mi-web.com', 'ftp://mi-web.com', 'https://', 'mi-web.com'):
            self.assertFalse(lb.url_web_valida(url), url)   # la clave temporal nunca viaja sin cifrar
        ruta = Path(tempfile.mkdtemp()) / 'lector.lock'
        uno = lb.tomar_candado(ruta)
        self.assertIsNotNone(uno)
        self.assertIsNone(lb.tomar_candado(ruta))               # un segundo lector no arranca
        uno.close()
        dos = lb.tomar_candado(ruta)
        self.assertIsNotNone(dos)
        dos.close()

    def test_no_sigue_redirecciones_de_la_web(self):
        visitas = []

        class Redirige(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                visitas.append(self.headers.get('Authorization'))
                self.send_response(301)
                self.send_header('Location', 'http://127.0.0.1:9/robar')
                self.send_header('Content-Length', '0')
                self.end_headers()
            do_GET = do_POST
        srv = ThreadingHTTPServer(('127.0.0.1', 0), Redirige)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            privada = PrivateKey.generate()
            web = lb.Web(f'http://127.0.0.1:{srv.server_address[1]}', privada, base64.b64encode(bytes(privada.public_key)).decode())
            with self.assertRaises(lb.WebCaida) as e:
                web.registrar()
            self.assertIn('BDV_WEB_URL', str(e.exception))
            web.clave, web.vence = 'clave-temporal', time.time() + 600
            with self.assertRaises(lb.WebCaida):
                web.cuentas()
            self.assertEqual(len(visitas), 2)   # no siguió a la otra dirección
        finally:
            srv.shutdown()

    @unittest.skipUnless(os.name == 'posix', 'permisos de Linux')
    def test_cierra_los_permisos_de_la_llave(self):
        ruta = Path(tempfile.mkdtemp()) / 'llave_privada'
        generar_llaves.cargar_o_crear(ruta)
        os.chmod(ruta, 0o644)
        lb.cuidar_llave(ruta)
        self.assertEqual(ruta.stat().st_mode & 0o777, 0o600)


# ---------------------------------------------------------------------------
# Con Chromium de verdad contra el BDV falso y la web de verdad
# ---------------------------------------------------------------------------

class LectorExternoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from werkzeug.serving import make_server
        cls.banco = BancoFalso()
        lb.URL = cls.banco.url
        lb.CADA_SEG, lb.REVISAR_SEG, lb.ESCALON_SEG, lb.JUNTAR_SEG = 0.5, 0.3, 0.2, 0.1
        lb.ESPERA_RECHAZO_SEG, lb.NO_RESPONDE_BASE_SEG, lb.NO_RESPONDE_MAX_SEG = 1.5, 1, 2
        lb.TIMEOUT_MS, lb.MEDIR_CADA_SEG, lb.APROBACION_SEG = 4000, 0, 0.3
        cls.limites = dict(bdv_lector_api.LIMITES)
        bdv_lector_api.LIMITES.update({k: 10 ** 6 for k in bdv_lector_api.LIMITES})   # aquí todo va mucho más rápido
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        webapp.app.config['TESTING'] = True
        cls.servidor = make_server('127.0.0.1', 0, webapp.app, threaded=True)
        threading.Thread(target=cls.servidor.serve_forever, daemon=True).start()
        cls.web_url = f'http://127.0.0.1:{cls.servidor.server_port}'
        ad.init_tablas()
        cls.uid = q("INSERT INTO usuarios (nombre, apellido, telefono, correo, contraseña, saldo) "
                    "VALUES ('Lea', 'L', '0', 'lector-externo@test.local', 'x', 0) RETURNING id")[0][0]
        cls.privada, cls.publica, _ = generar_llaves.cargar_o_crear(Path(tempfile.mkdtemp()) / 'llave_privada')
        cls.codigo = generar_llaves.huella(cls.publica)

    @classmethod
    def tearDownClass(cls):
        cls.servidor.shutdown()
        bdv_lector_api.LIMITES.update(cls.limites)

    def setUp(self):
        self.banco.reiniciar()
        for t in ('usos_referencia', 'pagos_banco', 'cuentas_banco'):
            q(f'DELETE FROM {t}')
        q("DELETE FROM configuracion_redeemer WHERE clave LIKE 'bdv_lector_%'")
        bdv_lector_api._golpes.clear()
        self.dir = Path(tempfile.mkdtemp())
        self.gestor = None

    def tearDown(self):
        q("DELETE FROM configuracion_redeemer WHERE clave LIKE 'bdv_lector_%'")

    # ---------- ayudas ----------
    def aprobado(self, publica=None):
        """Como si el admin ya lo hubiera aprobado en el panel."""
        config(bdv.CONFIG_PUBLICA, publica or self.publica)

    def admin(self, accion, codigo=None):
        c = webapp.app.test_client()
        with c.session_transaction() as s:
            s.update(usuario='admin@test.local', user_db_id=self.uid, is_admin=True)
        return c.post(f'/admin/antiduplic/lector/{accion}', json={'codigo': codigo} if codigo else {})

    def conectar(self, usuario=U1, clave=C1):
        conn = get_db_connection()
        try:
            status, datos = bdv.guardar_conexion(conn, self.uid, usuario, clave, numero(usuario), '', True)
        finally:
            conn.close()
        self.assertEqual(status, 200, datos)
        return datos['cuenta_id']

    def accion(self, fn, *args):
        conn = get_db_connection()
        try:
            return fn(conn, *args)
        finally:
            conn.close()

    def cuenta(self, cid):
        return q('SELECT * FROM cuentas_banco WHERE id = ?', (cid,))[0]

    def estado(self, cid):
        return self.cuenta(cid)['lector_estado']

    def pagos(self, cid):
        return [r['referencia'] for r in q('SELECT referencia FROM pagos_banco WHERE cuenta_banco_id = ? ORDER BY referencia', (cid,))]

    def correr(self, escenario, max_cuentas=30):
        """Arranca el lector (Chromium real sin ventana) contra el BDV falso y la web, y corre el escenario."""
        async def principal():
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                async def abrir_navegador():
                    return await p.chromium.launch(headless=True)
                web = lb.Web(self.web_url, self.privada, self.publica)
                self.gestor = gestor = lb.Gestor(web, lb.Cola(self.dir / 'cola.json'), self.privada, abrir_navegador, max_cuentas)
                tarea = asyncio.create_task(gestor.correr())

                async def hasta(condicion, segundos=25, que=''):
                    fin = time.monotonic() + segundos
                    while time.monotonic() < fin:
                        if tarea.done():
                            raise AssertionError(f'El lector terminó: {tarea.exception()!r}')
                        if await asyncio.to_thread(condicion):
                            return
                        await asyncio.sleep(0.2)
                    raise AssertionError(f'No pasó a tiempo: {que}')
                try:
                    await escenario(hasta)
                finally:
                    tarea.cancel()   # como systemctl stop: se detiene y cierra las sesiones del banco
                    try:
                        await tarea
                    except BaseException:
                        pass
                    await gestor.cerrar_todo()
                    if gestor.browser is not None:
                        await gestor.browser.close()
        asyncio.run(principal())

    # ---------- pruebas ----------
    def test_primer_arranque_aprobacion_y_lectura(self):
        self.banco.movimientos[U1] = [mov('000111222333', '1.234,56 Bs.'), mov('444555', '-80,00 Bs.', tipo='DEBITO'),
                                      mov('777888999', '3,21 Bs.', desc='ABONO INTERESES'), mov('5550001112', '75,00 Bs.')]
        enviados = []

        async def escenario(hasta):
            # 1. se presenta solo y queda pendiente con su código; sin aprobar, nadie puede guardar datos del BDV
            await hasta(lambda: (info_lector()['pendiente'] or {}).get('codigo') == self.codigo, que='quedar pendiente')
            self.assertFalse(bdv.cifrado_disponible())
            self.assertEqual(self.admin('aprobar', 'AAAA-BBBB').status_code, 409)   # otro código: nada
            self.assertEqual(self.admin('aprobar', self.codigo).status_code, 200)    # el clic del admin
            await hasta(lambda: info_lector()['vivo'], que='empezar a trabajar tras la aprobación')
            # 2. el usuario guarda sus datos (cerrados con la llave del lector) y el lector lee
            original = self.gestor.web.pagos
            self.gestor.web.pagos = lambda cid, lote: (enviados.extend(x['referencia'] for x in lote), original(cid, lote))[1]
            cid = self.conectar()
            await hasta(lambda: len(self.pagos(cid)) == 2, que='guardar los 2 pagos (en 2 páginas)')
            self.assertEqual(self.estado(cid), 'conectado')
            self.banco.movimientos[U1].append(mov('9998887776', '10,00 Bs.', fecha='11-10-2026 10:40'))
            await hasta(lambda: len(self.pagos(cid)) == 3, que='leer el pago nuevo en la vuelta siguiente')
            lecturas = self.banco.lecturas[U1]
            await hasta(lambda: self.banco.lecturas[U1] >= lecturas + 3, que='seguir leyendo')
            self.assertEqual(sorted(enviados), ['000111222333', '5550001112', '9998887776'])   # cada pago una sola vez
            self.assertTrue(self.banco.dentro(U1))
            estado = info_lector()['estado']
            self.assertEqual((estado['cuentas_leyendo'], estado['max_cuentas'], estado['version']), (1, 30, lb.VERSION))
        self.correr(escenario)
        self.assertEqual(self.banco.entradas[U1], 1)    # entra una vez y se queda dentro
        self.assertFalse(self.banco.dentro(U1))         # al detenerse pulsó «Salir»
        self.assertEqual(self.banco.salidas[U1], 1)
        pago = q("SELECT * FROM pagos_banco WHERE referencia = '000111222333'")[0]
        self.assertEqual((pago['fecha'], pago['hora'], float(pago['monto']), pago['usuario_id']), ('2026-10-11', '09:15', 1234.56, self.uid))
        self.assertTrue(pago['tipo'].startswith('BDV '))
        fila_cuenta = q("SELECT * FROM cuentas_banco WHERE banco = 'BDV'")[0]
        self.assertTrue(fila_cuenta['lector_ultima_lectura'] and fila_cuenta['lector_ultimo_pago'])

    def test_si_la_web_cae_sigue_leyendo_y_manda_la_cola_al_volver(self):
        self.aprobado()
        self.banco.movimientos[U1] = [mov('1110001110', '10,00 Bs.')]
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: len(self.pagos(cid)) == 1, que='primer pago')
            web = self.gestor.web
            base, web.base = web.base, 'http://127.0.0.1:9'   # la web «se cae»
            await hasta(lambda: not self.gestor.web_ok, que='notar la caída')
            self.banco.movimientos[U1].append(mov('2220002220', '20,00 Bs.', fecha='11-10-2026 10:00'))
            await hasta(lambda: lb.Cola(self.dir / 'cola.json').por_enviar(), que='guardar el pago nuevo en la cola (en disco)')
            self.assertEqual(len(self.pagos(cid)), 1)
            self.assertTrue(self.banco.dentro(U1))          # no cierra la sesión del banco por una caída de la web
            self.assertEqual(self.banco.salidas[U1], 0)
            web.base = base                                  # la web vuelve
            await hasta(lambda: len(self.pagos(cid)) == 2, que='mandar lo que quedó en cola')
            await hasta(lambda: self.gestor.web_ok, que='volver a la normalidad')
        self.correr(escenario)
        self.assertEqual(self.banco.entradas[U1], 1)

    def test_pausa_cierra_con_salir_y_vuelve_a_entrar(self):
        self.aprobado()
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='entrar')
            inicio = time.monotonic()
            self.accion(bdv.cambiar_lectura, self.uid, cid, False)
            await hasta(lambda: self.estado(cid) == 'pausado', segundos=10, que='pausar')
            self.assertLess(time.monotonic() - inicio, 6)   # se entera en segundos
            self.assertEqual(self.banco.salidas[U1], 1)     # tocó «Salir»
            self.assertFalse(self.banco.dentro(U1))
            self.assertEqual(self.cuenta(cid)['lector_mensaje'], lb.MSG_PAUSADO)
            lecturas = self.banco.lecturas[U1]
            await asyncio.sleep(2)
            self.assertEqual(self.banco.lecturas[U1], lecturas)   # en pausa no lee ni se reanuda solo
            self.accion(bdv.cambiar_lectura, self.uid, cid, True)
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='volver a entrar')
        self.correr(escenario)
        self.assertEqual(self.banco.entradas[U1], 2)
        self.assertEqual(self.cuenta(cid)['lector_fallos_login'], 0)   # la pausa no cuenta como fallo

    def test_salir_por_el_menu_lateral_con_confirmacion(self):
        self.aprobado()
        self.banco.estilo = 'lateral'
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='entrar')
            self.accion(bdv.cambiar_lectura, self.uid, cid, False)
            await hasta(lambda: self.estado(cid) == 'pausado', segundos=15, que='pausar por el menú lateral')
        self.correr(escenario)
        self.assertEqual(self.banco.salidas[U1], 1)
        self.assertEqual(self.cuenta(cid)['lector_mensaje'], lb.MSG_PAUSADO)

    def test_pausada_con_el_lector_apagado_se_confirma_al_arrancar(self):
        self.aprobado()
        cid = self.conectar()
        self.accion(bdv.cambiar_lectura, self.uid, cid, False)

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'pausado', que='confirmar la pausa')
        self.correr(escenario)
        self.assertEqual(self.cuenta(cid)['lector_mensaje'], lb.MSG_PAUSADO_SIN_SESION)
        self.assertEqual(self.banco.intentos[U1], 0)   # no entra al banco

    def test_dos_rechazos_lo_detienen_y_no_reintenta_solo(self):
        self.aprobado()
        self.banco.rechazar.add(U1)
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'rechazado', que='primer rechazo')
            await hasta(lambda: self.estado(cid) == 'detenido', que='segundo rechazo → detenido')
            self.assertEqual(self.banco.intentos[U1], 2)
            await asyncio.sleep(3)
            self.assertEqual(self.banco.intentos[U1], 2)   # no vuelve a intentar solo
            self.assertEqual(self.cuenta(cid)['lector_mensaje'], lb.MSG_DETENIDO)
            self.assertEqual(self.cuenta(cid)['lector_fallos_login'], 2)
            self.banco.rechazar.clear()
            self.accion(bdv.reanudar, self.uid, cid)   # «Reanudar»
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='entrar tras reanudar')
        self.correr(escenario)
        self.assertEqual(self.cuenta(cid)['lector_fallos_login'], 0)

    def test_sesion_vencida_vuelve_a_entrar(self):
        self.aprobado()
        cid = self.conectar()

        async def escenario(hasta):
            # una cuenta sin movimientos se lee sin errores: «0 pagos»
            await hasta(lambda: 'última vuelta: 0 pagos' in (self.cuenta(cid)['lector_mensaje'] or ''), que='leer sin movimientos')
            self.banco.expirar(U1)
            await hasta(lambda: self.banco.entradas[U1] == 2 and self.banco.dentro(U1), que='volver a entrar')
        self.correr(escenario)
        self.assertEqual(self.cuenta(cid)['lector_fallos_login'], 0)

    def test_banco_que_no_responde_no_cuenta_como_rechazo(self):
        self.aprobado()
        self.banco.caido = True
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'banco_no_responde', que='marcar banco_no_responde')
            self.assertEqual(self.banco.intentos[U1], 0)
            self.banco.caido = False
            await hasta(lambda: self.estado(cid) == 'conectado', que='entrar cuando el banco vuelve')
        self.correr(escenario)
        self.assertEqual(self.cuenta(cid)['lector_fallos_login'], 0)

    def test_tope_de_cuentas_a_la_vez(self):
        self.aprobado()
        c1, c2 = self.conectar(U1, C1), self.conectar(U2, C2)

        async def escenario(hasta):
            await hasta(lambda: {self.estado(c1), self.estado(c2)} == {'conectado', 'en_espera'}, que='una dentro y otra en espera')
            dentro, espera = (c1, c2) if self.estado(c1) == 'conectado' else (c2, c1)
            self.accion(bdv.cambiar_lectura, self.uid, dentro, False)   # se libera un lugar
            await hasta(lambda: self.estado(espera) == 'conectado', que='la que esperaba entra')
            self.assertEqual(self.estado(dentro), 'pausado')
        self.correr(escenario, max_cuentas=1)

    def test_borrar_la_conexion_cierra_la_sesion(self):
        self.aprobado()
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='entrar')
            self.accion(bdv.borrar_conexion, self.uid, cid)
            await hasta(lambda: not self.banco.dentro(U1), que='salir del banco')
            await asyncio.sleep(1)
        self.correr(escenario)
        self.assertEqual(self.banco.salidas[U1], 1)
        self.assertEqual(self.estado(cid), 'sin_conexion')   # el «pausado» del lector no pisa el borrado

    def test_desconectar_el_lector_cierra_las_sesiones_y_espera_aprobacion(self):
        self.aprobado()
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='entrar')
            self.assertEqual(self.admin('desconectar').status_code, 200)
            await hasta(lambda: not self.banco.dentro(U1), que='cerrar la sesión del banco')
            await hasta(lambda: (info_lector()['pendiente'] or {}).get('codigo') == self.codigo, que='pedir aprobación otra vez')
            await asyncio.sleep(1)
            self.assertEqual(self.banco.entradas[U1], 1)   # sin aprobación no vuelve a entrar
            self.assertEqual(self.admin('aprobar', self.codigo).status_code, 200)
            await hasta(lambda: self.banco.entradas[U1] == 2 and self.banco.dentro(U1), que='volver a entrar tras aprobar')
        self.correr(escenario)
        self.assertEqual(self.banco.salidas[U1], 2)

    def test_si_el_navegador_se_cae_lo_vuelve_a_abrir(self):
        self.aprobado()
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'conectado' and self.banco.dentro(U1), que='entrar')
            await self.gestor.browser.close()   # Chromium «muere»
            await hasta(lambda: self.banco.entradas[U1] == 2, que='abrir otro navegador y volver a entrar')
            await hasta(lambda: self.estado(cid) == 'conectado', que='seguir leyendo')
        self.correr(escenario)

    def test_no_escribe_los_datos_fuera_del_dominio_del_banco(self):
        self.aprobado()
        self.banco.redirigir = True   # la página del banco manda a otro dominio con un formulario igual
        cid = self.conectar()

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'banco_no_responde', que='no entrar')
        self.correr(escenario)
        self.assertGreater(self.banco.otro_sitio, 0)      # sí llegó a la otra página…
        self.assertEqual(self.banco.intentos[U1], 0)      # …pero no escribió nada en ella

    def test_datos_cerrados_con_la_llave_de_otro_lector(self):
        otra = base64.b64encode(bytes(PrivateKey.generate().public_key)).decode()
        self.aprobado(otra)
        cid = self.conectar()     # guardados cuando estaba aprobado otro lector
        self.aprobado()           # ahora está aprobado este

        async def escenario(hasta):
            await hasta(lambda: self.estado(cid) == 'detenido', que='detener la cuenta')
        self.correr(escenario)
        self.assertEqual(self.cuenta(cid)['lector_mensaje'], lb.MSG_SIN_DATOS)
        self.assertEqual(self.banco.intentos[U1], 0)   # no prueba nada en el banco

    def test_nada_secreto_en_registros_ni_mensajes(self):
        self.aprobado()
        self.banco.rechazar.add(U1)
        cid = self.conectar()
        c2 = self.conectar(U2, C2)
        registros = []

        class Captura(logging.Handler):
            def emit(self, record):
                registros.append(record.getMessage())
        manejador = Captura()
        raiz = logging.getLogger()
        nivel = raiz.level
        raiz.addHandler(manejador)
        raiz.setLevel(logging.INFO)
        claves = []
        try:
            async def escenario(hasta):
                await hasta(lambda: self.estado(cid) == 'detenido', que='detenido')
                await hasta(lambda: self.estado(c2) == 'conectado', que='la otra entra')
                self.accion(bdv.cambiar_lectura, self.uid, c2, False)
                await hasta(lambda: self.estado(c2) == 'pausado', que='pausada')
                claves.append(self.gestor.web.clave)
            self.correr(escenario)
        finally:
            raiz.removeHandler(manejador)
            raiz.setLevel(nivel)
        cifrados = [v for f in q("SELECT * FROM cuentas_banco WHERE banco = 'BDV'")
                    for v in (f['bdv_usuario_cifrado'], f['bdv_clave_cifrada'], f['numero_cuenta_cifrado']) if v]
        filas = [{k: str(v) for k, v in dict(self.cuenta(x)).items() if k.startswith('lector_')} for x in (cid, c2)]
        texto = ' '.join(registros) + ' ' + json.dumps(filas, ensure_ascii=False)
        self.assertTrue(registros)
        for secreto in (U1, C1, U2, C2, numero(U1), numero(U2), '0102' + '0' * 12, *claves, *cifrados):
            self.assertNotIn(secreto, texto)


if __name__ == '__main__':
    unittest.main()

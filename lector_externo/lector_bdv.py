"""
Lector externo del BDV (Banco de Venezuela en línea, personas) para Referencias de 3sRevendedores.

Corre en SU PROPIO servidor (no en el de la web) y NO importa nada de la web: solo habla con ella por HTTPS.

    python lector_bdv.py

Al arrancar:
  1. Carga su llave privada (o la crea la primera vez, con permisos 600) y muestra su código, p. ej. AB12-CD34.
  2. Se presenta a la web. La primera vez queda «pendiente»: el admin lo aprueba en Referencias (todos) comparando
     ese código. Después ya no hace falta nada: si se reinicia, sigue solo.
  3. Pide una clave temporal (un reto que solo él puede abrir con su llave privada) y empieza a trabajar.

Cada ~5 s pregunta a la web qué cuentas leer (ETag: casi siempre «nada cambió»). Por cada cuenta, igual que el
prototipo probado contra el banco real: un contexto propio de Chromium sin ventana; entra una vez y se queda dentro;
cada vuelta abre los movimientos, lee todas las páginas, guarda los pagos recibidos nuevos y espera 25 s con el
diálogo cerrado; si la sesión se cerró vuelve a entrar; si una vuelta falla 3 veces, recarga.

Protección de la cuenta del usuario: dos rechazos del banco → «detenido» (no reintenta hasta que el usuario guarde sus
datos o pulse «Reanudar»); banco que no responde → esperas de 1 a 10 min sin contarlo como rechazo.
Interruptor «Leer movimientos» apagado (o conexión borrada) → «Salir» en el banco y cierra el contexto.

Siempre en marcha:
  - si la web no responde, sigue leyendo y guarda los pagos en una cola local (archivo) para mandarlos cuando vuelva;
    no cierra sesiones del banco por una caída de la web;
  - si Chromium muere, lo vuelve a abrir; si el programa se queda trabado más de 5 min, se cierra solo y systemd lo
    vuelve a levantar (lector-bdv.service); un candado impide que haya dos lectores a la vez;
  - al detenerlo (systemctl stop/restart) pulsa «Salir» en el banco en todas las cuentas antes de terminar.

Variables (en /etc/lector-bdv.env, ver LEEME.md):
  BDV_WEB_URL         dirección de la web, p. ej. https://mi-web.com (obligatoria; https, salvo localhost en pruebas)
  BDV_MAX_CUENTAS     cuentas leyendo a la vez (por defecto 30); las demás quedan «en espera»
  BDV_CLAVE_PRIVADA   ruta de la llave privada (por defecto ~/.lector-bdv/llave_privada)
  BDV_DATOS           carpeta de la cola de pagos y el candado (por defecto ~/.lector-bdv)

Nunca imprime usuario, contraseña ni número completo de cuenta: solo el id de la cuenta y sus últimos 4 dígitos.
"""
import asyncio
import base64
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import generar_llaves  # noqa: E402

VERSION = '1.0'
USER_AGENT = f'3s-lector-bdv/{VERSION}'   # el de Python por defecto lo bloquea Cloudflare (error 1010)
logger = logging.getLogger('lector_bdv')

URL = os.environ.get('BDV_URL', '').strip() or 'https://bdvenlinea.banvenez.com/'

# Tiempos (las pruebas los acortan)
CADA_SEG = 25            # espera entre vueltas, con la ventana de movimientos cerrada
REVISAR_SEG = 5          # cada cuánto se pregunta a la web
JUNTAR_SEG = 1           # entre envíos a la web (junta avisos y respeta sus límites)
ESCALON_SEG = 4          # separación entre las entradas de cuentas nuevas
ESPERA_RECHAZO_SEG = 300  # antes del segundo intento cuando el banco rechaza la entrada
NO_RESPONDE_BASE_SEG = 60
NO_RESPONDE_MAX_SEG = 600
TIMEOUT_MS = 60000       # carga de páginas del banco
WEB_TIMEOUT_SEG = 15
APROBACION_SEG = 30      # cada cuánto se pregunta si ya está aprobado
MEDIR_CADA_SEG = 300     # memoria real, que se le cuenta a la web
VIGILANTE_SEG = 300      # trabado más que esto → se reinicia
MAX_RECHAZOS = 2

NO_SON_PAGOS = re.compile(r'INTERES|COMISION|REINTEGRO|AJUSTE', re.I)   # créditos que pone el propio banco
RECHAZO = re.compile(r'inv[aá]lid|incorrect|bloquead|suspendid|no coincide|intentos|no autorizad', re.I)

MSG_PAUSADO = 'En pausa — la sesión del banco está cerrada ✅. Ya puedes entrar al BDV desde tu teléfono o tu computadora'
MSG_PAUSADO_SIN_SESION = ('En pausa ✅. Ya puedes entrar al BDV desde tu teléfono o tu computadora '
                          '(si no te deja, espera unos minutos)')
MSG_PAUSA_SIN_CONFIRMAR = 'No se pudo confirmar el cierre de sesión en el banco. Si no te deja entrar, espera unos minutos'
MSG_DETENIDO = 'El banco rechazó tus datos: revísalos y guárdalos de nuevo'
MSG_CONECTANDO = 'Entrando al banco…'
MSG_EN_ESPERA = 'Hay muchas cuentas conectadas: entra en cuanto se libere un lugar'
MSG_RECHAZADO = 'El banco rechazó la entrada. Se reintenta una sola vez en 5 minutos'
MSG_SIN_CUENTA = 'No encontré en tu BDV la cuenta que termina en {}: revisa el número que guardaste'
MSG_SIN_DATOS = 'No se pudieron leer tus datos guardados: guárdalos de nuevo'


def ahora_local():
    """Hora de Venezuela (la de la web), aunque el servidor del lector esté en otra zona."""
    try:
        from zoneinfo import ZoneInfo
        zona = ZoneInfo('America/Caracas')
    except Exception:
        zona = timezone(timedelta(hours=-4))
    return datetime.now(zona).strftime('%Y-%m-%d %H:%M:%S')


# ---------------------------------------------------------------------------
# Lectura de la tabla (funciones puras)
# ---------------------------------------------------------------------------

def monto_numero(texto):
    """'-1.234,56 Bs.' → -1234.56 (None si no es un monto)."""
    t = re.sub(r'[^\d,.\-]', '', str(texto or '').replace('\xa0', ' '))
    try:
        return float(t.replace('.', '').replace(',', '.'))
    except ValueError:
        return None


def a_pago(fila):
    """Solo pagos recibidos (créditos que no son del banco), en el formato de la web. None si no corresponde."""
    monto = monto_numero(fila.get('monto'))
    tipo = str(fila.get('tipo') or '').lower()
    credito = 'cr' in tipo or (not tipo and monto is not None and monto > 0)
    if not credito or monto is None or monto <= 0 or NO_SON_PAGOS.search(str(fila.get('descripcion') or '')):
        return None
    m = re.match(r'(\d{2})-(\d{2})-(\d{4})\s+(\d{2}:\d{2})', str(fila.get('fecha') or ''))
    ref = re.sub(r'\D', '', str(fila.get('referencia') or ''))
    if not m or not ref:
        return None
    return {'referencia': ref, 'fecha': f'{m.group(3)}-{m.group(2)}-{m.group(1)}', 'hora': m.group(4),
            'monto': round(monto, 2), 'tipo': ('BDV ' + re.sub(r'\s+', ' ', str(fila.get('descripcion') or '')).strip())[:120]}


def clave_pago(p):
    return f"{p['referencia']}|{p['fecha']}|{p['hora']}|{float(p['monto']):.2f}"


# ---------------------------------------------------------------------------
# El banco (Playwright, asíncrono)
# ---------------------------------------------------------------------------

class BancoNoResponde(Exception):
    """La página no cargó (saturado, sin internet...): se espera y se reintenta. NO cuenta como rechazo."""


class LoginRechazado(Exception):
    """El banco dijo que no (datos inválidos, usuario bloqueado...): SÍ cuenta para detenerse."""


class CuentaNoEncontrada(Exception):
    pass


async def en_login(page):
    return await page.locator('input[formcontrolname="username"]:visible, input[formcontrolname="password"]:visible').count() > 0


async def en_sesion(page):
    return (await page.locator('table.table-saldo-cuenta').count() > 0
            or await page.locator('mat-row, .mat-row').count() > 0)


async def hay_rechazo(page):
    """¿Hay a la vista un aviso de rechazo del banco? (su texto no se guarda: podría traer datos del usuario)."""
    avisos = page.locator('mat-error:visible, simple-snack-bar:visible, mat-dialog-container:visible, .alert:visible')
    for i in range(await avisos.count()):
        try:
            if RECHAZO.search(await avisos.nth(i).inner_text()):
                return True
        except Exception:
            pass
    return False


async def entrar(page, usuario, clave):
    try:
        await page.goto(URL, wait_until='domcontentloaded', timeout=TIMEOUT_MS)
        await page.wait_for_selector('input[formcontrolname="username"], table.table-saldo-cuenta', timeout=TIMEOUT_MS)
    except Exception:
        raise BancoNoResponde('la página del banco no cargó')
    if await page.locator('table.table-saldo-cuenta').count():
        return
    campo = page.locator('input[formcontrolname="username"]')
    await campo.fill(usuario)
    await campo.locator('xpath=ancestor::form').locator('button[type="submit"]').click()
    campo_clave = page.locator('input[formcontrolname="password"]')
    try:
        await campo_clave.wait_for(state='visible', timeout=int(TIMEOUT_MS * 0.75))
    except Exception:
        if await hay_rechazo(page):
            raise LoginRechazado('el banco rechazó el usuario')
        raise BancoNoResponde('no apareció la casilla de la contraseña')
    await campo_clave.fill(clave)
    await campo_clave.locator('xpath=ancestor::form').locator('button[type="submit"]').click()
    fin = time.monotonic() + TIMEOUT_MS * 1.5 / 1000
    while time.monotonic() < fin:   # o entra (tabla de cuentas), o el banco dice que no, o no responde
        if await page.locator('table.table-saldo-cuenta').count():
            return
        if await hay_rechazo(page):
            raise LoginRechazado('el banco rechazó la contraseña')
        await page.wait_for_timeout(1000)
    raise BancoNoResponde('después de la contraseña la página no terminó de cargar')


async def cerrar_movimientos(page):
    """«Regresar» (por dentro se llama «Close dialog»: mat-dialog-close); si no se va, la tecla Escape."""
    if not await page.locator('mat-dialog-container').count():
        return
    boton = page.locator("mat-dialog-container button[mat-dialog-close], mat-dialog-container button:has-text('Regresar')")
    if await boton.count():
        await boton.first.click(timeout=10000)
    try:
        await page.wait_for_selector('mat-dialog-container', state='detached', timeout=10000)
    except Exception:
        await page.keyboard.press('Escape')
        await page.wait_for_selector('mat-dialog-container', state='detached', timeout=10000)
    await page.wait_for_timeout(500)


async def abrir_movimientos(page, ultimos4):
    await page.wait_for_selector('table.table-saldo-cuenta', timeout=30000)
    filas = page.locator('table.table-saldo-cuenta tbody tr')
    elegida = None
    for i in range(await filas.count()):
        numero = (await filas.nth(i).locator('td[headers="numero"]').inner_text()).strip()
        if not ultimos4 or numero.endswith(ultimos4):
            elegida = i
            break
    if elegida is None:
        raise CuentaNoEncontrada(ultimos4)
    await filas.nth(elegida).locator('td').nth(2).locator('mat-icon').click()
    await page.wait_for_selector('mat-dialog-container', timeout=TIMEOUT_MS)
    try:
        await page.wait_for_selector('mat-row .mat-column-referencia, .mat-row .mat-column-referencia',
                                     timeout=min(15000, TIMEOUT_MS))
    except Exception:
        if not await page.locator('mat-dialog-container').count():
            raise
        # La ventana abrió pero sin filas: cuenta sin movimientos (o tardó; se lee en la próxima vuelta)


async def leer_tabla(page):
    filas = []
    for _ in range(50):
        rows = page.locator('mat-row, .mat-row')
        for i in range(await rows.count()):
            r = rows.nth(i)
            fila = {}
            for clave, col in (('fecha', 'fecha'), ('referencia', 'referencia'), ('descripcion', 'descripcion'),
                               ('tipo', 'indicadorCargoAbono'), ('monto', 'importe')):
                c = r.locator(f'.mat-column-{col}')
                fila[clave] = (await c.inner_text()).strip() if await c.count() else ''
            filas.append(fila)
        siguiente = page.locator('button.mat-paginator-navigation-next')
        if not await siguiente.count() or await siguiente.is_disabled():
            break
        await siguiente.click()
        await page.wait_for_timeout(800)
    return filas


async def como_esta(page):
    """Qué hay en la página, sin ningún dato del usuario (solo conteos), para los registros."""
    async def n(sel):
        try:
            return await page.locator(sel).count()
        except Exception:
            return -1
    try:
        ruta = re.sub(r'[?#].*', '', page.url.split('//', 1)[-1].split('/', 1)[-1])[:40]
    except Exception:
        ruta = '?'
    return (f"ruta=/{ruta} login={await n('input[formcontrolname=username]:visible')} "
            f"clave={await n('input[formcontrolname=password]:visible')} tabla={await n('table.table-saldo-cuenta')} "
            f"dialogo={await n('mat-dialog-container')} filas={await n('mat-row, .mat-row')} "
            f"salir_arriba={await n('button[aria-label=Salir]:visible')} menu={await n('mat-sidenav')} "
            f"aviso={await n('simple-snack-bar:visible, mat-error:visible, .alert:visible')}")


async def salir(page):
    """Cierra la sesión en el banco con «Salir». True si se confirmó (vuelve la casilla de usuario)."""
    try:
        await cerrar_movimientos(page)
    except Exception:
        pass
    if await en_login(page):
        return True
    # Con la ventana ancha «Salir» está en la barra de arriba; en pantallas chicas, en el menú lateral
    arriba = page.locator('button[aria-label="Salir"]:visible')
    if await arriba.count():
        await arriba.first.click(timeout=10000)
    else:
        menu = page.locator('button:has(mat-icon:text-is("menu")):visible')
        if await menu.count():
            await menu.first.click(timeout=10000)
            await page.wait_for_timeout(600)
        lateral = page.locator('mat-sidenav a.menu-list-item:has(img[src*="salir"]), mat-sidenav a.menu-list-item:has-text("Salir")')
        await lateral.first.click(timeout=10000)
    # Lo que llegue primero: la casilla de usuario (salió) o una confirmación del banco (se acepta)
    confirmar = ('mat-dialog-container button:has-text("Sí"), mat-dialog-container button:has-text("Si"), '
                 'mat-dialog-container button:has-text("Aceptar"), mat-dialog-container button:has-text("Salir"), '
                 'mat-dialog-container button:has-text("Confirmar")')
    try:
        await page.locator(f'input[formcontrolname="username"]:visible, {confirmar}').first.wait_for(state='visible', timeout=20000)
        boton = page.locator(confirmar)
        if await boton.count() and await boton.first.is_visible():
            await boton.first.click(timeout=5000)
        await page.wait_for_selector('input[formcontrolname="username"]', state='visible', timeout=20000)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# La web (HTTPS)
# ---------------------------------------------------------------------------

class WebCaida(Exception):
    """La web no responde (o responde 5xx/429): se sigue leyendo el banco y se reintenta luego."""


class NoAprobado(Exception):
    """La web ya no reconoce a este lector (no aprobado o desconectado desde el panel)."""


class Web:
    """Cliente síncrono de /api/lector-bdv/* (desde el bucle se llama con asyncio.to_thread)."""

    def __init__(self, base, privada, publica):
        self.base = base.rstrip('/')
        self.privada = privada
        self.publica = publica
        self.sesion = requests.Session()
        self.sesion.headers['User-Agent'] = USER_AGENT
        self.clave = None
        self.vence = 0.0
        self.etag = None
        self.lock = threading.Lock()

    def _llamar(self, metodo, ruta, **kw):
        try:
            r = self.sesion.request(metodo, self.base + ruta, timeout=WEB_TIMEOUT_SEG, **kw)
        except requests.RequestException as e:
            raise WebCaida(type(e).__name__)
        if r.status_code >= 500 or r.status_code == 429:
            raise WebCaida(f'HTTP {r.status_code}')
        return r

    def registrar(self):
        """('aprobado' | 'pendiente' | 'rechazado', código)."""
        r = self._llamar('POST', '/api/lector-bdv/registro', json={'publica': self.publica})
        try:
            d = r.json()
        except ValueError:
            raise WebCaida(f'HTTP {r.status_code}')
        return (d.get('estado') or ('rechazado' if r.status_code == 403 else 'pendiente')), d.get('codigo')

    def autenticar(self):
        from nacl.public import SealedBox
        r = self._llamar('POST', '/api/lector-bdv/reto', json={'publica': self.publica})
        if r.status_code == 401:
            raise NoAprobado()
        if r.status_code != 200:
            raise WebCaida(f'HTTP {r.status_code}')
        d = r.json()
        self.clave = SealedBox(self.privada).decrypt(base64.b64decode(d['reto'])).decode('ascii')
        self.vence = time.time() + max(60, int(d.get('vence_en') or 3600) - 120)

    def _con_clave(self, metodo, ruta, headers=None, **kw):
        with self.lock:
            if not self.clave or time.time() > self.vence:
                self.autenticar()
            for _ in range(2):
                h = dict(headers or {})
                h['Authorization'] = f'Bearer {self.clave}'
                r = self._llamar(metodo, ruta, headers=h, **kw)
                if r.status_code != 401:
                    return r
                self.clave = None
                self.autenticar()   # si ya no está aprobado, lanza NoAprobado
            return r

    def cuentas(self):
        """Las cuentas a leer y a cerrar; None si nada cambió (304)."""
        r = self._con_clave('GET', '/api/lector-bdv/cuentas', headers={'If-None-Match': self.etag} if self.etag else None)
        if r.status_code == 304:
            return None
        if r.status_code != 200:
            raise WebCaida(f'HTTP {r.status_code}')
        self.etag = r.headers.get('ETag')
        return r.json()

    def estado(self, cuentas, lector=None):
        r = self._con_clave('POST', '/api/lector-bdv/estado', json={'cuentas': cuentas, 'lector': lector})
        if r.status_code != 200:
            raise WebCaida(f'HTTP {r.status_code}')
        return r.json()

    def pagos(self, cuenta_id, pagos):
        """La respuesta de la web, o None si la cuenta ya no existe (esos pagos se descartan)."""
        r = self._con_clave('POST', '/api/lector-bdv/pagos', json={'cuenta_id': cuenta_id, 'pagos': pagos})
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise WebCaida(f'HTTP {r.status_code}')
        return r.json()


# ---------------------------------------------------------------------------
# Cola local de pagos (si la web cae no se pierde nada) y pagos ya enviados (no se reenvían)
# ---------------------------------------------------------------------------

class Cola:
    MAX_ENVIADOS = 3000   # por cuenta: la tabla del banco muestra también días atrás

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.pendientes, self.enviados = {}, {}
        try:
            d = json.loads(self.ruta.read_text(encoding='utf-8'))
            self.pendientes = {str(k): v for k, v in (d.get('pendientes') or {}).items()}
            self.enviados = {str(k): v for k, v in (d.get('enviados') or {}).items()}
        except (OSError, ValueError):
            pass

    def _guardar(self):
        tmp = self.ruta.with_suffix('.tmp')
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({'pendientes': self.pendientes, 'enviados': self.enviados}, f)
        os.replace(tmp, self.ruta)

    def agregar(self, cuenta_id, pagos):
        """Pone en la cola los pagos que no se han enviado ni están ya en cola. Devuelve cuántos son nuevos."""
        cid = str(cuenta_id)
        ya = set(self.enviados.get(cid, [])) | {clave_pago(p) for p in self.pendientes.get(cid, [])}
        nuevos = [p for p in pagos if clave_pago(p) not in ya]
        if nuevos:
            self.pendientes.setdefault(cid, []).extend(nuevos)
            self._guardar()
        return len(nuevos)

    def enviados_ok(self, cuenta_id, pagos):
        cid = str(cuenta_id)
        claves = [clave_pago(p) for p in pagos]
        self.enviados[cid] = (self.enviados.get(cid, []) + claves)[-self.MAX_ENVIADOS:]
        hechos = set(claves)
        self.pendientes[cid] = [p for p in self.pendientes.get(cid, []) if clave_pago(p) not in hechos]
        if not self.pendientes[cid]:
            del self.pendientes[cid]
        self._guardar()

    def descartar(self, cuenta_id):
        self.pendientes.pop(str(cuenta_id), None)
        self._guardar()

    def por_enviar(self):
        return {int(k): list(v) for k, v in self.pendientes.items() if v}


# ---------------------------------------------------------------------------
# Memoria real (Linux: la suma de este proceso y su Chromium)
# ---------------------------------------------------------------------------

def memoria_mb():
    """MB que usan este proceso y todos sus hijos (PSS si está disponible, si no RSS). None fuera de Linux."""
    if not os.path.isdir('/proc'):
        return None
    padres = {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/stat', 'rb') as f:
                datos = f.read().decode('utf-8', 'replace')
            padres[int(pid)] = int(datos.rsplit(')', 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            pass
    mios, pendientes = set(), [os.getpid()]
    while pendientes:
        p = pendientes.pop()
        mios.add(p)
        pendientes += [h for h, pp in padres.items() if pp == p and h not in mios]
    total_kb = 0
    for pid in mios:
        kb = None
        for archivo, campo in ((f'/proc/{pid}/smaps_rollup', 'Pss:'), (f'/proc/{pid}/status', 'VmRSS:')):
            try:
                with open(archivo) as f:
                    for linea in f:
                        if linea.startswith(campo):
                            kb = int(linea.split()[1])
                            break
            except OSError:
                pass
            if kb is not None:
                break
        total_kb += kb or 0
    return round(total_kb / 1024, 1)


# ---------------------------------------------------------------------------
# Lectura de cada cuenta
# ---------------------------------------------------------------------------

class LectorCuenta:
    def __init__(self, gestor, info, retraso=0.0, antes=None):
        self.gestor = gestor
        self.info = dict(info)
        self.retraso = retraso
        self.antes = antes   # la tarea anterior de esta misma cuenta, si aún está saliendo del banco
        self.aviso = asyncio.Event()
        self.contexto = None
        self.page = None
        self.fallos = int(info.get('fallos_login') or 0)   # rechazos seguidos (la web los guarda: sobreviven a un reinicio)
        self.sin_respuesta = 0
        self.errores_seguidos = 0
        self.detenido = False
        self.reentrar = False
        self.terminar = False
        self.reportar_pausa = False
        self.version_cierre = ''
        self.en_espera = False
        self.tarea = None

    @property
    def id(self):
        return self.info['id']

    def actualizar(self, info):
        """La web mandó la cuenta otra vez: si cambió la versión (datos nuevos o «Reanudar»), sale y vuelve a entrar."""
        if info.get('version') != self.info.get('version'):
            self.info = dict(info)
            self.reentrar = True
            self.detenido = False
            self.fallos = int(info.get('fallos_login') or 0)
            self.sin_respuesta = 0
            self.aviso.set()

    def detener(self, version_cierre, reportar_pausa):
        """Pausada, borrada o detenida en la web: «Salir» en el banco y fin."""
        self.terminar = True
        self.version_cierre = version_cierre or ''
        self.reportar_pausa = reportar_pausa and not self.detenido
        self.aviso.set()

    async def esperar(self, segundos):
        """Espera, pero se despierta enseguida si llega un aviso (pausa, datos nuevos, lugar libre)."""
        if not self.aviso.is_set():
            try:
                await asyncio.wait_for(self.aviso.wait(), timeout=segundos)
            except asyncio.TimeoutError:
                pass
        self.aviso.clear()

    def estado(self, estado, mensaje, version=None, **extra):
        self.info['estado'] = estado
        self.gestor.reportar(self.id, version if version is not None else self.info.get('version', ''), estado, mensaje,
                             fallos=self.fallos, **extra)

    async def abrir(self):
        self.contexto = await self.gestor.browser.new_context(
            viewport={'width': 1280, 'height': 900}, locale='es-VE', timezone_id='America/Caracas')

        async def sin_imagenes(route):   # menos memoria y menos tráfico: el lector no necesita verlas
            if route.request.resource_type in ('image', 'media', 'font'):
                await route.abort()
            else:
                await route.continue_()
        await self.contexto.route('**/*', sin_imagenes)
        self.page = await self.contexto.new_page()

    async def cerrar(self, cerrar_sesion=True):
        """Cierra la sesión en el banco («Salir») y el contexto. True si se confirmó la salida."""
        confirmado = True
        if self.page is not None and cerrar_sesion:
            confirmado = False
            for intento in range(2):   # si no se confirma, se intenta una vez más
                try:
                    if await salir(self.page):
                        confirmado = True
                        break
                except Exception:
                    pass
                logger.warning(f'[BDV] Cuenta {self.id}: no se confirmó «Salir» ({intento + 1}/2) · {await como_esta(self.page)}')
                # La página pudo quedar trabada: se recarga. Si vuelve la casilla de usuario, la sesión ya estaba cerrada
                try:
                    await self.page.goto(URL, wait_until='domcontentloaded', timeout=TIMEOUT_MS)
                    await self.page.wait_for_selector('input[formcontrolname="username"], table.table-saldo-cuenta',
                                                      timeout=TIMEOUT_MS)
                    if await self.page.locator('input[formcontrolname="username"]:visible').count():
                        confirmado = True
                        break
                except Exception:
                    pass
        if self.contexto is not None:
            try:
                await self.contexto.close()
            except Exception:
                pass
        self.contexto = self.page = None
        self.gestor.soltar_lugar(self)
        return confirmado

    async def correr(self):
        try:
            if self.antes is not None and not self.antes.done():   # que termine de salir la sesión anterior
                await asyncio.wait([self.antes], timeout=120)
            if self.info.get('estado') == 'rechazado':   # tras un reinicio se respeta la espera antes del 2.º intento
                await self.esperar(ESPERA_RECHAZO_SEG)
            elif self.retraso:
                await self.esperar(self.retraso)   # escalonado: no entran todas en el mismo segundo
            while not self.terminar:
                if self.detenido and not self.reentrar:
                    await self.esperar(REVISAR_SEG * 12)
                    continue
                if self.reentrar:   # datos nuevos o «Reanudar»: sale y vuelve a entrar con ellos
                    self.reentrar = False
                    if self.page is not None:
                        await self.cerrar()
                if self.page is None:
                    if not self.gestor.pedir_lugar(self):
                        if not self.en_espera:
                            self.en_espera = True
                            self.estado('en_espera', MSG_EN_ESPERA)
                        await self.esperar(REVISAR_SEG * 2)
                        continue
                    self.en_espera = False
                    await self.abrir()
                await self.vuelta()
        except asyncio.CancelledError:
            raise
        except Exception as e:   # nunca con datos del usuario: solo el tipo de error
            logger.error(f'[BDV] Cuenta {self.id}: el lector de esta cuenta se detuvo por un error ({type(e).__name__})')
        finally:
            confirmado = True
            if self.page is not None or self.contexto is not None:
                try:
                    confirmado = await self.cerrar()
                except Exception:
                    confirmado = False
            self.gestor.soltar_lugar(self)
            if self.reportar_pausa:
                self.estado('pausado', MSG_PAUSADO if confirmado else MSG_PAUSA_SIN_CONFIRMAR, version=self.version_cierre)
                logger.info(f'[BDV] Cuenta {self.id}: en pausa (sesión cerrada{"" if confirmado else " sin confirmar"})')

    async def vuelta(self):
        page = self.page
        paso = 'entrar'
        try:
            if await en_login(page) or not await en_sesion(page):
                if not await self.entrar():
                    return
            paso = 'cerrar el diálogo anterior'
            await cerrar_movimientos(page)   # por si quedó abierta de una vuelta con error
            paso = 'abrir los movimientos'
            await abrir_movimientos(page, self.info.get('ultimos_digitos', ''))
            paso = 'leer la tabla'
            filas = await leer_tabla(page)
            paso = 'cerrar el diálogo'
            await cerrar_movimientos(page)   # leído: la espera se hace con la ventana cerrada
            pagos = [p for p in (a_pago(f) for f in filas) if p]
            nuevos = self.gestor.cola.agregar(self.id, pagos)   # solo los no enviados; la web ignora repetidos igual
            if nuevos:
                logger.info(f'[BDV] Cuenta {self.id}: {nuevos} pagos nuevos')
                self.gestor.despertar()
            self.estado('conectado', f'Leyendo tus movimientos cada {CADA_SEG} s · última vuelta: {len(pagos)} pagos recibidos',
                        ultima_lectura=ahora_local())
            self.errores_seguidos = 0
        except CuentaNoEncontrada:
            self.estado('conectado', MSG_SIN_CUENTA.format(self.info.get('ultimos_digitos', '')))
        except Exception as e:
            if self.terminar or self.page is None:   # se está cerrando (pausa, navegador caído): no es un problema
                return
            self.errores_seguidos += 1
            logger.warning(f'[BDV] Cuenta {self.id}: problema en la vuelta ({self.errores_seguidos}) al {paso}: '
                           f'{type(e).__name__} · {await como_esta(page)}')
            if self.errores_seguidos >= 3:   # algo raro en la página: se vuelve a cargar
                self.errores_seguidos = 0
                try:
                    await page.goto(URL, wait_until='domcontentloaded', timeout=TIMEOUT_MS)
                except Exception:
                    pass
        await self.esperar(CADA_SEG + random.uniform(0, 3))

    def _datos_de_acceso(self):
        """(usuario, contraseña) abiertos con la llave privada, solo en memoria."""
        from nacl.public import SealedBox
        caja = SealedBox(self.gestor.privada)
        return (caja.decrypt(base64.b64decode(self.info['bdv_usuario_cifrado'])).decode('utf-8'),
                caja.decrypt(base64.b64decode(self.info['bdv_clave_cifrada'])).decode('utf-8'))

    async def entrar(self):
        """True si quedó dentro. Aplica las esperas y la protección de la cuenta."""
        self.estado('conectando', MSG_CONECTANDO)
        try:
            usuario, clave = self._datos_de_acceso()
        except Exception:   # cerrados con la llave de otro lector (se cambió el lector) o dañados
            self.detenido = True
            self.estado('detenido', MSG_SIN_DATOS)
            logger.warning(f'[BDV] Cuenta {self.id}: sus datos no se abren con la llave de este lector')
            await self.cerrar(cerrar_sesion=False)
            return False
        try:
            await entrar(self.page, usuario, clave)
        except BancoNoResponde:
            self.sin_respuesta += 1
            espera = min(NO_RESPONDE_MAX_SEG, NO_RESPONDE_BASE_SEG * 2 ** (self.sin_respuesta - 1))
            self.estado('banco_no_responde', f'El banco no responde. Reintento en {max(1, round(espera / 60))} min.')
            logger.info(f'[BDV] Cuenta {self.id}: el banco no responde, reintento en {espera} s')
            await self.esperar(espera)
            return False
        except LoginRechazado:
            self.fallos += 1
            logger.warning(f'[BDV] Cuenta {self.id}: el banco rechazó la entrada ({self.fallos}/{MAX_RECHAZOS})')
            if self.fallos >= MAX_RECHAZOS:
                self.detenido = True
                self.estado('detenido', MSG_DETENIDO)
                await self.cerrar(cerrar_sesion=False)
                return False
            self.estado('rechazado', MSG_RECHAZADO)
            await self.esperar(ESPERA_RECHAZO_SEG)
            return False
        finally:
            usuario = clave = None   # noqa: F841 — no quedan en memoria más de lo necesario
        self.fallos = 0
        self.sin_respuesta = 0
        self.estado('conectado', f'Sesión abierta: leyendo tus movimientos cada {CADA_SEG} s')
        logger.info(f'[BDV] Cuenta {self.id} (··{self.info.get("ultimos_digitos", "")}): sesión abierta')
        return True


# ---------------------------------------------------------------------------
# Gestor de todas las cuentas
# ---------------------------------------------------------------------------

class Gestor:
    def __init__(self, web, cola, privada, abrir_navegador, max_cuentas=30):
        self.web = web
        self.cola = cola
        self.privada = privada
        self.abrir_navegador = abrir_navegador   # async () → browser de Playwright
        self.browser = None
        self.max = max_cuentas
        self.lectores = {}       # cuenta → su lector
        self.cerrando = {}       # cuenta → tarea que aún está saliendo del banco
        self.tareas = set()      # referencia a todas las tareas (asyncio solo guarda referencias débiles)
        self.con_lugar = set()   # lectores (objetos) con un contexto abierto
        self.lista = None        # la última lista de la web (con 304 se sigue usando)
        self.reportes = {}
        self.lector_por_enviar = False
        self.web_ok = True
        self.ultima_medida = float('-inf')
        self.memoria = None
        self.ultimo_tic = time.monotonic()   # para el vigilante
        self.despierta = asyncio.Event()

    # --- lugares (tope de cuentas a la vez) ---
    def pedir_lugar(self, lector):
        if lector in self.con_lugar:
            return True
        if len(self.con_lugar) >= self.max:
            return False
        self.con_lugar.add(lector)
        return True

    def soltar_lugar(self, lector):
        if lector in self.con_lugar:
            self.con_lugar.discard(lector)
            for otro in self.lectores.values():   # la que estaba esperando entra ya
                if otro.en_espera:
                    otro.aviso.set()

    # --- lo que se le cuenta a la web ---
    def reportar(self, cuenta_id, version, estado, mensaje, fallos=None, **extra):
        item = {'id': cuenta_id, 'version': version, 'estado': estado, 'mensaje': mensaje, **extra}
        if fallos is not None:
            item['fallos_login'] = fallos
        self.reportes[cuenta_id] = item
        self.despertar()

    def despertar(self):
        self.despierta.set()

    def info_lector(self):
        abiertas = len(self.con_lugar)
        return {'memoria_mb': self.memoria, 'cuentas_leyendo': abiertas, 'max_cuentas': self.max,
                'en_espera': sum(1 for x in self.lectores.values() if x.en_espera), 'version': VERSION,
                'por_cuenta_mb': round(self.memoria / abiertas, 1) if self.memoria and abiertas else None}

    # --- ciclo ---
    async def esperar_aprobacion(self):
        """Hasta que la web reconozca a este lector. Mientras tanto no lee nada."""
        ultimo_aviso = float('-inf')
        while True:
            self.ultimo_tic = time.monotonic()
            try:
                estado, codigo = await asyncio.to_thread(self.web.registrar)
                if estado == 'aprobado':
                    await asyncio.to_thread(self.web.autenticar)
                    logger.info(f'[BDV] Lector aprobado por la web (código {codigo})')
                    return
                if time.monotonic() - ultimo_aviso > 300:
                    ultimo_aviso = time.monotonic()
                    if estado == 'rechazado':
                        logger.error('[BDV] La web tiene fijada otra llave del lector (BDV_CLAVE_PUBLICA): este lector no '
                                     'puede conectarse hasta que se quite o se cambie esa variable')
                    else:
                        logger.warning(f'[BDV] Esperando aprobación: en la web, Referencias (todos) → Conexiones del BDV, '
                                       f'aprueba el lector con el código {codigo}')
            except (WebCaida, NoAprobado) as e:
                if time.monotonic() - ultimo_aviso > 300:
                    ultimo_aviso = time.monotonic()
                    logger.warning(f'[BDV] No se pudo hablar con la web para registrarse ({str(e) or type(e).__name__}); se reintenta')
            await asyncio.sleep(APROBACION_SEG)

    async def asegurar_navegador(self):
        if self.browser is not None and self.browser.is_connected():
            return
        if self.browser is not None:
            logger.error('[BDV] El navegador se cerró inesperadamente: se vuelve a abrir y se entra de nuevo')
            await self.cerrar_todo(salir_del_banco=False)
        self.browser = await self.abrir_navegador()

    def _nueva_tarea(self, lector):
        lector.tarea = asyncio.create_task(lector.correr())
        self.tareas.add(lector.tarea)
        lector.tarea.add_done_callback(self.tareas.discard)

    async def sincronizar(self):
        datos = await asyncio.to_thread(self.web.cuentas)
        if datos is not None:
            self.lista = datos
        if self.lista is None:
            return
        self.cerrando = {c: t for c, t in self.cerrando.items() if not t.done()}
        leer = {int(c['id']): c for c in self.lista.get('cuentas') or []}
        cerrar = {int(x['id']): x for x in self.lista.get('cerrar') or []}
        nuevas = 0
        for cid, info in leer.items():
            lector = self.lectores.get(cid)
            if lector is None or lector.tarea is None or lector.tarea.done():   # nueva (o su lector se cayó)
                lector = LectorCuenta(self, info, retraso=nuevas * ESCALON_SEG, antes=self.cerrando.pop(cid, None))
                self.lectores[cid] = lector
                self._nueva_tarea(lector)
                nuevas += 1
            else:
                lector.actualizar(info)
        for cid in [c for c in self.lectores if c not in leer]:   # pausada, borrada o detenida: «Salir» y fuera
            lector = self.lectores.pop(cid)
            x = cerrar.get(cid) or {}
            lector.detener(x.get('version', ''), reportar_pausa=bool(x.get('falta_pausa')))
            if lector.tarea is not None and not lector.tarea.done():
                self.cerrando[cid] = lector.tarea
        # Pausadas que este lector no tenía abiertas (p. ej. se pausaron con el lector apagado): se confirma la pausa
        for cid, x in cerrar.items():
            if x.get('falta_pausa') and cid not in self.lectores and cid not in self.cerrando and cid not in self.reportes:
                self.reportar(cid, x.get('version', ''), 'pausado', MSG_PAUSADO_SIN_SESION)

    async def enviar(self):
        if self.reportes or self.lector_por_enviar:
            items = list(self.reportes.values())
            for i in range(0, max(len(items), 1), 200):
                lote = items[i:i + 200]
                await asyncio.to_thread(self.web.estado, lote, self.info_lector() if self.lector_por_enviar else None)
                self.lector_por_enviar = False
                for it in lote:   # solo se quitan los enviados (pudo llegar uno nuevo mientras tanto)
                    if self.reportes.get(it['id']) is it:
                        del self.reportes[it['id']]
        for cid, pagos in self.cola.por_enviar().items():
            for i in range(0, len(pagos), 500):
                lote = pagos[i:i + 500]
                respuesta = await asyncio.to_thread(self.web.pagos, cid, lote)
                if respuesta is None:   # la cuenta ya no existe en la web
                    self.cola.descartar(cid)
                    break
                self.cola.enviados_ok(cid, lote)

    async def medir(self):
        if time.monotonic() - self.ultima_medida < MEDIR_CADA_SEG:
            return
        self.memoria = await asyncio.to_thread(memoria_mb)
        self.ultima_medida = time.monotonic()
        self.lector_por_enviar = True
        logger.info(f'[BDV] Cuentas leyendo: {len(self.con_lugar)} · memoria del lector: {self.memoria} MB')

    async def correr(self):
        await self.esperar_aprobacion()
        ultima_sync = float('-inf')
        while True:
            self.ultimo_tic = time.monotonic()
            try:
                await self.asegurar_navegador()
                await self.medir()
                if time.monotonic() - ultima_sync >= REVISAR_SEG:
                    ultima_sync = time.monotonic()
                    await self.sincronizar()
                await self.enviar()
                if not self.web_ok:
                    logger.info('[BDV] La web volvió: se mandaron los pagos que estaban en cola')
                    self.web_ok = True
            except WebCaida as e:
                if self.web_ok:
                    logger.warning(f'[BDV] La web no responde ({e}): sigo leyendo el banco y guardo los pagos en la cola')
                    self.web_ok = False
            except NoAprobado:
                logger.error('[BDV] La web ya no reconoce a este lector (desconectado en el panel): cierro las sesiones '
                             'del banco y espero una nueva aprobación')
                await self.cerrar_todo()
                await self.esperar_aprobacion()
                ultima_sync = float('-inf')
                continue
            except Exception as e:
                logger.error(f'[BDV] Error en la vuelta del gestor: {type(e).__name__}')
            if not self.web_ok:   # web caída: se reintenta con calma
                await asyncio.sleep(REVISAR_SEG)
                continue
            # Hasta la próxima revisión, o antes si hay pagos o estados que mandar (como mucho uno por segundo:
            # así se respetan los límites de la web)
            self.despierta.clear()
            try:
                await asyncio.wait_for(self.despierta.wait(), timeout=max(0.1, REVISAR_SEG - (time.monotonic() - ultima_sync)))
                await asyncio.sleep(JUNTAR_SEG)
            except asyncio.TimeoutError:
                pass

    async def cerrar_todo(self, salir_del_banco=True):
        lectores = list(self.lectores.values())
        self.lectores.clear()
        for lector in lectores:
            lector.terminar = True
            if not salir_del_banco:   # el navegador murió: no hay página donde pulsar «Salir»
                lector.page = lector.contexto = None
            lector.aviso.set()
        tareas = [x.tarea for x in lectores if x.tarea] + list(self.cerrando.values())
        self.cerrando = {}
        if tareas:
            await asyncio.wait(tareas, timeout=60)
        self.con_lugar.clear()
        self.lista = None
        self.web.etag = None   # al volver se pide la lista completa y todas vuelven a entrar


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------

def url_web_valida(url):
    """https siempre; http solo hacia esta misma máquina (pruebas): la clave temporal no viaja sin cifrar."""
    try:
        partes = urlsplit(url)
    except ValueError:
        return False
    if not partes.hostname:
        return False
    return partes.scheme == 'https' or (partes.scheme == 'http' and partes.hostname in ('localhost', '127.0.0.1', '::1'))


def tomar_candado(ruta):
    """Un solo lector a la vez en este servidor. Devuelve el archivo abierto (mantenerlo) o None si ya hay otro."""
    f = open(ruta, 'a+')
    try:
        if os.name == 'nt':
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def vigilante(gestor):
    """Si el programa se queda trabado (sin dar vueltas) más de VIGILANTE_SEG, se cierra para que systemd lo levante."""
    while True:
        time.sleep(30)
        if time.monotonic() - gestor.ultimo_tic > VIGILANTE_SEG:
            logger.error('[BDV] El lector se quedó trabado: se reinicia')
            logging.shutdown()
            os._exit(3)


async def principal(web_url, privada, publica, datos, max_cuentas):
    from playwright.async_api import async_playwright
    if os.name == 'posix':   # systemctl stop/restart: primero «Salir» en el banco en todas las cuentas
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    async with async_playwright() as p:
        async def abrir_navegador():
            return await p.chromium.launch(headless=True, args=['--disable-dev-shm-usage'])
        web = Web(web_url, privada, publica)
        gestor = Gestor(web, Cola(Path(datos) / 'cola.json'), privada, abrir_navegador, max_cuentas)
        threading.Thread(target=vigilante, args=(gestor,), daemon=True).start()
        logger.info(f'[BDV] Lector del BDV {VERSION} arrancado (máximo {max_cuentas} cuentas a la vez) · web: {web_url}')
        try:
            await gestor.correr()
        finally:
            logger.info('[BDV] Deteniendo: cierro las sesiones del banco')
            gestor.ultimo_tic = time.monotonic()
            await gestor.cerrar_todo()
            if gestor.browser is not None:
                try:
                    await gestor.browser.close()
                except Exception:
                    pass


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    web_url = os.environ.get('BDV_WEB_URL', '').strip().rstrip('/')
    if not url_web_valida(web_url):
        logger.error('[BDV] BDV_WEB_URL falta o no es https (la dirección de la web, p. ej. https://mi-web.com)')
        time.sleep(60)   # para que systemd no lo reinicie en bucle
        sys.exit(1)
    datos = Path(os.environ.get('BDV_DATOS', '').strip() or (Path.home() / '.lector-bdv'))
    datos.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(datos, 0o700)
    except OSError:
        pass
    candado = tomar_candado(datos / 'lector.lock')
    if candado is None:
        logger.error('[BDV] Ya hay otro lector del BDV funcionando en este servidor: este no arranca')
        time.sleep(60)
        sys.exit(1)
    privada, publica, creada = generar_llaves.cargar_o_crear(os.environ.get('BDV_CLAVE_PRIVADA', '').strip() or None)
    if creada:
        logger.warning(f'[BDV] Se creó la llave del lector en {generar_llaves.ruta_por_defecto()} (permisos 600): respáldala')
    logger.info(f'[BDV] Código de este lector: {generar_llaves.huella(publica)}')
    try:
        max_cuentas = max(1, int(os.environ.get('BDV_MAX_CUENTAS', '').strip() or 30))
    except ValueError:
        max_cuentas = 30
    try:
        asyncio.run(principal(web_url, privada, publica, datos, max_cuentas))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    logger.info('[BDV] Lector detenido')


if __name__ == '__main__':
    main()

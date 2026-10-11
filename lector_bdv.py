"""
Lector del BDV (Banco de Venezuela en línea, personas) para Referencias. Corre APARTE de la web, nunca dentro de
gunicorn:

    python lector_bdv.py

Un solo Chromium sin ventana, con un contexto (cookies propias, sin compartir) por cada cuenta BDV conectada en el
panel. Cada cuenta, igual que el prototipo probado contra el banco real:
  1. entra una vez y se queda dentro;
  2. cada vuelta abre los movimientos, lee todas las páginas, guarda los pagos recibidos nuevos
     (antiduplic.guardar_pagos ignora los repetidos), cierra el diálogo («Regresar») y espera 25 s con él cerrado;
  3. si la sesión se cerró, vuelve a entrar; si una vuelta falla 3 veces seguidas, recarga la página.

Protección de la cuenta del usuario:
  - banco que rechaza la entrada → espera 5 min y prueba UNA vez más; si vuelve a rechazar → «detenido» y no
    reintenta hasta que el usuario guarde sus datos de nuevo o pulse «Reanudar»;
  - banco que no responde → esperas de 1, 2, 4, 8 y hasta 10 min, sin contarlo como rechazo.

Interruptor «Leer movimientos»: se revisa cada 5 s. Apagado → cierra la sesión en el banco con «Salir» (así el dueño
puede entrar desde su teléfono) y cierra el contexto; encendido → vuelve a entrar.

Variables:
  BDV_LLAVE_CIFRADO   obligatoria: la misma llave con la que la web cifró los datos (sin ella no arranca)
  BDV_MAX_CUENTAS     cuentas leyendo a la vez (por defecto 30); las demás quedan «en espera»
  DATABASE_URL        la misma base de datos de la web
  BDV_URL             solo para pruebas (un BDV falso)

Los datos de acceso solo se descifran aquí, en memoria, al entrar. Nunca van a los registros ni a lector_mensaje.
"""
import asyncio
import json
import logging
import os
import random
import re
import secrets
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import antiduplic as ad  # noqa: E402
import bdv  # noqa: E402
from pg_compat import get_db_connection  # noqa: E402

logger = logging.getLogger('lector_bdv')

URL = os.environ.get('BDV_URL', '').strip() or 'https://bdvenlinea.banvenez.com/'
MAX_CUENTAS = max(1, int(os.environ.get('BDV_MAX_CUENTAS', '30') or 30))

# Tiempos (las pruebas los acortan)
CADA_SEG = 25            # espera entre vueltas, con la ventana de movimientos cerrada
REVISAR_SEG = 5          # cada cuánto se revisan el panel y los interruptores
ESCALON_SEG = 4          # separación entre las entradas de cuentas nuevas
ESPERA_RECHAZO_SEG = 300  # antes del segundo intento cuando el banco rechaza la entrada
NO_RESPONDE_BASE_SEG = 60
NO_RESPONDE_MAX_SEG = 600
TIMEOUT_MS = 60000       # carga de páginas del banco
MEDIR_CADA_SEG = 300     # medición de memoria
MAX_RECHAZOS = 2
LATIDO_VENCE_SEG = 30    # otro lector con un latido más reciente que esto → este no arranca

NO_SON_PAGOS = re.compile(r'INTERES|COMISION|REINTEGRO|AJUSTE', re.I)   # créditos que pone el propio banco
RECHAZO = re.compile(r'inv[aá]lid|incorrect|bloquead|suspendid|no coincide|intentos|no autorizad', re.I)


# ---------------------------------------------------------------------------
# Lectura de la tabla (funciones puras, probadas aparte)
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


# ---------------------------------------------------------------------------
# El banco (Playwright, asíncrono)
# ---------------------------------------------------------------------------

class BancoNoResponde(Exception):
    """La página no cargó (saturado, sin internet...): se espera y se reintenta. NO cuenta como rechazo."""


class LoginRechazado(Exception):
    """El banco dijo que no (datos inválidos, usuario bloqueado...): SÍ cuenta para detenerse."""


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


class CuentaNoEncontrada(Exception):
    pass


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
    await page.wait_for_selector('mat-row .mat-column-referencia, .mat-row .mat-column-referencia', timeout=TIMEOUT_MS)


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
# Base de datos (síncrona; desde el bucle se llama con asyncio.to_thread)
# ---------------------------------------------------------------------------

def _db(fn):
    conn = get_db_connection()
    try:
        r = fn(conn)
        conn.commit()
        return r
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def leer_cuentas():
    """Cuentas BDV con su configuración (sin datos de acceso)."""
    def q(conn):
        return [{'id': r['id'], 'usuario_id': r['usuario_id'], 'ultimos': r['ultimos_digitos'] or '',
                 'activa': bool(r['activo']), 'con_datos': bool(r['bdv_clave_cifrada']),
                 'encendido': r['lector_encendido'] is None or bool(r['lector_encendido']),
                 'cambio': r['lector_cambio'] or '', 'estado': r['lector_estado'] or 'pendiente',
                 'fallos': int(r['lector_fallos_login'] or 0)}
                for r in conn.execute("SELECT * FROM cuentas_banco WHERE banco = 'BDV' ORDER BY id").fetchall()]
    return _db(q)


def leer_acceso(cuenta_id):
    """(usuario, clave) descifrados, solo en memoria. Lanza bdv.SinLlave si no se pueden leer."""
    def q(conn):
        return conn.execute('SELECT bdv_usuario_cifrado, bdv_clave_cifrada FROM cuentas_banco WHERE id = ?', (cuenta_id,)).fetchone()
    fila = _db(q)
    if not fila or not fila['bdv_clave_cifrada']:
        raise bdv.SinLlave('sin datos')
    return bdv.descifrar(fila['bdv_usuario_cifrado']), bdv.descifrar(fila['bdv_clave_cifrada'])


def poner_estado(cuenta_id, estado, mensaje, **campos):
    """Estado del lector de una cuenta (lo que ve el usuario). Nunca con datos secretos."""
    sets, params = ['lector_estado = ?', 'lector_mensaje = ?'], [estado, mensaje[:200]]
    for k, v in campos.items():
        sets.append(f'{k} = ?')
        params.append(v)
    params.append(cuenta_id)
    # Si la conexión se borró mientras se leía, no se toca: se queda «sin conexión»
    _db(lambda conn: conn.execute(f"UPDATE cuentas_banco SET {', '.join(sets)} WHERE id = ? AND banco = 'BDV' "
                                  'AND bdv_clave_cifrada IS NOT NULL', tuple(params)))


def guardar(cuenta, pagos):
    """Guarda los pagos leídos (los repetidos se ignoran) y anota la lectura. Devuelve cuántos son nuevos."""
    _, insertados, _ = ad.guardar_pagos(pagos, cuenta['usuario_id'], cuenta['id'])
    ahora = ad.ahora_local()
    campos = {'lector_ultima_lectura': ahora}
    if insertados:
        campos['lector_ultimo_pago'] = ahora
    poner_estado(cuenta['id'], 'conectado', f'Leyendo tus movimientos cada {CADA_SEG} s · última vuelta: {len(pagos)} pagos recibidos',
                 **campos)
    return insertados


def tomar_turno(instancia):
    """Un solo lector a la vez: False si otro lector dio señales de vida hace poco."""
    def q(conn):
        actual = ad._config_get(conn, 'bdv_lector_instancia')
        try:
            datos = json.loads(actual) if actual else {}
        except ValueError:
            datos = {}
        if datos.get('id') not in (None, instancia) and time.time() - float(datos.get('t', 0)) < LATIDO_VENCE_SEG:
            return False
        ad._config_set(conn, 'bdv_lector_instancia', json.dumps({'id': instancia, 't': time.time()}))
        return True
    return _db(q)


def guardar_estado_lector(datos):
    _db(lambda conn: ad._config_set(conn, 'bdv_lector_estado', json.dumps(datos)))


# ---------------------------------------------------------------------------
# Memoria real (Linux: la suma de lo que usan los procesos de este lector y su Chromium)
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

MSG_CONECTANDO = 'Entrando al banco…'
MSG_EN_ESPERA = 'Hay muchas cuentas conectadas: entra en cuanto se libere un lugar'
MSG_RECHAZADO = 'El banco rechazó la entrada. Se reintenta una sola vez en 5 minutos'
MSG_SIN_CUENTA = 'No encontré en tu BDV la cuenta que termina en {}: revisa el número que guardaste'
MSG_SIN_DATOS = 'No se pudieron leer tus datos guardados: guárdalos de nuevo'


class LectorCuenta:
    def __init__(self, gestor, info, retraso=0.0):
        self.gestor = gestor
        self.info = dict(info)
        self.cambio_visto = info['cambio']
        self.retraso = retraso
        self.aviso = asyncio.Event()
        self.contexto = None
        self.page = None
        self.fallos = info['fallos']        # rechazos seguidos (se guardan en la base: sobreviven a un reinicio)
        self.sin_respuesta = 0
        self.errores_seguidos = 0
        self.reentrar = False
        self.terminar = False
        self.tarea = None

    @property
    def id(self):
        return self.info['id']

    def actualizar(self, info):
        anterior = self.info
        self.info = dict(info)
        if info['cambio'] != self.cambio_visto:      # guardó datos nuevos o pulsó «Reanudar»
            self.cambio_visto = info['cambio']
            self.reentrar = True
            self.fallos = info['fallos']
            self.sin_respuesta = 0
            self.aviso.set()
        if anterior['encendido'] != info['encendido'] or (anterior['estado'] == 'detenido') != (info['estado'] == 'detenido'):
            self.aviso.set()

    async def esperar(self, segundos):
        """Espera, pero se despierta enseguida si el usuario toca el interruptor o guarda datos nuevos
        (también si el aviso llegó mientras se leía: así una pausa tarda como mucho unos segundos)."""
        if not self.aviso.is_set():
            try:
                await asyncio.wait_for(self.aviso.wait(), timeout=segundos)
            except asyncio.TimeoutError:
                pass
        self.aviso.clear()

    async def estado(self, estado, mensaje, **campos):
        await asyncio.to_thread(poner_estado, self.id, estado, mensaje, **campos)

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
        """Cierra la sesión en el banco («Salir») y el contexto. Devuelve True si se confirmó la salida."""
        confirmado = True
        if self.page is not None and cerrar_sesion:
            confirmado = False
            for _ in range(2):   # si no se confirma, se intenta una vez más
                try:
                    if await salir(self.page):
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
            if self.info['estado'] == 'rechazado':   # tras un reinicio se respeta la espera antes del 2.º intento
                await self.esperar(ESPERA_RECHAZO_SEG)
            elif self.retraso:
                await self.esperar(self.retraso)   # escalonado: no entran todas en el mismo segundo
            while not self.terminar:
                info = self.info
                if not info['con_datos'] or not info['activa']:
                    break
                if not info['encendido']:
                    if self.page is not None:
                        ok = await self.cerrar()
                        await self.estado('pausado', bdv.MSG_PAUSADO if ok else bdv.MSG_PAUSA_SIN_CONFIRMAR)
                        logger.info(f'[BDV] Cuenta {self.id}: en pausa (sesión cerrada{"" if ok else " sin confirmar"})')
                    elif info['estado'] != 'pausado':
                        await self.estado('pausado', bdv.MSG_PAUSADO)
                    await self.esperar(REVISAR_SEG * 12)
                    continue
                if info['estado'] == 'detenido' and not self.reentrar:
                    if self.page is not None:
                        await self.cerrar()
                    await self.esperar(REVISAR_SEG * 12)
                    continue
                if self.reentrar:   # datos nuevos: sale y vuelve a entrar con ellos
                    self.reentrar = False
                    if self.page is not None:
                        await self.cerrar()
                if self.page is None:
                    if not self.gestor.pedir_lugar(self):
                        if info['estado'] != 'en_espera':
                            await self.estado('en_espera', MSG_EN_ESPERA)
                            self.info['estado'] = 'en_espera'
                        await self.esperar(REVISAR_SEG * 2)
                        continue
                    await self.abrir()
                await self.vuelta()
        except asyncio.CancelledError:
            raise
        except Exception as e:   # nunca con datos del usuario: solo el tipo de error
            logger.error(f'[BDV] Cuenta {self.id}: el lector de esta cuenta se detuvo por un error ({type(e).__name__})')
        finally:
            if self.page is not None or self.contexto is not None:
                try:
                    await self.cerrar()
                except Exception:
                    pass
            self.gestor.soltar_lugar(self)

    async def vuelta(self):
        page = self.page
        try:
            if await en_login(page) or not await en_sesion(page):
                if not await self.entrar():
                    return
            await cerrar_movimientos(page)   # por si quedó abierta de una vuelta con error
            await abrir_movimientos(page, self.info['ultimos'])
            filas = await leer_tabla(page)
            await cerrar_movimientos(page)   # leído: la espera se hace con la ventana cerrada
            pagos = [p for p in (a_pago(f) for f in filas) if p]
            nuevos = await asyncio.to_thread(guardar, self.info, pagos)
            if nuevos:
                logger.info(f'[BDV] Cuenta {self.id}: {nuevos} pagos nuevos')
            self.info['estado'] = 'conectado'
            self.errores_seguidos = 0
        except CuentaNoEncontrada:
            await self.estado('conectado', MSG_SIN_CUENTA.format(self.info['ultimos']))
        except Exception as e:
            self.errores_seguidos += 1
            logger.warning(f'[BDV] Cuenta {self.id}: problema en la vuelta ({self.errores_seguidos}): {type(e).__name__}')
            if self.errores_seguidos >= 3:   # algo raro en la página: se vuelve a cargar
                self.errores_seguidos = 0
                try:
                    await page.goto(URL, wait_until='domcontentloaded', timeout=TIMEOUT_MS)
                except Exception:
                    pass
        await self.esperar(CADA_SEG + random.uniform(0, 3))

    async def entrar(self):
        """True si quedó dentro. Aplica las esperas y la protección de la cuenta."""
        await self.estado('conectando', MSG_CONECTANDO)
        try:
            usuario, clave = await asyncio.to_thread(leer_acceso, self.id)
        except bdv.SinLlave:
            await self.estado('detenido', MSG_SIN_DATOS)
            self.info['estado'] = 'detenido'
            await self.cerrar(cerrar_sesion=False)
            return False
        try:
            await entrar(self.page, usuario, clave)
        except BancoNoResponde:
            self.sin_respuesta += 1
            espera = min(NO_RESPONDE_MAX_SEG, NO_RESPONDE_BASE_SEG * 2 ** (self.sin_respuesta - 1))
            minutos = max(1, round(espera / 60))
            await self.estado('banco_no_responde', f'El banco no responde. Reintento en {minutos} min.')
            logger.info(f'[BDV] Cuenta {self.id}: el banco no responde, reintento en {espera} s')
            await self.esperar(espera)
            return False
        except LoginRechazado:
            self.fallos += 1
            logger.warning(f'[BDV] Cuenta {self.id}: el banco rechazó la entrada ({self.fallos}/{MAX_RECHAZOS})')
            if self.fallos >= MAX_RECHAZOS:
                await self.estado('detenido', bdv.MSG_DETENIDO, lector_fallos_login=self.fallos)
                self.info['estado'] = 'detenido'
                await self.cerrar(cerrar_sesion=False)
                return False
            await self.estado('rechazado', MSG_RECHAZADO, lector_fallos_login=self.fallos)
            await self.esperar(ESPERA_RECHAZO_SEG)
            return False
        finally:
            usuario = clave = None   # noqa: F841 — no quedan en memoria más de lo necesario
        self.fallos = 0
        self.sin_respuesta = 0
        await self.estado('conectado', f'Sesión abierta: leyendo tus movimientos cada {CADA_SEG} s', lector_fallos_login=0)
        logger.info(f'[BDV] Cuenta {self.id}: sesión abierta')
        return True


# ---------------------------------------------------------------------------
# Gestor de todas las cuentas
# ---------------------------------------------------------------------------

class Gestor:
    def __init__(self, browser, max_cuentas=MAX_CUENTAS):
        self.browser = browser
        self.max = max_cuentas
        self.lectores = {}
        self.con_lugar = set()
        self.instancia = secrets.token_hex(6)
        self.ultima_medida = 0.0
        self.memoria = None

    def pedir_lugar(self, lector):
        if lector.id in self.con_lugar:
            return True
        if len(self.con_lugar) >= self.max:
            return False
        self.con_lugar.add(lector.id)
        return True

    def soltar_lugar(self, lector):
        if lector.id in self.con_lugar:
            self.con_lugar.discard(lector.id)
            for otro in self.lectores.values():   # la que estaba esperando entra ya
                if otro.id not in self.con_lugar and otro.info.get('estado') == 'en_espera':
                    otro.aviso.set()

    async def sincronizar(self):
        cuentas = await asyncio.to_thread(leer_cuentas)
        vivas = {c['id']: c for c in cuentas if c['con_datos'] and c['activa']}
        nuevas = 0
        for cid, info in vivas.items():
            lector = self.lectores.get(cid)
            if lector is None or lector.tarea is None or lector.tarea.done():
                lector = LectorCuenta(self, info, retraso=nuevas * ESCALON_SEG)
                lector.tarea = asyncio.create_task(lector.correr())
                self.lectores[cid] = lector
                nuevas += 1
            else:
                lector.actualizar(info)
        for cid in [c for c in self.lectores if c not in vivas]:   # conexión borrada: «Salir» y fuera
            lector = self.lectores.pop(cid)
            lector.terminar = True
            lector.info['con_datos'] = False
            lector.aviso.set()

    async def medir(self):
        if time.monotonic() - self.ultima_medida < MEDIR_CADA_SEG:
            return
        self.ultima_medida = time.monotonic()
        abiertas = len(self.con_lugar)
        self.memoria = await asyncio.to_thread(memoria_mb)
        datos = {'latido': ad.ahora_local(), 'cuentas_leyendo': abiertas, 'max_cuentas': self.max,
                 'en_espera': sum(1 for x in self.lectores.values() if x.info.get('estado') == 'en_espera'),
                 'memoria_mb': self.memoria,
                 'por_cuenta_mb': round(self.memoria / abiertas, 1) if self.memoria and abiertas else None}
        await asyncio.to_thread(guardar_estado_lector, datos)
        logger.info(f'[BDV] Cuentas leyendo: {abiertas} · memoria del lector: {self.memoria} MB')

    async def correr(self):
        avisado = False
        while True:
            # Un solo lector a la vez. Si hay otro, este espera (no termina: con honcho/foreman, que un proceso
            # termine cerraría también la web).
            if not await asyncio.to_thread(tomar_turno, self.instancia):
                if not avisado:
                    logger.error('[BDV] Ya hay otro lector del BDV funcionando: este queda en espera')
                    avisado = True
                await self.cerrar_todo()
                self.lectores.clear()
                await asyncio.sleep(LATIDO_VENCE_SEG)
                continue
            avisado = False
            try:
                await self.sincronizar()
                await self.medir()
            except Exception as e:
                logger.error(f'[BDV] Error revisando las cuentas: {type(e).__name__}')
            await asyncio.sleep(REVISAR_SEG)

    async def cerrar_todo(self):
        for lector in list(self.lectores.values()):
            lector.terminar = True
            lector.aviso.set()
        tareas = [x.tarea for x in self.lectores.values() if x.tarea]
        if tareas:
            await asyncio.wait(tareas, timeout=60)


async def principal():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--disable-dev-shm-usage'])
        gestor = Gestor(browser)
        logger.info(f'[BDV] Lector del BDV arrancado (máximo {gestor.max} cuentas a la vez)')
        try:
            await gestor.correr()
        finally:
            await gestor.cerrar_todo()
            await browser.close()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    if not bdv.cifrado_disponible():
        # Se queda quieto en vez de terminar: con honcho/foreman, que un proceso termine cerraría también la web
        while True:
            logger.error('[BDV] Falta BDV_LLAVE_CIFRADO (o no es válida): el lector del BDV no lee nada. '
                         'Ponla en el .env y reinicia el lector.')
            time.sleep(600)
    ad.init_tablas()
    try:
        asyncio.run(principal())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

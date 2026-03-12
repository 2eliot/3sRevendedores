"""
Pin Redeemer - Automatiza el canje de pines en redeempins.com
Usa Playwright para controlar un navegador y completar el proceso de redención.

Flujo:
1. Navega a redeempins.com
2. Ingresa el código PIN
3. Hace clic en "Canjear"
4. Completa el formulario (nombre, fecha nacimiento, nacionalidad, ID jugador)
5. Acepta términos y condiciones
6. Hace clic en "Verificar ID"
7. Espera confirmación
"""

import asyncio
import logging
import os
import subprocess
import shutil
from datetime import datetime
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    async_playwright = None
    PlaywrightTimeout = None

logger = logging.getLogger(__name__)

_chromium_installed = False

def ensure_chromium_installed():
    """Instala Chromium de Playwright si no está disponible."""
    global _chromium_installed
    if _chromium_installed:
        return
    try:
        result = subprocess.run(
            ['playwright', 'install', 'chromium'],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            logger.info('[PinRedeemer] Chromium instalado/verificado correctamente')
        else:
            logger.warning(f'[PinRedeemer] playwright install chromium salió con código {result.returncode}: {result.stderr}')
    except Exception as e:
        logger.warning(f'[PinRedeemer] No se pudo instalar Chromium automáticamente: {e}')
    _chromium_installed = True

# Configuración por defecto para el formulario de redención
DEFAULT_REDEEMER_CONFIG = {
    'nombre_completo': 'Usuario Revendedor',
    'fecha_nacimiento': '01/01/1995',
    'nacionalidad': 'Chile',
    'url_base': 'https://redeem.hype.games/',
    'timeout_ms': 30000,
    'headless': True,
    'fast_mode': True,
    'typing_delay_ms': 0,
}


class PinRedeemResult:
    """Resultado de un intento de redención de pin"""
    def __init__(self, success, message, pin_code='', player_id='', screenshot_path=None, player_name=''):
        self.success = success
        self.message = message
        self.pin_code = pin_code
        self.player_id = player_id
        self.player_name = player_name
        self.screenshot_path = screenshot_path
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            'success': self.success,
            'message': self.message,
            'pin_code': self.pin_code,
            'player_id': self.player_id,
            'player_name': self.player_name,
            'screenshot_path': self.screenshot_path,
            'timestamp': self.timestamp
        }


async def redeem_pin_async(pin_code, player_id, config=None):
    """
    Redime un pin en redeempins.com de forma asíncrona.
    
    Args:
        pin_code: Código del pin (ej: "41D26298-7BF3-4308-A576-17CBEE8373BC")
        player_id: ID del jugador en Free Fire
        config: Dict con configuración (nombre_completo, fecha_nacimiento, nacionalidad, etc.)
    
    Returns:
        PinRedeemResult con el resultado de la operación
    """
    cfg = {**DEFAULT_REDEEMER_CONFIG, **(config or {})}
    
    # Forzar headless en servidores sin pantalla (Render, etc.)
    if not os.environ.get('DISPLAY'):
        cfg['headless'] = True
    
    logger.info(f"[PinRedeemer] Iniciando redencion - PIN: {pin_code[:8]}... Player: {player_id} (headless={cfg['headless']})")
    
    # Asegurar que Chromium esté instalado
    ensure_chromium_installed()
    
    async with async_playwright() as p:
        browser = None
        try:
            # Lanzar navegador con configuración anti-detección
            browser = await p.chromium.launch(
                headless=cfg['headless'],
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                ]
            )
            
            # Crear contexto con user agent real
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1366, 'height': 768},
                locale='es-CL',
                timezone_id='America/Santiago',
            )
            
            page = await context.new_page()

            # Ajustar timeouts por defecto
            try:
                page.set_default_timeout(cfg.get('timeout_ms', 30000))
                page.set_default_navigation_timeout(cfg.get('timeout_ms', 30000))
            except Exception:
                pass
            
            # Ocultar webdriver
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            """)
            
            timeout = cfg['timeout_ms']
            
            # ========== PASO 1: Navegar a redeem.hype.games ==========
            logger.info("[PinRedeemer] Paso 1: Navegando a redeem.hype.games")
            try:
                await page.goto(cfg['url_base'], wait_until='domcontentloaded', timeout=timeout)
            except Exception:
                # Reintentar una vez
                await page.wait_for_timeout(1000)
                await page.goto(cfg['url_base'], wait_until='domcontentloaded', timeout=timeout)
            await page.wait_for_timeout(200 if cfg.get('fast_mode') else 500)  # Esperar carga completa
            
            # ========== PASO 2: Ingresar el PIN ==========
            logger.info("[PinRedeemer] Paso 2: Ingresando PIN")
            
            # Buscar el campo de PIN por diferentes selectores
            pin_input = None
            pin_selectors = [
                'input[placeholder*="pin" i]',
                'input[name*="pin" i]',
                'input[id*="pin" i]',
                'input[type="text"]',
                'input:not([type="hidden"]):not([type="submit"])',
            ]
            
            for selector in pin_selectors:
                try:
                    pin_input = await page.wait_for_selector(selector, timeout=5000)
                    if pin_input:
                        break
                except PlaywrightTimeout:
                    continue
            
            if not pin_input:
                return PinRedeemResult(False, "No se encontro el campo de PIN en la pagina", pin_code, player_id)
            
            # Escribir el PIN de forma natural (con delay entre teclas)
            await pin_input.click()
            await pin_input.fill('')  # Limpiar primero
            # En modo rápido, usar fill directo (mucho más rápido y estable en servidores)
            if cfg.get('fast_mode'):
                await pin_input.fill(pin_code)
            else:
                await pin_input.type(pin_code, delay=int(cfg.get('typing_delay_ms', 5)))
            await page.wait_for_timeout(30 if cfg.get('fast_mode') else 100)
            
            # ========== PASO 3: Hacer clic en "Canjear" ==========
            logger.info("[PinRedeemer] Paso 3: Haciendo clic en Canjear")
            
            canjear_btn = None
            btn_selectors = [
                'button:has-text("Canjear")',
                'input[value*="Canjear" i]',
                'button[type="submit"]',
                'a:has-text("Canjear")',
            ]
            
            for selector in btn_selectors:
                try:
                    canjear_btn = await page.wait_for_selector(selector, timeout=5000)
                    if canjear_btn:
                        break
                except PlaywrightTimeout:
                    continue
            
            if not canjear_btn:
                return PinRedeemResult(False, "No se encontro el boton Canjear", pin_code, player_id)
            
            await page.wait_for_timeout(30 if cfg.get('fast_mode') else 100)
            await canjear_btn.click()
            
            # ========== PASO 4: Esperar redirección al formulario ==========
            logger.info("[PinRedeemer] Paso 4: Esperando redireccion al formulario")
            
            # Esperar a que la página cambie o aparezca el formulario
            try:
                await page.wait_for_load_state('networkidle', timeout=timeout)
                await page.wait_for_timeout(200 if cfg.get('fast_mode') else 500)
            except PlaywrightTimeout:
                pass
            
            # Verificar si hay un error (PIN inválido, ya usado, etc.)
            page_content = await page.content()
            error_keywords = ['error', 'invalido', 'inválido', 'no encontrado', 'ya fue', 'expirado', 'usado']
            page_text_lower = (await page.inner_text('body')).lower()
            
            for keyword in error_keywords:
                if keyword in page_text_lower and 'verificar' not in page_text_lower:
                    # Tomar screenshot del error
                    ss_path = f'static/redeem_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                    try:
                        await page.screenshot(path=ss_path)
                    except:
                        ss_path = None
                    return PinRedeemResult(False, f"Error en redeempins: PIN posiblemente invalido o ya usado", pin_code, player_id, ss_path)
            
            # ========== PASO 4.5: Cerrar popup de cookies si existe ==========
            logger.info("[PinRedeemer] Paso 4.5: Cerrando popup de cookies si existe")
            try:
                await page.evaluate('''() => {
                    // Buscar botones de aceptar cookies
                    const btns = Array.from(document.querySelectorAll('button, a'));
                    const keywords = ['acceptar', 'aceptar', 'accept', 'aceitar', 'ok', 'agree', 'entendido', 'continuar'];
                    for (const btn of btns) {
                        const text = (btn.textContent || '').toLowerCase().trim();
                        for (const kw of keywords) {
                            if (text.includes(kw)) {
                                btn.click();
                                return 'Cookie popup closed: ' + text;
                            }
                        }
                    }
                    // Buscar por clase común de cookie banners
                    const cookieBanners = document.querySelectorAll('[class*="cookie"], [class*="consent"], [id*="cookie"], [id*="consent"]');
                    for (const banner of cookieBanners) {
                        const acceptBtn = banner.querySelector('button');
                        if (acceptBtn) {
                            acceptBtn.click();
                            return 'Cookie banner button clicked';
                        }
                    }
                    return 'No cookie popup found';
                }''')
                await page.wait_for_timeout(200)
            except:
                pass
            
            # ========== PASO 5: Completar formulario de datos ==========
            logger.info("[PinRedeemer] Paso 5: Completando formulario de datos")
            
            # Esperar que el formulario cargue
            await page.wait_for_timeout(120 if cfg.get('fast_mode') else 300)
            
            # Scroll al inicio de la página para asegurar visibilidad
            await page.evaluate('window.scrollTo(0, 0)')
            await page.wait_for_timeout(50 if cfg.get('fast_mode') else 100)
            
            fecha_valor = cfg['fecha_nacimiento']  # formato DD/MM/YYYY
            fecha_iso = fecha_valor
            try:
                parts = fecha_valor.split('/')
                if len(parts) == 3:
                    fecha_iso = f"{parts[2]}-{parts[1]}-{parts[0]}"
            except:
                pass
            
            # Mapeo de nacionalidades a códigos ISO alpha-2
            country_codes = {
                'chile': 'CL', 'argentina': 'AR', 'colombia': 'CO', 'mexico': 'MX',
                'méxico': 'MX', 'peru': 'PE', 'perú': 'PE', 'venezuela': 'VE',
                'ecuador': 'EC', 'bolivia': 'BO', 'uruguay': 'UY', 'paraguay': 'PY',
                'brasil': 'BR', 'brazil': 'BR', 'panama': 'PA', 'panamá': 'PA',
                'costa rica': 'CR', 'guatemala': 'GT', 'honduras': 'HN',
                'el salvador': 'SV', 'nicaragua': 'NI', 'cuba': 'CU',
                'republica dominicana': 'DO', 'puerto rico': 'PR',
            }
            nac_code = country_codes.get(cfg['nacionalidad'].lower(), cfg['nacionalidad'])
            
            # Rellenar formulario usando IDs exactos de Hype Games
            fill_result = await page.evaluate('''(data) => {
                const results = [];
                
                function setVal(el, val) {
                    el.scrollIntoView({ behavior: 'instant', block: 'center' });
                    el.focus();
                    // Usar nativeInputValueSetter para frameworks reactivos
                    const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    nativeSetter.call(el, val);
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    // Simular keyup para triggers de jQuery
                    el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
                    el.dispatchEvent(new Event('blur', { bubbles: true }));
                    // Marcar como filled para el CSS de Hype
                    const formItem = el.closest('.form-item');
                    if (formItem) formItem.classList.add('filled');
                }
                
                // 1. Nombre Completo (#Name)
                const nameEl = document.getElementById('Name');
                if (nameEl) {
                    setVal(nameEl, data.nombre);
                    results.push('Nombre: OK -> #Name');
                } else {
                    results.push('Nombre: ERROR - #Name no encontrado');
                }
                
                // 2. Fecha de Nacimiento (#BornAt) - formato DD/MM/YYYY
                const bornEl = document.getElementById('BornAt');
                if (bornEl) {
                    setVal(bornEl, data.fecha);
                    results.push('Fecha: OK -> #BornAt = ' + data.fecha);
                } else {
                    results.push('Fecha: ERROR - #BornAt no encontrado');
                }
                
                // 3. Nacionalidad (#NationalityAlphaCode) - usa códigos ISO
                const nacSel = document.getElementById('NationalityAlphaCode');
                if (nacSel) {
                    nacSel.scrollIntoView({ behavior: 'instant', block: 'center' });
                    nacSel.value = data.nac_code;
                    nacSel.dispatchEvent(new Event('change', { bubbles: true }));
                    results.push('Nacionalidad: OK -> ' + data.nac_code);
                } else {
                    results.push('Nacionalidad: ERROR - #NationalityAlphaCode no encontrado');
                }
                
                // 4. Player ID (#GameAccountId)
                const gameEl = document.getElementById('GameAccountId');
                if (gameEl) {
                    setVal(gameEl, data.player_id);
                    // Trigger keyup para checkValidationForm() de Hype
                    gameEl.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
                    results.push('Player ID: OK -> #GameAccountId = ' + data.player_id);
                } else {
                    results.push('Player ID: ERROR - #GameAccountId no encontrado');
                }
                
                // 5. Checkbox de privacidad (#privacy)
                const privacyEl = document.getElementById('privacy');
                if (privacyEl && !privacyEl.checked) {
                    privacyEl.click();
                    results.push('Checkbox: OK -> #privacy');
                } else if (privacyEl) {
                    results.push('Checkbox: ya marcado');
                }
                
                // 6. Habilitar botón Verificar ID
                const verifyBtn = document.getElementById('btn-verify');
                if (verifyBtn) {
                    verifyBtn.removeAttribute('disabled');
                    results.push('Btn Verificar: habilitado');
                }
                
                return results;
            }''', {
                'nombre': cfg['nombre_completo'],
                'fecha': fecha_valor,
                'nac_code': nac_code,
                'player_id': str(player_id)
            })
            
            for r in fill_result:
                logger.info(f"[PinRedeemer] JS: {r}")
            
            await page.wait_for_timeout(50 if cfg.get('fast_mode') else 100)
            
            # ========== PASO 6: Checkbox ya marcado en JS del PASO 5 ==========
            logger.info("[PinRedeemer] Paso 6: Checkbox ya procesado en JS")
            
            # ========== PASO 7: Clic en "Verificar ID" (#btn-verify) ==========
            logger.info("[PinRedeemer] Paso 7: Haciendo clic en Verificar ID")
            
            await page.wait_for_timeout(80 if cfg.get('fast_mode') else 200)
            
            # Clic en #btn-verify via JS
            await page.evaluate('''() => {
                const btn = document.getElementById('btn-verify');
                if (btn) {
                    btn.removeAttribute('disabled');
                    btn.scrollIntoView({ behavior: 'instant', block: 'center' });
                    btn.click();
                }
            }''')
            logger.info("[PinRedeemer] Clic en #btn-verify enviado")
            
            # ========== PASO 7.5: Esperar verificación AJAX ==========
            logger.info("[PinRedeemer] Paso 7.5: Esperando verificacion AJAX...")
            
            # Esperar a que aparezca el botón "Canjear Ahora" (#btn-redeem) visible
            # o que aparezca el nombre del jugador (.redeem-data visible)
            redeem_ready = False
            captured_player_name = ''
            for attempt in range(20):  # Máximo 10 segundos
                await page.wait_for_timeout(500)
                
                check = await page.evaluate('''() => {
                    const redeemBtn = document.getElementById('btn-redeem');
                    const redeemData = document.querySelector('.redeem-data');
                    const playerName = document.getElementById('btn-player-game-data');
                    const errorEl = document.querySelector('.error-message, .alert-danger, .text-danger');
                    
                    // Verificar si hay error
                    if (errorEl && errorEl.offsetParent !== null && errorEl.textContent.trim().length > 5) {
                        return { status: 'error', message: errorEl.textContent.trim(), player_name: '' };
                    }
                    
                    // Verificar si el botón Canjear está visible
                    if (redeemBtn && redeemData && redeemData.style.display !== 'none') {
                        const name = playerName ? playerName.textContent.trim() : '';
                        return { status: 'ready', message: 'Player: ' + name, player_name: name };
                    }
                    
                    // Verificar si el botón verify sigue en loading
                    const verifyBtn = document.getElementById('btn-verify');
                    if (verifyBtn && verifyBtn.classList.contains('loading')) {
                        return { status: 'loading', message: 'Verificando...', player_name: '' };
                    }
                    
                    return { status: 'waiting', message: 'Esperando...', player_name: '' };
                }''')
                
                logger.info(f"[PinRedeemer] Verificacion [{attempt+1}]: {check['status']} - {check['message']}")
                
                if check['status'] == 'ready':
                    redeem_ready = True
                    captured_player_name = check.get('player_name', '')
                    logger.info(f"[PinRedeemer] Nombre del jugador capturado: {captured_player_name}")
                    break
                elif check['status'] == 'error':
                    ss_path = f'static/redeem_verify_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                    try:
                        await page.screenshot(path=ss_path)
                    except:
                        ss_path = None
                    return PinRedeemResult(False, f"Error en verificacion: {check['message']}", pin_code, player_id, ss_path)
            
            if not redeem_ready:
                ss_path = f'static/redeem_timeout_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                try:
                    await page.screenshot(path=ss_path)
                except:
                    ss_path = None
                return PinRedeemResult(False, "Timeout esperando verificacion de ID", pin_code, player_id, ss_path)
            
            # ========== PASO 8: Clic en "Canjear Ahora" (#btn-redeem) ==========
            logger.info("[PinRedeemer] Paso 8: Haciendo clic en Canjear Ahora!")
            
            await page.wait_for_timeout(50 if cfg.get('fast_mode') else 100)
            
            await page.evaluate('''() => {
                const btn = document.getElementById('btn-redeem');
                if (btn) {
                    btn.removeAttribute('disabled');
                    btn.scrollIntoView({ behavior: 'instant', block: 'center' });
                    btn.click();
                }
            }''')
            logger.info("[PinRedeemer] Clic en #btn-redeem enviado")
            
            # ========== PASO 9: Esperar confirmación final ==========
            logger.info("[PinRedeemer] Paso 9: Esperando confirmacion final")
            
            url_before_redeem = page.url

            try:
                await page.wait_for_load_state('networkidle', timeout=timeout)
                await page.wait_for_timeout(1500)
            except PlaywrightTimeout:
                pass

            # Dar un margen extra para que aparezcan modales/toasts de confirmación
            await page.wait_for_timeout(2000)

            # Polling: esperar señales claras de éxito/error o cambios de estado.
            # En Render a veces no hay navegación, solo un toast o cambio de UI.
            final_state = None
            for attempt in range(20):  # ~10s
                final_state = await page.evaluate('''(urlBefore) => {
                    const isVisible = (el) => {
                        if (!el) return false;
                        const style = window.getComputedStyle(el);
                        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                        const r = el.getBoundingClientRect();
                        return r.width > 0 && r.height > 0;
                    };

                    const successSelectors = [
                        '.alert-success',
                        '.success-message',
                        '.text-success',
                        '.swal2-icon.swal2-success',
                        '.swal2-success',
                        '[role="alert"][class*="success" i]'
                    ];
                    const errorSelectors = [
                        '.error-message',
                        '.alert-danger',
                        '.text-danger',
                        '.swal2-icon.swal2-error',
                        '.swal2-error',
                        '[role="alert"][class*="danger" i]',
                        '[role="alert"][class*="error" i]'
                    ];

                    for (const sel of successSelectors) {
                        const el = document.querySelector(sel);
                        if (isVisible(el) && (el.textContent || '').trim().length > 0) {
                            return { status: 'success', message: (el.textContent || '').trim() };
                        }
                    }
                    for (const sel of errorSelectors) {
                        const el = document.querySelector(sel);
                        if (isVisible(el) && (el.textContent || '').trim().length > 0) {
                            return { status: 'error', message: (el.textContent || '').trim() };
                        }
                    }

                    const redeemBtn = document.getElementById('btn-redeem');
                    const redeemDisabled = redeemBtn ? redeemBtn.hasAttribute('disabled') : null;
                    const redeemLoading = redeemBtn ? redeemBtn.classList.contains('loading') : null;

                    const urlChanged = (window.location && window.location.href) ? (window.location.href !== urlBefore) : false;

                    const bodyText = (document.body && document.body.innerText) ? document.body.innerText : '';
                    const snippet = bodyText.replace(/\s+/g, ' ').trim().slice(0, 220);

                    return {
                        status: 'waiting',
                        message: snippet,
                        redeem_disabled: redeemDisabled,
                        redeem_loading: redeemLoading,
                        url_changed: urlChanged
                    };
                }''', url_before_redeem)

                logger.info(
                    f"[PinRedeemer] Confirmacion [{attempt+1}]: {final_state.get('status')} | "
                    f"disabled={final_state.get('redeem_disabled')} loading={final_state.get('redeem_loading')} url_changed={final_state.get('url_changed')}"
                )

                if final_state.get('status') in ('success', 'error'):
                    break

                # Heurística: si el botón quedó disabled o en loading y el body menciona 'gracias'/'success',
                # muchas veces el sitio ya completó el canje aunque no haya alert visible.
                msg_lower = (final_state.get('message') or '').lower()
                if (final_state.get('redeem_disabled') or final_state.get('url_changed')) and any(k in msg_lower for k in ('gracias', 'success', 'exito', 'éxito', 'completad')):
                    final_state = { 'status': 'success', 'message': final_state.get('message', '') }
                    break

                await page.wait_for_timeout(500)
            
            final_text = ((final_state or {}).get('message', '') or '').lower()
            
            # Tomar screenshot del resultado
            ss_path = f'static/redeem_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            try:
                await page.screenshot(path=ss_path, full_page=True)
            except:
                ss_path = None

            # Si detectamos error por selector visible, fallar explícitamente
            if final_state.get('status') == 'error':
                logger.error(f"[PinRedeemer] ERROR - Confirmacion final indica error visible: {final_state.get('message', '')}")
                return PinRedeemResult(False, f"Error en confirmacion final: {final_state.get('message', '')}", pin_code, player_id, ss_path)

            # Si detectamos éxito por selector visible, confirmar éxito
            if final_state.get('status') == 'success':
                logger.info(f"[PinRedeemer] EXITO - Confirmacion final indica exito visible")
                return PinRedeemResult(True, f"Pin canjeado exitosamente para jugador {player_id}", pin_code, player_id, ss_path, captured_player_name)
            
            # Buscar indicadores de éxito
            success_keywords = ['exitoso', 'exitosa', 'completado', 'completada', 'exito', 'éxito', 
                              'success', 'gracias', 'recarga', 'confirmado', 'confirmada', 'entregado']
            
            for keyword in success_keywords:
                if keyword in final_text:
                    logger.info(f"[PinRedeemer] EXITO - Pin redencion completada para player {player_id} ({captured_player_name})")
                    return PinRedeemResult(True, f"Pin canjeado exitosamente para jugador {player_id}", pin_code, player_id, ss_path, captured_player_name)
            
            # Si no detectamos éxito ni error, reportar como pendiente de verificación
            logger.warning("[PinRedeemer] Resultado no determinado - verificar screenshot")
            return PinRedeemResult(True, "Proceso completado - verificar resultado en screenshot", pin_code, player_id, ss_path, captured_player_name)
            
        except PlaywrightTimeout as e:
            logger.error(f"[PinRedeemer] Timeout: {str(e)}")
            return PinRedeemResult(False, f"Timeout durante el proceso: {str(e)}", pin_code, player_id)
        except Exception as e:
            logger.error(f"[PinRedeemer] Error inesperado: {str(e)}")
            return PinRedeemResult(False, f"Error inesperado: {str(e)}", pin_code, player_id)
        finally:
            if browser:
                await browser.close()


def redeem_pin(pin_code, player_id, config=None):
    """
    Wrapper síncrono para redeem_pin_async.
    Puede llamarse desde código síncrono (Flask).
    
    Args:
        pin_code: Código del pin
        player_id: ID del jugador en Free Fire
        config: Dict con configuración opcional
    
    Returns:
        PinRedeemResult
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(redeem_pin_async(pin_code, player_id, config))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"[PinRedeemer] Error en wrapper sincrono: {str(e)}")
        return PinRedeemResult(False, f"Error interno: {str(e)}", pin_code, player_id)


def redeem_pin_threaded(pin_code, player_id, config=None, callback=None):
    """
    Ejecuta la redención en un hilo separado para no bloquear Flask.
    
    Args:
        pin_code: Código del pin
        player_id: ID del jugador en Free Fire
        config: Dict con configuración opcional
        callback: Función a llamar con el PinRedeemResult cuando termine
    """
    import threading
    
    def _run():
        result = redeem_pin(pin_code, player_id, config)
        if callback:
            callback(result)
    
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


# ========== Configuración desde base de datos ==========

def get_redeemer_config_from_db(db_connection_func):
    """
    Obtiene la configuración del redeemer desde la base de datos.
    
    Args:
        db_connection_func: Función que retorna una conexión a la DB
    """
    config = dict(DEFAULT_REDEEMER_CONFIG)
    try:
        conn = db_connection_func()
        row = conn.execute('''
            SELECT clave, valor FROM configuracion_redeemer
        ''').fetchall()
        conn.close()
        
        for r in row:
            config[r['clave']] = r['valor']
        
        # Convertir headless a bool
        if isinstance(config.get('headless'), str):
            config['headless'] = config['headless'].lower() in ('true', '1', 'yes')
        
        # Convertir timeout a int
        if isinstance(config.get('timeout_ms'), str):
            config['timeout_ms'] = int(config['timeout_ms'])
            
    except Exception as e:
        logger.warning(f"[PinRedeemer] No se pudo leer config de DB, usando defaults: {e}")
    
    return config


# ========== Test/CLI ==========

if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if len(sys.argv) < 3:
        print("Uso: python pin_redeemer.py <PIN_CODE> <PLAYER_ID> [--visible]")
        print("Ejemplo: python pin_redeemer.py 41D26298-7BF3-4308-A576-17CBEE8373BC 123456789")
        sys.exit(1)
    
    pin = sys.argv[1]
    pid = sys.argv[2]
    
    cfg = dict(DEFAULT_REDEEMER_CONFIG)
    if '--visible' in sys.argv:
        cfg['headless'] = False
    
    print(f"Redimiendo PIN: {pin}")
    print(f"Player ID: {pid}")
    print(f"Headless: {cfg['headless']}")
    print("-" * 50)
    
    result = redeem_pin(pin, pid, cfg)
    
    print("-" * 50)
    print(f"Resultado: {'EXITO' if result.success else 'FALLO'}")
    print(f"Mensaje: {result.message}")
    if result.screenshot_path:
        print(f"Screenshot: {result.screenshot_path}")

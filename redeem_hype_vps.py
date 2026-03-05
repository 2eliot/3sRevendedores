"""
Redención de PIN via VPS remoto con Playwright.
Tu web solo envía PIN + Player ID al VPS, y el VPS hace todo el trabajo pesado
(navegador, captcha, redención). Respuesta en ~20s.
"""
import logging
import os

import requests

from pin_redeemer import PinRedeemResult

logger = logging.getLogger(__name__)

VPS_DEFAULT_URL = "http://74.208.158.70:5000/redeem"
VPS_TIMEOUT_S = 120


def redeem_pin_vps(pin_code, player_id, config=None, request_id=None):
    """
    Envía PIN + Player ID al VPS y recoge el resultado.
    El VPS ejecuta Playwright + captcha y devuelve éxito/error.
    """
    cfg = dict(config or {})
    vps_url = cfg.get("vps_url") or os.environ.get("VPS_REDEEM_URL", VPS_DEFAULT_URL)
    timeout = int(cfg.get("vps_timeout_s") or os.environ.get("VPS_TIMEOUT_S", VPS_TIMEOUT_S))

    nombre = cfg.get("nombre_completo") or cfg.get("nombre_cliente") or "Usuario Revendedor"
    born_at = cfg.get("fecha_nacimiento") or "01/01/1995"

    # Validación mínima antes de enviar
    if not pin_code or len(str(pin_code).strip()) < 10:
        return PinRedeemResult(False, "PIN inválido o vacío", pin_code, player_id)
    if not player_id or not str(player_id).strip().isdigit():
        return PinRedeemResult(False, "Player ID inválido (debe ser numérico)", pin_code, player_id)

    country = cfg.get("pais") or cfg.get("country") or "Chile"

    payload = {
        "pin_key": str(pin_code).strip(),
        "player_id": str(player_id).strip(),
        "full_name": nombre,
        "birth_date": born_at,
        "country": country,
    }
    if request_id:
        payload["request_id"] = str(request_id)

    logger.info(f"[VPS] Enviando PIN {pin_code[:8]}... + ID {player_id} a {vps_url}")

    try:
        resp = requests.post(
            vps_url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.Timeout:
        logger.error(f"[VPS] Timeout ({timeout}s) esperando respuesta del VPS")
        return PinRedeemResult(False, f"El VPS no respondió en {timeout}s. Reintenta.", pin_code, player_id)
    except requests.exceptions.ConnectionError:
        logger.error(f"[VPS] No se pudo conectar al VPS en {vps_url}")
        return PinRedeemResult(False, "No se pudo conectar al VPS. Verifica que esté encendido.", pin_code, player_id)
    except Exception as e:
        logger.error(f"[VPS] Error inesperado: {e}")
        return PinRedeemResult(False, f"Error de conexión: {str(e)}", pin_code, player_id)

    logger.info(f"[VPS] Respuesta HTTP {resp.status_code}")

    # Parsear respuesta JSON del VPS
    try:
        data = resp.json()
    except Exception:
        logger.warning(f"[VPS] Respuesta no-JSON: {resp.text[:300]}")
        if resp.status_code == 200:
            return PinRedeemResult(True, "Recarga procesada (VPS)", pin_code, player_id)
        return PinRedeemResult(False, f"VPS HTTP {resp.status_code}: respuesta inválida", pin_code, player_id)

    logger.info(f"[VPS] Respuesta: {data}")

    # Interpretar resultado del VPS
    exito = data.get("success") or data.get("exito") or data.get("status") == "ok"
    mensaje = data.get("message") or data.get("mensaje") or data.get("error") or ""
    player_name = data.get("player_name") or data.get("nombre_jugador") or ""

    if exito:
        logger.info(f"[VPS] Redención exitosa! Player: {player_name}")
        return PinRedeemResult(True, mensaje or "Recarga completada (VPS)", pin_code, player_id, player_name=player_name)
    else:
        logger.warning(f"[VPS] Redención fallida: {mensaje}")
        return PinRedeemResult(False, mensaje or f"Error del VPS (HTTP {resp.status_code})", pin_code, player_id)

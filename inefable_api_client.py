import requests
import os
import logging
from datetime import datetime
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InefableAPIClient:
    """Cliente para conectar con la API externa de Inefable Shop"""
    
    def __init__(self):
        # Configuración de la API externa
        self.base_url = "https://inefableshop.net/conexion_api/api.php"
        self.usuario = os.environ.get('INEFABLE_USUARIO', 'inefableshop')
        self.clave = os.environ.get('INEFABLE_CLAVE', '')
        
        # Mapeo de monto_id local a monto de la API externa
        self.monto_mapping = {
            1: 1,  # 110 💎 -> monto 1
            2: 2,  # 341 💎 -> monto 2
            3: 3,  # 572 💎 -> monto 3
            4: 4,  # 1.166 💎 -> monto 4
            5: 5,  # 2.376 💎 -> monto 5
            6: 6,  # 6.138 💎 -> monto 6
            7: 7,  # Tarjeta básica -> monto 7
            8: 8,  # Tarjeta semanal -> monto 8
            9: 9   # Tarjeta mensual -> monto 9
        }
        
        # Timeout para las peticiones
        self.timeout = 30
        
    def _make_request(self, params):
        """Realiza una petición a la API externa"""
        try:
            logger.info(f"Realizando petición a API externa con parámetros: {params}")
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'InefablePines/1.0',
                    'Accept': 'application/json, text/plain, */*'
                }
            )
            
            logger.info(f"Respuesta de API externa - Status: {response.status_code}")
            logger.info(f"Respuesta de API externa - Content: {response.text[:500]}...")
            
            response.raise_for_status()
            
            # Intentar parsear como JSON, si falla devolver texto plano
            try:
                return response.json()
            except json.JSONDecodeError:
                return {
                    'status': 'success' if response.status_code == 200 else 'error',
                    'data': response.text,
                    'raw_response': True
                }
                
        except requests.exceptions.Timeout:
            logger.error("Timeout al conectar con API externa")
            return {
                'status': 'error',
                'message': 'Timeout al conectar con la API externa',
                'error_type': 'timeout'
            }
        except requests.exceptions.ConnectionError:
            logger.error("Error de conexión con API externa")
            return {
                'status': 'error',
                'message': 'Error de conexión con la API externa',
                'error_type': 'connection'
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en petición a API externa: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error en petición: {str(e)}',
                'error_type': 'request'
            }
        except Exception as e:
            logger.error(f"Error inesperado en API externa: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error inesperado: {str(e)}',
                'error_type': 'unexpected'
            }
    
    def test_connection(self):
        """Prueba la conexión con la API externa"""
        params = {
            'action': 'recarga',
            'usuario': self.usuario,
            'clave': self.clave,
            'tipo': 'recargaPinFreefirebs',
            'monto': 1,
            'numero': 0  # Número de prueba
        }
        
        result = self._make_request(params)
        
        if result.get('status') == 'success':
            logger.info("Conexión con API externa exitosa")
            return True, "Conexión exitosa con API externa"
        else:
            logger.error(f"Error en conexión con API externa: {result.get('message', 'Error desconocido')}")
            return False, result.get('message', 'Error desconocido')
    
    def request_pin(self, monto_id, numero_destino=0):
        """
        Solicita un pin de Free Fire a la API externa
        
        Args:
            monto_id (int): ID del monto local (1-9)
            numero_destino (int): Número de destino (0 para pines)
            
        Returns:
            dict: Resultado de la operación
        """
        try:
            # Validar monto_id
            if monto_id not in self.monto_mapping:
                return {
                    'status': 'error',
                    'message': f'Monto ID {monto_id} no válido. Debe estar entre 1 y 9.',
                    'error_type': 'validation'
                }
            
            # Mapear monto_id local al monto de la API externa
            monto_externo = self.monto_mapping[monto_id]
            
            params = {
                'action': 'recarga',
                'usuario': self.usuario,
                'clave': self.clave,
                'tipo': 'recargaPinFreefirebs',
                'monto': monto_externo,
                'numero': numero_destino
            }
            
            logger.info(f"Solicitando pin para monto_id {monto_id} (monto externo: {monto_externo})")
            
            result = self._make_request(params)
            
            if result.get('status') == 'success':
                # Procesar respuesta exitosa
                pin_data = self._process_pin_response(result, monto_id)
                logger.info(f"Pin obtenido exitosamente para monto_id {monto_id}")
                return pin_data
            else:
                logger.error(f"Error al obtener pin: {result.get('message', 'Error desconocido')}")
                return result
                
        except Exception as e:
            logger.error(f"Error inesperado al solicitar pin: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error inesperado: {str(e)}',
                'error_type': 'unexpected'
            }
    
    def _process_pin_response(self, response, monto_id):
        """Procesa la respuesta de la API externa para extraer el pin"""
        try:
            # Si la respuesta es texto plano (raw_response)
            if response.get('raw_response'):
                response_text = response.get('data', '')
                response_text_lower = response_text.lower()
                
                logger.info(f"Procesando respuesta de API externa para monto_id {monto_id}")
                logger.info(f"Respuesta recibida: {response_text[:200]}...")
                
                # Verificar si la respuesta contiene JSON con error
                try:
                    # Intentar extraer JSON de la respuesta (puede tener HTML antes)
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        json_data = json.loads(json_str)
                        
                        # Verificar si el JSON indica error
                        if json_data.get('alerta') == 'red' or 'error' in json_data.get('mensaje', '').lower():
                            logger.error(f"API externa devolvió error JSON: {json_data.get('mensaje', 'Error desconocido')}")
                            return {
                                'status': 'error',
                                'message': f'Sin stock en API externa: {json_data.get("mensaje", "Error desconocido")}',
                                'raw_response': response_text,
                                'error_type': 'no_stock'
                            }
                        
                        # Buscar pin en diferentes campos del JSON
                        pin_code = None
                        
                        # 1. Buscar en campo 'pin' directo
                        if json_data.get('pin') and json_data.get('pin') != 'null' and json_data.get('pin') is not None:
                            pin_code = json_data.get('pin')
                            logger.info(f"Pin encontrado en campo 'pin': {pin_code[:4]}****")
                        
                        # 2. Buscar en campo 'codigo'
                        elif json_data.get('codigo') and json_data.get('codigo') != '':
                            pin_code = json_data.get('codigo')
                            logger.info(f"Pin encontrado en campo 'codigo': {pin_code[:4]}****")
                        
                        # 3. Buscar en campo 'mensaje' (HTML)
                        elif json_data.get('mensaje'):
                            mensaje = json_data.get('mensaje', '')
                            # Extraer pin del HTML usando regex
                            import re
                            pin_match = re.search(r'<b>Pin:</b>\s*([A-Z0-9]{6,20})', mensaje, re.IGNORECASE)
                            if pin_match:
                                pin_code = pin_match.group(1).strip()
                                logger.info(f"Pin encontrado en mensaje HTML: {pin_code[:4]}****")
                        
                        if pin_code:
                            return {
                                'status': 'success',
                                'pin_code': pin_code,
                                'monto_id': monto_id,
                                'source': 'inefable_api',
                                'timestamp': datetime.now().isoformat(),
                                'raw_response': response_text
                            }
                except (json.JSONDecodeError, AttributeError):
                    logger.info("No se encontró JSON válido en la respuesta")
                
                # Verificar indicadores de falta de stock en texto plano
                no_stock_keywords = [
                    'sin stock', 'no stock', 'agotado', 'no disponible',
                    'out of stock', 'unavailable', 'insufficient',
                    'no hay', 'temporalmente no disponible', 'error',
                    'no se pudo', 'fallido', 'failed', 'saldo insuficiente',
                    'balance insuficiente', 'no funds', 'insufficient funds',
                    'alerta":"red"', 'error desconocido', 'respuesta inválida'
                ]
                
                # Si hay indicadores de falta de stock, devolver error
                for keyword in no_stock_keywords:
                    if keyword in response_text_lower:
                        logger.error(f"Detectado indicador de falta de stock: {keyword}")
                        return {
                            'status': 'error',
                            'message': f'Sin stock en API externa: {keyword}',
                            'raw_response': response_text,
                            'error_type': 'no_stock'
                        }
                
                # Buscar patrones comunes de pines en la respuesta (solo si no hay errores)
                pin_code = self._extract_pin_from_text(response_text)
                
                if pin_code:
                    logger.info(f"Pin extraído de texto: {pin_code[:4]}****")
                    return {
                        'status': 'success',
                        'pin_code': pin_code,
                        'monto_id': monto_id,
                        'source': 'inefable_api',
                        'timestamp': datetime.now().isoformat(),
                        'raw_response': response_text
                    }
                else:
                    logger.error("No se pudo extraer pin válido de la respuesta")
                    return {
                        'status': 'error',
                        'message': 'No se pudo extraer el pin de la respuesta - posible falta de stock',
                        'raw_response': response_text,
                        'error_type': 'no_pin_found'
                    }
            
            # Si la respuesta es JSON estructurado
            elif isinstance(response.get('data'), dict):
                data = response['data']
                
                # Verificar si hay indicadores de error o falta de stock en JSON
                if data.get('error') or data.get('status') == 'error':
                    return {
                        'status': 'error',
                        'message': f'Error en API externa: {data.get("message", "Error desconocido")}',
                        'api_response': data,
                        'error_type': 'api_error'
                    }
                
                pin_code = data.get('pin') or data.get('codigo') or data.get('pin_code')
                
                if pin_code:
                    return {
                        'status': 'success',
                        'pin_code': pin_code,
                        'monto_id': monto_id,
                        'source': 'inefable_api',
                        'timestamp': datetime.now().isoformat(),
                        'api_response': data
                    }
                else:
                    return {
                        'status': 'error',
                        'message': 'Pin no encontrado en respuesta JSON - posible falta de stock',
                        'api_response': data,
                        'error_type': 'no_pin_in_json'
                    }
            
            else:
                return {
                    'status': 'error',
                    'message': 'Formato de respuesta no reconocido',
                    'response': response,
                    'error_type': 'format_error'
                }
                
        except Exception as e:
            logger.error(f"Error al procesar respuesta de pin: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error al procesar respuesta: {str(e)}',
                'error_type': 'processing_error'
            }
    
    def _extract_pin_from_text(self, text):
        """Extrae el código de pin de un texto de respuesta"""
        import re
        
        # Patrones comunes para códigos de pin
        patterns = [
            r'PIN[:\s]*([A-Z0-9]{10,20})',  # PIN: XXXXXXXXXX
            r'CODIGO[:\s]*([A-Z0-9]{10,20})',  # CODIGO: XXXXXXXXXX
            r'CODE[:\s]*([A-Z0-9]{10,20})',  # CODE: XXXXXXXXXX
            r'([A-Z0-9]{12,16})',  # Código alfanumérico de 12-16 caracteres
            r'Pin[:\s]*([A-Z0-9]{10,20})',  # Pin: XXXXXXXXXX
            r'Código[:\s]*([A-Z0-9]{10,20})',  # Código: XXXXXXXXXX
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                pin_code = match.group(1).strip()
                # Validar que el pin tenga una longitud razonable
                if 10 <= len(pin_code) <= 20:
                    logger.info(f"Pin extraído exitosamente: {pin_code[:4]}****{pin_code[-4:]}")
                    return pin_code
        
        logger.warning("No se pudo extraer pin del texto de respuesta")
        return None
    
    def get_balance(self):
        """Obtiene el saldo disponible en la API externa (si está disponible)"""
        try:
            # Algunos proveedores tienen endpoint para consultar saldo
            params = {
                'action': 'saldo',
                'usuario': self.usuario,
                'clave': self.clave
            }
            
            result = self._make_request(params)
            
            if result.get('status') == 'success':
                return result
            else:
                # Si no hay endpoint de saldo, devolver información básica
                return {
                    'status': 'info',
                    'message': 'Endpoint de saldo no disponible',
                    'connection_status': 'active'
                }
                
        except Exception as e:
            logger.error(f"Error al obtener saldo de API externa: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error al obtener saldo: {str(e)}',
                'error_type': 'balance_error'
            }
    
    def check_stock_availability(self, monto_id):
        """
        Verifica si hay stock disponible para un monto específico en la API externa
        
        Args:
            monto_id (int): ID del monto local (1-9)
            
        Returns:
            dict: Estado del stock
        """
        try:
            # Validar monto_id
            if monto_id not in self.monto_mapping:
                return {
                    'status': 'error',
                    'available': False,
                    'message': f'Monto ID {monto_id} no válido'
                }
            
            # Hacer una petición de prueba para verificar disponibilidad
            monto_externo = self.monto_mapping[monto_id]
            
            params = {
                'action': 'recarga',
                'usuario': self.usuario,
                'clave': self.clave,
                'tipo': 'recargaPinFreefirebs',
                'monto': monto_externo,
                'numero': 0  # Número de prueba
            }
            
            result = self._make_request(params)
            
            if result.get('status') == 'success':
                # Analizar la respuesta para determinar disponibilidad
                response_text = str(result.get('data', '')).lower()
                
                # Palabras clave que indican falta de stock
                no_stock_keywords = [
                    'sin stock', 'no stock', 'agotado', 'no disponible',
                    'out of stock', 'unavailable', 'insufficient',
                    'no hay', 'temporalmente no disponible', 'error',
                    'no se pudo', 'fallido', 'failed'
                ]
                
                # Verificar si hay indicadores de falta de stock
                has_stock = True
                for keyword in no_stock_keywords:
                    if keyword in response_text:
                        has_stock = False
                        break
                
                # Si la respuesta contiene un código de pin válido, asumir que hay stock
                if not has_stock:
                    # Buscar patrones de pin para confirmar
                    import re
                    pin_patterns = [
                        r'[A-Z0-9]{10,20}',  # Código alfanumérico
                        r'PIN[:\s]*[A-Z0-9]+',
                        r'CODIGO[:\s]*[A-Z0-9]+'
                    ]
                    
                    for pattern in pin_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            has_stock = True
                            break
                
                return {
                    'status': 'success',
                    'available': has_stock,
                    'message': 'Stock disponible' if has_stock else 'Sin stock en API externa',
                    'monto_id': monto_id
                }
            else:
                # Si hay error en la API, asumir que no hay stock
                return {
                    'status': 'error',
                    'available': False,
                    'message': f'Error al verificar stock: {result.get("message", "Error desconocido")}',
                    'monto_id': monto_id
                }
                
        except Exception as e:
            logger.error(f"Error al verificar stock para monto_id {monto_id}: {str(e)}")
            return {
                'status': 'error',
                'available': False,
                'message': f'Error al verificar stock: {str(e)}',
                'monto_id': monto_id
            }
    
    def is_available(self):
        """Verifica si la API externa está disponible"""
        try:
            success, message = self.test_connection()
            return success
        except Exception:
            return False

# Instancia global del cliente
inefable_client = InefableAPIClient()

def get_inefable_client():
    """Obtiene la instancia del cliente de Inefable API"""
    return inefable_client

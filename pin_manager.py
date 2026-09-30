import sqlite3
import logging
from datetime import datetime
from inefable_api_client import get_inefable_client

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PinManager:
    """Gestor de pines que maneja stock local + API externa manual para admin"""
    
    def __init__(self, database_path):
        self.database_path = database_path
        self.inefable_client = get_inefable_client()
        
    def get_db_connection(self):
        """Obtiene una conexión a la base de datos (PostgreSQL en producción, SQLite en local)"""
        from pg_compat import get_db_connection
        return get_db_connection()
    
    def get_local_stock(self, monto_id=None):
        """Obtiene el stock local de pines"""
        conn = self.get_db_connection()
        
        if monto_id:
            # Stock para un monto específico
            count = conn.execute('''
                SELECT COUNT(*) FROM pines_freefire 
                WHERE monto_id = ? AND usado = FALSE
            ''', (monto_id,)).fetchone()[0]
            conn.close()
            return count
        else:
            # Stock para todos los montos
            stock = {}
            for i in range(1, 10):
                count = conn.execute('''
                    SELECT COUNT(*) FROM pines_freefire 
                    WHERE monto_id = ? AND usado = FALSE
                ''', (i,)).fetchone()[0]
                stock[i] = count
            conn.close()
            return stock
    
    def get_local_pin(self, monto_id):
        """Obtiene un pin del stock local"""
        conn = self.get_db_connection()
        pin = conn.execute('''
            SELECT * FROM pines_freefire 
            WHERE monto_id = ? AND usado = FALSE 
            LIMIT 1
        ''', (monto_id,)).fetchone()
        conn.close()
        return pin
    
    def take_local_pin(self, monto_id):
        """Saca un pin del stock local de forma atómica: dos compras simultáneas nunca reciben el mismo."""
        for _ in range(5):
            pin = self.get_local_pin(monto_id)
            if not pin:
                return None
            conn = self.get_db_connection()
            try:
                deleted = conn.execute('DELETE FROM pines_freefire WHERE id = ? AND usado = FALSE', (pin['id'],))
                conn.commit()
                if deleted.rowcount == 1:
                    return pin
            finally:
                conn.close()
        return None

    def remove_local_pin(self, pin_id):
        """Elimina un pin del stock local completamente"""
        conn = self.get_db_connection()
        conn.execute('DELETE FROM pines_freefire WHERE id = ?', (pin_id,))
        conn.commit()
        conn.close()
    
    def add_local_pin(self, monto_id, pin_code, source='manual'):
        """Añade un pin al stock local"""
        conn = self.get_db_connection()
        try:
            # Verificar si el pin ya existe
            existing = conn.execute('''
                SELECT id FROM pines_freefire 
                WHERE pin_codigo = ? AND monto_id = ?
            ''', (pin_code, monto_id)).fetchone()
            
            if existing:
                conn.close()
                return False, "Pin ya existe en el stock"
            
            # Agregar pin
            conn.execute('''
                INSERT INTO pines_freefire (monto_id, pin_codigo)
                VALUES (?, ?)
            ''', (monto_id, pin_code))
            conn.commit()
            conn.close()
            
            logger.info(f"Pin agregado al stock local - Monto: {monto_id}")
            return True, "Pin agregado exitosamente"
            
        except Exception as e:
            conn.rollback()
            conn.close()
            logger.error(f"Error al agregar pin local: {str(e)}")
            return False, f"Error al agregar pin: {str(e)}"
    
    def get_pin_source_config(self, monto_id):
        """Obtiene la configuración de fuente para un monto específico"""
        conn = self.get_db_connection()
        result = conn.execute('''
            SELECT fuente FROM configuracion_fuentes_pines 
            WHERE monto_id = ? AND activo = TRUE
        ''', (monto_id,)).fetchone()
        conn.close()
        return result['fuente'] if result else 'local'
    
    def request_pin(self, monto_id):
        """
        Solicita un pin según la configuración de fuente para el monto_id
        
        Args:
            monto_id (int): ID del monto (1-9)
            
        Returns:
            dict: Resultado de la operación
        """
        try:
            # Obtener configuración de fuente para este monto
            source_config = self.get_pin_source_config(monto_id)
            
            logger.info(f"Solicitando pin para monto_id {monto_id} usando fuente: {source_config}")
            
            if source_config == 'api_externa':
                # SOLO usar API externa, sin respaldo
                logger.info(f"Usando SOLO API externa para monto_id {monto_id} (sin respaldo)")
                api_result = self.inefable_client.request_pin(monto_id)
                
                if api_result.get('status') == 'success':
                    logger.info(f"Pin obtenido exitosamente de API externa para monto_id {monto_id}")
                    return api_result
                else:
                    # Si API externa falla, devolver error directamente
                    logger.error(f"API externa falló para monto_id {monto_id}: {api_result.get('message', 'Error desconocido')}")
                    return {
                        'status': 'error',
                        'message': f'Error en API externa: {api_result.get("message", "Error desconocido")}',
                        'api_error': api_result.get('message'),
                        'monto_id': monto_id,
                        'source_attempted': 'api_externa'
                    }
            else:
                # Usar solo stock local
                logger.info(f"Usando stock local para monto_id {monto_id}")
                local_stock = self.get_local_stock(monto_id)
                
                if local_stock > 0:
                    # Hay stock local disponible
                    local_pin = self.take_local_pin(monto_id)
                    if local_pin:
                        
                        logger.info(f"Pin obtenido del stock local - Monto: {monto_id}")
                        return {
                            'status': 'success',
                            'pin_code': local_pin['pin_codigo'],
                            'monto_id': monto_id,
                            'source': 'local_stock',
                            'timestamp': datetime.now().isoformat(),
                            'stock_remaining': local_stock - 1
                        }
                
                # No hay stock local
                logger.info(f"Sin stock local para monto {monto_id}")
                return {
                    'status': 'error',
                    'message': 'Sin stock disponible',
                    'error_type': 'no_stock',
                    'local_stock': local_stock,
                    'monto_id': monto_id
                }
                
        except Exception as e:
            logger.error(f"Error inesperado al solicitar pin para monto_id {monto_id}: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error inesperado: {str(e)}',
                'monto_id': monto_id
            }
    
    def request_multiple_pins(self, monto_id, cantidad):
        """
        Solicita múltiples pines según la configuración de fuente
        Para API externa: hace múltiples llamadas individuales (1 pin por llamada)
        Para stock local: obtiene múltiples pines del stock
        
        Args:
            monto_id (int): ID del monto (1-9)
            cantidad (int): Cantidad de pines solicitados
            
        Returns:
            dict: Resultado de la operación
        """
        try:
            logger.info(f"Solicitando {cantidad} pines para monto_id {monto_id}")
            
            if cantidad <= 0:
                return {
                    'status': 'error',
                    'message': 'La cantidad debe ser mayor a 0',
                    'error_type': 'validation'
                }
            
            # Obtener configuración de fuente para este monto
            source_config = self.get_pin_source_config(monto_id)
            
            if source_config == 'api_externa':
                # Para API externa: hacer múltiples llamadas individuales
                logger.info(f"Usando API externa para {cantidad} pines (múltiples llamadas)")
                return self._request_multiple_pins_from_api(monto_id, cantidad)
            else:
                # Para stock local: obtener múltiples pines del stock
                logger.info(f"Usando stock local para {cantidad} pines")
                return self._request_multiple_pins_from_local(monto_id, cantidad)
                
        except Exception as e:
            logger.error(f"Error inesperado al solicitar múltiples pines: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error inesperado: {str(e)}',
                'error_type': 'unexpected'
            }
    
    def _request_multiple_pins_from_api(self, monto_id, cantidad):
        """
        Solicita múltiples pines de la API externa haciendo llamadas individuales
        """
        pines_obtenidos = []
        errores = []
        
        for i in range(cantidad):
            logger.info(f"Solicitando pin {i+1}/{cantidad} de API externa para monto_id {monto_id}")
            
            api_result = self.inefable_client.request_pin(monto_id)
            
            if api_result.get('status') == 'success':
                pines_obtenidos.append({
                    'pin_code': api_result.get('pin_code'),
                    'source': 'inefable_api'
                })
                logger.info(f"Pin {i+1}/{cantidad} obtenido exitosamente de API externa")
            else:
                error_msg = api_result.get('message', 'Error desconocido')
                errores.append(f"Pin {i+1}: {error_msg}")
                logger.error(f"Error al obtener pin {i+1}/{cantidad} de API externa: {error_msg}")
                
                # Si falla una solicitud, detener el proceso
                break
        
        # Resultado final
        if len(pines_obtenidos) == cantidad:
            return {
                'status': 'success',
                'pins': pines_obtenidos,
                'cantidad_solicitada': cantidad,
                'cantidad_obtenida': len(pines_obtenidos),
                'monto_id': monto_id,
                'source': 'inefable_api',
                'timestamp': datetime.now().isoformat()
            }
        elif len(pines_obtenidos) > 0:
            return {
                'status': 'partial_success',
                'pins': pines_obtenidos,
                'cantidad_solicitada': cantidad,
                'cantidad_obtenida': len(pines_obtenidos),
                'monto_id': monto_id,
                'source': 'inefable_api',
                'message': f'Solo se obtuvieron {len(pines_obtenidos)} de {cantidad} pines',
                'errores': errores,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'status': 'error',
                'message': f'No se pudo obtener ningún pin de la API externa',
                'error_type': 'api_failure',
                'cantidad_solicitada': cantidad,
                'cantidad_obtenida': 0,
                'errores': errores,
                'monto_id': monto_id
            }
    
    def _request_multiple_pins_from_local(self, monto_id, cantidad):
        """
        Solicita múltiples pines del stock local
        """
        local_stock = self.get_local_stock(monto_id)
        
        if local_stock < cantidad:
            return {
                'status': 'error',
                'message': f'Stock insuficiente. Disponible: {local_stock}, Solicitado: {cantidad}',
                'error_type': 'insufficient_stock',
                'local_stock': local_stock,
                'cantidad_solicitada': cantidad
            }
        
        pines_obtenidos = []
        
        # Obtener pines del stock local
        for i in range(cantidad):
            local_pin = self.take_local_pin(monto_id)
            if local_pin:
                pines_obtenidos.append({
                    'pin_code': local_pin['pin_codigo'],
                    'source': 'local_stock'
                })
            else:
                logger.warning(f"Pin local esperado no encontrado en iteración {i+1}")
                break
        
        # Resultado final
        if len(pines_obtenidos) == cantidad:
            return {
                'status': 'success',
                'pins': pines_obtenidos,
                'cantidad_solicitada': cantidad,
                'cantidad_obtenida': len(pines_obtenidos),
                'monto_id': monto_id,
                'source': 'local_stock',
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'status': 'error',
                'message': f'Solo se pudieron obtener {len(pines_obtenidos)} de {cantidad} pines',
                'error_type': 'partial_stock',
                'cantidad_solicitada': cantidad,
                'cantidad_obtenida': len(pines_obtenidos),
                'pins': pines_obtenidos
            }
    
    
    def test_external_api(self):
        """
        Función SOLO para admin: Prueba la conexión con la API externa
        
        Returns:
            dict: Estado de la API externa
        """
        try:
            logger.info("🧪 ADMIN: Probando conexión con API externa")
            
            # Probar conexión con API externa
            result = self.inefable_client.test_connection()
            
            if result.get('status') == 'success':
                return {
                    'status': 'success',
                    'message': 'API externa disponible y funcionando',
                    'connection': True
                }
            else:
                return {
                    'status': 'error',
                    'message': f'API externa no disponible: {result.get("message", "Error desconocido")}',
                    'connection': False
                }
                
        except Exception as e:
            logger.error(f"Error al probar API externa: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error al conectar con API externa: {str(e)}',
                'connection': False
            }

def create_pin_manager(database_path):
    """Crea una instancia del gestor de pines"""
    return PinManager(database_path)

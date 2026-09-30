#!/usr/bin/env python3
"""
Script para probar la respuesta real de la API de Inefable
y ver exactamente qué está devolviendo
"""

import os
import requests
import json
import re

def test_inefable_api():
    """Prueba la API de Inefable y muestra la respuesta completa"""
    
    # Configuración de la API
    base_url = "https://inefableshop.net/conexion_api/api.php"
    usuario = "inefableshop"
    clave = os.environ.get("INEFABLE_CLAVE", "")
    
    # Parámetros para la prueba
    params = {
        'action': 'recarga',
        'usuario': usuario,
        'clave': clave,
        'tipo': 'recargaPinFreefirebs',
        'monto': 1,  # Probar con monto 1 (110 diamantes)
        'numero': 0
    }
    
    print("🧪 Probando API de Inefable...")
    print(f"URL: {base_url}")
    print(f"Parámetros: {params}")
    print("-" * 50)
    
    try:
        response = requests.get(
            base_url,
            params=params,
            timeout=30,
            headers={
                'User-Agent': 'InefablePines/1.0',
                'Accept': 'application/json, text/plain, */*'
            }
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print("-" * 50)
        print("Respuesta completa:")
        print(response.text)
        print("-" * 50)
        
        # Intentar parsear como JSON
        try:
            json_data = response.json()
            print("✅ Respuesta es JSON válido:")
            print(json.dumps(json_data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("❌ Respuesta NO es JSON válido")
            
            # Buscar JSON dentro del texto
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    print("✅ JSON encontrado dentro del texto:")
                    print(json.dumps(json_data, indent=2, ensure_ascii=False))
                    
                    # Buscar el pin en el JSON
                    pin_fields = ['pin', 'codigo', 'pin_code', 'code']
                    for field in pin_fields:
                        if field in json_data:
                            print(f"🎯 PIN encontrado en campo '{field}': {json_data[field]}")
                            
                except json.JSONDecodeError:
                    print("❌ JSON extraído no es válido")
            else:
                print("❌ No se encontró JSON en la respuesta")
        
        print("-" * 50)
        
        # Buscar patrones de pin en texto plano
        print("🔍 Buscando patrones de PIN en texto plano:")
        
        patterns = [
            (r'PIN[:\s]*([A-Z0-9]{10,20})', 'PIN: XXXXXXXXXX'),
            (r'CODIGO[:\s]*([A-Z0-9]{10,20})', 'CODIGO: XXXXXXXXXX'),
            (r'CODE[:\s]*([A-Z0-9]{10,20})', 'CODE: XXXXXXXXXX'),
            (r'([A-Z0-9]{12,16})', 'Código alfanumérico 12-16 chars'),
            (r'Pin[:\s]*([A-Z0-9]{10,20})', 'Pin: XXXXXXXXXX'),
            (r'Código[:\s]*([A-Z0-9]{10,20})', 'Código: XXXXXXXXXX'),
        ]
        
        found_pins = []
        for pattern, description in patterns:
            matches = re.findall(pattern, response.text, re.IGNORECASE)
            if matches:
                print(f"✅ Patrón '{description}' encontró: {matches}")
                found_pins.extend(matches)
        
        if not found_pins:
            print("❌ No se encontraron patrones de PIN")
        
        print("-" * 50)
        
        # Verificar indicadores de error
        print("🚨 Verificando indicadores de error:")
        
        error_keywords = [
            'sin stock', 'no stock', 'agotado', 'no disponible',
            'out of stock', 'unavailable', 'insufficient',
            'no hay', 'temporalmente no disponible', 'error',
            'no se pudo', 'fallido', 'failed', 'saldo insuficiente',
            'balance insuficiente', 'no funds', 'insufficient funds',
            'alerta":"red"', 'error desconocido', 'respuesta inválida'
        ]
        
        response_lower = response.text.lower()
        found_errors = []
        for keyword in error_keywords:
            if keyword in response_lower:
                found_errors.append(keyword)
        
        if found_errors:
            print(f"⚠️ Indicadores de error encontrados: {found_errors}")
        else:
            print("✅ No se encontraron indicadores de error")
            
        return response.text
        
    except Exception as e:
        print(f"❌ Error al probar API: {str(e)}")
        return None

if __name__ == "__main__":
    test_inefable_api()

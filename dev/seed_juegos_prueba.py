"""
Crea juegos y paquetes FICTICIOS para probar la tienda en LOCAL — NO usar en producción.
Usa las mismas rutas del panel admin (no escribe directo en la base de datos).

  Juegos por ID  -> recarga vía proveedor (dev/mock_revendedor.py debe estar corriendo)
  Juegos PIN     -> stock local de códigos, sección "Juegos PIN" del menú
  Gift Cards     -> stock local de códigos (y una vía proveedor), sección "Gift Cards"

Uso (con la app en http://127.0.0.1:5000 y sin DATABASE_URL):
    .venv\\Scripts\\python.exe dev\\seed_juegos_prueba.py
"""
import os
import re
import sys

import requests

BASE = os.environ.get('SEED_BASE_URL', 'http://127.0.0.1:5000')
ADMIN = ('admin', '123456')  # login de desarrollo (solo sin DATABASE_URL)

JUEGOS = [
    # ---- Juegos por ID (proveedor) ----
    {'nombre': 'Mobile Legends (Prueba)', 'icono': '💎', 'modo': 'id', 'dual_id': True,
     'campo_id2_label': 'Zone ID', 'campo_id2_placeholder': 'Ej: 2451',
     'paquetes': [('86 Diamantes', 1.60), ('172 Diamantes', 3.20), ('257 Diamantes', 4.75), ('706 Diamantes', 12.50)]},
    {'nombre': 'PUBG Mobile (Prueba)', 'icono': '🎯', 'modo': 'id',
     'paquetes': [('60 UC', 0.99), ('325 UC', 4.90), ('660 UC', 9.80), ('1800 UC', 24.50)]},
    {'nombre': 'Genshin Impact (Prueba)', 'icono': '✨', 'modo': 'id',
     'servidor_enabled': True, 'servidor_opciones': 'America, Europe, Asia, TW/HK/MO',
     'paquetes': [('60 Cristales', 0.99), ('300+30 Cristales', 4.90), ('980+110 Cristales', 14.50), ('Bendición Lunar', 4.90)]},
    # ---- Juegos PIN (stock local) ----
    {'nombre': 'Call of Duty Mobile PIN (Prueba)', 'icono': '🎖️', 'stock': True, 'categoria': 'pin',
     'paquetes': [('80 CP', 0.99), ('420 CP', 4.90), ('880 CP', 9.80)]},
    {'nombre': 'Honor of Kings PIN (Prueba)', 'icono': '👑', 'stock': True, 'categoria': 'pin',
     'paquetes': [('80 Tokens', 1.00), ('400 Tokens', 5.00)]},
    # ---- Gift Cards (stock local) ----
    {'nombre': 'Google Play (Prueba)', 'icono': '🎁', 'stock': True, 'categoria': 'giftcard',
     'paquetes': [('Tarjeta $5', 5.25), ('Tarjeta $10', 10.40), ('Tarjeta $25', 25.80)]},
    {'nombre': 'Steam (Prueba)', 'icono': '🕹️', 'stock': True, 'categoria': 'giftcard',
     'paquetes': [('Tarjeta $5', 5.30), ('Tarjeta $10', 10.50), ('Tarjeta $20', 20.90)]},
    {'nombre': 'PlayStation Store (Prueba)', 'icono': '🎮', 'stock': True, 'categoria': 'giftcard',
     'paquetes': [('Tarjeta $10', 10.60), ('Tarjeta $25 (sin stock)', 26.20)], 'sin_stock': ['Tarjeta $25 (sin stock)']},
    # ---- Gift Card vía proveedor (el mock devuelve un código) ----
    {'nombre': 'Netflix (Prueba proveedor)', 'icono': '🎬', 'modo': 'pin', 'categoria': 'giftcard',
     'remote_product': 'GC-NETFLIX', 'paquetes': [('1 mes Básico', 7.50), ('1 mes Estándar', 12.00)]},
]
CODIGOS_POR_PAQUETE = 3


def main():
    s = requests.Session()
    s.post(f'{BASE}/login', data={'correo': ADMIN[0], 'contraseña': ADMIN[1]})
    page = s.get(f'{BASE}/admin/dynamic-games')
    if page.status_code != 200 or 'Juegos Din' not in page.text:
        sys.exit('No se pudo entrar como admin. ¿Está la app corriendo en modo local?')

    for j in JUEGOS:
        if j['nombre'] in page.text:
            print(f'= ya existe: {j["nombre"]}')
            continue
        payload = {
            'nombre': j['nombre'], 'icono': j['icono'], 'modo': j.get('modo', 'pin'),
            'gamepoint_product_id': 0 if j.get('stock') else 999,
            'usa_stock_local': bool(j.get('stock')), 'ganancia_default': 0.10,
            'campo_id_label': 'ID de Jugador', 'campo_id_placeholder': 'Ingresa tu ID',
            'descripcion': 'Juego FICTICIO para pruebas locales.',
        }
        for k in ('dual_id', 'campo_id2_label', 'campo_id2_placeholder', 'servidor_enabled', 'servidor_opciones', 'categoria'):
            if k in j:
                payload[k] = j[k]
        r = s.post(f'{BASE}/admin/dynamic-games/create', json=payload).json()
        if not r.get('success'):
            print(f'! error creando {j["nombre"]}: {r}')
            continue
        gid, slug = r['game_id'], r['slug']

        mappings = []
        for orden, (nombre, precio) in enumerate(j['paquetes']):
            pr = s.post(f'{BASE}/admin/dynamic-games/{gid}/packages/add',
                        json={'nombre': nombre, 'precio': precio, 'orden': orden}).json()
            pkg_id = pr.get('package_id')
            if not pkg_id:
                print(f'! error paquete {nombre}: {pr}')
                continue
            if j.get('stock') and nombre not in j.get('sin_stock', []):
                pref = re.sub(r'[^A-Z0-9]', '', j['nombre'].upper())[:6]
                codigos = '\n'.join(f'PRUEBA-{pref}-{pkg_id:03d}-{n:02d}' for n in range(1, CODIGOS_POR_PAQUETE + 1))
                s.post(f'{BASE}/admin/add_pins_batch',
                       data={'game_type': f'dyn_{slug}', 'batch_monto_id': pkg_id, 'pins_batch': codigos})
            if not j.get('stock'):
                mappings.append({'juego_id': gid, 'paquete_id': pkg_id, 'auto_enabled': True,
                                 'remote_product_id': j.get('remote_product', str(100 + gid)),
                                 'remote_package_id': str(pkg_id)})
        if mappings:
            s.post(f'{BASE}/admin/revendedores/mappings/bulk', json={'mappings': mappings})
        s.post(f'{BASE}/admin/dynamic-games/{gid}/toggle')  # activar (se crean desactivados)
        print(f'+ creado: {j["nombre"]}  ->  {BASE}/juego/d/{slug}')


if __name__ == '__main__':
    main()

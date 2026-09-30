"""
Proveedor (revendedor) FALSO para pruebas locales — NO usar en producción.

Imita la API que usa dynamic_games._purchase_via_reseller:
  POST /api/v1/recharge          (cabecera X-API-Key)
  GET  /api/v1/order-status?external_order_id=...

Resultado según el ID de jugador enviado:
  termina en 0000  -> error ("ID no encontrado")
  termina en 1111  -> en proceso; pasa a "completada" a los 60 s
  cualquier otro   -> éxito inmediato
Si remote_product_id empieza por "GC" devuelve además un código de tarjeta.

Uso:  .venv\\Scripts\\python.exe dev\\mock_revendedor.py
Y en .env:  REVENDEDORES_BASE_URL=http://127.0.0.1:5055
            REVENDEDORES_API_KEY=clave-prueba-local
"""
import os
import secrets
import time

from flask import Flask, jsonify, request

API_KEY = os.environ.get('MOCK_API_KEY', 'clave-prueba-local')
PENDING_SECONDS = 60

app = Flask(__name__)
orders = {}  # external_order_id -> dict


def _auth_ok():
    return request.headers.get('X-API-Key') == API_KEY


@app.post('/api/v1/recharge')
def recharge():
    if not _auth_ok():
        return jsonify(ok=False, error='API key inválida (mock)'), 401
    data = request.get_json(silent=True) or {}
    player = str(data.get('player_id') or '')
    order_id = str(data.get('external_order_id') or '')
    product = str(data.get('product_id') or '')
    ref = 'MOCK-' + secrets.token_hex(4).upper()
    name = f'Jugador_{player[-4:] or "0000"}'

    if player.endswith('0000'):
        orders[order_id] = {'status': 'fallida', 'error': 'ID de jugador no encontrado (simulado)'}
        return jsonify(ok=False, error='ID de jugador no encontrado (simulado)'), 400

    if player.endswith('1111'):
        orders[order_id] = {'status': 'procesando', 'created': time.time(), 'reference_no': ref, 'player_name': name}
        return jsonify(ok=False, pending=True, status='procesando', error='Recarga en proceso (simulado)'), 202

    pin = ''
    if product.upper().startswith('GC'):
        pin = f'GC-{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}'
    orders[order_id] = {'status': 'completada', 'reference_no': ref, 'player_name': name}
    return jsonify(ok=True, reference_no=ref, player_name=name, pin=pin)


@app.get('/api/v1/verify-player')
def verify_player():
    """Verificación de ID falsa: /api/v1/verify-player?player_id=...&zone=...
    Respuesta: {"ok": true, "data": {"nickname": "..."}}; IDs terminados en 0000 no existen."""
    if not _auth_ok():
        return jsonify(ok=False, message='API key inválida (mock)'), 401
    player = request.args.get('player_id', '')
    if not player or player.endswith('0000'):
        return jsonify(ok=False, message='Jugador no encontrado (simulado)'), 404
    zone = request.args.get('zone', '')
    return jsonify(ok=True, data={'nickname': f'Jugador_{player[-4:]}' + (f'#{zone}' if zone else ''), 'region': 'LATAM'})


@app.get('/api/v1/balance')
def balance():
    if not _auth_ok():
        return jsonify(ok=False, error='API key inválida (mock)'), 401
    return jsonify(ok=True, balance=1000.00, currency='USD')


@app.get('/api/v1/order-status')
def order_status():
    if not _auth_ok():
        return jsonify(ok=False, error='API key inválida (mock)'), 401
    order = orders.get(request.args.get('external_order_id', ''))
    if not order:
        return jsonify(ok=True, found=False, status='')
    if order['status'] == 'procesando' and time.time() - order['created'] >= PENDING_SECONDS:
        order['status'] = 'completada'
    return jsonify(ok=True, found=True, status=order['status'], order=order)


@app.route('/api/v1/verify-name', methods=['POST'])
def verify_name():
    """Imita la API verify-name de Inefable. ID terminado en 404 = no existe."""
    if (request.headers.get('X-API-Key') or '') != 'inef-prueba':
        return jsonify(ok=False, error='invalid api key'), 401
    d = request.get_json(silent=True) or {}
    game = str(d.get('game') or '').lower()
    alias = {'freefire': 'freefire', 'free-fire': 'freefire', 'inefable-bloodstriker': 'bs', 'bloodstrike': 'bs',
             'blood-strike': 'bs', 'inefable-mobilelegends': 'ml', 'mobilelegends': 'ml', 'mobile-legends': 'ml'}
    if game not in alias:
        return jsonify(ok=False, error='unknown game'), 400
    pid = str(d.get('player_id') or '')
    if alias[game] == 'ml' and not d.get('player_id2'):
        return jsonify(ok=False, error='player_id2 required'), 400
    if pid.endswith('404'):
        return jsonify(ok=False, error='not found'), 404
    return jsonify(ok=True, player_id=pid, player_id2=d.get('player_id2'), player_name=f'Jugador_{pid[-4:]}', cached=False)


if __name__ == '__main__':
    print('Proveedor FALSO escuchando en http://127.0.0.1:5055  (clave: %s)' % API_KEY)
    app.run(host='127.0.0.1', port=5055, debug=False)

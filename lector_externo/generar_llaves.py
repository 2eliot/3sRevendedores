"""
Crea el par de llaves del lector del BDV (PyNaCl, Curve25519) en el servidor del lector.

    python generar_llaves.py                 # guarda la privada en BDV_CLAVE_PRIVADA o ~/.lector-bdv/llave_privada
    python generar_llaves.py --ruta /otra/ruta

- La PRIVADA se guarda en un archivo con permisos 600 y NUNCA se muestra en pantalla. Respáldala: si se pierde, todos
  los usuarios tendrán que volver a conectar su BDV.
- Muestra la PÚBLICA (no es secreta) y su código. Normalmente no hace falta copiarla: al arrancar, el lector se
  presenta solo a la web y el admin lo aprueba en el panel comparando el código. Si prefieres no usar la aprobación,
  pon la pública en el .env de la web como BDV_CLAVE_PUBLICA.
- Si ya existe una llave, no la reemplaza (perderías las conexiones guardadas) salvo con --forzar.

No importa nada de la web: es parte del lector externo.
"""
import base64
import hashlib
import os
import sys
from pathlib import Path


def ruta_por_defecto():
    return Path(os.environ.get('BDV_CLAVE_PRIVADA', '').strip() or (Path.home() / '.lector-bdv' / 'llave_privada'))


def huella(publica_b64):
    """El mismo código que muestra la web en el panel: AB12-CD34."""
    h = hashlib.sha256(publica_b64.strip().encode('ascii')).hexdigest()[:8].upper()
    return f'{h[:4]}-{h[4:]}'


def _guardar_privada(ruta, crudo):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(ruta.parent, 0o700)
    except OSError:
        pass
    # Se crea ya con permisos 600 (sin un instante en que otros puedan leerla)
    fd = os.open(str(ruta), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(base64.b64encode(crudo).decode('ascii'))
    try:
        os.chmod(ruta, 0o600)
    except OSError:
        pass


def cargar_o_crear(ruta=None, crear=True):
    """(llave_privada, publica_b64, creada). La privada es un objeto PyNaCl; nunca se imprime."""
    from nacl.public import PrivateKey
    ruta = Path(ruta) if ruta else ruta_por_defecto()
    creada = False
    if ruta.exists():
        privada = PrivateKey(base64.b64decode(ruta.read_text().strip()))
    elif crear:
        privada = PrivateKey.generate()
        _guardar_privada(ruta, bytes(privada))
        creada = True
    else:
        raise FileNotFoundError(str(ruta))
    publica = base64.b64encode(bytes(privada.public_key)).decode('ascii')
    return privada, publica, creada


def main(argv):
    ruta = ruta_por_defecto()
    if '--ruta' in argv:
        ruta = Path(argv[argv.index('--ruta') + 1])
    if ruta.exists() and '--forzar' not in argv:
        _, publica, _ = cargar_o_crear(ruta, crear=False)
        print(f'Ya hay una llave en {ruta} (no se cambia; usa --forzar si de verdad quieres una nueva).')
    else:
        if ruta.exists():
            ruta.unlink()
        _, publica, _ = cargar_o_crear(ruta)
        print(f'Llave privada guardada en {ruta} (permisos 600). Respáldala en un lugar seguro.')
    print(f'Llave pública (no es secreta): {publica}')
    print(f'Código para aprobar el lector en el panel: {huella(publica)}')


if __name__ == '__main__':
    main(sys.argv[1:])

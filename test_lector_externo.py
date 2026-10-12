"""
Lector externo del BDV (lector_externo/): llaves, conversación con la web y lectura del banco.
Nunca usa el BDV real ni la cuenta del dueño: un banco falso y la web de verdad sobre una base temporal.

    python -m unittest test_lector_externo -v
"""
import base64
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

AQUI = Path(__file__).resolve().parent
LECTOR = AQUI / 'lector_externo'
sys.path.insert(0, str(LECTOR))

import generar_llaves  # noqa: E402  (del lector externo: no importa nada de la web)


class LlavesTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.ruta = self.dir / 'privada' / 'llave_privada'

    def correr(self, *args):
        env = dict(os.environ, BDV_CLAVE_PRIVADA=str(self.ruta), PYTHONIOENCODING='utf-8')
        r = subprocess.run([sys.executable, str(LECTOR / 'generar_llaves.py'), *args], capture_output=True, text=True,
                           env=env, encoding='utf-8')
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_crea_el_par_sin_mostrar_la_privada(self):
        salida = self.correr()
        privada = self.ruta.read_text().strip()
        self.assertNotIn(privada, salida)               # la privada nunca sale en pantalla
        self.assertIn('Llave pública', salida)
        _, publica, creada = generar_llaves.cargar_o_crear(self.ruta, crear=False)
        self.assertFalse(creada)
        self.assertIn(publica, salida)
        self.assertIn(generar_llaves.huella(publica), salida)
        if os.name == 'posix':
            self.assertEqual(self.ruta.stat().st_mode & 0o777, 0o600)

    def test_no_pisa_una_llave_que_ya_existe(self):
        self.correr()
        antes = self.ruta.read_text()
        salida = self.correr()
        self.assertIn('Ya hay una llave', salida)
        self.assertEqual(self.ruta.read_text(), antes)
        self.correr('--forzar')
        self.assertNotEqual(self.ruta.read_text(), antes)

    def test_la_web_cierra_y_el_lector_abre(self):
        from nacl.public import SealedBox
        privada, publica, _ = generar_llaves.cargar_o_crear(self.ruta)
        os.environ['BDV_CLAVE_PUBLICA'] = publica
        try:
            import bdv   # lo que hace la web al guardar la conexión
            cifrado = bdv.cifrar('Clave-Secreta-1')
            self.assertEqual(bdv.huella(publica), generar_llaves.huella(publica))   # mismo código en panel y lector
        finally:
            os.environ.pop('BDV_CLAVE_PUBLICA', None)
        self.assertEqual(SealedBox(privada).decrypt(base64.b64decode(cifrado)).decode(), 'Clave-Secreta-1')


if __name__ == '__main__':
    unittest.main()

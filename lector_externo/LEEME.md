# Lector del BDV (servidor aparte)

Programa que entra al BDV de cada usuario, lee sus movimientos y manda los pagos recibidos a la web
(Referencias). Corre en **su propio servidor**, no en el de la web. No importa nada de la web: solo habla con ella
por HTTPS.

- Tiene su propia llave: los datos del BDV que guarda la web solo los puede abrir este lector.
- La primera vez se presenta solo a la web y el admin lo aprueba **una vez** con un clic. Después todo es automático,
  también para los usuarios nuevos.
- Si se cae, systemd lo vuelve a levantar. Si la web no responde, sigue leyendo y guarda los pagos en una cola
  hasta que la web vuelva.

## Instalar (como root en el servidor del lector)

```bash
# 1. Programas del sistema
apt update && apt install -y git python3-venv

# 2. Usuario sin privilegios para el lector (su llave y su cola quedan en /var/lib/lector-bdv)
useradd --system --create-home --home-dir /var/lib/lector-bdv --shell /usr/sbin/nologin lectorbdv

# 3. El código (solo la carpeta lector_externo)
git clone --depth 1 --branch rediseno-sebastian --filter=blob:none --sparse \
    https://github.com/2eliot/3sRevendedores.git /opt/lector-bdv
git -C /opt/lector-bdv sparse-checkout set lector_externo

# 4. Python, librerías y Chromium
python3 -m venv /opt/lector-bdv/venv
/opt/lector-bdv/venv/bin/pip install -r /opt/lector-bdv/lector_externo/requirements.txt
PLAYWRIGHT_BROWSERS_PATH=/opt/lector-bdv/navegadores /opt/lector-bdv/venv/bin/playwright install --with-deps chromium

# 5. Configuración (no tiene nada secreto). Cambia la dirección por la de la web
cat > /etc/lector-bdv.env <<'FIN'
BDV_WEB_URL=https://LA-DIRECCION-DE-LA-WEB
BDV_MAX_CUENTAS=30
FIN

# 6. Arrancarlo (y que arranque solo cada vez que se prenda el servidor)
cp /opt/lector-bdv/lector_externo/lector-bdv.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now lector-bdv
```

Cuando `main` ya tenga el lector, cambia `rediseno-sebastian` por `main` en el paso 3.

## Aprobarlo (una sola vez)

1. Mira su código: `journalctl -u lector-bdv -n 20 --no-pager` muestra `Código de este lector: AB12-CD34`.
2. En la web, como admin: **Referencias (todos) → Conexiones del BDV**. Aparece «Un lector pide conectarse» con un
   código. Si es **el mismo** que mostró el lector, pulsa **Aprobar**. Si no es el mismo, pulsa **Rechazar**.

## Respaldar la llave

La llave privada se crea sola en `/var/lib/lector-bdv/.lector-bdv/llave_privada` (permisos 600). Guarda una copia
en un lugar seguro. Si se pierde, el lector nuevo necesita otra aprobación y cada usuario tiene que volver a guardar
sus datos del BDV.

## ¿Está vivo?

- `systemctl status lector-bdv`: debe decir `active (running)`.
- `journalctl -u lector-bdv -f`: lo que va haciendo, en vivo (nunca muestra usuarios, claves ni números de cuenta).
- En la web, en Conexiones del BDV: «último contacto hace X s». Si pasan más de 2 minutos sin contacto, la pantalla
  dice «El lector no responde».

## Otras órdenes

| Para | Orden |
|---|---|
| Reiniciarlo | `systemctl restart lector-bdv` (antes pulsa «Salir» en el banco en todas las cuentas) |
| Detenerlo | `systemctl stop lector-bdv` |
| Actualizarlo | `git -C /opt/lector-bdv pull && /opt/lector-bdv/venv/bin/pip install -r /opt/lector-bdv/lector_externo/requirements.txt && systemctl restart lector-bdv` |
| Memoria que usa | `systemctl status lector-bdv` (línea Memory), o en el panel de la web |

`BDV_MAX_CUENTAS` es cuántas cuentas lee a la vez; las demás quedan «en espera» hasta que se libere un lugar. Si
la cambias, reinícialo.

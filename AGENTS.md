# AGENTS.md — Contexto operativo de 3sRevendedores

> **Qué es este archivo:** el contexto que OpenCode debe leer al abrir este proyecto. Sirve para no confundir 3sRevendedores con 3srecargas, 3S-Ticket u otros proyectos del mismo dueño.
>
> **Origen:** código real del repo (revisado el 20-nov-2026), historial de Claude Code de este proyecto (30-sep-2026 a 02-nov-2026, 266 mensajes) y 17 notas de memoria del operador. Si este archivo choca con el código, **manda el código**: verifica `git status`, `git log` y los archivos antes de actuar.
>
> **Repo público:** `github.com/2eliot/3sRevendedores` es **público**. Aquí no van claves, `.env`, bases de datos, IPs, usuarios SSH, correos de clientes ni datos de producción. Los datos de despliegue están en `AGENTS.local.md` (ignorado por Git) y en la memoria local del operador.

---

## 0. Resumen en 12 líneas

- **3sRevendedores** es una plataforma **privada para revendedores** de recargas de videojuegos: cada revendedor entra con su cuenta, recarga saldo, compra recargas por ID o códigos/PIN y lleva sus cuentas.
- **No es una tienda pública:** `/` exige sesión y redirige a `/auth`; el catálogo y los precios solo se ven después de entrar. El registro está abierto en `/auth`.
- **Producción:** `https://3srevendedores.uk` (VPS propio, PostgreSQL, gunicorn detrás de proxy). Datos exactos de acceso y servicio: `AGENTS.local.md`.
- **Repo:** `https://github.com/2eliot/3sRevendedores`. Dueño original: Eliot (`2eliot`). Desde el 12-oct-2026, **Sebastián** opera y despliega con ayuda del agente.
- **Rama de trabajo:** `rediseno-sebastian`. `main` es producción. Al 20-nov-2026, `rediseno-sebastian`, `main`, `origin/main` y `origin/rediseno-sebastian` están todos en **`7fc7dc9`** (2-nov-2026), árbol limpio.
- **Local (Windows):** `.venv\Scripts\python.exe app.py` → **puerto 5000**, SQLite `revendedores_local.db`.
- **Ojo con el puerto:** el 5000 es de **este** proyecto. `3srecargas` usa el **5001**. Nunca los mezcles.
- **Stack:** Flask 2.3.3 + Jinja2 + JavaScript/CSS vanilla; PostgreSQL en producción y SQLite en local mediante `pg_compat.py`; gunicorn 2 workers; Playwright solo para canjes concretos, no para servir la web.
- **Núcleo del negocio:** billetera, catálogo de juegos dinámicos, Mapeo a proveedores (Inefable / Central One), Referencias bancarias, API para CRM/webs externas, Ganancias y proveedores propios de cada usuario.
- **Regla de oro:** no romper producción, no mezclar proyectos y no subir/desplegar nada sin un "sube" explícito del operador.
- **No tiene Google Authenticator/2FA.** Eso es de `3srecargas`; no copies ese código aquí sin pedirlo.
- **Documentación vieja:** `README.md`, `README_API.md`, `API_*`, `SIMPLE_API_GUIDE.md`, `CONNECTION_API_GUIDE.md`, `production_config.py` y los `*_standalone.py` son de etapas anteriores. No los tomes como estado actual ni los edites creyendo que cambian la web.

---

## 1. Cómo trabajar con el operador

- **Idioma:** siempre español, claro y sin tecnicismos innecesarios. Él escribe rápido y sin ñ; no le corrijas el texto ni lo copies con errores.
- **Capturas:** su forma principal de pedir un cambio es un pantallazo. Mira **qué parte de la pantalla está señalando** antes de asumir otra cosa.
- **Cambios pequeños y exactos:** "más grande", "centrado", "más abajo", "que no se vea pegado". Haz eso, sin rediseñar de más.
- **Si dice "solo pregunta" / "no hagas nada" / "avísame antes":** investiga en modo lectura y responde; **no edites** hasta que lo autorice.
- **Cambios grandes, que borran cosas o que tocan partes en uso:** explica primero qué vas a hacer y por qué pasó el problema, y espera su OK. Los cambios pequeños de estilo/texto se pueden hacer directo.
- **No lances agentes, auditorías ni revisiones completas** mientras él sigue pidiendo cambios. Las revisiones se hacen al cerrar la tanda o cuando las pida.
- **No pidas contraseñas ni llaves por chat.** No las guardes en el repo ni en archivos de contexto.
- **No mezcles proyectos:** si el pedido huele a CRM (`C:\3STICKET`), a `3srecargas` o a `gamerflash`, párate y pregunta. Ya pasó que se hicieron cambios del CRM en la pestaña equivocada (25-oct-2026).
- **Precios y datos vivos:** el admin edita precios, paquetes, stock y cuentas a mano. No los sobrescribas en un despliegue, script o prueba.
- **Ante dos interpretaciones:** elige la más conservadora o pregunta.
- **Antes de decir "listo":** verifica de verdad (código, pruebas, páginas 200, logs). No afirmes que algo funciona solo porque el cambio "se ve bien".

---

## 2. Frases del operador y qué significan

| Él dice | Qué hacer |
|---|---|
| "abre / levanta / activa el server" | Arrancar en local en el **5000** con `.venv\Scripts\python.exe app.py`. Comprobar que `http://127.0.0.1:5000/auth` responde. |
| "terminamos", "haz lo de siempre" | `git status` / `git diff`, commit **solo** de los archivos tocados en la sesión, mensaje en español. **No push, no deploy.** |
| "sube al repo", "sube a GitHub" | Commit + push a la rama de trabajo (`rediseno-sebastian`). Sin desplegar. |
| "sube todo", "sube a la web", "sube al VPS" | Commit + push + desplegar al VPS + verificar. Solo si lo dice explícitamente. |
| "sube todo y dame el link para Eliot" | Todo lo anterior y responder con repo + hash completo del commit. |
| "¿subiste?" | Verificar de verdad (git, servidor, página, logs); no responder solo porque se hizo el commit. |
| "esto es solo pregunta" | Responder sin tocar nada. |
| "avísame antes" | Revisar en solo lectura, explicar y esperar decisión. |
| "no me gustó, borra todo lo que hiciste" | Revertir **solo** lo de esa petición y verificar que quedó como antes. |
| "ya lo hice / yo lo hago manual" | No repetir la acción; dejar registro y seguir. |

---

## 3. Qué es el producto y cómo fluye

### 3.1 Lado revendedor (usuario normal)

- **Registro e inicio:** `/auth` (login y registro). La sesión dura **30 minutos**.
- **Panel `/`:** historial de compras con buscador y paginación; saldo; accesos a juegos y notificaciones.
- **Billetera `/billetera`:** recarga con **Binance Pay** (monto exacto + código de referencia), crédito manual del admin, bono configurable por usuario y aviso de saldo acreditado.
- **Catálogo:** juegos **dinámicos** creados por el admin o por el propio usuario. Dos modos:
  - **ID:** se pide ID de jugador (y Zone ID si el juego lo usa), se verifica el nombre y se recarga directo.
  - **PIN / código / gift card:** se entrega un código.
- **Compra:** descuenta saldo de forma atómica; si el proveedor falla, reembolsa. Si hay varios pasos de Mapeo, puede reembolsar la parte no entregada.
- **Dashboard `/dashboard`:** actividad y sección **Ganancias** (gasto, venta y ganancia por paquete; precios y tasa propios del usuario; historial para que los días pasados no cambien).
- **Referencias `/referencias`:** verificación de pagos bancarios **por usuario y por banco** (Bancamiga o BDV). **No suma saldo ni toca la billetera.** Sirve para que el CRM/tienda del revendedor confirme que una referencia existe y no se usó dos veces.
- **Mis proveedores `/mis-proveedores`:** cada usuario puede conectar su propio **Central One** (varios), ver su saldo, sincronizar catálogo y crear sus juegos/paquetes privados. Esos juegos solo los ve él, se recargan con **su** llave y **no descuentan saldo de la web**; quedan con monto 0 y no entran en el profit de la web.
- **API `/mi-api`:** claves propias de Recargas (se muestran **una sola vez**, se guardan como hash) y documentación con PDF.
- **Notificaciones `/notificaciones` y campana:** avisos del admin, noticias y saldo acreditado **en vivo** cada 30 s. La campana es el **primer icono** de la barra, se sacude y el número late. **No** avisa las recargas de juegos.
- **Tema y marca:** selector de tema (sistema/oscuro/claro + colores); logo subible desde el admin y servido por `/brand.css` y `/brand/logo`.

### 3.2 Lado admin

- **Panel `/admin`:** SPA con pestañas de Precios/Paquetes, Pines, Usuarios, Mapeo, Noticias, etc.
- **Juegos dinámicos `/admin/dynamic-games`:** crear/editar juegos, paquetes, categorías (ID, PIN, gift card), activar/desactivar y sincronizar precios de GamePoint.
- **Mapeo:** cada paquete se conecta a uno o varios ítems de proveedor (Inefable o Central One), con pasos y respaldo. La compra intenta los pasos en orden.
- **Proveedores `/admin/proveedores`:** estado/saldo, sincronizar catálogo e interruptor de **respaldo automático**.
- **Bot de Free Fire `/admin/bot-freefire`:** respaldo del proveedor con PINs canjeados por el VPS. **Apagado por defecto.** Sus PINs **nunca** salen por la API.
- **Pines `/admin/pins`:** stock local de PINs; cargar en lote, ver y eliminar. El bot tiene su propio apartado.
- **Verificación de IDs `/admin/verificacion-ids`:** API configurable por juego (Inefable verify-name u otra), con prueba.
- **API `/admin/api`:** cuentas, permisos (recargas, verificar ID, verificar pagos) y documentación en vivo.
- **Referencias (todos) `/admin/antiduplic`:** visión global de pagos/revisiones, conexión y aprobación del **lector externo del BDV**.
- **Correo `/admin/correo`, Logo `/admin/marca`, Estadísticas `/admin/stats`, Rentabilidad `/admin/profitability`, Ventas semanales `/admin/weekly_sales`, Log de API `/admin/api_recharges_log`.**

### 3.3 Reglas de negocio que no se deben romper

- **Precio de Free Fire ID:** es el **costo del proveedor principal del Mapeo** (suma de pasos × cantidad), redondeado hacia arriba al centavo, **sin ganancia**. Se revisa cada **20 minutos** (`precios_auto.py`). Si el admin lo cambia a mano, a los 20 min vuelve al costo. No lo "arregles" poniendo un margen.
- **Demás juegos:** precio manual del admin; no se recalcula solo.
- **Respaldo:** si el proveedor principal no entrega **nada**, y el respaldo está encendido, se intenta el respaldo. Si el fallo es por tiempo agotado, la orden queda **en proceso** y se reconcilia; no se reembolsa a ciegas.
- **Varios pasos:** se recargan en orden; si una unidad falla, se reembolsa la parte no entregada (reembolso proporcional).
- **Juego apagado:** **todos** sus paquetes quedan cerrados, aunque el paquete siga activo. El checkout directo también los bloquea.
- **Códigos/PINs:** esta web **no** tiene el cifrado de códigos ni el bloqueo por juego apagado que se implementó en `3srecargas`. No prometas esa función aquí sin implementarla y probarla.
- **Blood Strike:** eliminado del código hardcodeado. Si el admin quiere Blood Strike, lo crea como **juego dinámico** nuevo; no reintroduzcas páginas viejas.
- **Free Fire ID viejo y `/juego/freefire`:** retirados para clientes; redirigen. El bot se gestiona solo en su apartado.
- **Free Fire Latam y `/api.php`:** legado que sigue en el código; no lo borres sin revisar ventas/historial. No es el flujo principal.
- **Referencias:** no suma saldo, no toca la billetera, no crea órdenes. Solo verifica pagos y registra usos.
- **Mis proveedores:** no usa los proveedores de la web, ni el respaldo, ni el bot; no genera profit de la web.

---

## 4. Stack, arranque local y pruebas

### Stack

- **Python 3.12.10** en `.venv`.
- **Flask 2.3.3**, Jinja2, Flask-Mail.
- **PostgreSQL** en producción (`psycopg` v3); **SQLite** en local.
- **gunicorn** con `gunicorn.conf.py`: 2 workers por defecto, timeout 120 s, bind `0.0.0.0:$PORT`.
- **Playwright/Chromium** solo para el canje de PINs (`pin_redeemer.py`, `redeem_hype_vps.py`) y el lector externo; no para renderizar la web.
- **Chart.js** y **html2pdf** servidos desde `static/vendor/` (no CDN).
- Sin build step: HTML/Jinja + CSS/JS propios.

### Arranque local

```powershell
cd "C:\Users\reals\Desktop\3SREVENDEDORES WEB\3sRevendedores-main"
.venv\Scripts\python.exe app.py
# http://127.0.0.1:5000/auth
```

- `.env` local puede tener `DEV_LOGIN=1` para el acceso de prueba `admin` / `123456`. **Nunca** en producción.
- `SECRET_KEY` es obligatoria salvo con `DEV_LOGIN=1`; sin ella la app no arranca.
- `ADMIN_PASSWORD` es obligatoria para entrar como admin; **no hay contraseña por defecto**.
- Base local: `revendedores_local.db` (ignorada por Git). **No la borres ni la uses para pruebas destructivas.**
- El admin entra por **`/login`**, no por `/admin/login` (esa ruta no existe en este proyecto).

### Pruebas

- Son **`unittest`**, no pytest (pytest no está instalado).
- Cada prueba crea su propia SQLite temporal y define el entorno **antes** de importar `app`. No apuntes `DATABASE_URL` a producción.
- Comando base:

```powershell
.venv\Scripts\python.exe -m unittest test_campana -v
```

- Hay pruebas por módulo: `test_antiduplic`, `test_bdv`, `test_dynamic_recharge`, `test_ganancias`, `test_precios_auto`, `test_proveedores_usuario`, `test_campana`, `test_api_panel`, `test_referencias`, `test_central_one`, `test_saldo_negativo_fix`, `test_stock_priority`, `test_dashboard_fecha` (calendario del Resumen del día), etc.
- No hay CI ni una suite única; corre las que toquen el cambio y, si el cambio es grande, varias. Al importar `app.py` se inician hilos de fondo (precios, reconciliación, backup); es normal ver sus logs en las pruebas.
- Verificado el 20-nov-2026: `test_campana` 5 OK, `test_dashboard_fecha` 7 OK; combinadas con `test_ganancias`, **26 pruebas OK**.

---

## 5. Mapa de archivos (los importantes)

| Archivo | Responsabilidad |
|---|---|
| `app.py` | Aplicación principal: configuración, login/registro, panel `/`, billetera, Free Fire Latam legado, `/api.php`, catálogo activo, recarga dinámica por API, admin grande, tareas de fondo. **Es enorme: usa búsqueda/lectura por zonas, no lo reescribas entero.** |
| `pg_compat.py` | Capa que hace que el código escrito para SQLite funcione con PostgreSQL (`?`→`%s`, `datetime('now')`→`NOW()`, filas tipo dict). **Siempre usa `get_db_connection()`; nunca `sqlite3.connect` directo.** |
| `dynamic_games.py` | Juegos/paquetes dinámicos, compra, Mapeo, pasos, respaldo, reconciliación, precios GamePoint, páginas `/juego/d/<slug>`. |
| `proveedores.py` | Admin → Proveedores: Inefable y Central One, saldo, catálogo, interruptor de respaldo. |
| `central_one.py` | Cliente de Central One (Bearer `co_live_...`, catálogo, órdenes con Idempotency-Key, códigos). Sin sandbox: gasta saldo real. |
| `proveedores_usuario.py` | «Mis proveedores»: Central One propio de cada usuario, catálogo, juegos/paquetes privados, llaves cifradas. |
| `precios_auto.py` | Precio de Free Fire ID = costo del proveedor principal, cada 20 min. |
| `bot_freefire.py` | Admin → Bot de Free Fire; PINs de `pines_freefire_global` canjeados por VPS. |
| `pin_manager.py` / `pin_redeemer.py` / `redeem_hype_vps.py` | Stock de PINs y canje (Playwright/VPS). |
| `antiduplic.py` | Pagos bancarios, usos, revisiones, claves de lector, `/api/pagos-banco`, `/api/verificar-pago`, admin antiduplic. |
| `referencias.py` | Apartado y API de Referencias por usuario/banco; gestión de cuentas y claves. |
| `bdv.py` | Conexión de cada usuario con el BDV; cifra con la llave **pública** del lector (NaCl SealedBox). La web no puede abrir esos datos. |
| `bdv_lector_api.py` | API del lector externo del BDV: registro, reto, cuentas, estado, pagos. |
| `id_verify.py` | Verificación de ID/nombre por juego, API configurable y proveedor Inefable. |
| `api_whitelabel.py` | API REST de Marca Blanca: `/api/v1/account`, `/products`, `/recharge`, `/orders/<id>`, `/order-status`, `/balance`; cuentas `webservice_accounts`. |
| `api_panel.py` | Admin/API del usuario: permisos, claves `wsk_` hasheadas, `/mi-api`, `/api/v1/verify-id`. |
| `admin_stats.py` | Estadísticas del admin: resumen, top clientes, series, historial de paquetes, profit. |
| `ganancias.py` | Ganancias del Dashboard: precios/costos/tasas por usuario con historial. |
| `campana.py` | Estado de la campana en vivo (avisos, noticias, saldo). |
| `branding.py` | Logo del sitio en BD y `/brand.css`. |
| `recuperar.py` | Recuperar contraseña por correo (`/recuperar`, `/restablecer/<token>`), Admin → Correo. |
| `admin.html` | Panel admin grande (pestañas). Los demás `admin_*.html` son apartados. |
| `index.html`, `dashboard.html`, `billetera.html`, `referencias.html`, `mis_proveedores.html`, `mi_api.html`, `juego_dinamico.html` | Pantallas principales del revendedor. |
| `static/` | CSS/JS propios (`theme.css`, `themes.css`, `ui.css`, `campana.js`, `ui-select.js`, etc.) y vendor (`chart.umd.min.js`, `html2pdf.bundle.min.js`). |
| `lector_externo/` | Programa **aparte** que lee el BDV en su propio servidor. No se despliega con la web. |
| `redeemer_service/` | Servicio FastAPI + Playwright para canjes Hype; se llama por `VPS_REDEEM_URL`. |
| `dev/` | Mock de proveedor y seeds de prueba. No producción. |

**Legado / no tocar creyendo que es la web actual:** `README.md`, `README_API.md`, `API_GUIDE.md`, `API_CONEXION_COMPLETADA.md`, `CONNECTION_API_GUIDE.md`, `SIMPLE_API_GUIDE.md`, `DESPLIEGUE_API.md`, `production_config.py`, `api_standalone.py`, `connection_api.py`, `simple_connection_api.py`, `start_api.py`, `start_connection_api.py`. Son etapas anteriores o servicios sueltos.

---

## 6. Base de datos (modelo mental)

- **Usuarios y dinero:** `usuarios` (saldo, `sin_ganancia`, bono), `creditos_billetera` (acreditaciones y aviso), `recargas_binance`, `historial_compras`, `transacciones`.
- **Catálogo dinámico:** `juegos_dinamicos` (incluye `usuario_id` para juegos privados), `paquetes_dinamicos`, `pines_dinamicos_stock`, `transacciones_dinamicas`.
- **Precios y profit:** `precios_paquetes`, `precios_freefire_global`, `precios_freefire_id`, `precios_compra`, `profit_ledger`, `profit_daily_aggregate`, `monthly_user_spending`, `ventas_semanales`.
- **Mapeo/proveedores:** `rev_catalog_items`, `rev_item_mappings`, `rev_item_mapping_steps`, `rev_item_mapping_respaldo`, `proveedores_usuario`, `catalogo_proveedor_usuario`.
- **PINs:** `pines_freefire` (Latam legado), `pines_freefire_global` (bot), `pines_dinamicos_stock`.
- **Referencias/bancos:** `cuentas_banco`, `pagos_banco`, `revisiones_pago`, `pagos_borrados`, `claves_lector`; los usos van en tablas de `antiduplic.py` (por usuario y banco).
- **API:** `webservice_accounts` (claves hasheadas, tipo, permisos), `api_orders`, `api_recharges_log`.
- **Ganancias:** `ganancias_precios`, `ganancias_tasas` (historial con `desde`).
- **Avisos:** `noticias`, `noticias_vistas`, `notificaciones_personalizadas`.
- **Configuración:** `configuracion_redeemer` (clave/valor: logo, respaldo, bot, avisos, precios auto, etc.), `recuperar_clave`.

**Migraciones:** `init_db()` y los `init_tablas()` corren al importar `app.py`. En producción, con 2 workers, se corre **una vez con la web detenida** para evitar carreras. Respaldar antes.

---

## 7. Rutas y APIs (resumen)

### Sesión de usuario

- `/`, `/dashboard`, `/billetera`, `/referencias`, `/referencias/docs`, `/referencias/docs/bdv`, `/mis-proveedores`, `/mi-api`, `/mi-api/docs`, `/notificaciones`, `/noticias`.
- Juegos: `/juego/d/<slug>` (dinámicos), `/juego/freefire_latam` (legado). `/juego/freefire` y `/juego/freefire_id` están retirados y redirigen.
- Acciones: `/login`, `/register`, `/logout`, `/recuperar`, `/restablecer/<token>`.

### Sesión de admin

- `/admin` (panel), `/admin/pins`, `/admin/stats/*`, `/admin/dynamic-games`, `/admin/proveedores`, `/admin/verificacion-ids`, `/admin/bot-freefire`, `/admin/api`, `/admin/antiduplic`, `/admin/correo`, `/admin/marca`, `/admin/profitability`, `/admin/weekly_sales`, `/admin/api_recharges_log`.

### APIs externas

- **Marca Blanca** (`X-API-Key` o `?api_key=`): `/api/v1/account`, `/api/v1/products`, `/api/v1/recharge`, `/api/v1/orders/<id>`, `/api/v1/order-status`, `/api/v1/balance`.
- **Permisos por clave:** `perm_recargas`, `perm_verificar_id`, `perm_verificar_pago`. Una clave de un tipo no sirve para el otro (403).
- **Referencias** (`X-API-Key` tipo Referencias): `/api/v1/referencias/resumen`, `/pagos`, `/pagos/<id>`, `/revisiones`, `/cuentas`, `/asignar`, `/revisiones/<id>/resolver`, `/revisiones/<id>/descartar`; más `/api/verificar-pago` y `/api/pagos-banco`.
- **Lector del BDV** (`Authorization: Bearer <clave temporal>`): `/api/lector-bdv/registro`, `/reto`, `/cuentas`, `/estado`, `/pagos`.
- **API interna/legado:** `/api.php` (GET/POST, email+contraseña, Free Fire Latam), `/api/catalog/active` y `/api/recharge/dynamic` (protegidas con `WEBB_API_KEY`).
- **Internas de la web:** `/api/verificar-id`, `/api/news/unread`, `/api/notifications/freefire_id_reload`, `/campana/estado`, `/campana/visto`, `/ganancias/datos`, `/ganancias/precios`.

**Regla:** las claves se guardan como hash y se muestran **una sola vez**. No las loguees, no las devuelvas en HTML y no las pongas en el repo.

---

## 8. Integraciones y variables (solo nombres)

Nunca imprimas ni commitees valores. En producción van en el `.env` del servidor.

- **App:** `SECRET_KEY`, `FLASK_ENV`, `DEFAULT_TZ` (America/Caracas), `PORT`, `WEB_CONCURRENCY`, `GUNICORN_TIMEOUT`, `DEV_LOGIN` (solo local), `FORCE_HTTPS`.
- **Base:** `DATABASE_URL` (PostgreSQL) o `DATABASE_PATH` / `DEV_DATABASE_PATH` (SQLite local).
- **Admin:** `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `ADMIN_USER_IDS`, `ADMIN_EMAILS`.
- **Correo:** `MAIL_USERNAME`, `MAIL_PASSWORD` (contraseña de aplicación de Gmail), `SITE_URL`.
- **Binance Pay:** `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_PAY_ID`, `BINANCE_PROXY` (opcional), `RECARGA_BONUS_PERCENT`, `RECARGA_EXPIRATION_MINUTES`, `RECARGA_MIN_USDT`, `RECARGA_MAX_USDT`.
- **Proveedor Inefable:** `REVENDEDORES_BASE_URL`, `REVENDEDORES_API_KEY`; legado `INEFABLE_USUARIO`, `INEFABLE_CLAVE`.
- **Central One:** `CENTRALONE_API_KEY`, `CENTRALONE_BASE_URL` (opcional). La llave del dueño; los usuarios usan la suya en «Mis proveedores».
- **GamePoint Club:** `GAMECLUB_BASE_URL`, `GAMECLUB_PARTNERID`, `GAMECLUB_SECRET`, `GAMECLUB_PROXY` (opcional).
- **Verificación de IDs:** `INEFABLE_VERIFY_URL`, `INEFABLE_VERIFY_KEY`.
- **API externa/CRM:** `REVENDEDORES_API_TOKEN`, `WEBB_API_KEY`, `WEBB_FF_TIPO`.
- **Pagos bancarios:** `PAGOS_BANCO_TOKEN` (bot del banco); las claves de usuario son `lec_...` / `wsk_...`.
- **BDV:** `BDV_CLAVE_PUBLICA` (opcional; si no está, se aprueba el lector en el panel).
- **Canje VPS:** `VPS_REDEEM_URL`.
- **Cifrado de llaves de «Mis proveedores»:** `PROVEEDORES_USUARIO_SECRETO` (si no está, usa `SECRET_KEY`). **No cambiar `SECRET_KEY` ni ese secreto en producción sin avisar:** las llaves guardadas dejan de leerse y cada usuario debe volver a pegar la suya.
- **Sync:** `DYN_SYNC_INTERVAL_HOURS`.

---

## 9. Dinero y seguridad: invariantes

- **Saldo atómico:** descuenta con `UPDATE usuarios SET saldo = saldo - ? WHERE id = ? AND saldo >= ?` y revisa `rowcount`. Nunca "leer saldo, restar en Python y guardar" sin condición.
- **Reembolso:** si el proveedor no entrega, devuelve exactamente lo cobrado. Si es multi-paso, devuelve la parte no entregada. No reembolses dos veces.
- **Idempotencia:** usa `transaccion_id` / `external_order_id` / `Idempotency-Key` para no cobrar ni entregar dos veces. Central One no tiene sandbox: cada orden mueve saldo real.
- **Estados:** `pendiente` / `procesando` / `completado` / `rechazado` / `en proceso`; la reconciliación decide con datos del proveedor, no a ciegas.
- **Admin:** toda ruta nueva de admin debe validar `session.get('is_admin')` antes de leer o escribir. No confíes en que el enlace esté oculto.
- **Usuario:** toda consulta de datos de usuario filtra por `usuario_id`; un ID ajeno responde **404**, no 403 (no confirmes que existe).
- **API:** clave por hash, permisos por tipo, límites por clave; no devuelvas la clave completa.
- **Referencias:** nunca sumar saldo ni crear órdenes desde ahí; solo verificar y registrar usos.
- **BDV:** la web solo tiene la llave **pública** del lector; no puede descifrar usuario/contraseña/cuenta. No agregues descifrado ni navegador en la web.
- **Secretos:** no leer ni imprimir el `.env`; no loguear claves, PINs completos, datos bancarios ni correos de clientes. Enmascara.
- **CSRF/XSS:** las plantillas escapan por defecto; no uses `|safe` con datos de usuario. No hay tokens CSRF explícitos: no agregues acciones sensibles por GET.
- **Sesión:** 30 minutos, cookie HttpOnly y SameSite=Lax; `Secure` solo con HTTPS (`FORCE_HTTPS`).
- **Errores:** no muestres tracebacks ni detalles internos al cliente; al admin, lo justo.

---

## 10. Despliegue (sin datos sensibles)

1. **Confirma con el operador** y asegúrate de que el árbol está limpio.
2. Corre las pruebas que toquen el cambio sobre SQLite temporal. No uses producción.
3. En el VPS: **respaldo de código y de PostgreSQL** antes de tocar nada.
4. Copia solo los archivos cambiados (el servidor **no es un repo git**; se copian archivos). No subas `test_*`, `.env`, bases ni respaldos.
5. Si hay migración, **detén el servicio**, corre las migraciones una vez y vuelve a arrancar (2 workers sin preload: si no, se pisan).
6. Verifica: servicio activo, `/auth` y páginas públicas en 200, login admin, logs sin tracebacks, y una compra/consulta de prueba si aplica.
7. Los estáticos salen con caché de **4 horas**: tras cambiar CSS/JS, `Ctrl+F5` o cerrar/reabrir la pestaña.
8. Con OK del operador, adelanta `main` a lo desplegado **solo con fast-forward** (sin forzar). No empujes `main` por tu cuenta.

**Datos exactos del servidor** (host, usuario SSH, ruta, servicio, puerto de gunicorn, base, respaldos): `AGENTS.local.md`, en la raíz del proyecto y **fuera de Git**. Si no existe, pídeselos al operador; no los inventes.

---

## 11. Estado actual (20-nov-2026)

- **Código desplegado:** `06f7a25` — "Dashboard: filtro por fecha en el resumen del día" (20-nov-2026). Los tres archivos de la web en el servidor coinciden con ese commit (verificado ignorando CRLF).
- **Rama:** `rediseno-sebastian`; `main` y `origin/main` incluyen `06f7a25` y el commit de contexto `6d6a4af` (puede ir uno o dos commits de documentación después del código). **`master` es un puntero viejo (`c221c3c`): no lo uses.**
- **Producción:** `https://3srevendedores.uk` responde 200 (pantalla de acceso); `/dashboard` sin sesión redirige (302) a `/auth`. Servicio activo.
- **Feature recientes ya desplegadas:** Referencias por usuario/banco, API con permisos y claves hasheadas, BDV con lector externo, «Mis proveedores» (Central One propio), precio automático de Free Fire ID, Ganancias en el Dashboard, campana de notificaciones en vivo y **calendario para ver el Resumen del día por fecha** (`/dashboard?fecha=YYYY-MM-DD`).
- **Sin pendientes de código desplegados.** Verifica `git status` y `git log` antes de asumir que este archivo está al día.
- **Deuda conocida:** README y guías viejas desactualizadas; repo público; sin CI; sin 2FA en esta web; el bot de Free Fire es antiguo y el operador dijo que lo actualizaría más adelante; las tablas de Blood Strike siguen en producción vacías (no se usan).

---

## 12. Trampas conocidas (no hacer)

- **No mezclar con 3srecargas:** esa web usa Flask 3/SQLAlchemy, puerto 5001, TOTP, cifrado de códigos y otro servidor. No copies sus archivos, rutas, migraciones ni reglas aquí.
- **No mezclar con 3S-Ticket:** el CRM vive en `C:\3STICKET` y en otro VPS. Si el pedido es de WhatsApp/IA/flujos, pregunta antes de tocar.
- **No usar el puerto 5001** para este proyecto ni matar el proceso de 3srecargas.
- **No hardcodear juegos, paquetes ni precios:** todo se crea en el admin y vive en la base.
- **No reintroducir Blood Strike** ni las páginas viejas de Free Fire ID/Free Fire de Códigos.
- **No sacar los PINs del bot por API** ni mezclarlos con el catálogo normal.
- **No cambiar el precio de Free Fire ID a mano** esperando que se quede: `precios_auto` lo devuelve al costo cada 20 min.
- **No recalcular días pasados** de Ganancias con precios/tasas actuales: el historial existe para que no cambien.
- **No meter lógica de saldo en Referencias** ni crear órdenes desde ahí.
- **No tocar `profit_ledger`** por compras de «Mis proveedores» (monto 0; el costo es manual).
- **No cambiar `SECRET_KEY`/`PROVEEDORES_USUARIO_SECRETO`** en producción sin avisar (rompe llaves guardadas).
- **No leer/imprimir el `.env`** ni commitear `revendedores_local.db`, backups, `test_*` generados o datos de clientes.
- **No poner IPs, usuarios SSH, correos ni claves en este archivo:** el repo es público.
- **No decir "subido" o "funciona" sin verificarlo** en git, servidor, página y logs.

---

## 13. Checklist antes de decir "listo"

- [ ] ¿Entendí qué parte exacta de la pantalla/pedido quiere cambiar?
- [ ] ¿El cambio es de 3sRevendedores y no de 3srecargas/3S-Ticket?
- [ ] ¿Toqué solo los archivos necesarios?
- [ ] ¿Respeté precios, stock, saldos y datos vivos?
- [ ] ¿Probé en local con SQLite temporal (o en una copia), no contra producción?
- [ ] ¿Revisé el diff y no hay claves, `.env`, bases ni datos personales?
- [ ] ¿Verifiqué páginas 200, login, logs sin tracebacks y el flujo tocado?
- [ ] ¿El operador autorizó subir/desplegar? Si no, dejé el cambio local y se lo dije.
- [ ] ¿Actualicé este archivo si cambió algo estructural (rama, estado, reglas)?

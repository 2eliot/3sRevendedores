"""
Marca del sitio: logo subido desde el panel admin.

El logo se guarda en configuracion_redeemer (base64) para que no se pierda
al redesplegar (el disco del servidor puede ser efímero). Se sirve en
/brand/logo y /brand.css reemplaza el texto "3sRevendedores" de la cabecera
pública y de la barra lateral del admin por la imagen.

Tamaño recomendado: 480 × 120 px (proporción 4:1), PNG/WEBP con fondo
transparente. Se muestra a 32 px de alto en la cabecera. Máximo 512 KB.
"""
import base64
import logging
import time

from flask import Blueprint, Response, flash, redirect, render_template, request, session

from pg_compat import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint('branding', __name__)

KEY_LOGO = 'site_logo'          # valor: "<mime>;<version>;<base64>"
MAX_BYTES = 512 * 1024
MIMES = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
         'webp': 'image/webp', 'svg': 'image/svg+xml', 'gif': 'image/gif'}


def _read_logo():
    try:
        conn = get_db_connection()
        row = conn.execute('SELECT valor FROM configuracion_redeemer WHERE clave = ?', (KEY_LOGO,)).fetchone()
        conn.close()
        if not row or not row['valor']:
            return None
        mime, ver, data = row['valor'].split(';', 2)
        return {'mime': mime, 'ver': ver, 'data': data}
    except Exception as e:
        logger.warning(f'[Branding] No se pudo leer el logo: {e}')
        return None


def _write_logo(value):
    conn = get_db_connection()
    if value is None:
        conn.execute('DELETE FROM configuracion_redeemer WHERE clave = ?', (KEY_LOGO,))
    else:
        conn.execute(
            "INSERT INTO configuracion_redeemer (clave, valor, fecha_actualizacion) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, fecha_actualizacion = EXCLUDED.fecha_actualizacion",
            (KEY_LOGO, value)
        )
    conn.commit()
    conn.close()


def logo_version():
    """Versión del logo actual ('' si no hay). Se usa para romper la caché."""
    try:
        conn = get_db_connection()
        row = conn.execute("SELECT valor FROM configuracion_redeemer WHERE clave = ?", (KEY_LOGO,)).fetchone()
        conn.close()
        if row and row['valor']:
            return row['valor'].split(';', 2)[1]
    except Exception:
        pass
    return ''


def brand_css_url():
    return f'/brand.css?v={logo_version() or "0"}'


@bp.route('/brand.css')
def brand_css():
    logo = _read_logo()
    if not logo:
        css = '/* sin logo personalizado */\n'
    else:
        url = f"/brand/logo?v={logo['ver']}"
        css = f"""
.header::before {{
  content: '' !important;
  height: 34px; padding-left: 0 !important;
  background: url("{url}") no-repeat left center / contain !important;
}}
.admin-sidebar::before {{
  content: '' !important;
  height: 30px; padding: 0 !important; margin: 4px 10px 12px !important;
  background: url("{url}") no-repeat left center / contain !important;
}}
.admin-shell.sb-collapsed .admin-sidebar::before {{ margin: 4px 4px 12px !important; }}
"""
    resp = Response(css, mimetype='text/css')
    resp.headers['Cache-Control'] = 'public, max-age=300'
    return resp


@bp.route('/brand/logo')
def brand_logo():
    logo = _read_logo()
    if not logo:
        return Response(status=404)
    resp = Response(base64.b64decode(logo['data']), mimetype=logo['mime'])
    resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    if logo['mime'] == 'image/svg+xml':
        # Un SVG servido directamente no debe poder ejecutar scripts
        resp.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'; sandbox"
    return resp


@bp.route('/admin/marca', methods=['GET', 'POST'])
def admin_marca():
    if not session.get('is_admin'):
        return redirect('/auth')
    if request.method == 'POST':
        if request.form.get('accion') == 'quitar':
            _write_logo(None)
            flash('Logo eliminado. Se muestra de nuevo el nombre del sitio.', 'success')
            return redirect('/admin/marca')
        f = request.files.get('logo')
        if not f or not f.filename:
            flash('Selecciona una imagen.', 'error')
            return redirect('/admin/marca')
        ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
        if ext not in MIMES:
            flash('Formato no válido. Usa PNG, JPG, WEBP, SVG o GIF.', 'error')
            return redirect('/admin/marca')
        raw = f.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            flash('La imagen pesa más de 512 KB. Redúcela e inténtalo de nuevo.', 'error')
            return redirect('/admin/marca')
        if not raw:
            flash('El archivo está vacío.', 'error')
            return redirect('/admin/marca')
        _write_logo(f"{MIMES[ext]};{int(time.time())};{base64.b64encode(raw).decode()}")
        flash('Logo actualizado.', 'success')
        return redirect('/admin/marca')
    return render_template('admin_marca.html', logo=_read_logo())

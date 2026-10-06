"""
Antiduplic (Referencias) — apartado del panel admin (Blueprint Flask).

Solo administradores. Por ahora es una página vacía; su función se define más adelante.
"""
from flask import Blueprint, flash, redirect, render_template, session

bp = Blueprint('antiduplic', __name__)


@bp.route('/admin/antiduplic')
def admin_antiduplic():
    if not session.get('is_admin'):
        flash('Acceso denegado. Solo administradores.', 'error')
        return redirect('/auth')
    return render_template('admin_antiduplic.html')

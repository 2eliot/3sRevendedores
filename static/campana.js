/* ============================================================
   Campana de notificaciones de la barra (campana.py)
   - Avisos del admin, noticias y saldo acreditado; el admin ve el saldo acreditado a todos.
   - Pregunta cada 30 s. Si llega algo nuevo, la campana se sacude y el número late hasta que se abre.
   - Al tocarla se abre la lista aquí mismo (antes llevaba a /notificaciones) y se marca todo como visto.
   ============================================================ */
(function () {
  'use strict';
  var bell = document.querySelector('.header .news-icon-container') || document.querySelector('.news-icon-container');
  if (!bell || window.__campanaLista) return;
  window.__campanaLista = true;

  bell.removeAttribute('onclick');
  bell.onclick = null;
  bell.setAttribute('role', 'button');
  bell.setAttribute('aria-haspopup', 'true');
  bell.setAttribute('aria-expanded', 'false');
  bell.setAttribute('aria-label', 'Notificaciones');
  bell.title = 'Notificaciones';
  bell.tabIndex = 0;

  var panel = document.createElement('div');
  panel.className = 'campana-panel';
  panel.hidden = true;
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'Notificaciones');
  document.body.appendChild(panel);

  var items = [], nuevos = null, abierto = false, esAdmin = false;
  var ICONO = {saldo: '💰', aviso: '📢', noticia: '📰'};

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"'`]/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '`': '&#96;'}[c];
    });
  }

  function numero(n) {
    var b = bell.querySelector('.news-notification-badge');
    if (!n) { if (b) b.remove(); return; }
    if (!b) { b = document.createElement('div'); b.className = 'news-notification-badge'; bell.appendChild(b); }
    b.textContent = n > 9 ? '9+' : String(n);
    b.classList.add('campana-late');   // late hasta que se abra
  }

  function sacudir() {
    bell.classList.remove('campana-sacude');
    void bell.offsetWidth;   // reinicia la animación
    bell.classList.add('campana-sacude');
  }
  bell.addEventListener('animationend', function () { bell.classList.remove('campana-sacude'); });

  function pintar() {
    panel.innerHTML = '<div class="campana-cab"><span>Notificaciones</span>' +
      (esAdmin ? '' : '<a href="/notificaciones">Ver todo</a>') + '</div>' +
      (items.length ? items.map(function (i) {
        return '<div class="campana-item' + (i.nuevo ? ' nuevo' : '') + '"><span class="campana-ico" aria-hidden="true">' + (ICONO[i.tipo] || '🔔') + '</span>' +
          '<div class="campana-cuerpo"><div class="campana-tit">' + esc(i.titulo) + '</div>' +
          (i.texto ? '<div class="campana-txt">' + esc(i.texto) + '</div>' : '') +
          '<div class="campana-fecha">' + esc(i.fecha) + '</div></div>' + (i.nuevo ? '<span class="campana-punto" aria-label="nuevo"></span>' : '') + '</div>';
      }).join('') : '<div class="campana-vacio">No tienes notificaciones.</div>');
  }

  function posicionar() {
    var r = bell.getBoundingClientRect(), ancho = Math.min(360, window.innerWidth - 16);
    panel.style.top = Math.round(r.bottom + 8) + 'px';
    panel.style.left = Math.round(Math.max(8, Math.min(r.right - ancho, window.innerWidth - ancho - 8))) + 'px';
  }

  async function estado(primera) {
    try {
      var r = await fetch('/campana/estado', {headers: {'Accept': 'application/json'}, credentials: 'same-origin'});
      if (!r.ok) return;
      var d = await r.json();
      if (!d.ok) return;
      items = d.items || [];
      esAdmin = !!d.admin;
      if ((nuevos !== null && d.nuevos > nuevos) || (primera && d.nuevos > 0)) sacudir();
      nuevos = d.nuevos;
      if (abierto) pintar(); else numero(nuevos);
    } catch (e) { /* sin conexión: se intenta en la próxima vuelta */ }
  }

  function abrir() {
    abierto = true;
    pintar();
    posicionar();
    panel.hidden = false;
    bell.setAttribute('aria-expanded', 'true');
    numero(0);
    if (nuevos) {
      fetch('/campana/visto', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}', credentials: 'same-origin'}).catch(function () {});
      nuevos = 0;
    }
  }

  function cerrar() {
    abierto = false;
    panel.hidden = true;
    bell.setAttribute('aria-expanded', 'false');
    items.forEach(function (i) { i.nuevo = false; });
  }

  bell.addEventListener('click', function (e) { e.stopPropagation(); if (abierto) cerrar(); else abrir(); });
  bell.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (abierto) cerrar(); else abrir(); }
  });
  panel.addEventListener('click', function (e) { e.stopPropagation(); });
  document.addEventListener('click', function () { if (abierto) cerrar(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && abierto) { cerrar(); bell.focus(); } });
  window.addEventListener('resize', function () { if (abierto) posicionar(); });
  window.addEventListener('scroll', function () { if (abierto) posicionar(); }, {passive: true});

  estado(true);
  setInterval(function () { if (!document.hidden) estado(false); }, 30000);
  document.addEventListener('visibilitychange', function () { if (!document.hidden) estado(false); });
})();

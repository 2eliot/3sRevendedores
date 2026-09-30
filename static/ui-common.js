/* Utilidades comunes de la interfaz.
   - Botones [data-copy]: copian su valor al portapapeles y muestran un ✓ breve. */
(function () {
  'use strict';
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy') ? resolve() : reject(); } catch (e) { reject(e); }
      ta.remove();
    });
  }
  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('[data-copy]');
    if (!btn) return;
    e.preventDefault();
    copyText(btn.getAttribute('data-copy')).then(function () {
      btn.classList.add('is-copied');
      setTimeout(function () { btn.classList.remove('is-copied'); }, 1400);
    });
  });
})();

/* Selector de tema: cualquier botón [data-open-theme] abre el diálogo.
   El tema lo aplica theme-init.js (window.siteTheme). */
(function () {
  'use strict';
  var MODES = [['system', 'Sistema', 'th-prev-sys'], ['dark', 'Oscuro', 'th-prev-dark'], ['light', 'Claro', 'th-prev-light']];
  var ACCENTS = [['original', 'Predeterminado', '#ffffff'], ['coral', 'Rojo', '#ff1a1a'], ['morado', 'Morado', '#a78bfa'], ['azul', 'Azul', '#56c2ff'],
                 ['verde', 'Verde', '#59d499'], ['ambar', 'Ámbar', '#ffc531'], ['rosa', 'Rosa', '#f472b6']];
  function label(t) {
    var m = MODES.filter(function (x) { return x[0] === t.mode; })[0];
    var a = ACCENTS.filter(function (x) { return x[0] === t.accent; })[0];
    return (m ? m[1] : 'Oscuro') + ' · ' + (a ? a[1] : 'Predeterminado');
  }
  function refreshLabels() {
    if (!window.siteTheme) return;
    var txt = label(window.siteTheme.get());
    document.querySelectorAll('.ub-theme-now').forEach(function (el) { el.textContent = txt; });
  }
  var dlg;
  function build() {
    dlg = document.createElement('dialog');
    dlg.className = 'th-dialog';
    dlg.setAttribute('aria-labelledby', 'th-title');
    var modes = MODES.map(function (m) {
      return '<button type="button" class="th-mode" role="radio" data-mode="' + m[0] + '"><span class="th-prev ' + m[2] + '"></span>' + m[1] + '</button>';
    }).join('');
    var accents = ACCENTS.map(function (a) {
      return '<button type="button" class="th-accent" role="radio" data-accent="' + a[0] + '"><span class="th-dot" style="background:' + a[2] + '"></span>' + a[1] + '</button>';
    }).join('');
    dlg.innerHTML = '<div class="th-body">' +
      '<div class="th-head"><h3 id="th-title">Tema</h3><button type="button" class="th-close" aria-label="Cerrar">×</button></div>' +
      '<div><p class="th-label" id="th-l-mode">Apariencia</p><div class="th-modes" role="radiogroup" aria-labelledby="th-l-mode">' + modes + '</div></div>' +
      '<div><p class="th-label" id="th-l-acc">Color</p><div class="th-accents" role="radiogroup" aria-labelledby="th-l-acc">' + accents + '</div></div>' +
      '<p class="th-note">"Sistema" sigue el modo claro u oscuro de tu dispositivo. Se guarda en este navegador.</p>' +
      '</div>';
    document.body.appendChild(dlg);
    dlg.addEventListener('click', function (e) {
      if (e.target === dlg || e.target.closest('.th-close')) { dlg.close(); return; }
      var m = e.target.closest('.th-mode'), a = e.target.closest('.th-accent');
      if (m) window.siteTheme.set({ mode: m.dataset.mode });
      if (a) window.siteTheme.set({ accent: a.dataset.accent });
      if (m || a) sync();
    });
  }
  function sync() {
    var t = window.siteTheme.get();
    dlg.querySelectorAll('.th-mode').forEach(function (b) { b.setAttribute('aria-checked', String(b.dataset.mode === t.mode)); });
    dlg.querySelectorAll('.th-accent').forEach(function (b) { b.setAttribute('aria-checked', String(b.dataset.accent === t.accent)); });
    refreshLabels();
  }
  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('[data-open-theme]');
    if (!btn || !window.siteTheme) return;
    e.preventDefault();
    e.stopPropagation();
    var box = document.getElementById('user-box');
    if (box) box.classList.remove('visible');
    if (!dlg) build();
    sync();
    dlg.showModal();
    var sel = dlg.querySelector('.th-mode[aria-checked="true"]');
    if (sel) sel.focus();
  }, true);
  document.addEventListener('site-theme-change', refreshLabels);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', refreshLabels); else refreshLabels();
})();

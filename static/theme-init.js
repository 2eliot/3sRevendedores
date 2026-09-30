/* Tema del sitio: se ejecuta en <head> (sin defer) para aplicar el tema antes de pintar.
   Preferencia en localStorage 'tema' = {"mode": "system"|"dark"|"light", "accent": "original"|"coral"|...}  (original = Predeterminado, coral = Rojo) */
(function () {
  var ACCENTS = ['original', 'coral', 'morado', 'azul', 'verde', 'ambar', 'rosa'];
  var root = document.documentElement;
  var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;

  function read() {
    var t = {};
    try { t = JSON.parse(localStorage.getItem('tema') || '{}') || {}; } catch (e) { t = {}; }
    if (['system', 'dark', 'light'].indexOf(t.mode) < 0) t.mode = 'dark';
    if (ACCENTS.indexOf(t.accent) < 0) t.accent = 'original';
    return t;
  }
  function apply(t) {
    var mode = t.mode === 'system' ? (mq && mq.matches ? 'light' : 'dark') : t.mode;
    root.setAttribute('data-theme', mode);
    root.setAttribute('data-accent', t.accent);
  }
  var current = read();
  apply(current);
  if (mq) {
    var onChange = function () { if (current.mode === 'system') apply(current); };
    if (mq.addEventListener) mq.addEventListener('change', onChange); else if (mq.addListener) mq.addListener(onChange);
  }
  window.siteTheme = {
    accents: ACCENTS,
    get: function () { return { mode: current.mode, accent: current.accent }; },
    set: function (t) {
      current = { mode: t.mode || current.mode, accent: t.accent || current.accent };
      try { localStorage.setItem('tema', JSON.stringify(current)); } catch (e) { /* modo privado */ }
      apply(current);
      document.dispatchEvent(new CustomEvent('site-theme-change', { detail: current }));
    }
  };
})();

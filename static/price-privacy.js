/* ============================================================
   Modo "Ocultar precios" (para capturas de recibos a clientes)
   - Preferencia guardada en este navegador (localStorage).
   - Se aplica antes de pintar (clase en <html>) para evitar parpadeos.
   - Envuelve cada importe "$123.45" en <span class="money"> y, en modo
     oculto, lo muestra como "$****" (también dentro de las opciones).
   - Solo visual: no cambia valores de formularios ni datos.
   ============================================================ */
(function () {
  'use strict';
  var KEY = 'ocultarPrecios';
  var MONEY = /\$\s?\d[\d.,]*/g;
  var SKIP = { SCRIPT: 1, STYLE: 1, TEXTAREA: 1, OPTION: 1, SELECT: 1, TITLE: 1, NOSCRIPT: 1 };
  var MASK = '$****';
  var root = document.documentElement;

  function isOn() { try { return localStorage.getItem(KEY) === '1'; } catch (e) { return false; } }
  function save(v) { try { localStorage.setItem(KEY, v ? '1' : '0'); } catch (e) {} }
  if (isOn()) root.classList.add('hide-prices');

  var busy = false;

  function skipNode(t) {
    var p = t.parentNode;
    if (!p || p.nodeType !== 1) return true;
    if (SKIP[p.tagName]) return true;
    return !!(p.closest && p.closest('.money, [data-no-money], [contenteditable]'));
  }

  function wrapText(t) {
    if (skipNode(t)) return;
    var s = t.nodeValue;
    MONEY.lastIndex = 0;
    if (!MONEY.test(s)) return;
    MONEY.lastIndex = 0;
    var frag = document.createDocumentFragment(), last = 0, m;
    while ((m = MONEY.exec(s))) {
      if (m.index > last) frag.appendChild(document.createTextNode(s.slice(last, m.index)));
      var span = document.createElement('span');
      span.className = 'money';
      span.setAttribute('data-v', m[0]);
      span.textContent = isOn() ? MASK : m[0];
      frag.appendChild(span);
      last = m.index + m[0].length;
    }
    if (last < s.length) frag.appendChild(document.createTextNode(s.slice(last)));
    t.parentNode.replaceChild(frag, t);
  }

  function scan(node) {
    if (!node) return;
    if (node.nodeType === 3) { wrapText(node); return; }
    if (node.nodeType !== 1 || SKIP[node.tagName]) return;
    var w = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null), list = [], t;
    while ((t = w.nextNode())) list.push(t);
    list.forEach(wrapText);
  }

  function maskMoney(text) {
    return text.replace(/\$\s?\d[\d.,]*/g, MASK);
  }

  function applyOptions() {
    var hide = isOn();
    document.querySelectorAll('select option').forEach(function (o) {
      if (o.dataset.fullText == null) {
        MONEY.lastIndex = 0;
        if (!MONEY.test(o.text)) return;
        o.dataset.fullText = o.text;
      }
      var want = hide ? maskMoney(o.dataset.fullText) : o.dataset.fullText;
      if (o.text !== want) o.text = want;
    });
  }

  function apply() {
    busy = true;
    var hide = isOn();
    root.classList.toggle('hide-prices', hide);
    document.querySelectorAll('.money').forEach(function (s) {
      var want = hide ? MASK : s.getAttribute('data-v');
      if (s.textContent !== want) s.textContent = want;
    });
    applyOptions();
    document.querySelectorAll('.price-toggle-input').forEach(function (i) {
      i.checked = hide;
      i.setAttribute('aria-checked', hide ? 'true' : 'false');
    });
    busy = false;
  }

  function start() {
    busy = true;
    scan(document.body);
    busy = false;
    apply();

    new MutationObserver(function (muts) {
      if (busy) return;
      busy = true;
      var options = false;
      muts.forEach(function (m) {
        if (m.type === 'characterData') { wrapText(m.target); return; }
        m.addedNodes.forEach(function (n) {
          if (n.nodeType === 1 && (n.tagName === 'OPTION' || n.querySelector && n.querySelector('option'))) options = true;
          scan(n);
        });
      });
      busy = false;
      if (options) { busy = true; applyOptions(); busy = false; }
    }).observe(document.body, { childList: true, subtree: true, characterData: true });

    document.addEventListener('change', function (e) {
      if (!e.target.classList || !e.target.classList.contains('price-toggle-input')) return;
      save(e.target.checked);
      apply();
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();

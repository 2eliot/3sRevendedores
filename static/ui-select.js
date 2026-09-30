/* ============================================================
   Lista desplegable moderna para <select>
   - El <select> nativo sigue siendo el control real (formularios,
     onchange y scripts funcionan igual); solo se reemplaza la lista.
   - La lista se construye al abrir, con las opciones del momento.
   - En pantallas táctiles se deja el selector nativo del sistema.
   - Excluir un select: atributo data-native.
   ============================================================ */
(function () {
  'use strict';
  if (window.__uiSelectLoaded) return;
  window.__uiSelectLoaded = true;
  if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) return;

  var SEARCH_MIN = 9;     // mostrar buscador desde este número de opciones
  var open = null;        // { select, panel, list, search, items, active }

  function enhanceable(sel) {
    return sel && sel.tagName === 'SELECT' && !sel.multiple && !(sel.size > 1) &&
      !sel.disabled && !sel.hasAttribute('data-native');
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function build(sel) {
    var panel = el('div', 'ui-select-panel');
    panel.setAttribute('role', 'presentation');
    var search = null;
    if (sel.options.length >= SEARCH_MIN) {
      search = el('input', 'ui-select-search');
      search.type = 'text';
      search.placeholder = 'Buscar…';
      search.setAttribute('aria-label', 'Buscar opción');
      search.autocomplete = 'off';
      panel.appendChild(search);
    }
    var list = el('div', 'ui-select-list');
    list.setAttribute('role', 'listbox');
    list.id = 'ui-select-list';
    panel.appendChild(list);

    var items = [];
    Array.prototype.forEach.call(sel.children, function (child) {
      if (child.tagName === 'OPTGROUP') {
        list.appendChild(el('div', 'ui-select-group', child.label));
        Array.prototype.forEach.call(child.children, function (o) { addItem(o, true); });
      } else if (child.tagName === 'OPTION') {
        addItem(child, false);
      }
    });

    function addItem(o, grouped) {
      if (o.hidden) return;
      var item = el('div', 'ui-select-option' + (grouped ? ' is-grouped' : ''));
      item.setAttribute('role', 'option');
      item.id = 'ui-select-opt-' + items.length;
      var label = el('span', 'ui-select-label', o.textContent.trim() || ' ');
      item.appendChild(label);
      if (o.value === '' && !o.disabled) item.classList.add('is-placeholder');
      if (o.disabled) { item.classList.add('is-disabled'); item.setAttribute('aria-disabled', 'true'); }
      if (o.selected) { item.classList.add('is-selected'); item.setAttribute('aria-selected', 'true'); }
      item._option = o;
      item._text = o.textContent.trim().toLowerCase();
      list.appendChild(item);
      items.push(item);
    }

    var empty = el('div', 'ui-select-empty', 'Sin resultados');
    empty.hidden = true;
    list.appendChild(empty);
    return { panel: panel, list: list, search: search, items: items, empty: empty };
  }

  function place(state) {
    var r = state.select.getBoundingClientRect();
    var p = state.panel;
    var vw = window.innerWidth, vh = window.innerHeight;
    var width = Math.max(r.width, 200);
    p.style.minWidth = width + 'px';
    p.style.maxWidth = Math.max(width, Math.min(420, vw - 16)) + 'px';
    var left = Math.min(r.left, vw - p.offsetWidth - 8);
    p.style.left = Math.max(8, left) + 'px';
    var below = vh - r.bottom - 8, above = r.top - 8;
    var h = Math.min(p.scrollHeight, 360);
    if (below < Math.min(h, 240) && above > below) {
      p.style.top = ''; p.style.bottom = (vh - r.top + 6) + 'px';
      p.style.maxHeight = Math.min(360, above - 6) + 'px';
      p.classList.add('is-up');
    } else {
      p.style.bottom = ''; p.style.top = (r.bottom + 6) + 'px';
      p.style.maxHeight = Math.min(360, below - 6) + 'px';
      p.classList.remove('is-up');
    }
  }

  function visibleItems(state) {
    return state.items.filter(function (i) { return !i.hidden && !i.classList.contains('is-disabled'); });
  }

  function setActive(state, item, scroll) {
    if (state.active) state.active.classList.remove('is-active');
    state.active = item || null;
    if (item) {
      item.classList.add('is-active');
      (state.search || state.select).setAttribute('aria-activedescendant', item.id);
      if (scroll) item.scrollIntoView({ block: 'nearest' });
    }
  }

  function move(state, delta) {
    var vis = visibleItems(state);
    if (!vis.length) return;
    var i = vis.indexOf(state.active);
    i = i < 0 ? (delta > 0 ? 0 : vis.length - 1) : Math.max(0, Math.min(vis.length - 1, i + delta));
    setActive(state, vis[i], true);
  }

  function filter(state, q) {
    q = q.trim().toLowerCase();
    var any = false;
    state.items.forEach(function (i) {
      var show = !q || i._text.indexOf(q) !== -1;
      i.hidden = !show;
      if (show) any = true;
    });
    state.list.querySelectorAll('.ui-select-group').forEach(function (g) { g.hidden = !!q; });
    state.empty.hidden = any;
    var vis = visibleItems(state);
    setActive(state, vis[0], true);
  }

  function choose(state, item) {
    if (!item || item.classList.contains('is-disabled')) return;
    var sel = state.select, opt = item._option;
    var changed = !opt.selected;
    close(true);
    if (changed) {
      opt.selected = true;
      sel.dispatchEvent(new Event('input', { bubbles: true }));
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }

  function openFor(sel) {
    if (open && open.select === sel) { close(true); return; }
    close(false);
    var b = build(sel);
    var state = { select: sel, panel: b.panel, list: b.list, search: b.search, items: b.items, empty: b.empty, active: null };
    var host = sel.closest('dialog[open]') || document.body;
    host.appendChild(b.panel);
    sel.classList.add('ui-select-open');
    sel.setAttribute('aria-expanded', 'true');
    sel.setAttribute('aria-controls', b.list.id);
    open = state;
    place(state);
    var current = state.items.filter(function (i) { return i.classList.contains('is-selected'); })[0];
    setActive(state, current || visibleItems(state)[0], true);

    b.list.addEventListener('mousedown', function (e) { e.preventDefault(); });
    b.list.addEventListener('click', function (e) {
      var item = e.target.closest('.ui-select-option');
      if (item) choose(state, item);
    });
    b.list.addEventListener('mousemove', function (e) {
      var item = e.target.closest('.ui-select-option');
      if (item && item !== state.active && !item.classList.contains('is-disabled')) setActive(state, item, false);
    });
    if (b.search) {
      b.search.addEventListener('input', function () { filter(state, b.search.value); });
      b.search.addEventListener('keydown', function (e) { keys(e, state); });
      b.search.focus();
    }
  }

  function close(refocus) {
    if (!open) return;
    var s = open;
    open = null;
    s.panel.remove();
    s.select.classList.remove('ui-select-open');
    s.select.setAttribute('aria-expanded', 'false');
    s.select.removeAttribute('aria-activedescendant');
    if (refocus) s.select.focus({ preventScroll: true });
  }

  function keys(e, state) {
    switch (e.key) {
      case 'ArrowDown': e.preventDefault(); move(state, 1); break;
      case 'ArrowUp': e.preventDefault(); move(state, -1); break;
      case 'Home': if (!state.search) { e.preventDefault(); setActive(state, visibleItems(state)[0], true); } break;
      case 'End': if (!state.search) { e.preventDefault(); var v = visibleItems(state); setActive(state, v[v.length - 1], true); } break;
      case 'Enter': e.preventDefault(); choose(state, state.active); break;
      case ' ': if (!state.search) { e.preventDefault(); choose(state, state.active); } break;
      case 'Escape': e.preventDefault(); e.stopPropagation(); close(true); break;
      case 'Tab': close(false); break;
      default:
        // Escribir para saltar (sin buscador): primera opción que empieza por esa letra
        if (!state.search && e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
          var k = e.key.toLowerCase(), vis = visibleItems(state), start = vis.indexOf(state.active);
          for (var n = 1; n <= vis.length; n++) {
            var it = vis[(start + n) % vis.length];
            if (it._text.charAt(0) === k) { setActive(state, it, true); break; }
          }
        }
    }
  }

  // Abrir con el ratón (se evita la lista nativa)
  document.addEventListener('mousedown', function (e) {
    var sel = e.target.closest && e.target.closest('select');
    if (open && !open.panel.contains(e.target) && sel !== open.select) close(false);
    if (!enhanceable(sel) || e.button !== 0) return;
    e.preventDefault();
    sel.focus({ preventScroll: true });
    openFor(sel);
  }, true);

  // Teclado sobre el select
  document.addEventListener('keydown', function (e) {
    var sel = e.target;
    if (open && sel === open.select) { keys(e, open); return; }
    if (!enhanceable(sel)) return;
    if (e.key === ' ' || e.key === 'Enter' || e.key === 'F4' || (e.altKey && (e.key === 'ArrowDown' || e.key === 'ArrowUp'))) {
      e.preventDefault();
      openFor(sel);
    }
  }, true);

  window.addEventListener('resize', function () { close(false); });
  document.addEventListener('scroll', function (e) {
    if (open && !open.panel.contains(e.target)) place(open);
  }, true);
  // Si el select desaparece o se desactiva mientras está abierto
  setInterval(function () {
    if (open && (!document.contains(open.select) || open.select.disabled || !open.select.offsetParent)) close(false);
  }, 400);
})();

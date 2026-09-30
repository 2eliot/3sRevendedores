/* ============================================================
   Juegos por ID: verificación de ID, paquetes en tarjetas y
   ventana de confirmación antes de recargar.
   Se activa en <form data-id-game="..." data-game-name="..." data-verify="0|1">.
   El <select name="monto"> sigue siendo el campo real que se envía.
   ============================================================ */
(function () {
  'use strict';

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // "86 Diamantes / $1.60 ✅" -> { name: "86 Diamantes", price: "1.60" }
  function parseOption(text) {
    var t = text.replace(/\s+/g, ' ').trim().replace(/[✅❌]/g, '').trim();
    var m = t.match(/^(.*?)\s*\/\s*\$\s*([\d.,]+)\s*$/);
    return m ? { name: m[1], price: m[2] } : { name: t, price: '' };
  }

  function saldoActual() {
    var chip = document.querySelector('#saldo-display, .chip-saldo .chip-value');
    if (!chip) return null;
    var v = parseFloat(chip.getAttribute('data-saldo') || chip.textContent.replace(/[^\d.]/g, ''));
    return isNaN(v) ? null : v;
  }

  function init(form) {
    var select = form.querySelector('select[name=monto]');
    var pid = form.querySelector('input[name=player_id]');
    var pid2 = form.querySelector('input[name=player_id2]');
    var srv = form.querySelector('select[name=servidor]');
    var submitBtn = form.querySelector('button[type=submit]');
    if (!select || !pid || !submitBtn) return;
    form.classList.add('idg');

    // ---- Paquetes en tarjetas ----
    var col = select.closest('.form-col');
    if (col) col.classList.add('idg-hidden-col');
    var box = el('div', 'idg-packages');
    box.appendChild(el('div', 'idg-packages-title', 'Elige un paquete'));
    var grid = el('div', 'idg-grid');
    grid.setAttribute('role', 'radiogroup');
    grid.setAttribute('aria-label', 'Paquetes');
    box.appendChild(grid);
    Array.prototype.forEach.call(select.options, function (o) {
      if (!o.value) return;
      var p = parseOption(o.textContent);
      var card = el('button', 'idg-pkg');
      card.type = 'button';
      card.setAttribute('role', 'radio');
      card.dataset.value = o.value;
      card.appendChild(el('span', 'idg-pkg-name', p.name));
      if (p.price) card.appendChild(el('span', 'idg-pkg-price', '$' + p.price));
      if (o.disabled) { card.disabled = true; card.classList.add('is-disabled'); }
      grid.appendChild(card);
    });
    if (!grid.children.length) grid.appendChild(el('p', 'idg-empty', 'No hay paquetes disponibles por ahora.'));
    form.insertBefore(box, submitBtn);

    function syncSelection() {
      grid.querySelectorAll('.idg-pkg').forEach(function (c) {
        var on = c.dataset.value === select.value;
        c.classList.toggle('is-selected', on);
        c.setAttribute('aria-checked', on ? 'true' : 'false');
      });
      form.classList.toggle('idg-has-pkg', !!select.value);
    }
    grid.addEventListener('click', function (e) {
      var card = e.target.closest('.idg-pkg');
      if (!card || card.disabled) return;
      select.value = card.dataset.value;
      select.dispatchEvent(new Event('change', { bubbles: true }));
      syncSelection();
    });
    select.addEventListener('change', syncSelection);
    syncSelection();

    // ---- Verificar ID ----
    var result = null;
    if (form.dataset.verify === '1') {
      var row = el('div', 'idg-id-row');
      pid.parentNode.insertBefore(row, pid);
      row.appendChild(pid);
      var vbtn = el('button', 'idg-verify-btn', 'Verificar ID');
      vbtn.type = 'button';
      row.appendChild(vbtn);
      result = el('div', 'idg-verify-result');
      result.setAttribute('aria-live', 'polite');
      row.parentNode.appendChild(result);

      vbtn.addEventListener('click', function () {
        if (!pid.value.trim()) { pid.focus(); pid.reportValidity(); return; }
        if (pid2 && !pid2.value.trim()) { pid2.focus(); pid2.reportValidity(); return; }
        vbtn.disabled = true; vbtn.classList.add('is-loading');
        result.className = 'idg-verify-result'; result.textContent = 'Verificando…';
        fetch('/api/verificar-id', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            game: form.dataset.idGame, player_id: pid.value.trim(),
            player_id2: pid2 ? pid2.value.trim() : '', servidor: srv ? srv.value : ''
          })
        }).then(function (r) { return r.json(); }).then(function (d) {
          if (d.ok) {
            form.dataset.playerName = d.name;
            result.className = 'idg-verify-result is-ok';
            result.textContent = '✓ ' + d.name;
          } else {
            delete form.dataset.playerName;
            result.className = 'idg-verify-result is-error';
            result.textContent = '✗ ' + (d.error || 'No se pudo verificar el ID');
          }
        }).catch(function () {
          result.className = 'idg-verify-result is-error';
          result.textContent = '✗ Error de conexión. Intenta de nuevo.';
        }).then(function () { vbtn.disabled = false; vbtn.classList.remove('is-loading'); });
      });
    }
    // Si cambian el ID, la verificación anterior deja de valer
    [pid, pid2].forEach(function (inp) {
      if (!inp) return;
      inp.addEventListener('input', function () {
        delete form.dataset.playerName;
        if (result) { result.className = 'idg-verify-result'; result.textContent = ''; }
      });
    });

    form._idg = { select: select, pid: pid, pid2: pid2, srv: srv, submitBtn: submitBtn };
  }

  // ---- Ventana de confirmación ----
  var dlg = null;
  function getDialog() {
    if (dlg) return dlg;
    dlg = el('dialog', 'idg-dialog');
    dlg.setAttribute('aria-labelledby', 'idg-dlg-title');
    dlg.innerHTML =
      '<div class="idg-dlg-head"><div><h3 id="idg-dlg-title">Confirmar recarga</h3><p class="idg-dlg-game"></p></div>' +
      '<button type="button" class="idg-dlg-x" data-idg-close aria-label="Cerrar">×</button></div>' +
      '<dl class="idg-dlg-rows"></dl>' +
      '<div class="idg-dlg-total"><span>Total a pagar</span><strong class="idg-dlg-price"></strong></div>' +
      '<p class="idg-dlg-saldo"></p>' +
      '<div class="idg-dlg-actions"><button type="button" class="idg-dlg-cancel" data-idg-close>Cancelar</button>' +
      '<button type="button" class="idg-dlg-confirm">Confirmar recarga</button></div>';
    document.body.appendChild(dlg);
    dlg.addEventListener('click', function (e) {
      if (e.target.closest('[data-idg-close]')) { dlg.close(); return; }
      if (e.target === dlg) {
        var r = dlg.getBoundingClientRect();
        if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dlg.close();
      }
    });
    return dlg;
  }

  function row(dl, label, value, cls) {
    var r = el('div', 'idg-dlg-row');
    r.appendChild(el('dt', null, label));
    r.appendChild(el('dd', cls || null, value));
    dl.appendChild(r);
  }

  function openConfirm(form) {
    var f = form._idg, d = getDialog();
    var opt = f.select.options[f.select.selectedIndex];
    var p = parseOption(opt ? opt.textContent : '');
    d.querySelector('.idg-dlg-game').textContent = form.dataset.gameName || '';
    var dl = d.querySelector('.idg-dlg-rows');
    dl.innerHTML = '';
    row(dl, 'Juego', form.dataset.gameName || '');
    row(dl, labelOf(form, f.pid, 'ID de jugador'), f.pid.value.trim(), 'is-mono');
    if (f.pid2) row(dl, labelOf(form, f.pid2, 'Zone ID'), f.pid2.value.trim(), 'is-mono');
    if (f.srv && f.srv.value) row(dl, labelOf(form, f.srv, 'Servidor'), f.srv.value);
    if (form.dataset.playerName) row(dl, 'Nombre del jugador', form.dataset.playerName, 'is-ok');
    else row(dl, 'Nombre del jugador', form.dataset.verify === '1' ? 'Sin verificar' : 'No disponible', 'is-muted');
    row(dl, 'Paquete', p.name);
    d.querySelector('.idg-dlg-price').textContent = p.price ? '$' + p.price : '';
    var s = saldoActual(), precio = parseFloat((p.price || '').replace(',', '.'));
    var saldoP = d.querySelector('.idg-dlg-saldo');
    if (s != null && !isNaN(precio)) {
      var despues = s - precio;
      saldoP.textContent = 'Tu saldo: $' + s.toFixed(2) + ' → $' + despues.toFixed(2) + ' después de la recarga';
      saldoP.classList.toggle('is-error', despues < 0);
      if (despues < 0) saldoP.textContent = 'Saldo insuficiente: tienes $' + s.toFixed(2) + ' y el paquete cuesta $' + precio.toFixed(2);
    } else { saldoP.textContent = ''; }
    var confirmBtn = d.querySelector('.idg-dlg-confirm');
    confirmBtn.onclick = function () {
      form._idgConfirmed = true;
      d.close();
      if (form.requestSubmit) form.requestSubmit(f.submitBtn); else f.submitBtn.click();
    };
    d.showModal();
    confirmBtn.focus();
  }

  function labelOf(form, input, fallback) {
    var lab = input.id && form.querySelector('label[for="' + input.id + '"]');
    return lab ? lab.textContent.replace(/:\s*$/, '').trim() : fallback;
  }

  // Interceptar el envío: primero confirmar
  document.addEventListener('submit', function (e) {
    var form = e.target;
    if (!form._idg || form._idgConfirmed) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    openConfirm(form);
  }, true);

  function start() { document.querySelectorAll('form[data-id-game]').forEach(init); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();

/* Descarga una sección de la página como PDF, generado en el navegador (html2pdf, servido desde la web).
   Uso: <button onclick="descargarPDF('.dc-wrap', 'documentacion.pdf', this)">Descargar PDF</button>
   Mientras se genera, la página pasa a tema claro y se ocultan los botones (clase .pdf-modo en <body>). */
(function () {
  var LIB = '/static/vendor/html2pdf.bundle.min.js';

  function cargarLib() {
    if (window.html2pdf) return Promise.resolve();
    return new Promise(function (ok, mal) {
      var s = document.createElement('script');
      s.src = LIB; s.onload = ok; s.onerror = function () { mal(new Error('No se pudo cargar el generador de PDF')); };
      document.head.appendChild(s);
    });
  }

  window.descargarPDF = function (selector, archivo, boton) {
    var el = document.querySelector(selector);
    if (!el) return;
    var root = document.documentElement;
    var tema = root.getAttribute('data-theme');
    var texto = boton ? boton.textContent : '';
    if (boton) { boton.disabled = true; boton.textContent = 'Generando PDF…'; }
    cargarLib().then(function () {
      root.setAttribute('data-theme', 'light');
      document.body.classList.add('pdf-modo');
      return window.html2pdf().set({
        margin: [10, 10, 12, 10],
        filename: archivo || 'documentacion.pdf',
        image: {type: 'jpeg', quality: 0.95},
        html2canvas: {scale: 2, backgroundColor: '#ffffff', useCORS: true},
        jsPDF: {unit: 'mm', format: 'a4', orientation: 'portrait'},
        pagebreak: {mode: ['css', 'legacy'], avoid: ['pre', 'tr', 'h3', '.dc-note', '.dc-step']}
      }).from(el).save();
    }).catch(function (e) {
      alert((e && e.message) || 'No se pudo generar el PDF. Usa «Imprimir» y guarda como PDF.');
    }).then(function () {
      if (tema) root.setAttribute('data-theme', tema); else root.removeAttribute('data-theme');
      document.body.classList.remove('pdf-modo');
      if (boton) { boton.disabled = false; boton.textContent = texto; }
    });
  };
})();

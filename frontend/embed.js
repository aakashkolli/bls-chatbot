(function () {
  'use strict';

  // Derive the base URL from this script's own src so no manual config is needed.
  // e.g. <script src="https://your-domain.com/embed.js"></script>
  var scriptEl = document.currentScript ||
    (function () {
      var scripts = document.getElementsByTagName('script');
      return scripts[scripts.length - 1];
    })();

  var baseUrl = scriptEl.src.replace(/\/embed\.js(\?.*)?$/, '');

  var iframe = document.createElement('iframe');
  iframe.src = baseUrl + '/widget';
  iframe.title = 'BLS Virtual Advisor';
  iframe.setAttribute('allow', 'clipboard-write');
  iframe.setAttribute('aria-label', 'BLS Virtual Advisor chat widget');

  // Size covers the full widget (button + expanded panel).
  // Widget HTML manages its own open/close state inside.
  iframe.style.cssText = [
    'position:fixed',
    'bottom:0',
    'right:0',
    'width:420px',
    'height:640px',
    'border:none',
    'z-index:2147483647',
    'background:transparent',
    // Prevent layout shift while loading
    'pointer-events:none'
  ].join(';');

  // Enable interaction once the widget has loaded
  iframe.addEventListener('load', function () {
    iframe.style.pointerEvents = 'auto';
  });

  // Insert after DOM is ready
  function insert() {
    document.body.appendChild(iframe);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', insert);
  } else {
    insert();
  }
})();

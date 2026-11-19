/* Interactive map for the study page: this study's 2025 land cover, ESA WorldCover 2021, Esri land cover 2023 and
   Hansen tree cover loss over Esri World Imagery. Layer files, bounds and legends come from data/overlays.json,
   which scripts/s10_figures.py writes. No numbers are typed here. */
(() => {
    'use strict';
    const box = document.getElementById('explore-map');
    if (!box || !window.L) return;
    const esc = t => String(t).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const map = L.map(box, { scrollWheelZoom: false, zoomSnap: 0.5, attributionControl: true });
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 18,
        attribution: 'Imagery: Esri, Maxar, Earthstar Geographics and the GIS user community',
    }).addTo(map);

    fetch('data/overlays.json').then(r => r.json()).then(cfg => {
        const studyBox = L.latLngBounds(cfg.box);
        map.fitBounds(studyBox, { padding: [8, 8] });
        L.rectangle(studyBox, { color: '#111', weight: 1.5, fill: false, interactive: false }).addTo(map);
        const slider = document.getElementById('xp-opacity');
        const legend = document.getElementById('xp-legend');
        const buttons = document.getElementById('xp-buttons');
        const layers = {};
        let current = null;
        cfg.layers.forEach(l => {
            layers[l.id] = L.imageOverlay(l.file, l.bounds, { opacity: slider.value / 100, alt: l.name, className: 'lf-pixelated' });
            const b = document.createElement('button');
            b.type = 'button';
            b.className = 'xp-btn';
            b.textContent = l.name;
            b.dataset.id = l.id;
            b.setAttribute('aria-pressed', 'false');
            b.addEventListener('click', () => show(l.id));
            buttons.append(b);
        });
        const show = id => {
            if (current) map.removeLayer(layers[current]);
            current = id;
            layers[id].setOpacity(slider.value / 100).addTo(map);
            buttons.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.id === id)));
            const l = cfg.layers.find(x => x.id === id);
            legend.innerHTML = `<span class="lf-legend-title">${esc(l.name)}</span>` + l.legend
                .map(([name, col]) => `<span class="lf-key"><i style="background:${esc(col)}"></i>${esc(name)}</span>`).join('');
        };
        slider.addEventListener('input', () => { if (current) layers[current].setOpacity(slider.value / 100); });
        show(cfg.layers[0].id);
    }).catch(() => {
        box.insertAdjacentHTML('afterend', '<p class="xp-unit">The map layers could not be loaded.</p>');
    });
})();

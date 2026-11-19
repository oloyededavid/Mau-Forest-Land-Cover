/* Labelling tool for the reference sample. Points and classes come from data/label_points.json, which
   scripts/s08_sample_design.py writes without the map class. Labels are kept in this browser's storage while
   working (if storage is allowed) and leave as a CSV file that scripts/s09_accuracy.py reads. */
(() => {
    'use strict';
    const $ = id => document.getElementById(id);
    const KEY = 'mau-forest-labels-v1';
    const store = {
        get() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; } },
        set(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch (e) { /* storage blocked: export often */ } },
    };
    const msg = t => { $('lb-msg').textContent = t; };
    const map = L.map('lb-map', { zoomSnap: 0.5, keyboard: false });
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19, maxNativeZoom: 18,
        attribution: 'Imagery: Esri, Maxar, Earthstar Geographics and the GIS user community',
    }).addTo(map);
    const cell = L.rectangle([[0, 0], [0, 0]], { color: '#ffd400', weight: 2, fill: false, interactive: false }).addTo(map);
    const ctx = L.rectangle([[0, 0], [0, 0]], { color: '#ffffff', weight: 1, dashArray: '4 4', fill: false, interactive: false }).addTo(map);

    // square of side `m` metres centred on a point (the map grid is UTM; over 60 m the difference is negligible)
    const square = (lat, lon, m) => {
        const dLat = (m / 2) / 111320, dLon = (m / 2) / (111320 * Math.cos(lat * Math.PI / 180));
        return [[lat - dLat, lon - dLon], [lat + dLat, lon + dLon]];
    };
    const csvCell = v => { const s = String(v ?? ''); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const parseCSV = text => {
        const rows = []; let row = [], f = '', q = false;
        for (let i = 0; i < text.length; i++) {
            const c = text[i];
            if (q) {
                if (c === '"' && text[i + 1] === '"') { f += '"'; i++; } else if (c === '"') q = false; else f += c;
            } else if (c === '"') q = true;
            else if (c === ',') { row.push(f); f = ''; }
            else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(f); rows.push(row); row = []; f = ''; }
            else f += c;
        }
        if (f !== '' || row.length) { row.push(f); rows.push(row); }
        return rows.filter(r => r.some(x => x !== ''));
    };

    fetch('data/label_points.json').then(r => r.json()).then(data => {
        const pts = data.points, classes = data.classes;
        let labels = store.get();
        let i = 0;
        const box = $('lb-classes');
        classes.forEach((c, k) => {
            const b = document.createElement('button');
            b.type = 'button'; b.className = 'xp-btn'; b.dataset.cls = c;
            b.setAttribute('aria-pressed', 'false');
            b.innerHTML = `<kbd>${k + 1}</kbd><span></span>`;
            b.querySelector('span').textContent = c;
            b.addEventListener('click', () => choose(c));
            box.append(b);
        });
        const done = () => pts.filter(p => labels[p.id] && labels[p.id].label).length;
        const render = (zoom = true) => {
            const p = pts[i], l = labels[p.id] || {};
            $('lb-id').textContent = p.id;
            $('lb-pos').textContent = `${i + 1} of ${pts.length} · ${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}`;
            const n = done();
            $('lb-count').textContent = `${n} of ${pts.length} labelled`;
            $('lb-bar').style.width = `${(100 * n / pts.length).toFixed(1)}%`;
            box.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.cls === l.label)));
            $('lb-note').value = l.note || '';
            cell.setBounds(square(p.lat, p.lon, 20));
            ctx.setBounds(square(p.lat, p.lon, 60));
            if (zoom) map.setView([p.lat, p.lon], 18);
        };
        const save = () => store.set(labels);
        const choose = c => {
            const p = pts[i];
            labels[p.id] = { ...(labels[p.id] || {}), label: c, note: $('lb-note').value.trim(), at: new Date().toISOString() };
            save();
            render(false);
            msg(`${p.id}: ${c}`);
        };
        const go = d => { i = Math.max(0, Math.min(pts.length - 1, i + d)); render(); };
        const nextOpen = () => {
            for (let k = 1; k <= pts.length; k++) {
                const j = (i + k) % pts.length;
                if (!(labels[pts[j].id] && labels[pts[j].id].label)) { i = j; render(); return; }
            }
            msg('Every point has a label. Export the CSV.');
        };
        $('lb-prev').addEventListener('click', () => go(-1));
        $('lb-next').addEventListener('click', () => go(1));
        $('lb-skip').addEventListener('click', nextOpen);
        $('lb-zoom').addEventListener('click', () => render());
        $('lb-note').addEventListener('change', () => {
            const p = pts[i];
            if (labels[p.id]) { labels[p.id].note = $('lb-note').value.trim(); save(); }
        });
        try { $('lb-who').value = localStorage.getItem(KEY + '-who') || ''; } catch (e) { /* storage blocked */ }
        $('lb-who').addEventListener('change', () => { try { localStorage.setItem(KEY + '-who', $('lb-who').value.trim()); } catch (e) { /* blocked */ } });
        $('lb-export').addEventListener('click', () => {
            const who = $('lb-who').value.trim();
            const lines = ['id,lon,lat,label,note,interpreter,labelled_at'];
            pts.forEach(p => {
                const l = labels[p.id] || {};
                lines.push([p.id, p.lon, p.lat, l.label || '', l.note || '', who, l.at || ''].map(csvCell).join(','));
            });
            const url = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/csv' }));
            const a = document.createElement('a');
            a.href = url; a.download = 'labels.csv'; document.body.append(a); a.click(); a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            msg(`Exported ${done()} labels for ${pts.length} points.`);
        });
        $('lb-import').addEventListener('change', e => {
            const f = e.target.files[0];
            if (!f) return;
            f.text().then(t => {
                const rows = parseCSV(t), head = rows.shift() || [];
                const ix = n => head.indexOf(n);
                if (ix('id') < 0 || ix('label') < 0) { msg('That file has no id and label columns.'); return; }
                const ids = new Set(pts.map(p => p.id));
                let n = 0, bad = 0;
                rows.forEach(r => {
                    const id = r[ix('id')], lab = r[ix('label')];
                    if (!ids.has(id) || !lab) return;
                    if (!classes.includes(lab)) { bad++; return; }
                    labels[id] = { label: lab, note: ix('note') >= 0 ? r[ix('note')] : '', at: ix('labelled_at') >= 0 ? r[ix('labelled_at')] : '' };
                    n++;
                });
                save(); render(false);
                msg(`Imported ${n} labels${bad ? `, skipped ${bad} with an unknown class` : ''}.`);
            });
            e.target.value = '';
        });
        $('lb-clear').addEventListener('click', () => {
            if (!window.confirm('Remove every label kept in this browser? Export first if you need them.')) return;
            labels = {}; save(); render(false); msg('All labels removed from this browser.');
        });
        document.addEventListener('keydown', e => {
            if (e.target.closest('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey) return;
            const k = e.key;
            if (/^[1-9]$/.test(k) && classes[Number(k) - 1]) { choose(classes[Number(k) - 1]); e.preventDefault(); }
            else if (k === 'ArrowRight' || k === 'n' || k === 'N') { go(1); e.preventDefault(); }
            else if (k === 'ArrowLeft' || k === 'p' || k === 'P') { go(-1); e.preventDefault(); }
            else if (k === 'u' || k === 'U') { nextOpen(); e.preventDefault(); }
            else if (k === 'z' || k === 'Z') { render(); e.preventDefault(); }
        });
        // start at the first unlabelled point
        const first = pts.findIndex(p => !(labels[p.id] && labels[p.id].label));
        i = first < 0 ? 0 : first;
        render();
    }).catch(() => msg('The reference points could not be loaded.'));
})();

"""Build docs/index.html from paper/paper.md. Numbers and figures come from the paper build (scripts/build_paper.py),
so the page and the paper cannot disagree. Also copies the result files the page offers for download into
docs/data/.

    python scripts/build_paper.py && python scripts/build_page.py
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import study  # noqa: E402

MAPS = {  # paper figure image -> (desktop webp, phone webp or None)
    "paper/figures/fig1_composite.jpg": ("assets/maps/m01_composite.webp", "assets/maps/m01_composite_phone.webp"),
    "paper/figures/fig2_landcover.jpg": ("assets/maps/m02_landcover.webp", "assets/maps/m02_landcover_phone.webp"),
    "paper/figures/fig3_agreement.jpg": ("assets/maps/m03_agreement.webp", "assets/maps/m03_agreement_phone.webp"),
    "paper/figures/fig4_hansen.jpg": ("assets/maps/m04_hansen_loss_years.webp", "assets/maps/m04_hansen_loss_years_phone.webp"),
}
FIG = re.compile(r"!\[(\*\*Figure (\d+)\.\*\* )?(.*?)\]\((paper/figures/[^)]+)\)")
DESCRIPTION = ("Land cover in 2025 around the Mau Forest Complex, Kenya, from Sentinel-2: a random forest trained on "
               "map-agreement labels, tested with spatially blocked cross-validation and compared with WorldCover, "
               "Esri and Hansen forest loss.")


def image_size(rel):
    from PIL import Image
    with Image.open(ROOT / "docs" / rel) as im:
        return im.size


def figure_html(m):
    caption, num, desc, src = m.group(0), m.group(2), m.group(3), m.group(4)
    desk, phone = MAPS[src]
    alt = re.sub(r"\*\*", "", desc).replace('"', "&quot;")
    src_tag = f'<source media="(max-width: 760px)" srcset="{phone}">' if phone else ""
    return (f'<figure class="wide reveal mapimg" id="fig-{num}">'
            f'<picture>{src_tag}<img src="{desk}" width="{image_size(desk)[0]}" height="{image_size(desk)[1]}" loading="lazy" alt="{alt}"></picture>'
            f'<figcaption><b>Figure {num}.</b> {desc}</figcaption></figure>')


EXPLORE = """<h3 id="explore">3.7 Interactive map</h3>
<p>Switch between this study's map, the two maps its training labels came from, and Hansen tree cover loss, over Esri World Imagery. Drag the slider to see the imagery through the layer. The black rectangle is the study box. Layers are shown at 40 m, half the resolution of the analysis grid.</p>
<div id="xp-buttons" class="xp-buttons" role="group" aria-label="Map layer"></div>
<p class="xp-pick"><label for="xp-opacity">Layer opacity</label> <input type="range" id="xp-opacity" min="0" max="100" value="80"></p>
<div id="explore-map" class="lf-map" role="region" aria-label="Interactive map of the study box"></div>
<div id="xp-legend" class="lf-legend" aria-live="polite"></div>
<p class="xp-unit">The labelling tool for the reference sample is on <a href="label.html">a separate page</a>.</p>
"""


def tidy(html):
    """Keep a number and its unit together on one line, and put the title and description on one line each."""
    import re
    parts = re.split(r"(<[^>]+>)", html)
    unit = re.compile(r"(\d) (km²|km\b|m\b|%)")
    parts = [p if p.startswith("<") else unit.sub("\\1\u00a0\\2", p) for p in parts]
    html = "".join(parts)
    html = re.sub(r"<title>(.*?)</title>", lambda m: "<title>" + " ".join(m.group(1).split()) + "</title>", html, flags=re.S)
    html = re.sub(r'(<meta name="description" content=")(.*?)(">)', lambda m: m.group(1) + " ".join(m.group(2).split()) + m.group(3), html, flags=re.S)
    return html


def main():
    md = (ROOT / "paper" / "paper.md").read_text(encoding="utf-8")
    # drop the YAML header: the page template prints the title and author itself
    m = re.match(r"^---\n(.*?)\n---\n", md, flags=re.S)
    meta = dict(re.findall(r'^(\w+): "?([^"\n]*)"?$', m.group(1), flags=re.M)) if m else {}
    body = md[m.end():] if m else md
    body = FIG.sub(figure_html, body)
    tmp = ROOT / "paper" / "_page_body.md"
    tmp.write_text(body, encoding="utf-8")
    out = ROOT / "docs" / "index.html"
    try:
        subprocess.run(["pandoc", str(tmp), "-f", "markdown-raw_html+raw_html", "-t", "html5", "--shift-heading-level-by=1",
                        "--template", str(ROOT / "scripts" / "page_template.html"), "-o", str(out),
                        "--metadata", f"pagetitle={meta.get('title', '')}", "--metadata", f"title={meta.get('title', '')}",
                        "--metadata", f"author={meta.get('author', '')}", "--metadata", f"date={meta.get('date', '')}",
                        "--metadata", f"description={DESCRIPTION}"],
                       check=True)
    finally:
        tmp.unlink(missing_ok=True)
    html = out.read_text(encoding="utf-8")
    marker = '<h2 id="discussion">'
    assert html.count(marker) == 1, "discussion heading not found"
    html = html.replace(marker, EXPLORE + marker)
    html = tidy(html)
    html = _study_links(_page_end(html), out.parent)
    assert "{{" not in html, "unfilled placeholder in the page"
    out.write_text(html, encoding="utf-8")
    downloads()
    print("wrote", out)


def downloads():
    """Copy the result files offered on the page into docs/data/."""
    data = ROOT / "docs" / "data"
    results = {p.stem: json.loads(p.read_text()) for p in sorted(study.OUT.glob("*.json"))}
    (data / "results.json").write_text(json.dumps(results, indent=1))
    shutil.copyfile(study.OUT / "reference_sample.csv", data / "reference_sample.csv")


def bbox_text():
    x0, y0, x1, y1 = study.BBOX
    return f"{x0:.2f} to {x1:.2f}°E and {abs(y1):.2f} to {abs(y0):.2f}°S"


def _study_links(html, docs):
    """Re-insert the cross-site links that the portfolio's tools/sync_nav.py keeps in docs/assets/study-*.html."""
    import re as _re
    for key, where in (("study-bar", "top"), ("study-nav", "end")):
        f = docs / "assets" / f"{key}.html"
        if not f.exists():
            continue
        block = f.read_text(encoding="utf-8").strip()
        html = _re.sub(rf"<!-- {key}:start -->.*?<!-- {key}:end -->\n?", "", html, flags=_re.S)
        if where == "top":
            html = _re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + "\n" + block, html, count=1)
        else:
            html = html.replace("</body>", block + "\n</body>", 1)
    f = docs / "assets" / "study-others.html"  # the "Other studies" line in the footer, kept in place
    if f.exists():
        block = f.read_text(encoding="utf-8").strip()
        html = _re.sub(r"<!-- study-others:start -->.*?<!-- study-others:end -->", lambda m: block, html, flags=_re.S)
    return html


def _web_ending(html, end):
    """Shape the end of the page like the other study sites: the paper's own data-and-code section becomes the
    opening paragraph of "Data, code and citation" (so the page says it once), the page end goes before the
    references, and the references become the same compact list, with DOIs shown as doi:..."""
    import re as _re
    m = _re.search(r'<h2 id="[^"]*data-and-code[^"]*">.*?</h2>\s*(.*?)(?=<h2 )', html, flags=_re.S)
    if not m:
        raise SystemExit("page end: the paper's data-and-code section was not found (its heading id changed?)")
    html = html[:m.start()] + html[m.end():]
    end, n = _re.subn(r'(<h2 id="h-dc">.*?</h2>)', lambda k: k.group(1) + "\n    " + m.group(1).strip(), end, count=1, flags=_re.S)
    if not n:
        raise SystemExit('page end: page_end.html has no <h2 id="h-dc"> heading')
    anchor = '<h2 id="references">'
    if anchor not in html:
        raise SystemExit("page end: no References heading to place the end of the page before")
    html = html.replace(anchor, end + "\n" + anchor, 1)
    r = _re.search(r'(<h2 id="references">References</h2>\s*)((?:<p>.*?</p>\s*)+)', html, flags=_re.S)
    if not r:
        raise SystemExit("page end: no reference paragraphs after the References heading")
    else:
        def item(p):
            p = " ".join(p.split())  # pandoc wraps lines, even inside tags
            p = _re.sub(r'<a href="https://doi\.org/([^"]+)"[^>]*>[^<]*</a>', lambda k: f'<a href="https://doi.org/{k.group(1)}">doi:{k.group(1)}</a>', p)
            trim = lambda u: (u[:-1] if u.endswith(")") and u.count("(") < u.count(")") else u).rstrip(".,;:")
            def bare(k, doi):
                u = trim(k.group(1)); rest = k.group(1)[len(u):]
                return (f'<a href="https://doi.org/{u}">doi:{u}</a>' if doi else f'<a href="{u}">{u}</a>') + rest
            p = _re.sub(r'(?<![">/])https://doi\.org/([^\s<]+)', lambda k: bare(k, True), p)
            p = _re.sub(r'(?<![">/])(https?://[^\s<]+)', lambda k: bare(k, False), p)
            return "<li>" + p + "</li>"
        items = [item(p) for p in _re.findall(r"<p>(.*?)</p>", r.group(2), flags=_re.S)]
        html = html[:r.start(2)] + '<ol class="refs" role="list">\n' + "\n".join(items) + "\n</ol>\n" + html[r.end(2):]
    return html


def _page_end(html):
    """The end of the page (key terms, data and citation, a note, feedback) lives in page_end.html next to this script."""
    end = Path(__file__).with_name("page_end.html").read_text(encoding="utf-8").strip()
    return _web_ending(html, end.replace("{{bbox_text}}", bbox_text()))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build a single self-contained HTML report with all 55 plots embedded as
base64 PNGs. One file, no external assets.

Reads:  reports/plots/<source_id>.png  (plus build_report.py captions)
Reads:  sources.yaml
Writes: reports/REPORT_single.html
"""
from __future__ import annotations

import base64
import datetime as dt
import subprocess
import sys
from pathlib import Path
from textwrap import shorten

import yaml

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
PLOTS = REPORTS / "plots"


def ensure_plots():
    if not PLOTS.exists() or len(list(PLOTS.glob("*.png"))) < 50:
        print("Plots missing — running build_report.py first")
        subprocess.check_call([sys.executable, "build_report.py"], cwd=ROOT)


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    ensure_plots()
    sources = yaml.safe_load(open(ROOT / "sources.yaml"))

    # Captions are already cached in REPORT.md; parse them out for parity.
    cap_by_id: dict[str, str] = {}
    for line in (REPORTS / "REPORT.md").read_text().splitlines():
        if line.startswith("_") and line.endswith("_") and "rows" in line:
            # Pair with the preceding `### \`<id>\`` block
            pass

    # Easier: just re-derive by scanning for `### \`...\`` then `_<cap>_` pattern
    md_lines = (REPORTS / "REPORT.md").read_text().splitlines()
    cur_id = None
    for i, line in enumerate(md_lines):
        if line.startswith("### `"):
            cur_id = line.strip("# `")
        elif cur_id and line.startswith("_") and line.endswith("_") and (
            "rows" in line or "snapshot" in line or "panels" in line
        ):
            cap_by_id[cur_id] = line.strip("_")
            cur_id = None

    by_domain: dict[str, list] = {}
    for s in sources:
        by_domain.setdefault(s["domain"], []).append(s)

    domain_order = ["econ_fin", "energy", "nature", "healthcare",
                    "sales", "transport", "web_cloudops"]

    out = []
    out.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    out.append("<title>Timeframe benchmark — all 55 source plots</title>")
    out.append("""<style>
:root { color-scheme: light dark; }
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px;
       margin: 2rem auto; padding: 0 1.4rem; line-height: 1.5;
       color: #1a1a1a; background: #fff; }
h1 { margin: 0 0 .2em 0; font-size: 1.9rem; }
h2 { margin-top: 3rem; padding-bottom: .3em; border-bottom: 2px solid #2a6; }
h3 { font-family: 'JetBrains Mono', ui-monospace, monospace; margin-top: 2rem;
     font-size: 1.05rem; color: #014; }
img { max-width: 100%; border: 1px solid #e0e0e0; margin: .6em 0 1.4em 0;
      display: block; }
.summary { background: #f7f9fc; padding: 1rem 1.2rem; border-left: 3px solid #2a6;
           margin: 1.5rem 0; font-size: .92rem; }
.toc { columns: 2; column-gap: 2rem; font-size: .92rem; margin: 1rem 0 2rem 0; }
.toc a { text-decoration: none; color: #014; }
.toc a:hover { text-decoration: underline; }
.meta { color: #666; font-size: .9rem; }
.tags code { background: #eef3f8; padding: .1em .4em; border-radius: .3em;
             font-size: .82rem; }
.cap { color: #888; font-size: .82rem; font-style: italic; margin-top: 0; }
.notes { color: #555; font-size: .85rem; max-width: 70ch; }
.domain-toc { margin: .6em 0 1.4em 1em; }
.domain-toc a { display: inline-block; margin: 0 .8em .3em 0; font-family: monospace;
                font-size: .85rem; color: #036; text-decoration: none; }
.domain-toc a:hover { text-decoration: underline; }
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1a; color: #ddd; }
  h3 { color: #8cf; }
  img { border-color: #444; background: #fff; }
  .summary { background: #222; border-left-color: #8e6; }
  .tags code { background: #2a2f3a; }
  .toc a, .domain-toc a { color: #8cf; }
}
</style></head><body>""")

    out.append(f"<h1>Timeframe benchmark — source plots</h1>")
    out.append(f"<p class='meta'>Self-contained HTML, all images embedded. "
               f"Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}. "
               f"{len(sources)} verified sources across 7 GIFT-Eval domains.</p>")

    # Summary stats
    from collections import Counter
    by_archetype = Counter()
    for s in sources:
        for a in s["archetypes"]:
            by_archetype[a] += 1
    by_novelty = Counter(s["pretraining_novelty"] for s in sources)
    out.append("<div class='summary'>")
    out.append(f"<strong>{len(sources)} sources</strong> — by domain: " +
               ", ".join(f"{d} {len(by_domain.get(d, []))}" for d in domain_order))
    out.append("<br>by novelty: " + ", ".join(f"{k} {v}" for k, v in by_novelty.most_common()))
    out.append("<br>by archetype: " +
               ", ".join(f"{k} {v}" for k, v in by_archetype.most_common()))
    out.append("</div>")

    # Top-level TOC
    out.append("<h2 style='margin-top:1.5rem;border:none;'>Contents</h2>")
    out.append("<div class='toc'>")
    for d in domain_order:
        if d not in by_domain:
            continue
        out.append(f"<div><a href='#{d}'><strong>{d}</strong></a> ({len(by_domain[d])} sources)</div>")
        out.append("<div class='domain-toc'>")
        for s in by_domain[d]:
            out.append(f"<a href='#{s['id']}'>{s['id']}</a>")
        out.append("</div>")
    out.append("</div>")

    # Sections
    for d in domain_order:
        if d not in by_domain:
            continue
        out.append(f"<h2 id='{d}'>{d} <span class='meta'>— {len(by_domain[d])} sources</span></h2>")
        for s in by_domain[d]:
            sid = s["id"]
            out.append(f"<h3 id='{sid}'>{sid}</h3>")
            out.append(f"<p><strong>{s['name']}</strong></p>")
            out.append(
                f"<p class='tags'>"
                f"archetypes <code>{', '.join(s['archetypes'])}</code> · "
                f"novelty <code>{s['pretraining_novelty']}</code> · "
                f"cadence <code>{s['frequency']}</code> · "
                f"history <code>{s.get('history_available', '?')}</code></p>"
            )
            if s.get("notes"):
                out.append(f"<p class='notes'>{shorten(s['notes'], 280)}</p>")
            cap = cap_by_id.get(sid, "")
            if cap:
                out.append(f"<p class='cap'>{cap}</p>")
            png = PLOTS / f"{sid}.png"
            if png.exists():
                b64 = img_b64(png)
                out.append(f"<img src='data:image/png;base64,{b64}' alt='{sid}'>")
            else:
                out.append(f"<p style='color:red'>plot missing: {png.name}</p>")

    out.append(f"<hr><p class='meta'>End of report — {len(sources)} sources, "
               f"all images embedded.</p>")
    out.append("</body></html>")

    target = REPORTS / "REPORT_single.html"
    target.write_text("\n".join(out))
    size_mb = target.stat().st_size / 1e6
    print(f"Wrote {target} ({size_mb:.2f} MB, {len(sources)} sources, all images embedded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

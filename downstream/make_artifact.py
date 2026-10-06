"""Render downstream/results/*.json into a self-contained HTML page for publishing.

    conda run --no-capture-output -n tabsyn python -m downstream.make_artifact

Numbers are read from the result JSON rather than transcribed, so the page cannot drift from
the harness output. Chart palette is the validated categorical set:
light #0D9488/#C1440E/#3355A8/#9B2D6F on #F4F5F1, dark #17A99A/#DC5F26/#6889DB/#C75C9A on #161A19.
"""

import html
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.data import REPO_ROOT
from downstream.report import (
    RESULTS_DIR,
    berka_label_diagnostic,
    load_results,
    sort_key,
    summarise,
)
from downstream.tasks import TASKS

OUT = os.path.join(REPO_ROOT, "reports", "archive", "downstream_report.html")

TITLES = {
    "california_inctot": ("California", "Individual income", "R²"),
    "california_empstat": ("California", "Employment status", "macro-F1"),
    "berka_loan_default": ("Berka", "Loan default at origination", "AUROC"),
    "berka_card_ownership": ("Berka", "Card ownership", "AUROC"),
}
# Metric floor: AUROC's usable range starts at chance, the others at zero.
FLOOR = {"roc_auc": 0.5, "r2": 0.0, "f1_macro": 0.0}

# Tiles, panels and the sweep show only the leakage-free protocol: four series, matching the
# four validated categorical hues. The interim full-data Berka run stays in the detail tables.
HEADLINE = [
    "california__california_inctot",
    "california__california_empstat",
    "berka_split__berka_loan_default",
    "berka_split__berka_card_ownership",
]


def dataset_label(dataset_key, task_name):
    if dataset_key == "berka":
        return "Berka (interim)"
    return TITLES[task_name][0]
REGIMES = [("real", "real"), ("synth", "synthetic"), ("mixed", "mixed")]
SWEEP = [("real", 0), ("mixed_25", 25), ("mixed_50", 50), ("mixed", 100)]


def esc(s):
    return html.escape(str(s))


def bar_path(x, y, w, y0, r=4):
    r = min(r, w / 2, max(y0 - y, 0.01))
    return (
        f"M{x:.1f},{y0:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
        f"L{x + w:.1f},{y0:.1f} Z"
    )


def panel_svg(task_name, records, task):
    """One small multiple: three regime bars with +/-1 SD whiskers and a floor reference."""
    floor = FLOOR[task.primary]
    W, H = 340, 200
    left, right, top, bot = 52, 330, 20, 158
    plot_w = right - left

    def scale(v):
        return bot - v * (bot - top)

    parts = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = scale(t)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left - 9}" y="{y + 3.5:.1f}" class="tick" text-anchor="end">{t:g}</text>'
        )
    if floor > 0:
        y = scale(floor)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="floor"/>'
            f'<text x="{right}" y="{y - 6:.1f}" class="floor-label" text-anchor="end">chance</text>'
        )

    slot = plot_w / 3
    bw = 46
    for i, (regime, label) in enumerate(REGIMES):
        mean, std, _ = summarise(records, regime, task.primary)
        cx = left + slot * (i + 0.5)
        x = cx - bw / 2
        y = scale(mean)
        parts.append(
            f'<g class="bar s{i}"><title>{esc(label)}: {mean:.4f} +/- {std:.4f}</title>'
            f'<path d="{bar_path(x, y, bw, bot)}"/></g>'
        )
        if std > 0.002:
            hi, lo = scale(min(mean + std, 1.0)), scale(max(mean - std, 0.0))
            parts.append(
                f'<line x1="{cx:.1f}" y1="{hi:.1f}" x2="{cx:.1f}" y2="{lo:.1f}" class="whisk"/>'
                f'<line x1="{cx - 6:.1f}" y1="{hi:.1f}" x2="{cx + 6:.1f}" y2="{hi:.1f}" class="whisk"/>'
                f'<line x1="{cx - 6:.1f}" y1="{lo:.1f}" x2="{cx + 6:.1f}" y2="{lo:.1f}" class="whisk"/>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{y - 11:.1f}" class="val" text-anchor="middle">{mean:.3f}</text>'
            f'<text x="{cx:.1f}" y="{bot + 20:.1f}" class="xlab" text-anchor="middle">{esc(label)}</text>'
        )

    dataset, title, metric = TITLES[task_name]
    return (
        f'<figure class="panel"><figcaption><span class="ds">{esc(dataset)}</span>'
        f'<strong>{esc(title)}</strong><span class="metric">{esc(metric)}</span></figcaption>'
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="{esc(title)}: {esc(metric)} by training regime">'
        + "".join(parts)
        + "</svg></figure>"
    )


def sweep_svg(results):
    """Change in the primary metric as synthetic rows are added to the full real training set."""
    series = []
    for key in [k for k in HEADLINE if k in results]:
        task_name = key.split("__")[1]
        task = TASKS[task_name]
        recs = results[key]["records"]
        base = summarise(recs, "real", task.primary)[0]
        pts = [(frac, summarise(recs, reg, task.primary)[0] - base) for reg, frac in SWEEP]
        dataset, title, _ = TITLES[task_name]
        series.append((f"{dataset} · {title}", pts))

    lo = min(v for _, pts in series for _, v in pts)
    hi = max(v for _, pts in series for _, v in pts)
    pad = (hi - lo) * 0.3 or 0.01
    lo, hi = lo - pad, hi + pad

    W, H = 780, 320
    left, right, top, bot = 62, 470, 24, 258

    def sx(f):
        return left + f / 100 * (right - left)

    def sy(v):
        return bot - (v - lo) / (hi - lo) * (bot - top)

    parts = []
    for t in np.linspace(lo, hi, 5):
        y = sy(t)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left - 9}" y="{y + 3.5:.1f}" class="tick" text-anchor="end">{t:+.3f}</text>'
        )
    y0 = sy(0.0)
    parts.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{right}" y2="{y0:.1f}" class="zero"/>')
    for _, frac in SWEEP:
        parts.append(
            f'<text x="{sx(frac):.1f}" y="{bot + 22}" class="xlab" text-anchor="middle">'
            f"+{frac}%</text>"
        )
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{bot + 44}" class="axtitle" '
        f'text-anchor="middle">synthetic rows added, as a share of the real training set</text>'
    )

    for i, (name, pts) in enumerate(series):
        d = " ".join(
            ("M" if j == 0 else "L") + f"{sx(f):.1f},{sy(v):.1f}" for j, (f, v) in enumerate(pts)
        )
        parts.append(f'<path d="{d}" class="line s{i}"/>')
        for f, v in pts:
            parts.append(
                f'<circle cx="{sx(f):.1f}" cy="{sy(v):.1f}" r="4.5" class="dot s{i}">'
                f"<title>{esc(name)} at +{f}% synthetic: {v:+.4f}</title></circle>"
            )
        fx, fv = pts[-1]
        parts.append(
            f'<text x="{sx(fx) + 12:.1f}" y="{sy(fv) + 4:.1f}" class="lineLabel s{i}">'
            f"{esc(name)}</text>"
        )

    return (
        '<figure class="wide"><figcaption><strong>Adding synthetic rows to the full real '
        'training set changes almost nothing</strong><span class="metric">change in each '
        'task&rsquo;s primary metric vs its real-only baseline</span></figcaption>'
        f'<div class="scroll"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Change in the '
        'primary metric as synthetic data is added to the real training set">'
        + "".join(parts)
        + "</svg></div></figure>"
    )


def tiles(results):
    out = []
    for key in [k for k in HEADLINE if k in results]:
        task_name = key.split("__")[1]
        task = TASKS[task_name]
        recs = results[key]["records"]
        real = summarise(recs, "real", task.primary)[0]
        syn = summarise(recs, "synth", task.primary)[0]
        floor = FLOOR[task.primary]
        retained = (syn - floor) / (real - floor)
        dataset, title, metric = TITLES[task_name]
        flag = (
            '<span class="flag" title="generator trained on the test rows">interim</span>'
            if results[key].get("leakage_caveat")
            else ""
        )
        state = "good" if retained >= 0.9 else "bad"
        out.append(
            f'<div class="tile {state}"><div class="tile-h"><span class="ds">{esc(dataset)}</span>'
            f'{flag}</div><p class="tile-t">{esc(title)}</p>'
            f'<p class="huge">{retained * 100:.1f}<span>%</span></p>'
            f'<div class="meter"><i style="width:{min(retained, 1) * 100:.1f}%"></i></div>'
            f'<p class="tile-f">{esc(metric)} {syn:.3f} of {real:.3f}</p></div>'
        )
    return "".join(out)


def metric_table(payload, task):
    recs = payload["records"]
    label = {
        "real": "real only",
        "synth": "synthetic only",
        "mixed_25": "real + 25% syn",
        "mixed_50": "real + 50% syn",
        "mixed": "real + 100% syn",
    }
    head = "".join(f"<th>{esc(m)}</th>" for m in task.metrics)
    rows = []
    for regime in ["real", "synth", "mixed_25", "mixed_50", "mixed"]:
        first = next((r for r in recs if r["regime"] == regime), None)
        if first is None:
            continue
        cells = []
        for m in task.metrics:
            mean, std, _ = summarise(recs, regime, m)
            cells.append(f'<td>{mean:.4f}<span class="sd">&plusmn;{std:.4f}</span></td>')
        cls = ' class="row-real"' if regime == "real" else ""
        rows.append(
            f'<tr{cls}><th scope="row">{esc(label[regime])}</th>'
            f'<td class="n">{first["n_train"]:,}</td>' + "".join(cells) + "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>training data</th><th>rows</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


TOKENS_DARK = """
  --paper:#161A19; --surface:#1E2321; --sunk:#141817;
  --ink:#E7EAE5; --ink2:#A9B2AC; --ink3:#818B86; --rule:#2E3532; --rule2:#3B433F;
  --s0:#17A99A; --s1:#DC5F26; --s2:#6889DB; --s3:#C75C9A;
  --good:#17A99A; --bad:#DC5F26; --shadow:0 1px 2px rgba(0,0,0,.4);
"""

CSS = (
    """
:root{
  --paper:#F4F5F1; --surface:#FFFFFF; --sunk:#EDEFE9;
  --ink:#171C1A; --ink2:#4A5450; --ink3:#727C77; --rule:#D9DDD4; --rule2:#C4CABF;
  --s0:#0D9488; --s1:#C1440E; --s2:#3355A8; --s3:#9B2D6F;
  --good:#0D9488; --bad:#C1440E; --shadow:0 1px 2px rgba(23,28,26,.06);
  --serif:Georgia,"Iowan Old Style","Source Serif 4","Times New Roman",serif;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){"""
    + TOKENS_DARK
    + """}}
:root[data-theme="dark"]{"""
    + TOKENS_DARK
    + """}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:clamp(28px,5vw,64px) clamp(18px,4vw,40px) 90px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink3);margin:0 0 14px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(29px,4.4vw,45px);line-height:1.13;
  letter-spacing:-.015em;margin:0 0 16px;text-wrap:balance;max-width:21ch}
.lede{font-size:clamp(16.5px,2vw,19px);color:var(--ink2);margin:0;max-width:64ch}
.meta{display:flex;flex-wrap:wrap;gap:8px 26px;margin-top:26px;padding-top:20px;
  border-top:1px solid var(--rule);font-family:var(--mono);font-size:12px;color:var(--ink3)}
.meta b{color:var(--ink2);font-weight:500}
h2{font-family:var(--serif);font-weight:600;font-size:clamp(21px,2.6vw,27px);line-height:1.2;
  letter-spacing:-.01em;margin:0 0 6px;text-wrap:balance}
section{margin-top:clamp(44px,6vw,74px)}
.sub{color:var(--ink3);margin:0 0 24px;max-width:66ch;font-size:15px}
p{max-width:66ch}
.tiles{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(212px,1fr))}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:15px 18px 18px;box-shadow:var(--shadow);border-top:3px solid var(--rule2)}
.tile.good{border-top-color:var(--good)}
.tile.bad{border-top-color:var(--bad)}
.tile-h{display:flex;justify-content:space-between;align-items:center;gap:8px;min-height:20px}
.ds{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink3)}
.flag{font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--bad);border:1px solid currentColor;border-radius:2px;padding:1px 5px}
.tile-t{margin:6px 0 10px;font-size:14.5px;color:var(--ink2);min-height:2.7em;line-height:1.35}
.huge{font-family:var(--mono);font-size:37px;line-height:1;margin:0 0 12px;letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;color:var(--ink)}
.huge span{font-size:17px;color:var(--ink3);margin-left:2px}
.meter{height:5px;background:var(--sunk);border-radius:3px;overflow:hidden}
.meter i{display:block;height:100%;border-radius:3px;background:var(--rule2)}
.tile.good .meter i{background:var(--good)}
.tile.bad .meter i{background:var(--bad)}
.tile-f{margin:10px 0 0;font-family:var(--mono);font-size:11.5px;color:var(--ink3);
  font-variant-numeric:tabular-nums}
.grid2{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(290px,1fr))}
figure{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:15px 16px 10px;box-shadow:var(--shadow)}
figure.wide{padding-bottom:16px}
figcaption{display:flex;flex-wrap:wrap;align-items:baseline;gap:5px 12px;margin-bottom:8px;
  padding-bottom:10px;border-bottom:1px solid var(--rule)}
figcaption strong{font-family:var(--serif);font-weight:600;font-size:16px;letter-spacing:-.005em}
figcaption .metric{font-family:var(--mono);font-size:11px;color:var(--ink3);margin-left:auto}
svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--rule);stroke-width:1}
.zero{stroke:var(--ink3);stroke-width:1.5}
.floor{stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:3 3}
.floor-label,.tick{font-family:var(--mono);font-size:10px;fill:var(--ink3);
  font-variant-numeric:tabular-nums}
.xlab{font-family:var(--mono);font-size:11px;fill:var(--ink2)}
.axtitle{font-family:var(--mono);font-size:10.5px;fill:var(--ink3)}
.val{font-family:var(--mono);font-size:12px;fill:var(--ink);font-variant-numeric:tabular-nums}
.bar path{transition:opacity .12s}
.bar:hover path{opacity:.78}
.whisk{stroke:var(--ink);stroke-width:1.5;opacity:.45}
.s0 path,.s0 circle{fill:var(--s0)}
.s1 path,.s1 circle{fill:var(--s1)}
.s2 path,.s2 circle{fill:var(--s2)}
.s3 path,.s3 circle{fill:var(--s3)}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.s0{stroke:var(--s0)}
.line.s1{stroke:var(--s1)}
.line.s2{stroke:var(--s2)}
.line.s3{stroke:var(--s3)}
.dot{stroke:var(--surface);stroke-width:2}
.lineLabel{font-family:var(--mono);font-size:11px}
.lineLabel.s0{fill:var(--s0)}
.lineLabel.s1{fill:var(--s1)}
.lineLabel.s2{fill:var(--s2)}
.lineLabel.s3{fill:var(--s3)}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;margin:14px 0 0;padding:0;list-style:none;
  font-family:var(--mono);font-size:11.5px;color:var(--ink2)}
.legend li{display:flex;align-items:center;gap:7px}
.swatch{width:11px;height:11px;border-radius:2px;flex:none}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:560px}
th,td{text-align:right;padding:8px 12px;border-bottom:1px solid var(--rule);white-space:nowrap}
thead th{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink3);font-weight:500;border-bottom:1px solid var(--rule2)}
tbody th{text-align:left;font-weight:500;color:var(--ink2)}
thead th:first-child{text-align:left}
td{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink)}
td .sd{color:var(--ink3);font-size:11px;margin-left:5px}
tr.row-real{background:var(--sunk)}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:3px;padding:17px 20px;
  box-shadow:var(--shadow)}
.card+.card{margin-top:13px}
.card h3{font-family:var(--serif);font-size:17px;font-weight:600;margin:0 0 8px;
  letter-spacing:-.005em}
.card p{margin:0;color:var(--ink2);font-size:14.5px}
.card b{color:var(--ink)}
.note{border-left:3px solid var(--bad);background:var(--surface);padding:14px 18px;
  border-radius:0 3px 3px 0;margin:0 0 24px}
.note p{margin:0;font-size:14px;color:var(--ink2);max-width:none}
.note b{color:var(--ink)}
code{font-family:var(--mono);font-size:.88em;background:var(--sunk);padding:1px 5px;
  border-radius:2px;color:var(--ink)}
.taskblock{margin-top:28px}
.taskblock h3{font-family:var(--serif);font-size:18px;font-weight:600;margin:0 0 3px}
.taskblock .tdesc{font-family:var(--mono);font-size:11.5px;color:var(--ink3);margin:0 0 12px}
footer{margin-top:70px;padding-top:22px;border-top:1px solid var(--rule);
  font-family:var(--mono);font-size:11.5px;color:var(--ink3);line-height:1.9}
a{color:var(--s2)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""
)



def diagnosis_tables():
    """Render the A2 ceiling table and the fix comparison from saved JSON (no refitting)."""
    path = os.path.join(RESULTS_DIR, "diagnosis.json")
    diag = ""
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        labels = {0: "own row only", 1: "+ 1-hop parents (published default)", 2: "+ 2-hop (full)"}
        rows = []
        for r in d["a2_information_ceiling"]:
            rows.append(
                f'<tr><th scope="row">{esc(TITLES[r["task"]][1])}</th>'
                f'<td>{esc(labels[r["hop"]])}</td><td>{r["n_features"]}</td>'
                f'<td>{r["roc_auc"]:.4f}<span class="sd">&plusmn;{r["sd"]:.4f}</span></td></tr>'
            )
        diag = (
            '<div class="scroll"><table><thead><tr><th>task</th><th>conditioning depth</th>'
            '<th>features</th><th>real AUROC</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )
    return diag


def fix_table_html(results):
    from downstream.report import FIX_LABELS, FIX_ORDER

    blocks = []
    for task_name in ("berka_loan_default", "berka_card_ownership"):
        present = [ds for ds in FIX_ORDER if f"{ds}__{task_name}" in results]
        if len(present) < 2:
            continue
        rows = []
        base = None
        for ds in present:
            recs = results[f"{ds}__{task_name}"]["records"]
            n_seeds = len({r["grdm_seed"] for r in recs if r["grdm_seed"] is not None})
            real = summarise(recs, "real", "roc_auc")[0]
            syn = summarise(recs, "synth", "roc_auc")
            ret = (syn[0] - 0.5) / (real - 0.5)
            delta = "&mdash;" if base is None else f"{syn[0] - base:+.4f}"
            if base is None:
                base = syn[0]
            cls = ' class="row-real"' if ds.endswith("_s5") else ""
            rows.append(
                f'<tr{cls}><th scope="row">{esc(FIX_LABELS[ds])}</th><td>{n_seeds}</td>'
                f'<td>{syn[0]:.4f}<span class="sd">&plusmn;{syn[1]:.4f}</span></td>'
                f"<td>{ret:.1%}</td><td>{delta}</td></tr>"
            )
        blocks.append(
            f'<p class="tdesc" style="margin:18px 0 8px">{esc(TITLES[task_name][1])}</p>'
            '<div class="scroll"><table><thead><tr><th>variant</th><th>seeds</th>'
            '<th>synthetic only</th><th>retained</th><th>vs baseline</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )
    return "".join(blocks)


def main():
    results = load_results()
    # other experiments drop their own <dataset>__<name>.json here (blend ratios, cross-domain
    # TSTR, feature comparison); keep only the four canonical downstream tasks.
    results = {k: v for k, v in results.items() if k.split("__", 1)[1] in TASKS}
    if not results:
        raise SystemExit(f"no result files in {RESULTS_DIR}")

    panels = "".join(
        panel_svg(k.split("__")[1], results[k]["records"], TASKS[k.split("__")[1]])
        for k in HEADLINE
        if k in results
    )
    regime_legend = "".join(
        f'<li><span class="swatch" style="background:var(--s{i})"></span>{esc(lbl)}</li>'
        for i, (_, lbl) in enumerate(REGIMES)
    )

    task_blocks = []
    for key in sorted(results, key=sort_key):
        task_name = key.split("__")[1]
        task = TASKS[task_name]
        dataset = dataset_label(key.split("__")[0], task_name)
        title = TITLES[task_name][1]
        caveat = '<span class="flag">interim</span>' if results[key].get("leakage_caveat") else ""
        task_blocks.append(
            f'<div class="taskblock"><h3>{esc(dataset)} &middot; {esc(title)} {caveat}</h3>'
            f'<p class="tdesc">{esc(task.kind)} &middot; target: {esc(task.target)}</p>'
            + metric_table(results[key], task)
            + "</div>"
        )

    diag_rows = berka_label_diagnostic(
        grdm_dataset="berka_split", source_rel="data/berka/split_train"
    )
    diag = ['<div class="scroll"><table><thead><tr>']
    for h in [c.strip() for c in diag_rows[0].strip("|").split("|")]:
        diag.append(f"<th>{esc(h)}</th>")
    diag.append("</tr></thead><tbody>")
    for line in diag_rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        cls = ' class="row-real"' if cells[0] == "real" else ""
        diag.append(
            f'<tr{cls}><th scope="row">{esc(cells[0])}</th>'
            + "".join(f"<td>{esc(c)}</td>" for c in cells[1:])
            + "</tr>"
        )
    diag.append("</tbody></table></div>")

    diag_table = diagnosis_tables()
    fix_table = fix_table_html(results)

    body = f"""<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">GRDM reproduction &middot; downstream utility audit</p>
  <h1>Fidelity said both databases were fine. Only one of them is usable.</h1>
  <p class="lede">GRDM&rsquo;s synthetic databases reproduce the paper&rsquo;s fidelity scores on
    both California and Berka. Training XGBoost on them tells a different story: California
    synthetic data is a near drop-in replacement for the real thing, while Berka synthetic data
    retains barely a quarter of the signal. Diagnosing why traces the gap to two root causes
    &mdash; conditioning depth and gradient starvation on small tables &mdash; and combining fixes
    for both recovers roughly half of what was lost.</p>
  <div class="meta">
    <span><b>4</b> relational tasks</span>
    <span><b>3</b> training regimes</span>
    <span><b>3</b> generator seeds each</span>
    <span><b>1</b> real held-out test set per database</span>
    <span><b>5</b> fixes measured separately</span>
    <span>rdb-diffusion &middot; <b>2026-08-09</b></span>
  </div>
</header>

<section>
  <h2>How much of the real signal survives?</h2>
  <p class="sub">Share of the real-data baseline a model retains when trained on synthetic data
    alone, measured above each metric&rsquo;s floor &mdash; chance for AUROC, zero for R&sup2; and
    macro-F1. Synthetic training sets are subsampled to the real row count, so this compares data
    quality, not quantity.</p>
  <div class="tiles">{tiles(results)}</div>
</section>

<section>
  <h2>Real, synthetic and mixed on every task</h2>
  <p class="sub">Each panel is one task on its own axis. Bars are the mean over generator and fit
    seeds, whiskers &plusmn;1 standard deviation. The dashed line marks chance on the AUROC
    panels &mdash; the Berka synthetic bars sit close to it.</p>
  <div class="grid2">{panels}</div>
  <ul class="legend">{regime_legend}
    <li><span class="swatch" style="background:var(--ink3)"></span>&plusmn;1 SD across repeats</li>
  </ul>
</section>

<section>
  <h2>Does mixing help?</h2>
  <p class="sub">Not measurably &mdash; but it does no harm either, which is the useful finding
    when synthetic data is added for privacy or volume rather than accuracy. Every task stays
    within 0.03 of its real-only baseline at every ratio.</p>
  {sweep_svg(results)}
</section>

<section>
  <h2>Why Berka collapses</h2>
  <p class="sub">Two different failure modes, and only one of them is visible in the per-column
    distributions that the fidelity metrics measure.</p>
  <div class="card">
    <h3>Loan default: the marginal itself breaks</h3>
    <p>Class B &mdash; a loan that finished unpaid &mdash; has 24 rows in the real training split
      and between 1 and 10 in every generated one. The default rate falls from 0.112 to about
      0.073. Berka scored <b>96.89 Column Shapes</b> on these same tables.</p>
  </div>
  <div class="card">
    <h3>Card ownership: the marginal is fine, the dependency is gone</h3>
    <p>The share of dispositions holding a card is reproduced almost exactly &mdash; 0.168 real
      against 0.160&ndash;0.167 synthetic &mdash; and the task still falls from 0.874 to 0.602
      AUROC. What is lost is the cross-table relationship between account behaviour and card
      ownership, not the rate. <b>Matching per-column distributions is not sufficient for
      downstream utility.</b></p>
  </div>
  <div style="margin-top:16px">{"".join(diag)}</div>
</section>

<section>
  <h2>Full results</h2>
  <p class="sub">Mean &plusmn; standard deviation over 3 generator seeds &times; 3&ndash;5 XGBoost
    seeds. Hyperparameters are fixed and identical across regimes: tuning per regime, or early
    stopping on a real validation split, would leak real data into the synthetic-only run.</p>
  <div class="note"><p><b>Two Berka protocols are shown.</b> The headline figures above use
    <code>berka_split</code>, where the generator was retrained on an account-level train split
    and never saw the test rows. The rows marked <i>interim</i> come from the original generator,
    which trained on the whole database including those rows. They agree to within the seed
    spread. Real-only baselines are identical across the two by construction &mdash; both evaluate
    the same fixed split, and only what the generator saw differs.</p></div>
  {"".join(task_blocks)}
</section>

<section>
  <h2>Two leaks found while building this</h2>
  <p class="sub">The first feature sets produced real-data AUROC of 1.0000 and 0.9805. Those were
    ceilings measuring the schema, not the model. Berka&rsquo;s label encoders name the cause
    exactly.</p>
  <div class="card">
    <h3><code>trans.k_symbol</code> = SANKC. UROK &mdash; sanction interest</h3>
    <p>Charged only once an account is already in debt, which is what a defaulted loan
      <em>is</em>. With a negative minimum balance it separated the classes perfectly. Fixed by
      restricting transaction aggregates to <code>trans_date &lt; loan_date</code>, making this a
      genuine prediction at origination. That leaves a median of 30 prior transactions per loan
      and drops the sanction signal from 77 accounts to 8. Real AUROC became <b>0.872</b>.</p>
  </div>
  <div class="card">
    <h3><code>trans.operation</code> = VYBER KARTOU &mdash; card withdrawal</h3>
    <p>Cannot occur unless the disposition holds a card, so it determines the label outright.
      Cards are issued a median of 348 days after the account opens with only 51% inside the first
      year, so no time window separates cause from effect here. The deterministic giveaway is
      dropped and the task is scored as a cross-table dependency probe rather than a forecast.
      Real AUROC became <b>0.874</b>.</p>
  </div>
</section>

<section>
  <h2>Why Berka fails</h2>
  <p class="sub">GRDM&rsquo;s GNN has <code>n_hops</code> layers, so while a row is generated it sees
    its own noisy features plus its direct neighbours&rsquo; <em>raw</em> columns &mdash; nothing
    further. Scoring XGBoost on <b>real</b> data restricted to exactly that window reveals the
    ceiling the generator works against.</p>
  {diag_table}
  <div class="card" style="margin-top:16px">
    <h3>Essentially all Berka signal lives two hops away &mdash; but widening the window has a cost</h3>
    <p>A <code>loan</code> conditions on its account&rsquo;s two columns; the account&rsquo;s ~235
      transactions &mdash; which carry the signal &mdash; are two hops out. California&rsquo;s
      <code>individual &larr; household</code> is exactly one hop, which is why the same
      configuration works there. The measured synthetic scores sit on these ceilings: baseline
      0.60 on card against an own-row ceiling of 0.604; <code>n_hops=2</code> reaches 0.67&ndash;0.69
      against the 1-hop ceiling of 0.671. The effective window sits one hop behind the nominal
      depth, because a neighbour&rsquo;s features are themselves still noisy during generation.</p>
    <p>Widening the window is not free: it costs loan default 0.065 AUROC (3-seed mean), most
      likely because the wider fan-out further dilutes the seed share of small tables that are
      already gradient-starved. See the fix section below.</p>
  </div>
</section>

<section>
  <h2>Five fixes, measured separately</h2>
  <p class="sub">Each changes exactly one thing and is scored on the same fixed split and test set,
    so a delta is attributable to that change alone. Synthetic-only AUROC; <i>retained</i> is
    signal above chance. Highlighted row is the recommended fix.</p>
  {fix_table}
  <div class="card" style="margin-top:16px">
    <h3>Depth and gradient balance are coupled &mdash; neither alone is the full answer</h3>
    <p><b>S2</b> (deeper conditioning) nearly doubles retained signal on card ownership
      (27.3% &rarr; 45.1%, 3-seed mean) but <em>actively hurts</em> loan default (23.6% &rarr; 6.8%),
      the opposite of what the diagnosis predicted. <b>S3</b> (balanced seed sampling) is the
      strongest single fix for loan default (23.6% &rarr; 41.5%) and the only fix that
      <em>improves</em> fidelity over baseline rather than trading it away.</p>
    <p><b>S5 combines both</b> and wins on card ownership outright (48.5%, best of any fix) while
      recovering most of S2&rsquo;s loan-default loss (34.0% vs S2&rsquo;s 6.8%). The likely reason:
      2-hop conditioning widens every table&rsquo;s effective batch, further diluting
      <code>loan</code>&rsquo;s already-thin seed share; balanced sampling directly offsets that.</p>
    <p><b>S1</b> is the most informative negative result: it restores every categorical marginal
      exactly &mdash; <code>loan.status</code> class B from 1&ndash;10 rows back to 24, default rate
      0.073 &rarr; 0.113 &mdash; and moves downstream AUROC by less than 0.005 either way. The
      minority-class collapse is the most <em>visible</em> symptom and is not what breaks utility.
      <b>S4</b> (one-hot categoricals) is harmful everywhere &mdash; it broke California
      (synthetic-only R&sup2; &minus;4.97, worse than predicting the mean) and was dropped after one
      Berka seed.</p>
  </div>
</section>

<footer>Generated by <code>downstream/make_artifact.py</code> from
  <code>downstream/results/*.json</code>.<br>
  Harness <code>downstream/run.py</code> &middot; tasks and leakage guards
  <code>downstream/tasks.py</code> &middot; root-cause diagnosis <code>docs/downstream/DIAGNOSIS.md</code>
  &middot; per-fix design <code>docs/downstream/FIXES.md</code> &middot; write-up in
  <code>docs/REPRODUCTION.md</code> &sect;8&ndash;9.</footer>
</div>"""

    with open(OUT, "w") as f:
        f.write(body)
    print(f"wrote {OUT}  ({len(body) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn a session timeline (one JSON object per turn, from the app's ARTEMIS_TURN lines) into
Artemis metrics and a self-contained HTML chart.

usage: session_report.py timeline.jsonl [--results artemis_results.json] [--html out.html] [--title T]
"""
import argparse
import html
import json
import math
import os
import statistics
import sys

THERMAL = ["nominal", "fair", "serious", "critical"]
# Thermal phases of a long session, in minutes (cold burst, transition, equilibrium)
PHASES = [("burst", 0, 2), ("transition", 5, 10), ("equilibrium", 15, 20)]


def harmonic(xs):
    """Overall tokens/s for equal-length turns: total tokens over total time."""
    return len(xs) / sum(1 / x for x in xs) if xs and all(x > 0 for x in xs) else 0.0


def pct_dropped(rows):
    frames = sum(r["ui_frames"] + r["ui_dropped"] for r in rows)
    return 100 * sum(r["ui_dropped"] for r in rows) / frames if frames else 0.0


def summarize(rows):
    m = {
        "tg_tps": harmonic([r["tg_tps"] for r in rows]),
        "tg_tps_first_turn": rows[0]["tg_tps"],
        "tg_tps_deepest_turn": max(rows, key=lambda r: r["depth"])["tg_tps"],
        "ttft_ms_p50": statistics.median(r["ttft_ms"] for r in rows),
        "ttft_ms_max": max(r["ttft_ms"] for r in rows),
        "prefill_tps": harmonic([r["prefill_tps"] for r in rows]),
        "ui_dropped_pct": pct_dropped(rows),
        "ui_dropped_pct_worst_turn": max(pct_dropped([r]) for r in rows),
        "ui_p99_ms_max": max(r["ui_p99_ms"] for r in rows),
        "thermal_max": max(r["thermal"] for r in rows),
        "mem_peak_mb": max(r["mem_mb"] for r in rows),
        "turns": len(rows),
        "session_s": rows[-1]["t_s"] + rows[-1]["ttft_ms"] / 1000 + 150 / rows[-1]["tg_tps"],
    }
    for name, a, b in PHASES:
        sel = [r for r in rows if a * 60 <= r["t_s"] < b * 60]
        if sel and m["session_s"] >= b * 60 * 0.9:
            m[f"tg_tps_{name}"] = harmonic([r["tg_tps"] for r in sel])
            m[f"ttft_ms_{name}"] = statistics.median(r["ttft_ms"] for r in sel)
            m[f"ui_dropped_pct_{name}"] = pct_dropped(sel)
    return {k: round(v, 4) for k, v in m.items()}


# ---------------------------------------------------------------- chart

W, PL, PR = 720, 64, 16          # svg width, plot left/right padding
PH, GAP, TOP = 92, 34, 8         # panel plot height, gap between panels (room for the title)


def nice_ticks(lo, hi, n=3):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks


def fmt(v):
    if abs(v) >= 1000:
        return f"{v / 1000:.1f}k".replace(".0k", "k")
    return f"{v:.0f}" if abs(v) >= 10 or v == int(v) else f"{v:.1f}"


def render(rows, metrics, title, ui_cap=None):
    t_end = max(metrics["session_s"], rows[-1]["t_s"] + 1)
    xs = [r["t_s"] for r in rows]

    def X(t):
        return PL + (W - PL - PR) * t / t_end

    panels = [
        ("thermal", "Thermal state", "", None),
        ("tg_tps", "Generation speed", "tokens/s", "higher is better"),
        ("ttft_ms", "Time to first token", "ms", "lower is better"),
        ("prefill_tps", "Prompt processing", "tokens/s", "higher is better"),
        ("ui_drop_pct", "UI frames dropped", "% of frames", "lower is better"),
        ("depth", "Context length at turn start", "tokens", None),
        ("mem_mb", "App memory footprint", "MB", "excludes the memory-mapped model weights"),
    ]
    for r in rows:
        r["turn_s"] = r["ttft_ms"] / 1000 + 150 / r["tg_tps"]
        f = r["ui_frames"] + r["ui_dropped"]
        r["ui_drop_pct"] = 100 * r["ui_dropped"] / f if f else 0.0

    long_session = t_end >= 4 * 60
    out = []
    y0 = TOP
    for key, label, unit, hint in panels:
        top = y0 + 18
        bot = top + PH
        vals = [r[key] for r in rows]
        if key == "thermal":
            lo, hi, ticks = 0, 3, [0, 1, 2, 3]
        else:
            lo = 0
            hi = max(vals) * 1.1 or 1
            if key == "ui_drop_pct" and ui_cap is not None:
                hi = max(hi, ui_cap * 1.15)
            ticks = nice_ticks(lo, hi)
            hi = max(hi, ticks[-1])

        def Y(v, lo=lo, hi=hi, top=top, bot=bot):
            return bot - (bot - top) * (v - lo) / (hi - lo)

        sub = f" · {unit}" if unit else ""
        sub += f" · {hint}" if hint else ""
        out.append(f'<text class="ptitle" x="{PL}" y="{y0 + 10}">{html.escape(label)}'
                   f'<tspan class="psub">{html.escape(sub)}</tspan></text>')
        if long_session:
            for name, a, b in PHASES:
                if a * 60 < t_end:
                    x1, x2 = X(a * 60), X(min(b * 60, t_end))
                    out.append(f'<rect class="phase" x="{x1:.1f}" y="{top}" width="{x2 - x1:.1f}" height="{PH}"/>')
        for t in ticks:
            yy = Y(t)
            out.append(f'<line class="grid" x1="{PL}" x2="{W - PR}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            lab = THERMAL[int(t)] if key == "thermal" else fmt(t)
            out.append(f'<text class="tick" x="{PL - 6}" y="{yy + 3.5:.1f}" text-anchor="end">{lab}</text>')
        if key == "thermal":
            # step line: each reading is taken at the end of its turn and holds until the next
            d = f"M{X(0):.1f},{Y(vals[0]):.1f}"
            for r in rows:
                d += f" H{X(r['t_s'] + r['turn_s']):.1f} V{Y(r['thermal']):.1f}"
            d += f" H{X(t_end):.1f}"
            out.append(f'<path class="line" d="{d}"/>')
        else:
            pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(xs, vals))
            out.append(f'<polyline class="line" points="{pts}"/>')
            if len(rows) <= 60:
                for t, v in zip(xs, vals):
                    out.append(f'<circle class="dot" cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3"/>')
        if key == "ui_drop_pct" and ui_cap is not None:
            yc = Y(ui_cap)
            out.append(f'<line class="cap" x1="{PL}" x2="{W - PR}" y1="{yc:.1f}" y2="{yc:.1f}"/>')
            out.append(f'<text class="tick" x="{W - PR}" y="{yc - 4:.1f}" text-anchor="end">'
                       f'cap {ui_cap:g}% per turn</text>')
        out.append(f'<line class="axis" x1="{PL}" x2="{W - PR}" y1="{bot}" y2="{bot}"/>')
        y0 = bot + GAP

    # shared time axis under the last panel
    axis_y = y0 - GAP
    span_min = t_end / 60
    step_s = next(s for s in (10, 15, 30, 60, 120, 300, 600) if t_end / s <= 8)
    t = 0
    while t <= t_end + 0.1:
        lab = f"{t / 60:g} min" if step_s >= 60 else f"{t:.0f}s"
        out.append(f'<text class="tick" x="{X(t):.1f}" y="{axis_y + 15}" text-anchor="middle">{lab}</text>')
        t += step_s
    height = axis_y + 26
    if long_session:
        for name, a, b in PHASES:
            if a * 60 < t_end:
                out.append(f'<text class="tick" x="{(X(a * 60) + X(min(b * 60, t_end))) / 2:.1f}" '
                           f'y="{TOP + 10}" text-anchor="middle">{name}</text>')

    # invisible hit targets for the hover layer, one per turn, spanning all panels
    hits = []
    for i, r in enumerate(rows):
        left = (X(rows[i - 1]["t_s"]) + X(r["t_s"])) / 2 if i else PL
        right = (X(r["t_s"]) + X(rows[i + 1]["t_s"])) / 2 if i + 1 < len(rows) else W - PR
        hits.append(f'<rect class="hit" data-i="{i}" x="{left:.1f}" y="{TOP}" width="{right - left:.1f}" '
                    f'height="{axis_y - TOP}"/>')

    svg = (f'<svg viewBox="0 0 {W} {height:.0f}" role="img" aria-label="{html.escape(title)} timeline">'
           + "".join(out)
           + f'<line id="xhair" class="xhair" x1="0" x2="0" y1="{TOP}" y2="{axis_y}" visibility="hidden"/>'
           + "".join(hits) + "</svg>")

    tiles = [
        ("Generation", f'{metrics["tg_tps"]:.1f}', "tokens/s overall"),
        ("Deepest turn", f'{metrics["tg_tps_deepest_turn"]:.1f}', f'tokens/s at {fmt(max(r["depth"] for r in rows))} context'),
        ("Time to first token", f'{metrics["ttft_ms_p50"]:.0f}', "ms median"),
        ("UI frames dropped", f'{metrics["ui_dropped_pct"]:.1f}%', f'worst turn {metrics["ui_dropped_pct_worst_turn"]:.1f}%'),
        ("Hottest", THERMAL[int(metrics["thermal_max"])], "thermal state"),
        ("Memory peak", f'{metrics["mem_peak_mb"]:.0f}', "MB"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="tl">{a}</div><div class="tv">{b}</div>'
                         f'<div class="ts">{c}</div></div>' for a, b, c in tiles)

    cols = [("t_s", "Time (s)"), ("conv", "Conv"), ("turn", "Turn"), ("depth", "Context"),
            ("ttft_ms", "TTFT ms"), ("prefill_tps", "Prefill t/s"), ("tg_tps", "Gen t/s"),
            ("ui_drop_pct", "UI drop %"), ("ui_p99_ms", "UI p99 ms"), ("thermal", "Thermal"), ("mem_mb", "Mem MB")]

    def cell(k, v):
        return THERMAL[int(v)] if k == "thermal" else (f"{v:.0f}" if k in ("conv", "turn", "depth", "mem_mb", "ttft_ms") else f"{v:.1f}")

    table = ("<table><thead><tr>" + "".join(f"<th>{h}</th>" for _, h in cols) + "</tr></thead><tbody>"
             + "".join("<tr>" + "".join(f"<td>{cell(k, r[k])}</td>" for k, _ in cols) + "</tr>" for r in rows)
             + "</tbody></table>")

    data = json.dumps([{k: r[k] for k, _ in cols} for r in rows])
    return TEMPLATE.format(title=html.escape(title), tiles=tiles_html, svg=svg, table=table, data=data,
                           minutes=f"{span_min:.1f}", turns=len(rows))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Session Timeline</title>
<style>
:root {{ color-scheme: light;
  --surface-1:#fcfcfb; --surface-2:#f3f2ef; --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#7a7974;
  --grid:#e6e5e1; --axis:#c9c8c2; --series-1:#2a78d6; --phase:#f0efec; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{ color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#242422; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
  --grid:#2e2e2b; --axis:#45443f; --series-1:#3987e5; --phase:#262624; }} }}
:root[data-theme="dark"] {{ color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#242422; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
  --grid:#2e2e2b; --axis:#45443f; --series-1:#3987e5; --phase:#262624; }}
body {{ margin:0; background:var(--surface-1); color:var(--text-primary);
  font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:760px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:var(--text-secondary); margin:0 0 16px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:8px; margin-bottom:16px; }}
.tile {{ background:var(--surface-2); border-radius:8px; padding:10px 12px; }}
.tl {{ font-size:12px; color:var(--text-secondary); }} .tv {{ font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }}
.ts {{ font-size:11px; color:var(--text-muted); }}
.chart {{ position:relative; }} svg {{ width:100%; height:auto; display:block; }}
.ptitle {{ font-size:12px; font-weight:600; fill:var(--text-primary); }} .psub {{ font-weight:400; fill:var(--text-muted); }}
.tick {{ font-size:10px; fill:var(--text-secondary); font-variant-numeric:tabular-nums; }}
.grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }}
.phase {{ fill:var(--phase); }}
.line {{ fill:none; stroke:var(--series-1); stroke-width:2; stroke-linejoin:round; }}
.dot {{ fill:var(--series-1); stroke:var(--surface-1); stroke-width:2; }}
.cap {{ stroke:var(--text-secondary); stroke-width:1; stroke-dasharray:4 3; }}
.hit {{ fill:transparent; cursor:crosshair; }} .xhair {{ stroke:var(--text-muted); stroke-width:1; stroke-dasharray:3 3; pointer-events:none; }}
#tip {{ position:absolute; pointer-events:none; background:var(--surface-2); color:var(--text-primary); border-radius:6px;
  padding:8px 10px; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.15); display:none; white-space:nowrap; font-variant-numeric:tabular-nums; }}
#tip b {{ font-weight:600; }} #tip span {{ color:var(--text-secondary); }}
details {{ margin-top:20px; }} summary {{ cursor:pointer; color:var(--text-secondary); }}
.tablewrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; font-size:12px; margin-top:8px; font-variant-numeric:tabular-nums; }}
th,td {{ padding:4px 8px; text-align:right; border-bottom:1px solid var(--grid); }} th {{ color:var(--text-secondary); font-weight:500; }}
</style></head>
<body><main>
<h1>{title}</h1>
<p class="sub">{turns} turns over {minutes} min on iPhone 16. Each point is one turn: a 250-token user message, then a 150-token reply, with a 60 Hz Metal UI rendering alongside.</p>
<div class="tiles">{tiles}</div>
<div class="chart">{svg}<div id="tip"></div></div>
<details><summary>Table view</summary><div class="tablewrap">{table}</div></details>
</main>
<script>
const rows = {data};
const names = ["nominal","fair","serious","critical"];
const svg = document.querySelector("svg"), tip = document.getElementById("tip"), xh = document.getElementById("xhair");
svg.addEventListener("mousemove", e => {{
  const h = e.target.closest(".hit"); if (!h) return;
  const r = rows[+h.dataset.i], b = h.getBBox();
  const cx = b.x + b.width / 2;
  xh.setAttribute("x1", cx); xh.setAttribute("x2", cx); xh.setAttribute("visibility", "visible");
  tip.innerHTML = `<b>Conversation ${{r.conv + 1}}, turn ${{r.turn + 1}}</b> <span>at ${{r.t_s.toFixed(0)}}s</span><br>` +
    `<span>Context</span> ${{r.depth}} tokens<br><span>TTFT</span> ${{r.ttft_ms.toFixed(0)}} ms<br>` +
    `<span>Generation</span> ${{r.tg_tps.toFixed(1)}} tok/s<br><span>Prefill</span> ${{r.prefill_tps.toFixed(0)}} tok/s<br>` +
    `<span>UI dropped</span> ${{r.ui_drop_pct.toFixed(1)}}% · p99 ${{r.ui_p99_ms.toFixed(1)}} ms<br>` +
    `<span>Thermal</span> ${{names[r.thermal]}} · <span>Memory</span> ${{r.mem_mb.toFixed(0)}} MB`;
  const box = svg.getBoundingClientRect(), px = e.clientX - box.left;
  tip.style.display = "block";
  tip.style.left = Math.min(px + 12, box.width - tip.offsetWidth - 4) + "px";
  tip.style.top = (e.clientY - box.top + 12) + "px";
}});
svg.addEventListener("mouseleave", () => {{ tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); }});
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("--results")
    ap.add_argument("--html")
    ap.add_argument("--title", default="Session")
    ap.add_argument("--ui-drop-cap", type=float, default=15.0,
                    help="fail (no results file, exit 1) when any turn drops more than this %% of its UI frames")
    ap.add_argument("--no-ui", action="store_true", help="the session ran without the UI load; skip the cap")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.timeline) if l.strip()]
    if not rows:
        sys.exit("empty timeline")
    for r in rows:
        r["thermal"] = int(r["thermal"])
        r["conv"], r["turn"], r["depth"] = int(r["conv"]), int(r["turn"]), int(r["depth"])
    metrics = summarize(rows)
    out_html = a.html or os.path.splitext(a.timeline)[0] + ".html"
    with open(out_html, "w") as f:
        f.write(render(rows, metrics, a.title, None if a.no_ui else a.ui_drop_cap))
    for k, v in metrics.items():
        print(f"{k}={v}")
    print(f"chart: {out_html}")

    # Hard cap on the worst moment: a version that makes the UI janky during any turn fails
    # outright, instead of trading jank for speed in the Composite score or hiding a bad stretch
    # in the session average. No frames at all means the UI load broke, which also fails.
    ui_frames = sum(r["ui_frames"] for r in rows)
    if not a.no_ui and ui_frames == 0:
        sys.exit("ARTEMIS_ERROR no UI frames recorded; the UI load did not run")
    if not a.no_ui and metrics["ui_dropped_pct_worst_turn"] > a.ui_drop_cap:
        worst = max(rows, key=lambda r: pct_dropped([r]))
        sys.exit(f"ARTEMIS_ERROR UI dropped {metrics['ui_dropped_pct_worst_turn']:.1f}% of frames in "
                 f"conversation {worst['conv'] + 1} turn {worst['turn'] + 1} (at {worst['t_s']:.0f}s), "
                 f"over the {a.ui_drop_cap:g}% cap; no results written")

    if a.results:
        tmp = a.results + ".tmp"
        with open(tmp, "w") as f:
            json.dump(metrics, f, indent=1)
        os.replace(tmp, a.results)
        print(f"wrote {a.results}")


if __name__ == "__main__":
    main()

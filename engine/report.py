"""Standalone HTML dashboard builder."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from engine.bias import NUMBER_POPULARITY, STAR_POPULARITY, uniqueness_score, score_tier, pop_color
from engine.data import load_picks, load_draws, compute_stats


def _bar(value: int, max_val: int, color: str = "#4fc3f7", width: int = 200) -> str:
    pct = int((value / max(max_val, 1)) * width)
    return (
        f'<div style="display:inline-block;width:{width}px;height:8px;background:#1a2540;border-radius:3px;vertical-align:middle">'
        f'<div style="width:{pct}px;height:100%;background:{color};border-radius:3px"></div></div>'
    )


def build_report(output_path: str | Path = "report.html") -> Path:
    output_path = Path(output_path)
    picks = load_picks()
    draws = load_draws()
    stats = compute_stats(draws) if draws else {}

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = []

    # ── Summary ───────────────────────────────────────────────────────────────
    sections.append(f"""
<section>
  <h2>Summary</h2>
  <table>
    <tr><td>Report generated</td><td>{now}</td></tr>
    <tr><td>Saved picks</td><td>{len(picks)}</td></tr>
    <tr><td>Draw history</td><td>{stats.get("totalDraws", 0)} draws</td></tr>
    {"<tr><td>Date range</td><td>" + stats["dateRange"]["first"] + " → " + stats["dateRange"]["last"] + "</td></tr>" if stats else ""}
  </table>
</section>""")

    # ── Picks ─────────────────────────────────────────────────────────────────
    if picks:
        rows = []
        for p in sorted(picks, key=lambda x: x["timestamp"], reverse=True)[:20]:
            nums = " ".join(str(n) for n in p["numbers"])
            stars = " ".join(str(s) for s in p["stars"])
            score = p["uniquenessScore"]
            label, emoji = score_tier(score)
            rows.append(
                f"<tr>"
                f"<td>{p['timestamp'][:10]}</td>"
                f"<td style='font-family:monospace'>{nums}</td>"
                f"<td style='font-family:monospace'>★{stars}</td>"
                f"<td>{score}</td>"
                f"<td>{emoji} {label}</td>"
                f"<td>{p['source']}</td>"
                f"</tr>"
            )
        sections.append(f"""
<section>
  <h2>Recent Picks (last 20)</h2>
  <table>
    <thead><tr><th>Date</th><th>Numbers</th><th>Stars</th><th>Score</th><th>Tier</th><th>Source</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>""")

    # ── Number popularity ─────────────────────────────────────────────────────
    pop_rows = []
    for n in range(1, 51):
        pop = NUMBER_POPULARITY[n]
        color = pop_color(pop)
        pop_rows.append(
            f"<tr>"
            f"<td style='color:{color}'>{n}</td>"
            f"<td>{pop}</td>"
            f"<td>{_bar(pop, 100, color)}</td>"
            f"</tr>"
        )

    sections.append(f"""
<section>
  <h2>Number Popularity Model</h2>
  <table>
    <thead><tr><th>Number</th><th>Popularity</th><th>Visual</th></tr></thead>
    <tbody>{"".join(pop_rows)}</tbody>
  </table>
</section>""")

    # ── Draw frequency ────────────────────────────────────────────────────────
    if stats:
        freq = stats.get("numFrequency", {})
        max_freq = max(freq.values(), default=1)
        hot = set(stats.get("hotNumbers", []))
        cold = set(stats.get("coldNumbers", []))
        freq_rows = []
        for n in range(1, 51):
            cnt = freq.get(str(n), 0)
            color = "#ff9a3c" if n in hot else "#4fc3f7" if n in cold else "#4a5a7a"
            freq_rows.append(
                f"<tr>"
                f"<td style='color:{color}'>{n}</td>"
                f"<td>{cnt}</td>"
                f"<td>{_bar(cnt, max_freq, color)}</td>"
                f"</tr>"
            )

        sections.append(f"""
<section>
  <h2>Number Draw Frequency</h2>
  <p>Based on {stats['totalDraws']} draws. Hot = top 10 🟠, Cold = bottom 10 🔵</p>
  <table>
    <thead><tr><th>Number</th><th>Count</th><th>Visual</th></tr></thead>
    <tbody>{"".join(freq_rows)}</tbody>
  </table>
</section>""")

    # ── HTML shell ────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>EuroMillions Crowd Analyser — Report</title>
  <style>
    body {{ background:#05080f; color:#d0d8f0; font-family:'Segoe UI',sans-serif; padding:32px; max-width:960px; margin:0 auto; }}
    h1 {{ color:#4fc3f7; font-size:22px; margin-bottom:4px; }}
    h2 {{ color:#d0d8f0; font-size:15px; margin:28px 0 10px; border-bottom:1px solid #1a2540; padding-bottom:6px; }}
    table {{ border-collapse:collapse; width:100%; margin-bottom:8px; }}
    th,td {{ padding:6px 10px; font-size:12px; border-top:1px solid #1a2540; text-align:left; }}
    th {{ color:#4a5a7a; }}
    section {{ margin-bottom:32px; }}
    .dim {{ color:#4a5a7a; font-size:11px; }}
  </style>
</head>
<body>
  <h1>EuroMillions Crowd Analyser</h1>
  <p class="dim">Report generated {now}</p>
  {"".join(sections)}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    path = build_report("report.html")
    print(f"Report saved to {path}")

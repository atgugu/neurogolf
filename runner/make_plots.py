#!/usr/bin/env python3
"""make_plots — derive small CSVs from the (private) run logs and render the README plots.

Two stages, runnable independently:

  python3 runner/make_plots.py --derive   # needs the private logs (see env vars below)
  python3 runner/make_plots.py            # renders docs/assets/plots/*.png from docs/assets/data/*.csv

The derive stage reads:
  $NEUROGOLF_CLEAN/logs/probe_outcomes.json + probe_log.jsonl   (Kaggle probe history)
  --results <dir>   a results/ tree with task*/ATTEMPTS.jsonl + LEDGER.csv (wave history)
and writes assets/data/{kaggle_score,points_per_day,model_stats,compression,deltas}.csv —
those CSVs are committed, so the plots regenerate on any machine without the logs.

Each figure is rendered twice (light + dark) for GitHub's <picture> element.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "assets" / "data"
PLOTS = ROOT / "docs" / "assets" / "plots"

# palette (validated with the dataviz six-checks validator in both modes)
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e5e4e0", "blue": "#2a78d6", "green": "#008300"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "grid": "#33322f", "blue": "#3987e5", "green": "#00a000"},
}


# ---------------------------------------------------------------- derive ----
def derive(results_dir: Path):
    sys.path.insert(0, str(ROOT / "runner"))
    from config import CLEAN
    clean = Path(CLEAN)
    DATA.mkdir(parents=True, exist_ok=True)

    # 1) Score history, from the two record types the project actually kept:
    #    - `probe`: one Kaggle probe's returned score (probe_outcomes.json). Probing ran
    #      Jun 5 – Jul 10; it stopped when the pipeline moved to the repin loop.
    #    - `pin`: the Kaggle-confirmed score recorded at each pin rotation (REPIN.log,
    #      `repin.py --score`). Jul 11 – Jul 16. The local grader replica tracks these
    #      within ~0.33 pts at every step, which is what corroborates them.
    outcomes = json.load(open(clean / "logs" / "probe_outcomes.json"))
    rows = sorted(("probe", v["date"], float(v["actual"])) for v in outcomes.values()
                  if v.get("date") and v.get("actual"))
    n_probe = len(rows)

    pin_re = re.compile(r"^(\S+)Z === REPIN start: \w+@[\d.]+ -> (\w+)@([\d.]+) ===")
    best = max(r[2] for r in rows)
    pins = []
    for line in open(results_dir / "REPIN.log", errors="replace"):
        m = pin_re.match(line)
        if m and float(m.group(3)) > best:      # running max: keep only real advances
            best = float(m.group(3))
            pins.append(("pin", m.group(1)[:10], best))
    rows += pins
    with open(DATA / "kaggle_score.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "date", "score"])
        w.writerows(rows)
    print(f"kaggle_score.csv: {n_probe} probes {rows[0][1]} → {rows[n_probe-1][1]} · "
          f"{len(pins)} confirmed pin rotations → {pins[-1][2]} ({pins[-1][1]})")

    # 2+3) wave history: per-day yield and per-model outcomes from ATTEMPTS.jsonl
    per_day = defaultdict(lambda: [0, 0, 0.0])       # day -> [attempts, wins, delta]
    per_model = defaultdict(lambda: [0, 0, 0.0])     # model/effort -> same
    for path in sorted(results_dir.glob("task*/ATTEMPTS.jsonl")):
        for line in open(path, errors="replace"):
            try:
                a = json.loads(line)
            except ValueError:
                continue
            ts = str(a.get("ts") or "")
            if len(ts) < 8:
                continue
            day = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
            mid = a.get("model_id") or "unknown"
            eff = a.get("effort") or "?"
            won = a.get("outcome") == "DONE" and (a.get("delta") or 0) > 0
            delta = float(a.get("delta") or 0)
            buckets = [per_day[day]]
            if str(mid).startswith("gpt"):   # per-model view covers the GPT fleet only
                buckets.append(per_model[f"{mid}/{eff}"])
            for bucket in buckets:
                bucket[0] += 1
                bucket[1] += won
                bucket[2] += delta if won else 0.0
    with open(DATA / "points_per_day.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "attempts", "wins", "points"])
        for day in sorted(per_day):
            n, wn, pts = per_day[day]
            w.writerow([day, n, wn, f"{pts:.3f}"])
    with open(DATA / "model_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "attempts", "wins", "points"])
        for m in sorted(per_model, key=lambda m: -per_model[m][0]):
            n, wn, pts = per_model[m]
            w.writerow([m, n, wn, f"{pts:.3f}"])
    print(f"points_per_day.csv: {len(per_day)} days · model_stats.csv: {len(per_model)} configs")

    # 4+5) compression + deltas: best banked build per task vs the pin it beat.
    # Δ = ln(pin/new) under score = 25 − ln(cost), so the pin cost at win time is
    # exactly new_cost·e^Δ — no dependence on later repins moving the baseline.
    best = {}                                        # task -> (delta, cost, lane)
    for r in csv.DictReader(open(results_dir / "LEDGER.csv")):
        try:
            d, c = float(r["delta"]), int(float(r["cost"]))
        except (KeyError, ValueError):
            continue
        if d > best.get(r["task"], (-9, 0, ""))[0]:
            best[r["task"]] = (d, c, r.get("lane", "?"))
    wins = 0
    with open(DATA / "compression.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "pin_cost", "best_cost", "delta", "lane"])
        for t, (d, c, lane) in sorted(best.items()):
            if d > 0 and c > 0:
                w.writerow([t, int(round(c * math.exp(d))), c, f"{d:.4f}", lane])
                wins += 1
    print(f"compression.csv: {wins} wins")


# ------------------------------------------------------------------ plots ----
def _style(ax, th):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["grid"])
    ax.tick_params(colors=th["ink2"], labelsize=9)
    ax.yaxis.grid(True, color=th["grid"], linewidth=0.7)
    ax.set_axisbelow(True)


def _fig(th, w=8.6, h=4.4):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(th["surface"])
    _style(ax, th)
    return fig, ax


def _finish(fig, ax, th, title, subtitle, name, mode):
    ax.set_title(title, color=th["ink"], fontsize=13, fontweight="bold",
                 loc="left", pad=26)
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, color=th["ink2"], fontsize=9.5)
    fig.tight_layout()
    out = PLOTS / f"{name}_{mode}.png"
    fig.savefig(out, facecolor=th["surface"], bbox_inches="tight")
    print("wrote", out.relative_to(ROOT))
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_kaggle_score(th, mode):
    rows = list(csv.DictReader(open(DATA / "kaggle_score.csv")))
    dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in rows]
    vals = [float(r["score"]) for r in rows]
    best, run = [], -math.inf
    for v in vals:
        run = max(run, v)
        best.append(run)
    probe = [i for i, r in enumerate(rows) if r["kind"] == "probe"]
    pin = [i for i, r in enumerate(rows) if r["kind"] == "pin"]

    fig, ax = _fig(th)
    ax.scatter([dates[i] for i in probe], [vals[i] for i in probe], s=7,
               color=th["blue"], alpha=0.28, linewidths=0, label="Kaggle probe result")
    ax.plot(dates, best, color=th["blue"], linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", zorder=3)
    ax.scatter([dates[i] for i in pin], [vals[i] for i in pin], s=22, marker="s",
               color=th["surface"], edgecolors=th["blue"], linewidths=1.4, zorder=4,
               label="confirmed pin rotation")
    ax.annotate(f"{best[0]:,.0f}", (dates[0], best[0]), xytext=(9, -12),
                textcoords="offset points", va="top", color=th["ink"], fontsize=10,
                fontweight="bold")
    ax.annotate(f"{best[-1]:,.2f}", (dates[-1], best[-1]), xytext=(11, 3),
                textcoords="offset points", ha="left", va="bottom",
                color=th["ink"], fontsize=11, fontweight="bold")
    ax.annotate("final · 64th / 3,061", (dates[-1], best[-1]), xytext=(11, -8),
                textcoords="offset points", ha="left", va="top",
                color=th["ink2"], fontsize=9)
    # room on the right for the final label
    ax.set_xlim(dates[0] - timedelta(days=1), dates[-1] + timedelta(days=8))
    ax.legend(loc="lower right", frameon=False, labelcolor=th["ink2"], fontsize=9,
              handletextpad=.4, borderaxespad=.2)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    _finish(fig, ax, th, f"Leaderboard score: +{best[-1]-best[0]:,.0f} points in six weeks",
            f"Dots: {len(probe)} Kaggle probes (Jun 5 – Jul 10) · squares: {len(pin)} "
            f"confirmed pin rotations · line: best confirmed score",
            "kaggle_score", mode)


def plot_points_per_day(th, mode):
    rows = list(csv.DictReader(open(DATA / "points_per_day.csv")))
    days = [r["day"][5:] for r in rows]
    pts = [float(r["points"]) for r in rows]
    wins = [int(r["wins"]) for r in rows]
    fig, ax = _fig(th)
    bars = ax.bar(days, pts, width=0.62, color=th["green"], zorder=3)
    for b, wn in zip(bars, wins):
        if wn:
            ax.annotate(f"{wn} win" + ("s" if wn != 1 else ""),
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        color=th["ink2"], fontsize=8)
    ax.set_ylabel("points banked", color=th["ink2"], fontsize=9)
    _finish(fig, ax, th, "Verified wins per day",
            "Sum of verified score deltas from every build that passed the full gate "
            "(re-attacks on the same task counted per attempt)",
            "points_per_day", mode)


def plot_model_stats(th, mode):
    rows = [r for r in csv.DictReader(open(DATA / "model_stats.csv"))
            if int(r["attempts"]) >= 30]
    rows.sort(key=lambda r: int(r["attempts"]))
    names = [r["model"] for r in rows]
    atts = [int(r["attempts"]) for r in rows]
    wins = [int(r["wins"]) for r in rows]
    fig, ax = _fig(th, h=0.62 * len(rows) + 1.8)
    y = range(len(rows))
    ax.barh(y, atts, height=0.56, color=th["blue"], alpha=0.30, zorder=3,
            label="attempts")
    ax.barh(y, wins, height=0.56, color=th["blue"], zorder=4, label="wins")
    for i, (a, wn) in enumerate(zip(atts, wins)):
        ax.annotate(f"{wn}/{a} · {wn / a * 100:.0f}%", (a, i), xytext=(6, 0),
                    textcoords="offset points", va="center", color=th["ink2"],
                    fontsize=8.5)
    ax.set_yticks(list(y), names, fontsize=9)
    ax.xaxis.grid(True, color=th["grid"], linewidth=0.7)
    ax.yaxis.grid(False)
    ax.legend(loc="lower right", frameon=False, labelcolor=th["ink2"], fontsize=9)
    _finish(fig, ax, th, "Attempts and wins by model configuration",
            "Build attempts (light) and full-gate wins (solid) per model×effort — "
            "≥30 attempts shown",
            "model_stats", mode)


def plot_compression(th, mode):
    rows = list(csv.DictReader(open(DATA / "compression.csv")))
    pin = [int(r["pin_cost"]) for r in rows]
    new = [int(r["best_cost"]) for r in rows]
    fig, ax = _fig(th, h=5.6)
    lo, hi = 30, max(pin) * 1.4
    ax.plot([lo, hi], [lo, hi], color=th["grid"], linewidth=1.2)
    ax.plot([lo, hi], [lo / 2, hi / 2], color=th["ink2"], linewidth=1,
            linestyle=(0, (4, 4)))
    ax.text(hi * 0.42, hi * 0.48, "no change", color=th["ink2"], fontsize=8.5,
            rotation=40, rotation_mode="anchor")
    ax.text(hi * 0.42, hi * 0.155, "2× cheaper (+0.69 pts)", color=th["ink2"],
            fontsize=8.5, rotation=40, rotation_mode="anchor")
    ax.scatter(pin, new, s=16, color=th["blue"], alpha=0.65, linewidths=0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("previous best cost (bytes, log)", color=th["ink2"], fontsize=9)
    ax.set_ylabel("rebuilt cost (bytes, log)", color=th["ink2"], fontsize=9)
    ax.xaxis.grid(True, color=th["grid"], linewidth=0.7)
    _finish(fig, ax, th, "Every win is a compression ratio",
            "Each dot is one task: the incumbent graph's cost vs the fleet's rebuilt "
            "graph. score = 25 − ln(cost), so distance below the line is points",
            "compression", mode)


def plot_deltas(th, mode):
    rows = list(csv.DictReader(open(DATA / "compression.csv")))
    deltas = sorted(float(r["delta"]) for r in rows)
    fig, ax = _fig(th)
    ax.hist(deltas, bins=40, color=th["blue"], zorder=3)
    structural = sum(d for d in deltas if d >= 0.69)
    total = sum(deltas)
    for x, label in ((0.15, "registration floor"), (0.69, "2× cut"), (1.39, "4× cut")):
        ax.axvline(x, color=th["ink2"], linewidth=1, linestyle=(0, (4, 4)))
        ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", color=th["ink2"],
                    fontsize=8.5)
    ax.set_xlabel("points gained per task (Δ)", color=th["ink2"], fontsize=9)
    ax.set_ylabel("tasks", color=th["ink2"], fontsize=9)
    _finish(fig, ax, th, "Why the prompts demand reconstruction",
            f"Distribution of per-task wins: builds at ≥2× compression are "
            f"{structural / total * 100:.0f}% of all points banked — shavings barely move "
            f"the score",
            "deltas", mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derive", action="store_true",
                    help="regenerate assets/data/*.csv from the private logs")
    ap.add_argument("--results", default=str(ROOT / "results"),
                    help="results/ tree with ATTEMPTS.jsonl + LEDGER.csv (derive stage)")
    a = ap.parse_args()
    if a.derive:
        derive(Path(a.results))
    import matplotlib
    matplotlib.use("Agg")
    PLOTS.mkdir(parents=True, exist_ok=True)
    for mode, th in THEMES.items():
        plot_kaggle_score(th, mode)
        plot_points_per_day(th, mode)
        plot_model_stats(th, mode)
        plot_compression(th, mode)
        plot_deltas(th, mode)


if __name__ == "__main__":
    main()

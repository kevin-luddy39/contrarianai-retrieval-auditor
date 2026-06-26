"""Step 6 of SPEC.md — generate one chart per query, side-by-side clean vs poisoned.

Reads:
  results/clean.json
  results/poisoned.json

Writes:
  results/chart_q*.png  (one per query, 2-panel: clean and poisoned)
  results/summary.png   (one big table-style heatmap of pathology firings)

Usage:
    python plot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def load(mode: str) -> dict:
    path = RESULTS / f"{mode}.json"
    if not path.exists():
        sys.exit(f"missing {path}; run `python run.py {'' if mode == 'clean' else '--poisoned'}` first")
    return json.loads(path.read_text(encoding="utf-8"))


def chart_query(qid: str, q_clean: dict, q_poisoned: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)

    for ax, run, q in zip(axes, ("clean", "poisoned"), (q_clean, q_poisoned)):
        retrieved = q["retrieved"]
        scores = [c["score"] for c in retrieved]
        # use auditor's independent alignment (TF-IDF) where available
        align = (q.get("audit") or {}).get("alignments")
        if align is None:
            align = scores
        flags = ", ".join(q["summary"]["flags"]) or "none"
        health = q["summary"]["health"]
        h_text = f"{health:.3f}" if isinstance(health, (int, float)) else "—"
        title = f"{run.upper()}\nflags: {flags}\nhealth: {h_text}  p@5: {q['summary']['p_at_k']:.2f}"

        x = np.arange(len(retrieved))
        width = 0.4
        ax.bar(x - width / 2, scores, width, label="retriever score", color="steelblue", alpha=0.9)
        ax.bar(x + width / 2, align, width, label="TF-IDF alignment", color="darkorange", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{i+1}" for i in range(len(retrieved))])
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("rank")
        if ax is axes[0]:
            ax.set_ylabel("score / alignment")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{qid}: {q_clean['query']}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = RESULTS / f"chart_{qid.lower()}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out.name}")


def summary_grid(clean: dict, poisoned: dict) -> None:
    queries = [q["id"] for q in clean["queries"]]
    flags_universe = sorted({
        f for run in (clean, poisoned) for q in run["queries"]
        for f in q["summary"]["flags"]
    })
    if not flags_universe:
        flags_universe = ["(no flags fired)"]

    grid = np.zeros((len(queries), 2 * len(flags_universe)))
    labels: list[str] = []
    for j, fk in enumerate(flags_universe):
        labels += [f"{fk}\n(clean)", f"{fk}\n(poisoned)"]

    for i, qid in enumerate(queries):
        c = next(q for q in clean["queries"] if q["id"] == qid)
        p = next(q for q in poisoned["queries"] if q["id"] == qid)
        for j, fk in enumerate(flags_universe):
            grid[i, 2 * j] = 1 if fk in c["summary"]["flags"] else 0
            grid[i, 2 * j + 1] = 1 if fk in p["summary"]["flags"] else 0

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), max(4, len(queries) * 0.45)))
    ax.imshow(grid, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels(queries, fontsize=9)
    for i in range(len(queries)):
        for j in range(len(labels)):
            if grid[i, j] > 0:
                ax.text(j, i, "✓", ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Pathology fire pattern: clean vs poisoned", fontsize=11)
    plt.tight_layout()
    out = RESULTS / "summary.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out.name}")


def main() -> int:
    clean = load("clean")
    poisoned = load("poisoned")

    by_id_c = {q["id"]: q for q in clean["queries"]}
    by_id_p = {q["id"]: q for q in poisoned["queries"]}
    common = [qid for qid in by_id_c if qid in by_id_p]

    print(f"Plotting {len(common)} queries ...")
    for qid in common:
        chart_query(qid, by_id_c[qid], by_id_p[qid])

    summary_grid(clean, poisoned)
    return 0


if __name__ == "__main__":
    sys.exit(main())

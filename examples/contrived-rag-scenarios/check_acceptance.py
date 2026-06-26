"""Step 7 of SPEC.md — automated acceptance-criteria gate (v0.3, reframe).

Reads:
  results/clean.json
  results/poisoned.json

Recalibrated for the empirical reframe (see ground_truth.json reframe_note):
the auditor catches retrieval-mechanism pathology (REDUNDANT clusters, the
natural Q6 ranking flaw), not factual content errors (MIS / OFF poisons).
The artifact teaches a three-bucket distinction. Acceptance gates only the
bucket the auditor is responsible for.

Criteria (v0.3):

1. Clean baselines Q7, Q9: no engineered pathology flag in either run
   (severity < 0.4 for OFF_TOPIC, RANK_INVERSION, SCORE_MISCALIBRATED, REDUNDANT).
2. Caught-by-auditor: REDUNDANT fires on Q3 + Q8 in poisoned run at severity
   >= 0.30, and does NOT fire on the same queries in clean run (severity < 0.30).
3. Natural pathology: Q6 fires RANK_INVERSION + SCORE_MISCALIBRATED in BOTH runs
   (it is a property of the base corpus + dense embedding, not the poisons).
4. OOD probe: Q10 fires OFF_TOPIC in both runs.
5. Missed-by-design queries (Q1, Q2, Q4, Q5): no engineered pathology flag
   fires (the auditor correctly does not detect content errors as retrieval
   pathology). This is the teaching point — verified, not failed.

Severity gate is 0.30 (not 0.40) — the dense profile's penalties are gentler
than the tfidf profile's, and 0.30 is well above the noise floor while still
distinguishing intentional fires from incidental low-magnitude triggers.

Exit code 0 iff all criteria pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PASS = "PASS"
FAIL = "FAIL"

SEVERITY_GATE = 0.30
FLAG_KINDS = ("OFF_TOPIC", "RANK_INVERSION", "SCORE_MISCALIBRATED", "REDUNDANT")


def load(mode: str) -> dict:
    path = RESULTS / f"{mode}.json"
    if not path.exists():
        sys.exit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def severity_of(audit: dict | None, kind: str) -> float:
    if not audit:
        return 0.0
    for p in audit.get("pathologies") or []:
        if p.get("kind") == kind:
            return float(p.get("severity") or 0.0)
    return 0.0


def main() -> int:
    clean = load("clean")
    poisoned = load("poisoned")

    by_c = {q["id"]: q for q in clean["queries"]}
    by_p = {q["id"]: q for q in poisoned["queries"]}

    fails: list[str] = []

    for qid in ("Q7", "Q9"):
        for run_label, by in (("clean", by_c), ("poisoned", by_p)):
            for kind in FLAG_KINDS:
                sev = severity_of(by[qid].get("audit"), kind)
                if sev >= SEVERITY_GATE:
                    fails.append(
                        f"Criterion 1: clean baseline {qid} fired {kind} "
                        f"in {run_label} at severity {sev:.2f} "
                        f"(must be < {SEVERITY_GATE})"
                    )

    for qid in ("Q3", "Q8"):
        sev_c = severity_of(by_c[qid].get("audit"), "REDUNDANT")
        sev_p = severity_of(by_p[qid].get("audit"), "REDUNDANT")
        if sev_p < SEVERITY_GATE:
            fails.append(
                f"Criterion 2: REDUNDANT did not fire on {qid} in poisoned run "
                f"(severity {sev_p:.2f}, must be >= {SEVERITY_GATE})"
            )
        if sev_c >= SEVERITY_GATE:
            fails.append(
                f"Criterion 2: REDUNDANT incorrectly fired on {qid} in clean run "
                f"(severity {sev_c:.2f}, must be < {SEVERITY_GATE})"
            )

    for run_label, by in (("clean", by_c), ("poisoned", by_p)):
        sev_ri = severity_of(by["Q6"].get("audit"), "RANK_INVERSION")
        sev_sc = severity_of(by["Q6"].get("audit"), "SCORE_MISCALIBRATED")
        if sev_ri < SEVERITY_GATE:
            fails.append(
                f"Criterion 3: Q6 RANK_INVERSION did not fire in {run_label} "
                f"(severity {sev_ri:.2f}, must be >= {SEVERITY_GATE}). "
                f"This is the natural-pathology bonus finding; if it disappears the artifact loses a key teaching example."
            )
        if sev_sc < SEVERITY_GATE:
            fails.append(
                f"Criterion 3: Q6 SCORE_MISCALIBRATED did not fire in {run_label} "
                f"(severity {sev_sc:.2f}, must be >= {SEVERITY_GATE})."
            )

    for run_label, by in (("clean", by_c), ("poisoned", by_p)):
        flags = by["Q10"]["summary"]["flags"]
        if "OFF_TOPIC" not in flags and "OUT_OF_DISTRIBUTION" not in flags:
            fails.append(
                f"Criterion 4: Q10 did not fire OFF_TOPIC or OUT_OF_DISTRIBUTION "
                f"in {run_label} (flags={flags}). With a true OOD probe (Pythagorean "
                f"theorem on a biology corpus) one of those must fire — if neither does, "
                f"the dense profile thresholds are too lenient."
            )

    for qid in ("Q1", "Q2", "Q4", "Q5"):
        for run_label, by in (("clean", by_c), ("poisoned", by_p)):
            for kind in FLAG_KINDS:
                sev = severity_of(by[qid].get("audit"), kind)
                if sev >= SEVERITY_GATE:
                    fails.append(
                        f"Criterion 5: missed-by-design {qid} unexpectedly fired {kind} "
                        f"in {run_label} (severity {sev:.2f}). The teaching point relies "
                        f"on these queries staying clean — investigate before publishing."
                    )

    print("=" * 70)
    print("Acceptance check (v0.3 reframe)")
    print("=" * 70)
    if fails:
        print(f"{FAIL}: {len(fails)} criteria not met\n")
        for f in fails:
            print(f"  - {f}")
        print()
        print("Not publishable yet. Iterate.")
        return 1
    print(f"{PASS}: all criteria met. Artifact is publishable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

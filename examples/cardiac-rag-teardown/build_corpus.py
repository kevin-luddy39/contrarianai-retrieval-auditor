"""Step 1 — assemble cardiac corpus from public Wikipedia cardiology pages.

Wikipedia is CC BY-SA 4.0; dense, well-cited cardiology coverage; predictable
URLs. Same fetch pattern as the LangChain teardown and the contrived-RAG
scenario.

Output: corpus/base.txt (concatenated, light-cleaned, paragraph-aware)

Run:
    python3 build_corpus.py

Future iteration: add StatPearls cardiology subset via NCBI Bookshelf if
corpus density needs boosting beyond Wikipedia.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT = Path(__file__).parent / "corpus"
OUT.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "contrarianAI-cardiac-corpus/0.1 (kevin.luddy39@gmail.com)"}

CARDIOLOGY_TOPICS = [
    "Heart_failure",
    "Myocardial_infarction",
    "Acute_coronary_syndrome",
    "Atrial_fibrillation",
    "Atrial_flutter",
    "Ventricular_tachycardia",
    "Ventricular_fibrillation",
    "Angina_pectoris",
    "Unstable_angina",
    "Coronary_artery_disease",
    "Aortic_stenosis",
    "Aortic_regurgitation",
    "Mitral_regurgitation",
    "Mitral_stenosis",
    "Hypertrophic_cardiomyopathy",
    "Dilated_cardiomyopathy",
    "Restrictive_cardiomyopathy",
    "Endocarditis",
    "Pericarditis",
    "Aortic_dissection",
    "Pulmonary_embolism",
    "Deep_vein_thrombosis",
    "Hypertension",
    "Hyperlipidemia",
    "Beta_blocker",
    "ACE_inhibitor",
    "Statin",
    "Heparin",
    "Warfarin",
    "Aspirin",
    "Clopidogrel",
    "Amiodarone",
    "Digoxin",
    "Electrocardiography",
    "Echocardiography",
    "Coronary_artery_bypass_surgery",
    "Percutaneous_coronary_intervention",
    "Cardiac_arrest",
    "Cardiopulmonary_resuscitation",
    "Cardiogenic_shock",
]

SOURCES = []


def fetch(url: str, sleep: float = 0.5) -> str:
    print(f"  GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(sleep)
    return r.text


def clean_text(t: str) -> str:
    t = re.sub(r"\[\d+\]", " ", t)
    t = re.sub(r"\[edit\]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def extract_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    scope = soup.select_one("#mw-content-text") or soup.find("main") or soup
    paras = []
    for p in scope.find_all("p"):
        txt = clean_text(p.get_text(" "))
        if len(txt) >= 80:
            paras.append(txt)
    return paras


def fetch_topic(topic: str) -> list[str]:
    url = f"https://en.wikipedia.org/wiki/{topic}"
    try:
        html = fetch(url)
    except Exception as e:
        print(f"  ! {topic}: {e}")
        return []
    paras = extract_paragraphs(html)
    SOURCES.append({
        "topic": topic.replace("_", " "),
        "url": url,
        "license": "CC BY-SA 4.0",
        "paragraphs": len(paras),
    })
    return paras


def write_outputs(paras: list[str]) -> None:
    base = OUT / "base.txt"
    with base.open("w", encoding="utf-8") as f:
        for p in paras:
            f.write(p + "\n\n")
    print(f"\nWrote {base} — {len(paras)} paragraphs, {sum(len(p) for p in paras):,} chars")
    src = OUT / "sources.json"
    src.write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")
    print(f"Wrote {src}")


def main() -> int:
    paras: list[str] = []
    for topic in CARDIOLOGY_TOPICS:
        print(f"Source: Wikipedia {topic.replace('_', ' ')}")
        paras += fetch_topic(topic)

    seen = set()
    deduped = []
    for p in paras:
        sig = p[:120]
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(p)
    print(f"\nDedup: {len(paras)} -> {len(deduped)} paragraphs")

    write_outputs(deduped)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Step 1 of SPEC.md — assemble base corpus from public sources.

Pulls clean photosynthesis text from three open sources:
  1. Wikipedia "Photosynthesis" (CC BY-SA)
  2. OpenStax Biology 2e, Ch. 8 (CC BY 4.0)
  3. Project Gutenberg public-domain botany text

Output: corpus/base.txt (concatenated, light-cleaned, paragraph-aware)

Run:
    python3 build_corpus.py

No external API keys required. Uses requests + beautifulsoup4 only.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

OUT = Path(__file__).parent / "corpus"
OUT.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "contrarianAI-corpus-builder/0.1 (kevin.luddy39@gmail.com)"}

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


def extract_paragraphs(html: str, container_selector: str | None = None) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    if container_selector:
        scope = soup.select_one(container_selector) or soup
    else:
        scope = soup.find("article") or soup.find("main") or soup
    paras = []
    for p in scope.find_all("p"):
        txt = clean_text(p.get_text(" "))
        if len(txt) >= 80:
            paras.append(txt)
    return paras


def fetch_wikipedia() -> list[str]:
    print("Source 1: Wikipedia 'Photosynthesis'")
    url = "https://en.wikipedia.org/wiki/Photosynthesis"
    html = fetch(url)
    paras = extract_paragraphs(html, container_selector="#mw-content-text")
    print(f"  -> {len(paras)} paragraphs")
    SOURCES.append({"name": "Wikipedia: Photosynthesis", "url": url, "license": "CC BY-SA 4.0", "paragraphs": len(paras)})
    return paras


def fetch_openstax() -> list[str]:
    print("Source 2: OpenStax Biology 2e Ch. 8 (Photosynthesis)")
    base = "https://openstax.org/books/biology-2e/pages/"
    pages = [
        "8-introduction",
        "8-1-overview-of-photosynthesis",
        "8-2-the-light-dependent-reactions-of-photosynthesis",
        "8-3-using-light-energy-to-make-organic-molecules",
    ]
    paras = []
    for slug in pages:
        try:
            html = fetch(base + slug, sleep=1.0)
            paras += extract_paragraphs(html)
        except Exception as e:
            print(f"  ! {slug}: {e}")
    print(f"  -> {len(paras)} paragraphs total")
    SOURCES.append({"name": "OpenStax Biology 2e Ch. 8", "url": base, "license": "CC BY 4.0", "paragraphs": len(paras)})
    return paras


def fetch_gutenberg() -> list[str]:
    print("Source 3: Project Gutenberg — Strasburger 'Lehrbuch der Botanik' English ed. (PG #58207, public domain)")
    candidates = [
        "https://www.gutenberg.org/cache/epub/58207/pg58207.txt",
        "https://www.gutenberg.org/files/19073/19073-0.txt",
    ]
    paras: list[str] = []
    for url in candidates:
        try:
            text = fetch(url, sleep=1.0)
            blocks = [p.strip() for p in text.split("\n\n") if p.strip()]
            blocks = [clean_text(b) for b in blocks if len(b) >= 200]
            keep = [b for b in blocks if any(k in b.lower() for k in ("photosynth", "chloroph", "light react", "leaf", "plant cell"))]
            print(f"  {url}: {len(blocks)} blocks, {len(keep)} on-topic")
            paras += keep[:50]
            SOURCES.append({"name": f"Project Gutenberg: {url.rsplit('/', 1)[-1]}", "url": url, "license": "public domain", "paragraphs": len(keep[:50])})
            if paras:
                break
        except Exception as e:
            print(f"  ! {url}: {e}")
    print(f"  -> {len(paras)} on-topic paragraphs")
    return paras


def write_base(paras: list[str]) -> None:
    base = OUT / "base.txt"
    with base.open("w", encoding="utf-8") as f:
        for p in paras:
            f.write(p + "\n\n")
    print(f"\nWrote {base} — {len(paras)} paragraphs, {sum(len(p) for p in paras):,} chars")


def write_sources() -> None:
    src = OUT / "sources.json"
    import json
    src.write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")
    print(f"Wrote {src}")


def main() -> int:
    paras: list[str] = []
    paras += fetch_wikipedia()
    paras += fetch_openstax()
    paras += fetch_gutenberg()

    seen = set()
    deduped = []
    for p in paras:
        sig = p[:120]
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(p)
    print(f"\nDedup: {len(paras)} -> {len(deduped)} paragraphs")

    write_base(deduped)
    write_sources()
    return 0


if __name__ == "__main__":
    sys.exit(main())

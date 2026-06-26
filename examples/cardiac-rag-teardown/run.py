"""Run cardiac retrieval audit.

Reads:
  corpus/base.txt              — Wikipedia cardiology corpus (build_corpus.py)
  queries_sampled_50.json      — 50 strict-cardiac MedQA-USMLE clinical vignettes
                                  filtered from MIRAGE benchmark

Embeds with sentence-transformers all-MiniLM-L6-v2 (matches LangChain teardown
+ contrived-RAG scenario for direct comparability), indexes in chromadb,
runs each query at top-K=5, pipes each retrieval payload through
retrieval-auditor's CLI with --profile dense for pathology analysis.

Note on query shape: MedQA queries are clinical vignettes (long-form, mean
~750 chars) — realistic for clinical-decision-support RAG. The auditor's
TF-IDF grader produces different absolute alignment magnitudes on long
clinical text than on short keyword queries; if OFF_TOPIC carpet-fires or
RANK_INVERSION misses, recalibrate dense profile thresholds (same loop as
contrived-RAG scenario; SPEC.md captures the iteration).

Output:
  results/cardiac.json
  results/q_*.json (per-query detail)

Usage (from this directory):
    python3 run.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RETRIEVAL_AUDITOR_CLI = (HERE / ".." / ".." / "cli.js").resolve()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
RNG_SEED = 42


def load_chunks() -> list[dict]:
    text = (HERE / "corpus" / "base.txt").read_text(encoding="utf-8")
    chunks: list[dict] = []
    i = 0
    while i < len(text):
        body = text[i : i + CHUNK_SIZE].strip()
        if body:
            chunks.append({"id": f"c_{len(chunks):05d}", "text": body})
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_queries() -> list[dict]:
    return json.loads((HERE / "queries_sampled_50.json").read_text(encoding="utf-8"))


def build_index(chunks: list[dict]):
    print(f"Loading embedding model {EMBED_MODEL}")
    np.random.seed(RNG_SEED)
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    client = chromadb.Client()
    name = f"cardiac_rag_{int(time.time())}"
    coll = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    coll.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embs.tolist(),
    )
    return model, coll


def retrieve(model, coll, query: str, k: int = TOP_K) -> dict:
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    res = coll.query(query_embeddings=[q_emb.tolist()], n_results=k)
    return {
        "query": query,
        "retrieved": [
            {
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "score": float(1.0 - res["distances"][0][i]),
            }
            for i in range(len(res["ids"][0]))
        ],
    }


def run_auditor(payload: dict) -> dict | None:
    proc = subprocess.run(
        ["node", str(RETRIEVAL_AUDITOR_CLI), "-", "--json", "--profile", "dense"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        print(f"  AUDITOR ERROR: {proc.stderr.strip()[:200]}")
        return None
    return json.loads(proc.stdout)


def short(query: str, n: int = 80) -> str:
    return (query[:n] + "...") if len(query) > n else query


def main() -> int:
    print("=== Cardiac RAG Audit ===")
    chunks = load_chunks()
    print(f"Chunks: {len(chunks)}")
    queries = load_queries()
    print(f"Queries: {len(queries)}")

    model, coll = build_index(chunks)

    out: dict = {"config": {
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "top_k": TOP_K,
        "embed_model": EMBED_MODEL, "n_chunks": len(chunks), "n_queries": len(queries),
    }, "queries": []}

    for i, q in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] {q['source']}/{q['qid']}: {short(q['question'])}")
        payload = retrieve(model, coll, q["question"])
        audit = run_auditor(payload)
        flags = [p["kind"] for p in (audit.get("pathologies") or [])] if audit else []
        health = audit.get("health") if audit else None
        rankR = (audit.get("retrieval") or {}).get("rankQualityR") if audit else None
        calibR = (audit.get("retrieval") or {}).get("scoreCalibrationR") if audit else None
        meanA = ((audit.get("domain") or {}).get("stats") or {}).get("mean") if audit else None

        out["queries"].append({
            "qid": q["qid"],
            "source": q["source"],
            "question": q["question"],
            "options": q.get("options"),
            "answer": q.get("answer"),
            "retrieved": payload["retrieved"],
            "audit": audit,
            "summary": {
                "flags": flags,
                "health": health,
                "rankQualityR": rankR,
                "scoreCalibrationR": calibR,
                "meanAlignment": meanA,
            },
        })
        print(f"  flags: {', '.join(flags) or 'none'} | health={health} | rankR={rankR} | calibR={calibR}")

    out_path = RESULTS_DIR / "cardiac.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

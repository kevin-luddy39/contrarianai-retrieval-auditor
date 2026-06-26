"""Step 4 of SPEC.md — runner for clean and poisoned modes.

Reads:
  corpus/base.txt           — base corpus (built by build_corpus.py)
  corpus/poisons.json       — 15 hand-authored poisons (used only in --poisoned mode)
  corpus/ground_truth.json  — 10 queries + per-query truth predicates

Embeds with sentence-transformers all-MiniLM-L6-v2 (matches the LangChain
quickstart teardown for direct comparability), indexes in chromadb, runs each
query at top-K=5, then pipes each retrieval payload through retrieval-auditor's
CLI for pathology analysis.

Outputs:
  results/clean.json     (run with no flag)
  results/poisoned.json  (run with --poisoned)

Usage (from this directory):
    python run.py             # clean mode
    python run.py --poisoned  # poisoned mode

Both modes must be run to produce the side-by-side comparison the teardown post
needs. Acceptance criteria are checked by check_acceptance.py (TBD), not here.

Reproducibility:
- Random seed is fixed via numpy + chromadb config below.
- Sentence-transformers determinism is high but not perfectly bit-exact across
  GPU/CPU; results may vary in last decimal place across machines. The
  pathology flag firing should be stable.
"""

from __future__ import annotations

import argparse
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
CORPUS_DIR = HERE / "corpus"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Resolve auditor CLI relative to repo (works on Linux/WSL/Windows)
RETRIEVAL_AUDITOR_CLI = (HERE / ".." / ".." / "cli.js").resolve()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
RNG_SEED = 42


def load_base_chunks() -> list[dict]:
    text = (CORPUS_DIR / "base.txt").read_text(encoding="utf-8")
    chunks: list[dict] = []
    i = 0
    while i < len(text):
        body = text[i : i + CHUNK_SIZE].strip()
        if body:
            chunks.append({"id": f"base_{len(chunks):04d}", "text": body, "source": "base"})
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_poison_chunks() -> list[dict]:
    pois = json.loads((CORPUS_DIR / "poisons.json").read_text(encoding="utf-8"))
    return [{"id": p["id"], "text": p["text"], "source": "poison",
             "target_pathology": p["target_pathology"], "intent": p["intent"]}
            for p in pois["poisons"]]


def load_queries() -> list[dict]:
    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    return gt["queries"]


def build_index(chunks: list[dict]):
    print(f"Loading embedding model {EMBED_MODEL} (one-time download first run)")
    np.random.seed(RNG_SEED)
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    client = chromadb.Client()
    name = f"contrived_rag_{int(time.time())}"
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


def precision_at_k(retrieved: list[dict], predicates: dict, k: int = TOP_K) -> dict:
    must_all = [t.lower() for t in (predicates.get("must_contain_all") or [])]
    must_any = [t.lower() for t in (predicates.get("must_contain_any") or [])]
    must_not_top = [t.lower() for t in (predicates.get("must_not_contain_at_top_rank") or [])]
    must_not_any = [t.lower() for t in (predicates.get("must_not_contain") or [])]

    hits = 0
    top_violations = 0
    any_violations = 0
    for rank, c in enumerate(retrieved[:k]):
        body = c["text"].lower()
        ok_all = all(t in body for t in must_all) if must_all else True
        ok_any = any(t in body for t in must_any) if must_any else True
        if ok_all and ok_any:
            hits += 1
        if rank == 0:
            top_violations += sum(1 for t in must_not_top if t in body)
        any_violations += sum(1 for t in must_not_any if t in body)

    return {
        "hits_at_k": hits,
        "precision_at_k": hits / k,
        "top1_violations": top_violations,
        "any_violations": any_violations,
    }


def run(mode: str) -> Path:
    assert mode in ("clean", "poisoned")
    print(f"=== Mode: {mode} ===")
    chunks = load_base_chunks()
    print(f"Base chunks: {len(chunks)}")
    if mode == "poisoned":
        poisons = load_poison_chunks()
        chunks += poisons
        print(f"Added {len(poisons)} poisons -> {len(chunks)} total chunks")
    queries = load_queries()
    print(f"Queries: {len(queries)}")

    model, coll = build_index(chunks)

    out: dict = {"mode": mode, "config": {
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "top_k": TOP_K,
        "embed_model": EMBED_MODEL, "n_chunks": len(chunks),
    }, "queries": []}

    for q in queries:
        print(f"\n[{q['id']}] {q['text']}")
        payload = retrieve(model, coll, q["text"])
        audit = run_auditor(payload)
        prec = precision_at_k(payload["retrieved"], q["true_chunk_predicates"])
        flags = [p["kind"] for p in (audit.get("pathologies") or [])] if audit else []
        health = audit.get("health") if audit else None
        out["queries"].append({
            "id": q["id"],
            "query": q["text"],
            "demonstration_category": q.get("demonstration_category"),
            "associated_poison": q.get("associated_poison"),
            "expected_behavior": q.get("expected_behavior"),
            "retrieved": payload["retrieved"],
            "audit": audit,
            "precision": prec,
            "summary": {
                "flags": flags,
                "health": health,
                "p_at_k": prec["precision_at_k"],
            },
        })
        print(f"  flags: {', '.join(flags) or 'none'} | health: {health} | p@5: {prec['precision_at_k']:.2f}")

    out_path = RESULTS_DIR / f"{mode}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poisoned", action="store_true")
    args = ap.parse_args()
    mode = "poisoned" if args.poisoned else "clean"
    run(mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())

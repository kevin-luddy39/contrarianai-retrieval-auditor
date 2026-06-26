# I built a RAG corpus with 12 deliberate landmines. retrieval-auditor caught some — and the ones it missed tell you exactly what the auditor is for.

A controlled follow-up to [the LangChain quickstart teardown](../langchain-quickstart-teardown/). Where that piece ran the auditor against a real public RAG repo, this one runs it against a corpus I engineered specifically to surface every pathology class the auditor advertises. It does not catch them all. The ones it misses are the most useful part.

[INSERT: results/summary.png here]

## TL;DR

I assembled a 170-chunk biology corpus (Wikipedia + OpenStax) on photosynthesis, hand-authored 15 poison chunks across four pathology classes, ran a sentence-transformers MiniLM retriever over it, and piped the results through `retrieval-auditor --profile dense`. Then I asked 10 queries.

The auditor produced three honest categories of result:

| Bucket | Queries | What happened |
|--------|---------|---------------|
| **Caught** | Q3, Q8 | Three near-duplicate paraphrases of one fact were retrieved together; `REDUNDANT` fires in poisoned mode only. |
| **Caught (no poison needed)** | Q6 | The base corpus alone produces a genuine ranking flaw on the chlorophyll query; auditor surfaces it without any landmine. Bonus finding. |
| **Missed by design** | Q1, Q2, Q4, Q5 | A factually-wrong chunk sits at rank 1 with retriever score 0.74. Auditor reports CLEAN. The chunk is wrong about the photosynthesis equation, the organelle, the gas exchange, or the comparison. The auditor does not measure correctness — only retrieval mechanics. |
| Clean baselines | Q7, Q9 | No flags. Validates the auditor isn't carpet-bombing. |
| OOD probe | Q10 | "What is the Pythagorean theorem?" against a biology corpus. Auditor fires `OUT_OF_DISTRIBUTION`, health 0. |

The headline lesson: **`retrieval-auditor` is a measurement of retrieval-mechanism pathology, not factual correctness.** Pair it with a content grader. If the chunk is retrieved well-ranked, well-calibrated, and non-redundant, but the chunk is wrong, the auditor cannot tell you. Knowing this is the price of admission for using the tool well.

---

## What the corpus contains

- **170 base chunks**, ~110K characters total, sourced from Wikipedia "Photosynthesis" (CC BY-SA 4.0) and OpenStax Biology 2e Chapter 8 (CC BY 4.0). Light cleanup, paragraph-aware split, then chunked at 800/150 overlap.
- **15 hand-authored poisons** (`corpus/poisons.json`):
  - 3 RANK_INVERSION baits (vocabulary-stuffed off-topic chunks)
  - 3 SCORE_MISCALIBRATED baits (factually wrong, surface-textbook-tone)
  - 3 OFF_TOPIC baits (adjacent topics that retriever ranks too high)
  - 6 REDUNDANT cluster members (two clusters of 3 paraphrases each)

Each poison ID, target pathology, and intent statement is in the JSON. No corpus is hidden; the entire experiment is reproducible from the included files.

---

## The headline finding — the auditor reports CLEAN on a wrong answer

**Query Q4: "What gases do plants exchange during photosynthesis?"**

The poisoned-mode top-5 retrieval:

```
rank 1   P-MIS-1     retriever score 0.741   POISON
rank 2   base_0076   retriever score 0.727
rank 3   P-DIV-1c    retriever score 0.702   POISON
rank 4   base_0079   retriever score 0.669
rank 5   base_0077   retriever score 0.667
```

What is `P-MIS-1`? Verbatim:

> "Photosynthesis is the central biological process by which green plants release carbon dioxide into the atmosphere and absorb oxygen from the surrounding air..."

Plants do the opposite. The chunk inverts the gas exchange.

What does the auditor report?

```
flags:           none
health:          1.000
mean alignment:  0.217
rankQualityR:   +0.826  (strong positive — ranking is consistent with TF-IDF alignment)
scoreCalibrationR: +0.882  (retriever scores agree with grader)
redundancyRatio: 0.372  (well below threshold)
```

Every retrieval-mechanism signal is healthy. The auditor and the retriever agree. The chunk is also dead wrong. Both signals can be true — they measure different things.

[INSERT: results/chart_q4.png here]

This is what the artifact is fundamentally about. The auditor is doing exactly what it's designed to do. What it's designed to do does not include catching factual errors. If you wire this tool into a production pipeline expecting it to flag hallucinations or content inversions, you'll be disappointed and surprised — twice.

---

## What the auditor *did* catch — REDUNDANT clusters

**Query Q3: "Explain the Calvin cycle."**

Poisoned retrieval pulled the entire P-DIV-1 cluster (three paraphrases of the same Calvin-cycle definition) into ranks 1, 2, 3. RedundancyRatio jumped from **0.474 (clean) to 0.752 (poisoned)**. `REDUNDANT` fires.

[INSERT: results/chart_q3.png here]

**Query Q8: "Describe the light-dependent reactions."**

Same shape. P-DIV-2 cluster lands at ranks 2, 4, 5. RedundancyRatio jumps **0.543 → 0.837**. `REDUNDANT` fires harder than on Q3.

[INSERT: results/chart_q8.png here]

REDUNDANT is the cleanest fire because it's the most mechanical pathology: count near-duplicates, threshold. The auditor's measurement maps directly to the failure mode. No interpretation gap.

---

## What the auditor caught with no poison at all — Q6 chlorophyll

**Query Q6: "What is the role of chlorophyll?"**

I engineered a poison for this query (`P-INV-1`, a kitchen appliance review padded with "chlorophyll-green ceramic coating" and similar surface-vocabulary stuffing). The retriever correctly ignored it — it never made top-5.

But the auditor still fires `RANK_INVERSION` and `SCORE_MISCALIBRATED` on this query, in **both clean and poisoned runs**. Without any landmine.

```
rankQualityR:    -0.435   (top-ranked chunks are LESS aligned than lower-ranked)
scoreCalibrationR: -0.508   (retriever scores anti-correlate with TF-IDF alignment)
health:          0.699
flags:           RANK_INVERSION, SCORE_MISCALIBRATED
```

The five chunks the retriever returned are all reasonable answers (precision@5 = 1.00 — every chunk mentions chlorophyll). But the dense embedding ranks them in the wrong order: a chunk about leaf mesophyll structure ranks above a chunk that actually defines chlorophyll's photosystem role. Auditor sees the inversion. The retriever's own confidence is anti-correlated with independent alignment.

This is the natural pathology that's most worth knowing about as an operator: **your retriever can be doing something quietly broken on a perfectly normal query, and you would not see it without this kind of measurement.** No poison, no contrived setup, no synthetic stress test required.

[INSERT: results/chart_q6.png here]

---

## OOD detection works when it should — Q10

To verify the auditor catches genuinely out-of-distribution queries, I asked it about geometry on a biology corpus.

**Q10: "What is the Pythagorean theorem?"**

Auditor:
```
flags:    OUT_OF_DISTRIBUTION, SCORE_MISCALIBRATED
health:   0.000
mean alignment: well below the 0.05 OOD threshold
```

Earlier I tried a softer OOD probe ("when did photosynthesis first evolve") — auditor missed it because the corpus has tangential mentions of evolution and timescales. Pythagorean theorem has zero overlap with photosynthesis vocabulary. OOD detection works on hard out-of-distribution; soft OOD against a domain-rich corpus is a coin flip. Worth knowing.

---

## Methodology — exactly what I ran

```bash
git clone https://github.com/kevin-luddy39/contrarianAI
cd contrarianAI/tools/retrieval-auditor/examples/contrived-rag-scenarios
pip install -r requirements.txt
python build_corpus.py        # fetches Wikipedia + OpenStax, writes corpus/base.txt
python run.py                  # clean mode → results/clean.json + results/clean charts
python run.py --poisoned       # poisoned mode → results/poisoned.json
python check_acceptance.py     # automated publish-gate; prints PASS or FAIL list
python plot.py                 # generates side-by-side per-query charts + summary heatmap
```

**Components**:
- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (free, local; matches the LangChain teardown for direct comparability)
- Vector store: `chromadb` in-memory, cosine distance
- Chunking: 800 chars, 150 overlap
- Top-K: 5
- Auditor: `tools/retrieval-auditor/cli.js --profile dense`

**Profile note**: `retrieval-auditor` ships with two threshold profiles. `tfidf` (default) is calibrated for sparse retrievers where mean alignment runs ~0.40. `dense` is calibrated for embedding retrievers where mean alignment runs ~0.20. Without a profile selection, OFF_TOPIC carpet-fires on every query when the retriever is dense. The profile flag is documented in the auditor's `--help`.

---

## Acceptance gate

The artifact does not publish until `check_acceptance.py` exits 0. The five gates are encoded in the script — they are not subjective:

1. Clean baselines Q7, Q9: zero engineered-pathology flags, either run.
2. REDUNDANT fires on Q3 + Q8 in poisoned only, severity ≥ 0.30.
3. Q6 natural pathology (RANK_INVERSION + SCORE_MISCALIBRATED) fires in both runs.
4. Q10 OOD fires either OFF_TOPIC or OUT_OF_DISTRIBUTION in both runs.
5. Missed-by-design queries (Q1, Q2, Q4, Q5) stay clean from the auditor's perspective.

If any future change to the corpus, retriever, or auditor breaks one of these, the post is wrong before it ships. The full criteria are in `SPEC.md`.

---

## What you take away from this if you ship RAG

1. **Distinguish retrieval pathology from content pathology.** They need different graders. `retrieval-auditor` is for the first.
2. **Run a controlled corpus experiment against your own retriever.** Whatever class of pathology lives in your stack will surface differently than mine. The cost is one weekend.
3. **The most valuable finding may not be a poison-driven one.** Q6 was a natural retrieval flaw the auditor caught with zero engineering input. Look at what fires on your *unmodified* corpus before you start engineering stress tests.
4. **Carpet-firing flags are a profile problem, not a retrieval problem.** If you wire a TF-IDF-calibrated grader on top of a dense retriever, every query will look broken. Use the matching profile.

---

## If your RAG pipeline matters to revenue

I run a paid 48-hour audit (Bell Tuning Rapid Audit, $2,500) that does the contrived-corpus experiment plus the natural-pathology scan plus a content-correctness pass against your live retriever. The deliverable is the same shape as this artifact — three buckets, honest about what was caught and what wasn't, with concrete next-step recommendations. [contrarianai-landing.onrender.com](https://contrarianai-landing.onrender.com).

If you just want the tool, it's MIT-licensed in [`tools/retrieval-auditor/`](../../) and the scenario in this directory is the recommended starting point.

---

## Reproducibility footnote

- Random seed: 42 (hardcoded in `run.py`)
- All deps pinned in `requirements.txt`
- Corpus is rebuilt from public sources by `build_corpus.py`; the version-of-record for an earlier ground-truth alignment freeze is the commit that contains this README
- Auditor commit shipped with this artifact is the one that introduces the `dense` profile; reverting to an earlier commit will reproduce the OFF_TOPIC carpet-firing artifact and is itself a useful exercise

If your numbers don't match within last-decimal-place noise, that's a real bug — file an issue.

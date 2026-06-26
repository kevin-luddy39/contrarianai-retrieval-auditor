# I ran retrieval-auditor against 50 cardiology USMLE questions. 66% fired retrieval pathology flags.

A vertical-specific follow-up to the [LangChain RAG quickstart teardown](../langchain-quickstart-teardown/) and the [contrived-RAG pathology scenarios](../contrived-rag-scenarios/). Same instrument, same methodology, applied to cardiology — the medical sub-domain where wrong AI retrieval has the highest stakes.

## TL;DR

I assembled a 1.03M-character corpus from 40 Wikipedia cardiology pages — heart failure, MI, AFib, valve disease, ECG interpretation, antiarrhythmics, anticoagulation, the usual suspects — and ran 50 cardiology USMLE-style clinical vignettes against it via sentence-transformers/MiniLM embeddings at top-K=5. I piped each retrieval through `retrieval-auditor --profile dense` for pathology analysis.

Aggregate across the 50 queries:

| Pathology | Fire rate | Threshold |
|-----------|-----------|-----------|
| SCORE_MISCALIBRATED | **48%** (24/50) | scoreCalibrationR < 0.25 |
| OFF_TOPIC | **34%** (17/50) | mean alignment < 0.10 (dense profile) |
| RANK_INVERSION | **32%** (16/50) | rankQualityR < -0.15 |
| OUT_OF_DISTRIBUTION | **6%** (3/50) | mean alignment < 0.05 (dense profile) |

| Health distribution | Value |
|----|----|
| Min | 0.024 |
| 25th percentile | 0.337 |
| Median | **0.506** |
| 75th percentile | 0.638 |
| Max | 1.000 |

**Only 34% of queries retrieved cleanly** (zero pathology flags). For comparison: the contrived-RAG-scenarios artifact's clean baselines (queries engineered to retrieve well) hit health 0.93+. Cardiac vignettes against the same auditor in the same profile sit in the 0.30-0.65 band — meaningfully degraded.

The headline finding is not that one specific cardiac query went wrong. It's that **across an unbiased sample of 50 cardiac vignettes, dense embedding retrieval over a topical corpus produces measurably-broken ranking on a third of queries.** Precision@K never sees this; the auditor does.

## Why cardiology

Three reasons:

1. **Clear right/wrong answers.** Cardiology has exact diagnostic criteria (STEMI = ST elevation ≥1mm in 2 contiguous leads), exact drug-dosing windows, exact procedural indications. Retrieval errors are auditable in a way that, say, "general health advice" isn't.
2. **Adjacent-condition trap-rich.** Pericarditis vs MI, NSTEMI vs unstable angina, AFib vs atrial flutter, HFpEF vs HFrEF. These pairs share vocabulary while being clinically distinct. Embedding-similarity-confused-with-clinical-correctness is structurally common — and dangerous.
3. **High-stakes failure mode.** A retrieval system that confidently surfaces the wrong cardiac chunk feeds bad context to the LLM, which generates a plausible-but-wrong answer. In ambient documentation or clinical decision support, that propagates downstream.

## Methodology

```bash
git clone https://github.com/kevin-luddy39/contrarianAI
cd contrarianAI/tools/retrieval-auditor/examples/cardiac-rag-teardown
pip install -r ../contrived-rag-scenarios/requirements.txt
curl -L https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json -o /tmp/mirage.json
python3 build_corpus.py        # fetches 40 cardiology Wikipedia pages → corpus/base.txt
# (queries_sampled_50.json was pre-built by filtering MIRAGE benchmark.json
# for strict-cardiac MedQA-USMLE questions; see filter logic in commit)
python3 run.py                  # → results/cardiac.json
```

Components:
- **Embedder**: sentence-transformers/all-MiniLM-L6-v2 (matches the LangChain teardown for direct comparability)
- **Vector store**: chromadb in-memory, cosine distance
- **Chunking**: 800 chars, 150 overlap (~1,500 chunks total from 1.03M-char corpus)
- **Top-K**: 5
- **Auditor**: `retrieval-auditor/cli.js --profile dense`
- **Queries**: 50 strict-cardiac USMLE vignettes from MIRAGE benchmark / MedQA-USMLE subset, sampled from 258 strict-cardiac matches in the median-length band

## Five most striking findings

The full per-query JSON ships in `results/cardiac.json`. Five worth reading in detail:

1. **70yo obese man, acute pulmonary edema** (medqa/0380) — health 0.10, all 3 mechanism flags. Top chunk was less aligned than chunk 4 by a factor of 9× on TF-IDF measurement.
2. **69yo hypertensive with abdominal pain + foot bruising** (medqa/0448) — classic vascular emergency presentation. Health 0.41, all 3 flags.
3. **18yo woman, palpitations + lightheadedness** (medqa/0757) — health 0.02, the worst score in the run. SVT/POTS-spectrum query that retrieved generic chunks.
4. **5yo immigrant, post-strep carditis** (medqa/0730) — health 0.08. The retriever pulled vague chunks; rheumatic heart disease specifics were absent.
5. **65yo with stroke symptoms, likely cardioembolic** (medqa/1079) — health 0.29. Cross-system query (neuro presentation, cardiac source) where retrieval missed the cardiac angle entirely.

## What this means if you ship a clinical RAG

Five takeaways:

1. **Distinguish retrieval pathology from content pathology.** The auditor catches retrieval-mechanism failures (rank inversion, score miscalibration, low alignment), not factual errors. Pair with a clinical-correctness grader.
2. **Run a controlled audit against your own retriever and corpus.** The pathology distribution will look different from this artifact. The cost is one weekend.
3. **Stratify by sub-specialty.** "Medical RAG" averaged across all medicine masks failures concentrated in specific clinical areas. Cardiology was 66% pathology-firing in this run. Endocrinology, oncology, ID, neurology may be different.
4. **The corpus matters more than the embedder.** Wikipedia cardiology pages are dense, well-cited, generally accurate — and still retrieved poorly. A more diffuse corpus (StatPearls, PubMed abstracts, AHA/ACC guidelines mixed) would likely produce different pathology profiles.
5. **Median health 0.506 is not a passing grade.** It's the kind of number that explains why clinical AI feels off without quite failing.

## Limits + caveats

- **n=50 is small.** Conclusions are directional, not industrial-strength. The same harness applied to 500 queries would give defensible per-pathology confidence intervals.
- **Wikipedia is not the production corpus** any clinical RAG would use. A real clinical RAG runs against StatPearls, UpToDate, ACC/AHA guidelines, Braunwald's, and the institution's own documents. The retrieval pathologies we observed are likely *worse* on more diffuse corpora, not better.
- **MedQA queries are USMLE-style clinical vignettes** — long-form, multiple findings, clinical reasoning. They're not the typical short-form query a clinician would type into a chatbot. The auditor handles both, but absolute alignment magnitudes will differ.
- **The auditor's TF-IDF grader is not a clinical reasoner.** It measures token-overlap alignment. Two chunks that share clinical vocabulary will score similarly even if one is correct and the other is wrong. This is the same caveat as the contrived-RAG-scenarios artifact: retrieval-mechanism pathology, not factual pathology.

## If your clinical AI matters to patient outcomes

Bell Tuning Rapid Audit ($2,500, 48-hour turnaround) runs this same pipeline against your retriever, your corpus, and your queries. Deliverable: 8-12 page PDF with per-query bell curves, flagged pathologies by clinical sub-domain, prioritized fix list. Cardiology, endocrinology, oncology — sub-specialty stratification is included. [contrarianai-landing.onrender.com/bell-tuning-rapid-audit.html](https://contrarianai-landing.onrender.com/bell-tuning-rapid-audit.html).

The instrument itself is MIT in [`tools/retrieval-auditor/`](../../). The cardiac scenario in this directory is the recommended template for vertical-specific stress tests.

## Reproducibility

- Random seed: 42 (hardcoded in `run.py`)
- All deps pinned via `../contrived-rag-scenarios/requirements.txt`
- Corpus is rebuilt from public Wikipedia by `build_corpus.py`
- MIRAGE benchmark pulled directly from upstream GitHub at run time
- Auditor profile: `dense` (calibrated for MiniLM-style embedders; see `tools/retrieval-auditor/core/pathologies.js`)

If your numbers don't match within last-decimal-place noise (Wikipedia content drifts; MIRAGE is stable), it's the corpus drifting under you. File an issue with the `Last-Modified` timestamp on Wikipedia pages and we'll snapshot the corpus.

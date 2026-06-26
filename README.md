# retrieval-auditor

> RAG-specific Bell Tuning instrument. Audits retrieval quality per query by measuring the bell curve of chunk-to-query alignment plus rank quality, diversity, score calibration, and pathology flags.

Companion to [`context-inspector`](../context-inspector) (which measures *domain*-aligned context) and [`predictor-corrector`](../predictor-corrector) (which forecasts health trajectories over time). The three tools share output shapes — an audit trace from this tool can be piped directly into the predictor-corrector.

## Why a RAG-specific tool

Retrieval has failure modes that per-chunk domain-alignment cannot see:

| Pathology | What it looks like | Why a domain-alignment-only tool misses it |
|---|---|---|
| **Rank inversion** | top-1 less relevant than top-5 | Aggregate bell curve is healthy; rank order is not tested |
| **Redundancy** | top-K are near-duplicates | All chunks score high individually; bell curve looks clean |
| **Score miscalibration** | retriever's scores don't match alignment | Context-side statistics don't reference retriever's scores |
| **Bimodal retrieval** | two clusters (relevant + incidentally-matching) | Flat moments can hide bimodality in the histogram |
| **Out-of-distribution query** | no chunk aligns well | Requires query-relative scoring |

Retrieval-auditor adds four retrieval-specific signals on top of the bell-curve stats context-inspector already produces.

## What it emits

```js
{
  domain: {
    stats: { mean, stdDev, skewness, kurtosis, histogram, ... },  // CI-compatible
    scores: [per-chunk alignment against query]
  },
  retrieval: {
    rankQualityR,        // Pearson: does rank position predict alignment? (higher = healthier)
    scoreCalibrationR,   // Pearson: do retriever scores agree with alignment?
    diversity,           // 1 − mean pairwise similarity
    redundancyRatio,     // max pairwise similarity
    bimodalSignal,       // two-cluster test on histogram
    similarityMatrix     // pairwise matrix for dashboards
  },
  pathologies: [          // named flags with severity
    { kind: 'RANK_INVERSION', severity: 0.6, description: '...' },
    ...
  ],
  health,                 // [0,1]
  regime                  // healthy | drift | contamination | rot
}
```

## Install

```bash
cd tools/retrieval-auditor
npm install
```

## CLI

```bash
# Audit a single retrieval trace
cat trace.json | node cli.js -

# With structured JSON output
node cli.js trace.json --json

# Trace shape:
#   { "query": "...", "retrieved": [ {"id": "...", "text": "...", "score": 0.9}, ... ] }
```

## MCP server

```json
{
  "mcpServers": {
    "retrieval-auditor": {
      "command": "node",
      "args": ["/path/to/tools/retrieval-auditor/mcp-server.js"]
    }
  }
}
```

Three tools:
- `audit_retrieval` — per-query audit (the primary entry point)
- `audit_corpus` — corpus-wide diversity/length sanity check
- `compare_retrievals` — A/B two retrievers on the same query

## Library

```js
const { auditRetrieval } = require('./core');

const audit = auditRetrieval({
  query: 'how do I treat varroa in winter',
  retrieved: [
    { id: 'd1', text: '...', score: 0.89 },
    { id: 'd2', text: '...', score: 0.77 },
    ...
  ],
});

console.log(audit.health, audit.regime);
console.log(audit.pathologies.map(p => p.kind));
```

## Composition with predictor-corrector

The `audit.domain.stats` shape matches context-inspector, so the predictor-corrector can consume a stream of per-query audits as a trajectory of bell curves — detecting *drift in retrieval health over time* in addition to per-query pathologies.

```js
const { auditRetrieval }   = require('../retrieval-auditor/core');
const { Forecaster }       = require('../predictor-corrector/core');

const fc = new Forecaster({ baseline: /* calibrated on clean retrieval */ });
for (const traceEvent of liveStream) {
  const audit = auditRetrieval(traceEvent);
  fc.observe(audit);  // forecasts the next expected bell curve
}
```

## Pathologies detected

| Kind | Fires when |
|---|---|
| `OFF_TOPIC` | mean query-chunk alignment is low |
| `OUT_OF_DISTRIBUTION` | alignment is near zero — query probably has no support in the corpus |
| `REDUNDANT` | max pairwise chunk similarity exceeds threshold |
| `RANK_INVERSION` | rank-quality correlation is negative |
| `SCORE_MISCALIBRATED` | retriever scores do not agree with independent alignment |
| `BIMODAL` | histogram is bimodal (two clusters of chunks) |
| `LONG_TAIL` | top-1 is good, rest are noise — effective K is 1 |

Each fires with a severity in [0, 1]; downstream code can promote above a threshold into user-visible warnings.

## Known limitations

1. **Lexical-only scoring.** Alignment is cosine on TF-IDF vectors. Semantically relevant chunks that share no tokens with the query score 0 — honest limit of lexical scoring. Hybrid retrievers or embedding-based alignment scoring would address this; both are v1.1 work.
2. **K-sensitivity of bimodal detection.** With small K (≤5), the histogram can't cleanly show two peaks. The experiment whitepaper documents this and shows how the pathology still surfaces as SCORE_MISCALIBRATED.
3. **Thresholds are retriever/corpus sensitive.** The default OFF_TOPIC threshold of 0.30 was calibrated against TF-IDF retrieval on an English beekeeping corpus. Production use should re-calibrate against a clean-retrieval sample from the target pipeline.

## Tests

```bash
npm test
```

## Experiment: The RAG Needle

```bash
npm run experiment
# → writes results to sim/rag-needle/results/
```

Two parts:
- **Pathology Fingerprint** — six controlled scenarios × one query, validates that each pathology flag fires on its designed scenario.
- **Progressive Degradation** — 10-turn session × four queries with progressive contamination, measuring Pearson correlation between auditor health and ground-truth precision@5.

See [`docs/whitepaper-rag-needle.md`](docs/whitepaper-rag-needle.md) for full results.

## License

MIT

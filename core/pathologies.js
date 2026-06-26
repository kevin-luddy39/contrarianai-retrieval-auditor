/**
 * Pathology detection — specific RAG failure modes diagnosed from the
 * combination of bell-curve moments, rank-quality, diversity, and
 * score calibration.
 *
 * Each pathology has a named flag plus a severity in [0, 1]. Downstream
 * code can promote any severity above a threshold into a user-facing
 * warning. Flags are non-exclusive — a single audit can surface
 * multiple pathologies simultaneously.
 */

const PATHOLOGY_DEFS = {
  OFF_TOPIC:           'mean query-chunk alignment is low — retriever is not finding relevant chunks for this query',
  REDUNDANT:           'top-K chunks are near-duplicates — retriever returned the same fact multiple times',
  RANK_INVERSION:      'top-ranked chunks are less aligned than lower-ranked ones — retriever scoring is broken',
  SCORE_MISCALIBRATED: 'retriever-reported scores do not match independent alignment — scoring function drift',
  BIMODAL:             'histogram is bimodal — chunks split into relevant and incidentally-matching clusters',
  LONG_TAIL:           'top-1 is good but remaining chunks are noise — K is effectively 1',
  OUT_OF_DISTRIBUTION: 'no chunk aligns well with the query — likely out-of-corpus query',
};

/**
 * Bimodal detection on a histogram. Looks for two local maxima with a
 * pronounced valley between them. Returns a "dip" in [0, 1].
 */
function detectBimodality(histogram) {
  if (!Array.isArray(histogram) || histogram.length < 5) return 0;
  const k = 2;
  const sm = histogram.map((_, i) => {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - k); j <= Math.min(histogram.length - 1, i + k); j++) {
      s += histogram[j]; n++;
    }
    return s / n;
  });
  const peaks = [];
  for (let i = 1; i < sm.length - 1; i++) {
    if (sm[i] > sm[i - 1] && sm[i] > sm[i + 1]) peaks.push({ i, v: sm[i] });
  }
  if (peaks.length < 2) return 0;
  peaks.sort((a, b) => b.v - a.v);
  const [p1, p2] = peaks;
  const lo = Math.min(p1.i, p2.i);
  const hi = Math.max(p1.i, p2.i);
  let minBetween = Infinity;
  for (let i = lo + 1; i < hi; i++) if (sm[i] < minBetween) minBetween = sm[i];
  if (!isFinite(minBetween)) return 0;
  const smallerPeak = Math.min(p1.v, p2.v);
  if (smallerPeak <= 0) return 0;
  return Math.max(0, Math.min(1, 1 - (minBetween / smallerPeak)));
}

/**
 * Run all pathology checks over computed signals.
 * Returns an array of { kind, severity, description }.
 */
// Profile presets — pick the one matching the retriever family used
// upstream. The auditor's grader is TF-IDF cosine, so dense (embedding)
// retrievers produce intrinsically lower TF-IDF alignment than sparse
// retrievers do, even on healthy retrievals. Without a profile, OFF_TOPIC
// fires constantly on dense-retrieved chunks.
//
// Use "tfidf" (default) when the upstream retriever is TF-IDF / BM25 / sparse.
// Use "dense" when the upstream retriever is sentence-transformers / OpenAI /
// any embedding-model cosine retrieval.
const PROFILES = {
  tfidf: {
    // Calibrated against RAG Needle baseline; clean TF-IDF retrieval on test
    // queries produces mean alignment around 0.38-0.45. OFF_TOPIC at 0.30
    // fires below baseline but not on query-to-query variation.
    offTopic: 0.30,
    redundant: 0.75,
    rankInversion: -0.15,
    scoreMiscalibrated: 0.25,
    bimodal: 0.35,
    longTail: 0.55,
    outOfDistribution: 0.10,
  },
  dense: {
    // Calibrated against dense-embedding retrieval (sentence-transformers
    // MiniLM) over the contrived RAG scenario corpus; clean retrieval
    // produces TF-IDF mean alignment around 0.15-0.25, so absolute alignment
    // thresholds (offTopic, outOfDistribution) drop accordingly. Most relative
    // measures (rankInversion, scoreMiscalibrated, bimodal) stay the same
    // since they are scale-free correlations.
    //
    // redundant drops from 0.75 to 0.60: dense retrieval over a domain-rich
    // corpus produces clean-retrieval redundancyRatio around 0.42-0.54
    // (chunks share domain vocabulary even when distinct). The threshold has
    // to leave room above clean and still discriminate intentional duplicate
    // clusters; 0.60 sits clean+1.5σ-ish.
    offTopic: 0.10,
    redundant: 0.60,
    rankInversion: -0.15,
    scoreMiscalibrated: 0.25,
    bimodal: 0.35,
    longTail: 0.40,
    outOfDistribution: 0.05,
  },
};

function detectPathologies(signals, thresholds = {}, profile = 'tfidf') {
  const base = PROFILES[profile] || PROFILES.tfidf;
  const t = { ...base, ...thresholds };

  const out = [];

  if (signals.meanAlignment < t.outOfDistribution) {
    out.push({
      kind: 'OUT_OF_DISTRIBUTION',
      severity: clip(1 - signals.meanAlignment / t.outOfDistribution),
      description: PATHOLOGY_DEFS.OUT_OF_DISTRIBUTION,
    });
  } else if (signals.meanAlignment < t.offTopic) {
    out.push({
      kind: 'OFF_TOPIC',
      severity: clip(1 - signals.meanAlignment / t.offTopic),
      description: PATHOLOGY_DEFS.OFF_TOPIC,
    });
  }

  if (signals.redundancyRatio > t.redundant) {
    out.push({
      kind: 'REDUNDANT',
      severity: clip((signals.redundancyRatio - t.redundant) / (1 - t.redundant)),
      description: PATHOLOGY_DEFS.REDUNDANT,
    });
  }

  if (signals.rankQualityR != null && signals.rankQualityR < t.rankInversion) {
    out.push({
      kind: 'RANK_INVERSION',
      severity: clip((t.rankInversion - signals.rankQualityR) / (1 + t.rankInversion)),
      description: PATHOLOGY_DEFS.RANK_INVERSION,
    });
  }

  if (
    signals.scoreCalibrationR != null &&
    signals.scoreCalibrationR < t.scoreMiscalibrated
  ) {
    out.push({
      kind: 'SCORE_MISCALIBRATED',
      severity: clip((t.scoreMiscalibrated - signals.scoreCalibrationR) / (1 + t.scoreMiscalibrated)),
      description: PATHOLOGY_DEFS.SCORE_MISCALIBRATED,
    });
  }

  if (signals.bimodalSignal > t.bimodal) {
    out.push({
      kind: 'BIMODAL',
      severity: clip((signals.bimodalSignal - t.bimodal) / (1 - t.bimodal)),
      description: PATHOLOGY_DEFS.BIMODAL,
    });
  }

  // Long-tail: high top-1, low rest
  if (
    signals.alignments?.length >= 3 &&
    signals.alignments[0] > t.longTail &&
    signals.alignments.slice(1).every(v => v < 0.5 * signals.alignments[0])
  ) {
    out.push({
      kind: 'LONG_TAIL',
      severity: 0.5,
      description: PATHOLOGY_DEFS.LONG_TAIL,
    });
  }

  return out;
}

function clip(v) { return Math.max(0, Math.min(1, v)); }

module.exports = { detectPathologies, detectBimodality, PATHOLOGY_DEFS, PROFILES };

/**
 * retrieval-auditor — main entry point.
 *
 *   auditRetrieval({ query, retrieved, options })
 *     Input:   query string + array of { id, text, score? }
 *     Output:  { domain: {stats, scores}, retrieval: { ...signals },
 *                health, regime, pathologies }
 *
 *   auditCorpus({ corpus })
 *     Aggregate statistics about a document corpus — mean document
 *     length, coverage diversity, topic-term concentration. Useful as
 *     a sanity check before the live retrieval audit.
 *
 * Design invariant: the returned `domain.stats` shape matches
 * context-inspector's so the predictor-corrector can consume a stream
 * of per-query audits as a trajectory and apply its forecasting machinery.
 */

const ciCore = require('contrarianai-context-inspector');

const { scoreChunks, scoreCalibration, rankQualityCorrelation } = require('./scoring');
const { diversityMetrics } = require('./diversity');
const { detectPathologies, detectBimodality } = require('./pathologies');
const { scoreFromSignals, regime, DEFAULT_TOLERANCE, TOLERANCE_PROFILES } = require('./health');

/**
 * Audit a single retrieval event.
 *
 * @param {object} input
 * @param {string} input.query      the user query
 * @param {Array<{id?, text, score?}>} input.retrieved  top-K chunks
 * @param {object} [input.options]
 * @param {object} [input.options.tolerance]   override health tolerances
 * @param {object} [input.options.thresholds]  override pathology thresholds
 * @returns {object} audit result
 */
function auditRetrieval({ query, retrieved, options = {} }) {
  if (typeof query !== 'string' || query.length === 0) {
    throw new Error('auditRetrieval: `query` must be a non-empty string');
  }
  if (!Array.isArray(retrieved) || retrieved.length === 0) {
    throw new Error('auditRetrieval: `retrieved` must be a non-empty array');
  }

  // 1. Score each chunk against the query (independent of retriever's own score).
  const alignments = scoreChunks(query, retrieved);

  // 2. Bell-curve stats — reuse context-inspector's exact stat pipeline
  //    so downstream predictor-corrector sees a familiar shape.
  const stats = ciCore.computeStats(alignments);

  // 3. RAG-specific signals
  const rankQualityR = rankQualityCorrelation(alignments);
  const scoreCalibrationR = scoreCalibration(retrieved, alignments);
  const { diversity, redundancyRatio, similarityMatrix } = diversityMetrics(retrieved);
  const bimodalSignal = detectBimodality(stats.histogram);

  const signals = {
    meanAlignment: stats.mean,
    stdDev: stats.stdDev,
    rankQualityR,
    scoreCalibrationR,
    diversity,
    redundancyRatio,
    bimodalSignal,
    alignments,
  };

  const profile = options.profile || 'tfidf';
  const baseTolerance = TOLERANCE_PROFILES[profile] || DEFAULT_TOLERANCE;
  const tolerance = { ...baseTolerance, ...(options.tolerance || {}) };
  const pathologies = detectPathologies(signals, options.thresholds, profile);
  const health = scoreFromSignals(signals, tolerance);
  const regimeLabel = regime(health);

  return {
    query,
    retrievedCount: retrieved.length,
    domain: {
      stats,
      scores: alignments,
    },
    retrieval: {
      rankQualityR,
      scoreCalibrationR,
      diversity,
      redundancyRatio,
      bimodalSignal,
      similarityMatrix,
    },
    pathologies,
    health,
    regime: regimeLabel,
  };
}

/**
 * Audit an entire corpus (not a retrieval event). Useful as a sanity
 * check on the document index — does the corpus have adequate
 * diversity and coverage?
 */
function auditCorpus({ corpus, options = {} }) {
  if (!Array.isArray(corpus) || corpus.length === 0) {
    throw new Error('auditCorpus: `corpus` must be a non-empty array of documents');
  }
  const texts = corpus.map(d => d.text || d);
  const tokenLists = texts.map(t => ciCore.tokenize(t));
  const lengths = tokenLists.map(t => t.length);
  const avgLen = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const matrix = ciCore.chunkSimilarityMatrix(tokenLists);
  let sum = 0, count = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = i + 1; j < matrix[i].length; j++) {
      sum += matrix[i][j]; count++;
    }
  }
  const meanPairwiseSim = count > 0 ? sum / count : 0;
  return {
    docCount: corpus.length,
    avgTokens: avgLen,
    minTokens: Math.min(...lengths),
    maxTokens: Math.max(...lengths),
    meanPairwiseSimilarity: meanPairwiseSim,
    corpusDiversity: 1 - meanPairwiseSim,
  };
}

/**
 * Compare two retrieval results over the same query. Useful for A/B
 * testing retrievers. Returns a side-by-side audit + delta on health
 * score and each retrieval-specific signal.
 */
function compareRetrievals({ query, retrievedA, retrievedB, labelA = 'A', labelB = 'B' }) {
  const a = auditRetrieval({ query, retrieved: retrievedA });
  const b = auditRetrieval({ query, retrieved: retrievedB });
  return {
    query,
    [labelA]: a,
    [labelB]: b,
    delta: {
      health: b.health - a.health,
      meanAlignment: b.domain.stats.mean - a.domain.stats.mean,
      rankQualityR: safeDelta(b.retrieval.rankQualityR, a.retrieval.rankQualityR),
      diversity: b.retrieval.diversity - a.retrieval.diversity,
      redundancyRatio: b.retrieval.redundancyRatio - a.retrieval.redundancyRatio,
    },
  };
}

function safeDelta(b, a) {
  if (b == null || a == null) return null;
  return b - a;
}

module.exports = {
  auditRetrieval,
  auditCorpus,
  compareRetrievals,
  DEFAULT_TOLERANCE,
};

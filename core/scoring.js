/**
 * Query-relevance scoring for retrieval auditing.
 *
 * Given a query and a set of retrieved chunks, compute per-chunk
 * alignment scores that are independent of the retriever's own scoring
 * function. The independence is what lets us detect score calibration
 * drift — if our independent score disagrees with the retriever's
 * score, the retriever is mis-ranking.
 *
 * Uses cosine similarity over TF-IDF vectors. Tokenisation and the
 * cosine primitive are borrowed from context-inspector for consistency.
 */

const ciCore = require('contrarianai-context-inspector');

/**
 * Score one chunk against a query via cosine similarity on tokenised
 * TF vectors. Returns a value in [0, 1].
 */
function scoreQueryRelevance(query, chunk) {
  if (!query || !chunk) return 0;
  const qTokens = ciCore.tokenize(query);
  const cTokens = ciCore.tokenize(chunk);
  if (qTokens.length === 0 || cTokens.length === 0) return 0;
  return ciCore.cosineSimilarity(qTokens, cTokens);
}

/**
 * Score all retrieved chunks against the query. Returns an array of
 * per-chunk scores in the same order as `retrieved`.
 */
function scoreChunks(query, retrieved) {
  return retrieved.map(r => scoreQueryRelevance(query, r.text || r.chunk || ''));
}

/**
 * Pearson correlation between retriever-reported scores and our
 * independent alignment scores. High positive = retriever is
 * well-calibrated. Near zero or negative = retriever scores disagree
 * with alignment, which is a strong signal the retriever is broken.
 * Returns null if retriever scores are missing.
 */
function scoreCalibration(retrieved, alignments) {
  const pairs = retrieved
    .map((r, i) => [r.score, alignments[i]])
    .filter(([s]) => typeof s === 'number' && Number.isFinite(s));
  if (pairs.length < 3) return null;
  const xs = pairs.map(p => p[0]);
  const ys = pairs.map(p => p[1]);
  return ciCore.sampleCorrelation(xs, ys);
}

/**
 * Rank-quality correlation: does the position in the top-K match the
 * independent alignment ranking? Uses Pearson between rank index
 * (0=top) and alignment. Negative = top-ranked chunks are in fact the
 * most aligned (healthy); near zero or positive = rank inversion.
 * We return the *sign-flipped* value so higher = healthier, in [-1, 1].
 */
function rankQualityCorrelation(alignments) {
  if (alignments.length < 3) return null;
  const ranks = alignments.map((_, i) => i);
  const rho = ciCore.sampleCorrelation(ranks, alignments);
  // rho near -1 means alignment decreases with rank (healthy).
  // Flip sign so healthier retrieval has higher rankQualityR.
  return -rho;
}

module.exports = {
  scoreQueryRelevance,
  scoreChunks,
  scoreCalibration,
  rankQualityCorrelation,
};

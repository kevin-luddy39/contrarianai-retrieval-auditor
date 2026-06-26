/**
 * Inter-chunk diversity / redundancy signals.
 *
 * Given the top-K retrieved chunks, compute how similar they are to
 * each other. Healthy retrieval returns complementary chunks that
 * cover different facets of the query. Pathological retrieval returns
 * near-duplicates that express the same fact K times.
 *
 * Uses context-inspector's chunkSimilarityMatrix which computes pairwise
 * cosine similarity over tokenised TF vectors.
 */

const ciCore = require('contrarianai-context-inspector');

/**
 * Returns { diversity, redundancyRatio, similarityMatrix }.
 *
 *   diversity        = 1 − mean off-diagonal pairwise similarity
 *                      (higher = chunks more distinct)
 *   redundancyRatio  = max off-diagonal pairwise similarity
 *                      (higher = at least one near-duplicate pair)
 */
function diversityMetrics(retrieved) {
  const texts = retrieved.map(r => r.text || r.chunk || '');
  if (texts.length < 2) {
    return { diversity: 1, redundancyRatio: 0, similarityMatrix: [[1]] };
  }
  const tokenLists = texts.map(t => ciCore.tokenize(t));
  const n = tokenLists.length;

  // Build pairwise matrix from primitive cosine — context-inspector's
  // chunkSimilarityMatrix returns averages, not the full matrix we need.
  const matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  let sum = 0, count = 0, max = 0;
  for (let i = 0; i < n; i++) {
    matrix[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const v = ciCore.cosineSimilarity(tokenLists[i], tokenLists[j]);
      matrix[i][j] = v;
      matrix[j][i] = v;
      sum += v; count++;
      if (v > max) max = v;
    }
  }
  const meanSim = count > 0 ? sum / count : 0;
  return {
    diversity: 1 - meanSim,
    redundancyRatio: max,
    similarityMatrix: matrix,
  };
}

module.exports = { diversityMetrics };

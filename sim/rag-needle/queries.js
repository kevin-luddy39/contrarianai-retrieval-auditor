/**
 * Test queries with ground-truth relevance annotations.
 *
 * Each query has a known set of relevant doc IDs in the corpus.
 * The runner uses this to compute precision@K and to measure whether
 * auditor health correlates with ground-truth quality over time.
 *
 * Queries are written to share distinctive lexical tokens with their
 * relevant docs, so that a TF-IDF lexical retriever can actually find
 * them — the auditor's job is to evaluate retrieval quality, not to do
 * the retrieval itself.
 */

module.exports = [
  {
    id: 'q1',
    text: 'how do I treat varroa mites with oxalic acid in winter',
    relevantIds: ['c01', 'c02', 'c03', 'c04', 'c05'],
  },
  {
    id: 'q2',
    text: 'what brood pattern should I look for when inspecting the queen',
    relevantIds: ['c06', 'c07', 'c08', 'c09', 'c10'],
  },
  {
    id: 'q3',
    text: 'how do I manage spring swarm cells and splits',
    relevantIds: ['c11', 'c12', 'c13', 'c14', 'c15'],
  },
  {
    id: 'q4',
    text: 'what do I need to winterize my hive stores and cluster',
    relevantIds: ['c16', 'c17', 'c18', 'c19', 'c20'],
  },
];

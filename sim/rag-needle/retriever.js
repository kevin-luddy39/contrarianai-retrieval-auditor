/**
 * Retrievers used by the RAG Needle experiment.
 *
 * createRetriever(index)
 *   Naive TF-IDF cosine retriever. Ranks by query-document cosine,
 *   returns top-K. This is the "clean" retriever used as control.
 *
 * createBiasedRetriever(index, biasFn)
 *   A retriever whose top-K is deliberately degraded by biasFn to
 *   simulate specific pathologies (embedding drift, vector-store
 *   contamination, score miscalibration). Used to stage scenarios.
 *
 * Both retrievers emit the same shape: { id, text, score, metadata }[].
 */

const path = require('path');
const ciCore = require(path.resolve(__dirname, '..', '..', '..', 'context-inspector', 'core.js'));

function createRetriever(initialIndex = []) {
  let index = initialIndex.slice();

  function setIndex(docs) { index = docs.slice(); }
  function addDocs(docs)  { index.push(...docs); }
  function removeByIds(ids) {
    const set = new Set(ids);
    index = index.filter(d => !set.has(d.id));
  }

  function retrieve(query, k = 5) {
    const qTokens = ciCore.tokenize(query);
    const scored = index.map(doc => {
      const dTokens = ciCore.tokenize(doc.text);
      const score = ciCore.cosineSimilarity(qTokens, dTokens);
      return { id: doc.id, text: doc.text, score, metadata: doc.metadata || {} };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  return { setIndex, addDocs, removeByIds, retrieve, getIndex: () => index };
}

/**
 * Biased retriever. Runs the clean retrieve then applies biasFn to the
 * top-K before returning. biasFn receives { topK, index, query } and
 * returns the modified top-K array. Used to inject specific
 * pathological retrieval patterns for the experiment.
 */
function createBiasedRetriever(initialIndex, biasFn) {
  const base = createRetriever(initialIndex);
  function retrieve(query, k = 5) {
    const topK = base.retrieve(query, k);
    return biasFn({ topK, index: base.getIndex(), query, k });
  }
  return {
    setIndex: base.setIndex,
    addDocs: base.addDocs,
    removeByIds: base.removeByIds,
    getIndex: base.getIndex,
    retrieve,
  };
}

// ── Bias function library ────────────────────────────────────────
// Each bias function takes the clean top-K and returns a degraded
// top-K to simulate a specific retrieval pathology.

/**
 * Swap N positions of the top-K with random poison docs from the
 * index. Simulates embedding drift — the retriever now believes
 * unrelated docs are relevant.
 */
function swapInPoison(poisonDocs, nSwap) {
  let i = 0;
  return ({ topK, k }) => {
    const out = topK.slice();
    const toSwap = Math.min(nSwap, poisonDocs.length, k);
    for (let s = 0; s < toSwap; s++) {
      const idx = k - 1 - s;  // replace from the bottom upward
      const poison = poisonDocs[(i + s) % poisonDocs.length];
      out[idx] = {
        id: poison.id,
        text: poison.text,
        score: topK[0].score * 0.85,   // poison reports a near-top score
        metadata: poison.metadata || {},
      };
    }
    i++;
    return out;
  };
}

/**
 * Replace top-K entirely with K near-duplicates of the top-1.
 * Simulates a retriever stuck on one document.
 */
function injectRedundancy() {
  return ({ topK, k }) => {
    if (topK.length === 0) return topK;
    const top = topK[0];
    const tweaks = [
      top.text,
      top.text + ' (Additional note.)',
      top.text + ' See also.',
      top.text + ' Details below.',
      top.text + ' For further reading.',
    ];
    return tweaks.slice(0, k).map((text, idx) => ({
      id: `${top.id}_dup${idx}`,
      text,
      score: top.score - idx * 0.001,
    }));
  };
}

/**
 * Reverse the rank order. Simulates a retriever whose scoring
 * function is inverted — least relevant ranked highest.
 */
function reverseRanking() {
  return ({ topK }) => topK.slice().reverse();
}

/**
 * Randomise the reported scores while keeping chunk order unchanged.
 * Simulates score calibration drift — ranking is still correct but
 * the numerical scores are no longer meaningful.
 */
function miscalibrateScores(seed = 42) {
  const rnd = mulberry32(seed);
  return ({ topK }) => topK.map(r => ({ ...r, score: rnd() }));
}

/**
 * Bimodal: pad top-K with highly-scored poison that is semantically
 * disjoint from the real relevant chunks. Produces two visible
 * clusters in the alignment histogram.
 */
function bimodalInject(poisonDocs) {
  let i = 0;
  return ({ topK, k }) => {
    const half = Math.floor(k / 2);
    const out = topK.slice(0, k - half);
    for (let s = 0; s < half; s++) {
      const poison = poisonDocs[(i + s) % poisonDocs.length];
      out.push({
        id: poison.id,
        text: poison.text,
        score: topK[0].score * 0.90,  // poison claims near-top score
        metadata: poison.metadata || {},
      });
    }
    i++;
    return out;
  };
}

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

module.exports = {
  createRetriever,
  createBiasedRetriever,
  swapInPoison,
  injectRedundancy,
  reverseRanking,
  miscalibrateScores,
  bimodalInject,
};

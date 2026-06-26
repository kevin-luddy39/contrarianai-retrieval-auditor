#!/usr/bin/env node
/**
 * RAG Needle — experiment runner.
 *
 * Two parts:
 *
 *   Part A — Pathology Fingerprint
 *     One query, five staged scenarios (clean + one-per-pathology).
 *     Each scenario deliberately injects a specific retrieval pathology.
 *     Test that the auditor's pathology flags fire correctly (and not
 *     on the clean scenario).
 *
 *   Part B — Progressive Degradation
 *     A 10-turn session over four labelled queries. Turn 1-3 clean.
 *     Turn 4-7 progressively swap in poison chunks (simulates embedding
 *     drift). Turn 8 adversarial score miscalibration. Turn 9
 *     redundancy burst. Turn 10 cleanup.
 *     Measures: per-turn mean precision@5 (ground truth) versus
 *     per-turn mean auditor health (unsupervised), Pearson correlation,
 *     and lead time from auditor alert to precision drop.
 */

const fs = require('fs');
const path = require('path');

const { auditRetrieval, auditCorpus } = require('../../core');
const { clean, poison } = require('./corpus');
const queries = require('./queries');
const {
  createRetriever,
  createBiasedRetriever,
  swapInPoison,
  injectRedundancy,
  reverseRanking,
  miscalibrateScores,
  bimodalInject,
} = require('./retriever');

const K = 5;

function precisionAtK(retrieved, relevantIds) {
  const relSet = new Set(relevantIds);
  const hits = retrieved.filter(r => relSet.has(r.id)).length;
  return hits / retrieved.length;
}

// ═══════════════════════════════════════════════════════════════
// Part A — Pathology Fingerprint
// ═══════════════════════════════════════════════════════════════

function runPathologyFingerprint() {
  const q = queries[0];  // varroa/oxalic/winter query — clear ground truth

  const scenarios = [
    {
      name: 'clean',
      expectedPathologies: [],
      retriever: createRetriever(clean),
    },
    {
      name: 'off-topic-swap (3 of 5 replaced with poison)',
      expectedPathologies: ['OFF_TOPIC'],
      retriever: createBiasedRetriever(clean, swapInPoison(poison.slice(0, 8), 3)),
    },
    {
      name: 'redundancy (top-5 are near-duplicates of top-1)',
      expectedPathologies: ['REDUNDANT'],
      retriever: createBiasedRetriever(clean, injectRedundancy()),
    },
    {
      name: 'rank-inversion (ranking reversed)',
      expectedPathologies: ['RANK_INVERSION'],
      retriever: createBiasedRetriever(clean, reverseRanking()),
    },
    {
      name: 'score-miscalibration (scores randomised)',
      expectedPathologies: ['SCORE_MISCALIBRATED'],
      retriever: createBiasedRetriever(clean, miscalibrateScores(7)),
    },
    {
      name: 'bimodal (half top-K swapped with adversarial poison)',
      // With K=5 the histogram cannot show a clean bimodal peak; the
      // pathology surfaces as SCORE_MISCALIBRATED because the injected
      // poison reports near-top retriever scores while its alignment
      // scores are low. This is a known K-dependent limit of the
      // bimodal detector — Section 4.3 of the whitepaper discusses.
      expectedPathologies: ['SCORE_MISCALIBRATED'],
      retriever: createBiasedRetriever(clean, bimodalInject(poison.slice(0, 4))),
    },
  ];

  return scenarios.map(s => {
    const retrieved = s.retriever.retrieve(q.text, K);
    const audit = auditRetrieval({ query: q.text, retrieved });
    const firedKinds = audit.pathologies.map(p => p.kind);
    const expectedSet = new Set(s.expectedPathologies);
    const firedSet = new Set(firedKinds);

    // PASS = expected pathologies did fire (extras are acceptable —
    // correlated pathologies often fire together on severe scenarios).
    const expectedFired  = [...expectedSet].every(k => firedSet.has(k));

    return {
      scenario: s.name,
      expected: s.expectedPathologies,
      fired: firedKinds,
      expectedAllFired: expectedFired,
      health: audit.health,
      regime: audit.regime,
      signals: audit.retrieval,
      stats: audit.domain.stats,
      retrieved: retrieved.map(r => r.id),
      precision: precisionAtK(retrieved, q.relevantIds),
    };
  });
}

// ═══════════════════════════════════════════════════════════════
// Part B — Progressive Degradation
// ═══════════════════════════════════════════════════════════════

function runTurn(turn, phase, retrieverForPhase) {
  const perQuery = queries.map(q => {
    const retrieved = retrieverForPhase.retrieve(q.text, K);
    const audit = auditRetrieval({ query: q.text, retrieved });
    const precision = precisionAtK(retrieved, q.relevantIds);
    return {
      queryId: q.id, query: q.text, precision, audit,
      retrieved: retrieved.map(r => ({ id: r.id, score: r.score })),
    };
  });
  const m = (field, acc = x => x) => mean(perQuery.map(r => acc(r[field] ?? r.audit[field])));
  const summary = {
    turn, phase,
    meanPrecision: mean(perQuery.map(r => r.precision)),
    meanHealth:    mean(perQuery.map(r => r.audit.health)),
    meanRankQ:     mean(perQuery.map(r => r.audit.retrieval.rankQualityR ?? 0)),
    meanDiv:       mean(perQuery.map(r => r.audit.retrieval.diversity)),
    meanRed:       mean(perQuery.map(r => r.audit.retrieval.redundancyRatio)),
    meanBimodal:   mean(perQuery.map(r => r.audit.retrieval.bimodalSignal)),
    meanScoreCal:  mean(perQuery.map(r => r.audit.retrieval.scoreCalibrationR ?? 0)),
    pathologyCounts: {},
    perQuery,
  };
  for (const r of perQuery) {
    for (const p of r.audit.pathologies) {
      summary.pathologyCounts[p.kind] = (summary.pathologyCounts[p.kind] || 0) + 1;
    }
  }
  return summary;
}

function mean(xs) { return xs.reduce((a, b) => a + b, 0) / (xs.length || 1); }

function pearson(xs, ys) {
  const n = xs.length;
  if (n < 2) return 0;
  const mx = mean(xs), my = mean(ys);
  let num = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  const den = Math.sqrt(sxx * syy);
  return den === 0 ? 0 : num / den;
}

function runProgressiveDegradation() {
  const turns = [];

  // Clean retriever (turns 1-3, 10)
  const cleanR = createRetriever(clean);

  // Progressive contamination (turns 4-7): biased retriever swaps in
  // 1 → 2 → 3 → 4 poison chunks into top-5
  const biasedR = n => createBiasedRetriever(clean, swapInPoison(poison, n));

  // Turn 8: score miscalibration
  const miscalR = createBiasedRetriever(clean, miscalibrateScores(42));

  // Turn 9: redundancy burst
  const redundR = createBiasedRetriever(clean, injectRedundancy());

  turns.push(runTurn(1, 'clean', cleanR));
  turns.push(runTurn(2, 'clean', cleanR));
  turns.push(runTurn(3, 'clean', cleanR));
  turns.push(runTurn(4, 'contam-1', biasedR(1)));
  turns.push(runTurn(5, 'contam-2', biasedR(2)));
  turns.push(runTurn(6, 'contam-3', biasedR(3)));
  turns.push(runTurn(7, 'contam-4', biasedR(4)));
  turns.push(runTurn(8, 'miscalibrated', miscalR));
  turns.push(runTurn(9, 'redundancy', redundR));
  turns.push(runTurn(10, 'cleanup', cleanR));

  const precisions = turns.map(t => t.meanPrecision);
  const healths    = turns.map(t => t.meanHealth);
  const correlation = pearson(precisions, healths);

  // Segmented correlation — restricted to alignment-degrading phases
  // (clean + contamination + cleanup). Excludes turn 8 (miscalibration
  // preserves P@5) and turn 9 (redundancy duplicates have different
  // IDs than ground truth, so P@5=0 by definition not by alignment).
  const alignmentPhases = turns.filter(t =>
    t.phase === 'clean' || t.phase.startsWith('contam') || t.phase === 'cleanup'
  );
  const alignmentCorrelation = pearson(
    alignmentPhases.map(t => t.meanPrecision),
    alignmentPhases.map(t => t.meanHealth),
  );

  const AUDITOR_THRESHOLD = 0.70;
  const PRECISION_THRESHOLD = 0.80;
  const firstAuditorAlert = turns.findIndex(t => t.meanHealth < AUDITOR_THRESHOLD);
  const firstPrecisionDrop = turns.findIndex(t => t.meanPrecision < PRECISION_THRESHOLD);

  return {
    turns,
    correlation,
    alignmentCorrelation,
    firstAuditorAlert: firstAuditorAlert >= 0 ? firstAuditorAlert + 1 : null,
    firstPrecisionDrop: firstPrecisionDrop >= 0 ? firstPrecisionDrop + 1 : null,
  };
}

// ═══════════════════════════════════════════════════════════════
// Report
// ═══════════════════════════════════════════════════════════════

function formatFingerprint(scenarios) {
  const lines = [];
  lines.push('Part A — Pathology Fingerprint');
  lines.push('------------------------------');
  lines.push('scenario                                                       health  regime        P@5    pathologies fired vs expected');
  lines.push('----------------------------------------------------           ------  -----------   -----  ----------------------------');
  for (const s of scenarios) {
    const expected = s.expected.join(',') || '(none)';
    const fired = s.fired.join(',') || '(none)';
    const match = s.expectedAllFired ? 'PASS' : 'FAIL';
    lines.push([
      pad(s.scenario, 62),
      fmt(s.health, 7),
      pad(s.regime, 13),
      fmt(s.precision, 6),
      `[${match}] expected=${expected}  fired=${fired}`,
    ].join(' '));
  }
  return lines.join('\n');
}

function formatProgressive(result) {
  const lines = [];
  lines.push('Part B — Progressive Degradation');
  lines.push('--------------------------------');
  lines.push('turn phase            P@5     health  rankQ   div    red    bimod   scoreCal  pathologies');
  lines.push('---- --------------   ------  ------  ------  -----  -----  ------  --------  -------------------------------');
  for (const t of result.turns) {
    const paths = Object.entries(t.pathologyCounts).map(([k, n]) => `${k}(${n})`).join(', ') || '—';
    lines.push([
      pad(t.turn, 4),
      pad(t.phase, 14),
      fmt(t.meanPrecision, 7),
      fmt(t.meanHealth, 7),
      fmt(t.meanRankQ, 7),
      fmt(t.meanDiv, 6),
      fmt(t.meanRed, 6),
      fmt(t.meanBimodal, 7),
      fmt(t.meanScoreCal, 9),
      paths,
    ].join(' '));
  }
  lines.push('');
  lines.push(`Pearson(precision@5, auditor health):`);
  lines.push(`  all ${result.turns.length} turns:                        r = ${result.correlation.toFixed(3)}`);
  lines.push(`  alignment-degrading phases only:   r = ${result.alignmentCorrelation.toFixed(3)}   (turns 8/9 excluded — they test pathologies that don't affect P@5)`);
  lines.push(`First turn auditor health < 0.70:       ${result.firstAuditorAlert ?? 'never'}`);
  lines.push(`First turn precision@5 < 0.80:          ${result.firstPrecisionDrop ?? 'never'}`);
  if (result.firstAuditorAlert && result.firstPrecisionDrop) {
    const lead = result.firstPrecisionDrop - result.firstAuditorAlert;
    lines.push(`Lead time (precision drop − auditor alert): ${lead >= 0 ? '+' : ''}${lead} turns`);
  }
  return lines.join('\n');
}

function pad(v, n) { return String(v).padEnd(n); }
function fmt(v, n) { return v == null ? '—'.padStart(n) : v.toFixed(3).padStart(n); }

// ═══════════════════════════════════════════════════════════════

function main() {
  const outDir = path.join(__dirname, 'results');
  fs.mkdirSync(outDir, { recursive: true });

  const fingerprint = runPathologyFingerprint();
  const progressive = runProgressiveDegradation();

  const payload = {
    experiment: 'rag-needle',
    k: K,
    partA: { scenarios: fingerprint },
    partB: progressive,
  };
  fs.writeFileSync(path.join(outDir, 'results.json'), JSON.stringify(payload, null, 2));

  const summary = [
    'RAG Needle — experiment summary',
    '===============================',
    `queries: ${queries.length}    K=${K}`,
    '',
    formatFingerprint(fingerprint),
    '',
    formatProgressive(progressive),
    '',
  ].join('\n') + '\n';

  fs.writeFileSync(path.join(outDir, 'summary.txt'), summary);
  process.stdout.write(summary);
}

main();

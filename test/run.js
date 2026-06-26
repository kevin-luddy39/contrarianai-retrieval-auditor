#!/usr/bin/env node
/**
 * retrieval-auditor test suite.
 */

const assert = require('assert');
const { auditRetrieval, compareRetrievals } = require('../core');

let pass = 0, fail = 0;

function test(name, fn) {
  try { fn(); pass++; console.log(`  ok  ${name}`); }
  catch (e) { fail++; console.log(`  FAIL ${name}\n       ${e.stack || e.message}`); }
}

test('auditRetrieval returns CI-compatible stats', () => {
  const result = auditRetrieval({
    query: 'varroa mite treatment',
    retrieved: [
      { id: '1', text: 'Varroa mite treatment with oxalic acid in winter.' },
      { id: '2', text: 'Varroa monitoring via sugar roll counts.' },
      { id: '3', text: 'Treatment thresholds for varroa based on mite count.' },
    ],
  });
  assert(result.domain && result.domain.stats, 'has domain.stats');
  assert(typeof result.domain.stats.mean === 'number');
  assert(typeof result.domain.stats.stdDev === 'number');
  assert(Array.isArray(result.domain.stats.histogram));
  assert(result.domain.stats.histogram.length === 20);
  assert(Array.isArray(result.domain.scores));
  assert(result.domain.scores.length === 3);
});

test('clean retrieval produces no pathologies', () => {
  const result = auditRetrieval({
    query: 'varroa mite oxalic acid treatment',
    retrieved: [
      { id: '1', text: 'Oxalic acid vapour is the preferred winter treatment for varroa mites.', score: 0.9 },
      { id: '2', text: 'Dribble method uses 3.5% oxalic acid syrup between frames for varroa.', score: 0.85 },
      { id: '3', text: 'Three-round oxalic vapour schedule targets varroa through brood cycles.', score: 0.78 },
    ],
  });
  assert(result.pathologies.length === 0,
    `expected no pathologies, got ${result.pathologies.map(p => p.kind).join(',')}`);
});

test('redundancy is detected', () => {
  const chunk = 'Varroa oxalic acid treatment vapour winter brood.';
  const result = auditRetrieval({
    query: 'varroa oxalic',
    retrieved: [
      { id: '1', text: chunk, score: 0.9 },
      { id: '2', text: chunk + ' See also.', score: 0.89 },
      { id: '3', text: chunk + ' Details below.', score: 0.88 },
      { id: '4', text: chunk + ' For more.', score: 0.87 },
    ],
  });
  const kinds = result.pathologies.map(p => p.kind);
  assert(kinds.includes('REDUNDANT'), `expected REDUNDANT, got ${kinds.join(',')}`);
});

test('rank inversion is detected', () => {
  const result = auditRetrieval({
    query: 'varroa oxalic acid treatment',
    retrieved: [
      { id: '5', text: 'Queen laying pattern inspection.', score: 0.40 },
      { id: '4', text: 'Honey harvest timing and extraction.', score: 0.30 },
      { id: '3', text: 'Treatment three-round schedule for varroa.', score: 0.60 },
      { id: '2', text: 'Dribble method oxalic acid application.', score: 0.80 },
      { id: '1', text: 'Oxalic acid vapour varroa winter treatment.', score: 0.90 },
    ],
  });
  const kinds = result.pathologies.map(p => p.kind);
  assert(kinds.includes('RANK_INVERSION'), `expected RANK_INVERSION, got ${kinds.join(',')}`);
});

test('compareRetrievals returns deltas', () => {
  const result = compareRetrievals({
    query: 'varroa',
    retrievedA: [{ id: '1', text: 'varroa mite treatment oxalic' }],
    retrievedB: [{ id: '1', text: 'basketball invented 1891 Naismith' }],
  });
  assert(result.delta.health <= 0, 'B is worse');
  assert(result.delta.meanAlignment <= 0);
});

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);

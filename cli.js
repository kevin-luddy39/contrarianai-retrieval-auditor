#!/usr/bin/env node
/**
 * retrieval-auditor CLI.
 *
 * Input JSON shape:
 *   {
 *     "query": "...",
 *     "retrieved": [
 *       { "id": "...", "text": "...", "score": 0.89 },
 *       ...
 *     ]
 *   }
 *
 * Or an array of such objects (one audit per element).
 *
 * Usage:
 *   retrieval-auditor trace.json
 *   cat trace.json | retrieval-auditor -
 */

const fs = require('fs');
const { auditRetrieval } = require('./core');

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith('--')) { args[key] = next; i++; }
      else args[key] = true;
    } else {
      args._.push(a);
    }
  }
  return args;
}

function readStdin() { return fs.readFileSync(0, 'utf8'); }

function loadInput(args) {
  if (args._[0] && args._[0] !== '-') {
    return JSON.parse(fs.readFileSync(args._[0], 'utf8'));
  }
  return JSON.parse(readStdin());
}

function summarizeOne(audit) {
  const s = audit.domain.stats;
  const r = audit.retrieval;
  const path = audit.pathologies.map(p => p.kind).join(', ') || '—';
  return [
    `query:           ${audit.query}`,
    `retrieved:       ${audit.retrievedCount} chunks`,
    `mean alignment:  ${s.mean.toFixed(4)}`,
    `stdDev:          ${s.stdDev.toFixed(4)}`,
    `skewness:        ${s.skewness.toFixed(4)}   kurtosis: ${s.kurtosis.toFixed(4)}`,
    `rank quality R:  ${fmt(r.rankQualityR)}`,
    `diversity:       ${r.diversity.toFixed(4)}`,
    `redundancy ratio:${r.redundancyRatio.toFixed(4)}`,
    `score calib R:   ${fmt(r.scoreCalibrationR)}`,
    `bimodal signal:  ${r.bimodalSignal.toFixed(4)}`,
    `pathologies:     ${path}`,
    `health:          ${audit.health.toFixed(3)}   regime: ${audit.regime}`,
  ].join('\n');
}

function fmt(v) { return v == null ? '—' : v.toFixed(4); }

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || args.h) {
    process.stdout.write(
      'retrieval-auditor <trace.json|->   [--json] [--profile tfidf|dense]\n' +
      '  Reads one retrieval trace (or array of traces) and prints audit report.\n' +
      '  With --json, emits structured JSON instead of the text summary.\n' +
      '  --profile selects pathology-threshold preset; default is "tfidf".\n' +
      '  Use "dense" when the upstream retriever is sentence-transformers /\n' +
      '  OpenAI / any embedding-model cosine retrieval.\n'
    );
    return;
  }

  const profile = args.profile || 'tfidf';
  if (profile !== 'tfidf' && profile !== 'dense') {
    process.stderr.write(`retrieval-auditor: unknown profile '${profile}' (expected tfidf|dense)\n`);
    process.exit(2);
  }

  const input = loadInput(args);
  const traces = Array.isArray(input) ? input : [input];
  const audits = traces.map(t => auditRetrieval({ ...t, options: { ...(t.options || {}), profile } }));

  if (args.json) {
    process.stdout.write(JSON.stringify(audits.length === 1 ? audits[0] : audits, null, 2) + '\n');
    return;
  }

  const sections = audits.map((a, i) =>
    (audits.length > 1 ? `── audit #${i + 1} ──\n` : '') + summarizeOne(a)
  );
  process.stdout.write(sections.join('\n\n') + '\n');
}

try { main(); }
catch (err) {
  process.stderr.write(`retrieval-auditor: ${err.message}\n`);
  process.exit(1);
}

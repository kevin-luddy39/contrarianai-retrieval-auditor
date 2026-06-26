#!/usr/bin/env node
/**
 * retrieval-auditor — MCP Server
 *
 * Three tools:
 *   audit_retrieval      — per-query audit; primary entry point
 *   audit_corpus         — corpus-wide diversity check
 *   compare_retrievals   — A/B two retrievers on the same query
 */

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { z } = require('zod');

const { auditRetrieval, auditCorpus, compareRetrievals } = require('./core');

const server = new McpServer({
  name: 'retrieval-auditor',
  version: '0.1.0',
});

const chunkSchema = z.object({
  id: z.string().optional(),
  text: z.string(),
  score: z.number().optional(),
  metadata: z.any().optional(),
});

server.tool(
  'audit_retrieval',
  'Audit a single retrieval event. Returns per-chunk alignment scores, bell-curve stats, rank quality, diversity, score calibration, bimodality signal, pathology flags, health score, and regime label.',
  {
    query: z.string().describe('The user query'),
    retrieved: z.array(chunkSchema).describe('Top-K retrieved chunks, ordered by retriever score (best first)'),
  },
  async ({ query, retrieved }) => {
    const audit = auditRetrieval({ query, retrieved });
    return {
      content: [{ type: 'text', text: JSON.stringify(audit, null, 2) }],
    };
  }
);

server.tool(
  'audit_corpus',
  'Audit an entire document corpus for diversity and length distribution. Useful as a sanity check on the RAG index.',
  {
    corpus: z.array(chunkSchema).describe('All documents in the corpus'),
  },
  async ({ corpus }) => {
    const report = auditCorpus({ corpus });
    return {
      content: [{ type: 'text', text: JSON.stringify(report, null, 2) }],
    };
  }
);

server.tool(
  'compare_retrievals',
  'Compare two retrievers on the same query. Returns side-by-side audits and deltas on health and retrieval signals.',
  {
    query: z.string(),
    retrievedA: z.array(chunkSchema),
    retrievedB: z.array(chunkSchema),
    labelA: z.string().optional().default('A'),
    labelB: z.string().optional().default('B'),
  },
  async ({ query, retrievedA, retrievedB, labelA, labelB }) => {
    const report = compareRetrievals({ query, retrievedA, retrievedB, labelA, labelB });
    return {
      content: [{ type: 'text', text: JSON.stringify(report, null, 2) }],
    };
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(err => {
  process.stderr.write(`retrieval-auditor MCP error: ${err.stack || err.message}\n`);
  process.exit(1);
});

# I ran retrieval-auditor against LangChain's RAG quickstart. 5 of 6 queries flagged.

The corpus is Lilian Weng's "LLM Powered Autonomous Agents" — the blog post that the LangChain RAG tutorial uses as its canonical demo. The retriever is the LangChain default (cosine similarity over `all-MiniLM-L6-v2` embeddings, top-5). The queries are six things you'd actually ask of that post: Chain of Thought, ReAct vs Reflexion, memory mechanisms, tool-use decisions, planning loops, and one adversarial query whose subject isn't in the corpus.

5 of the 6 came back with distributional pathology flags. One came back clean. One adversarial query was correctly identified as out-of-distribution.

This is exactly the gap I keep running into in production RAG audits. The eval suite says fine. Precision@K says "5 chunks retrieved." Users say "the answers feel off." This shows why.

Long-form on Medium: https://medium.com/p/967cc480ad74

**Want this run on your stack?** $2,500, 48hr, Stripe direct: https://buy.stripe.com/00w28sfq5gjS6Dg4Ia9IQ00?ref=langchain-teardown-top  ·  Or email kevin.luddy39@gmail.com

---

## The headline finding

**Query:** "When does an agent decide to use a tool vs respond directly?"

This is — literally — what the Lilian Weng post is about. It's the central theme of the agent-architecture section. There is no more on-topic query you could possibly ask of that corpus.

The default retriever returned 5 chunks. Their alignment scores against the query, in rank order:

```
rank 1: 0.1139
rank 2: 0.1497
rank 3: 0.2085
rank 4: 0.2349   ← actually the most relevant
rank 5: 0.1640
```

`retrieval-auditor` computes:

- **rankQualityR = -0.611** — the retriever's ranking is anti-correlated with actual alignment. The top-ranked chunk is LESS relevant than the chunk it ranked fourth.
- **scoreCalibrationR = -0.675** — the retriever's reported similarity scores anti-correlate with independent alignment. Its confidence is upside-down.
- Three flags fire simultaneously: `OFF_TOPIC` (severity 0.42), `RANK_INVERSION` (0.54), `SCORE_MISCALIBRATED` (0.74).
- Health score: 0.099. Regime: "rot."

Precision@5 against ground truth would say: 5 chunks retrieved. The eval suite passes.

[INSERT: chart_q4.png here]

---

## The other four flagged queries

| # | Query | Mean alignment | Health | Flag |
|---|-------|----------------|--------|------|
| 1 | What is Chain of Thought prompting? | 0.135 | 0.225 | OFF_TOPIC |
| 2 | How does ReAct differ from Reflexion? | 0.114 | 0.207 | OFF_TOPIC |
| 4 | When does an agent decide to use a tool vs respond directly? | 0.174 | 0.099 | OFF_TOPIC, RANK_INVERSION, SCORE_MISCALIBRATED |
| 5 | Show me the planning loop for a long-horizon task | 0.203 | 0.360 | OFF_TOPIC |

Every one of these queries has a direct, unambiguous answer in the source post. Lilian Weng's article literally defines Chain of Thought, walks through ReAct, contrasts it with Reflexion, and describes the planning loop. The default retriever pulls chunks averaging ~0.15 alignment for queries that should land on text averaging ~0.7+.

[INSERT: chart_q1.png — Chain of Thought, mean 0.135]

---

## The clean baseline

**Query:** "What memory mechanisms do LLM agents use?"

```
mean alignment: 0.391
stdDev:         0.132
health:         0.710
flags:          —
```

This is what a healthy retrieval looks like. Higher mean, distribution spread that suggests genuine relevance, no flags. The auditor doesn't false-fire on clean retrievals — Q3 here is the negative control.

The lesson is in the contrast. On the SAME corpus, with the SAME retriever, with the SAME embedding model, query-by-query the retrieval quality varies by an order of magnitude on the auditor's health metric. Most production RAG monitoring won't show you that variance because it's collapsing distributional signal into top-line averages.

[INSERT: chart_q3.png — clean baseline]

---

## The adversarial check

**Query:** "What is reward shaping?"

This term doesn't appear in the corpus. It's an RL concept, not an LLM-agents concept. I included it to test whether the auditor distinguishes "the retriever is failing" from "the corpus genuinely doesn't contain what was asked."

```
mean alignment: 0.036
stdDev:         0.049
health:         0.065
flag:           OUT_OF_DISTRIBUTION
```

Distinct flag. Distinct severity profile. The auditor correctly identifies this as a different failure mode than the four `OFF_TOPIC` queries above — the corpus genuinely lacks the topic, rather than the retriever picking poorly from available material.

[INSERT: chart_q6.png — out of distribution]

This matters for production triage. "Retriever is broken" and "user asked something the corpus doesn't cover" need different fixes. Output-side metrics conflate them.

---

## Why precision@K misses all of this

Precision@K answers a single question: of the K chunks retrieved, how many are labeled relevant against ground truth?

For Query 4 above, suppose the ground-truth set is "any chunk discussing tool-use decisions." All 5 retrieved chunks discuss tool-related material to some degree. Precision@5 = 1.0. Eval passes.

What precision@K cannot see:

- The retriever ranked the LEAST relevant chunk first
- The retriever's similarity scores anti-correlate with actual alignment
- The mean alignment is near the floor of what a useful retrieval should produce
- The variance is too tight to give the LLM useful signal about which chunk to weight

These distributional properties show up for free if you measure them. They are invisible if you don't.

This isn't a hypothetical. The four engagements I've taken in 2026 to debug "production RAG that feels off" all turned out to be variants of the same thing: the eval suite was measuring the wrong layer.

---

## Caveats

The LangChain RAG quickstart is meant to be simple, not production-ready. I'm not arguing LangChain ships broken RAG. Quickstart code is supposed to be quickstart code. The point is that the failure modes I just measured *do not get caught by precision@K, do not get caught by LLM-as-judge, and do not get caught by the eval suite that ships with most production RAG implementations*. They survive contact with prod. They produce the "answers feel off" complaint.

The auditor isn't claiming to be a ground-truth oracle. It's distributional analysis. It surfaces shape problems that K-fold precision is blind to, regardless of whether the underlying retriever is the LangChain default or something more sophisticated.

The embedding model is `all-MiniLM-L6-v2` because that's what the LangChain quickstart uses. A larger embedding model would shift the absolute alignment numbers but not the structural findings (rank inversion, score miscalibration are properties of the retriever's scoring function, not the embedding).

I ran one corpus, six queries, one retriever, one embedding model. This is one teardown. Run it on yours.

---

## Reproduction

The teardown script is ~80 lines of Python. Sentence-transformers (free, local) for embeddings. Chroma for the vector store. retrieval-auditor for the analysis. No paid APIs.

```bash
pip install sentence-transformers chromadb requests beautifulsoup4 numpy matplotlib
npx contrarianai-retrieval-auditor --help
```

Full script + raw results JSON: https://github.com/kevin-luddy39/contrarianAI/tree/main/tools/retrieval-auditor/examples/langchain-quickstart-teardown

retrieval-auditor itself is MIT-licensed: `npx contrarianai-retrieval-auditor`

If you run it against your own production retrieval and you get clean health scores across queries, your retriever is in better shape than the canonical RAG quickstart. If you get flags — especially RANK_INVERSION or SCORE_MISCALIBRATED — those are diagnosable, and the fixes are usually narrow (re-fit a small reranker, change chunking, or in some cases change the embedding model entirely).

---

## If you want this run against your own pipeline

I do this as a fixed-scope engagement: $2,500, 48-hour turnaround, 3 clients per week. Five sensors run against your stack. 8-12 page report with flagged pathologies, the bell curve of your retrieval distribution per query class, and one prioritized fix.

**Buy now (Stripe, direct):** https://buy.stripe.com/00w28sfq5gjS6Dg4Ia9IQ00?ref=langchain-teardown
**Or email me first:** kevin.luddy39@gmail.com (reply within 24h, no form, no sales sequence)

More detail / longer pitch page: https://contrarianai-landing.onrender.com/bell-tuning-rapid-audit.html

Or skip the audit entirely and run the tool yourself — it's MIT, one command, no account required. The point is that this layer of measurement exists and is cheap. Whether you do the work or I do isn't the important part. Measuring the layer is.

Worth getting both layers right before assuming you need to pay 5-10x per query for agentic search on everything.

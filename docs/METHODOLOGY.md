# Benchmark methodology

## Comparison unit

Freeze the corpus, query text, ordered stage-one candidates, rerank text, model weights, Python packages, machine, thread count, batch size, token limit and scoring depth. Compare only raw result rows with matching identities. Each cross-encoder evaluates a query-company pair independently; it does not read all 500 companies in a single prompt. Native logits/scores are sorted within one query and are not interpreted as calibrated probabilities or Yes/No decisions.

Stage one is deterministic BM25 over `description_keywords` for the public and controlled company pools, `rich` for the long-context diagnostics, and `title_text` for SciFact. The company pools contain up to 1,000 unique IDs per query (the 130-company pilot naturally has only 130). The SciFact pool similarly has up to 1,000 and uses title plus abstract, but the separate SciFact job reranks 100 to align more closely with public reranking practice. No judged positive is inserted into a retrieved pool. `build_candidates.py --external-jsonl` accepts ordered JSONL `{ "query_id": "...", "doc_ids": ["...", ...] }` exported from another retrieval system, validates IDs and uniqueness, then freezes the run. For production transfer, export the actual embedding system's stage-one top 1,000, use the same candidate file for every reranker, and score its original top 500.

The reranked prefix is sorted by descending native score; ties preserve stage-one order. Tail IDs are unchanged. Candidate recall at 500 is therefore fixed. Report stage-one ceiling and an oracle ordering restricted to the same prefix to distinguish missed candidates from ranking mistakes.

## Relevance and constraints

Report per-query and mean nDCG@10/20/100 (and larger cutoffs where available), known-positive recall at 25/50/100, MRR and eligible category slices. For partially judged data, unjudged pairs are **unknown**, not verified negatives. nDCG uses zero gain for unjudged hits as a pooled-label diagnostic, and the report labels it accordingly. Precision/MAP are suppressed unless the dataset explicitly claims exhaustive labels. The public pilot's strict relevance-2 positives and partial relevance-1 candidates must be distinguished; inspect the original rationale records before drawing any conclusion.

The controlled synthetic set defines all unlisted pairs as irrelevant by construction. Long-context diagnostics explicitly annotate `match`, `excluded`, `negative_other` and `unknown`; unknown pairs are omitted from qrels and counted separately. Exclusion counts describe documented violations in top-k, not a model's proof that the remaining companies comply. In the real workflow, verify each finalist's relevant company fields and source evidence; mark an unmentioned customer channel unknown.

## CPU and long context

Run one model/stage per fresh process on the VDI, with fixed batch size, threads and text representation. The quality stage records total pair scoring time, original token lengths and truncation rate. The speed stage warms up, scores and sorts complete query prefixes over repeated rounds, records pairs/s, query p50/p95, cold load and sampled process RSS. These are reranker-stage measurements, excluding stage-one retrieval and company-data loading. Compare runs only when machine and full workload match. Short local smoke tests and a different CPU are exploratory.

The regular plan uses a 512-token pair cap. The long-context diagnostic repeats at 4,096 tokens, with evidence placed at the beginning, middle or end of synthetic profiles. A long advertised pair limit does not prove the relevant text survives truncation or that long-context CPU latency is practical. Compare error cases and token counts, not just aggregate score. For final sizing, sweep batch 1/8/32 and threads 1/2/4/8 on representative short and long profiles, record p95, memory and sustained throughput, and respect the actual VDI core quota. This sweep can be run by editing a copied suite plan or invoking `reranker_bench.cli speed` directly.

## Paired model comparison

Use query-level paired differences and bootstrap intervals where both runs share candidate digest, dataset digest, representation and runtime settings. Inspect hard-negative rank changes, exclusion violations, unknowns and failures by query category. Re-run unstable speed cells; do not infer quantization benefit from model-size labels. This release contains FP32 PyTorch artifacts only, so a quantization comparison needs separate verified artifacts and a new paired run.

External MTEB numbers in `models.json` are reference metadata only. The six published nDCG@10 values are from the [Ettin reranker report](https://huggingface.co/blog/ettin-reranker); the Jina result is absent from that same comparison. nDCG@20 and @100 are measured by this kit, not taken from that report.

## Decision gate

Choose a deployment candidate only after all relevant assets pass file-hash and local score validation, company evaluations use independently adjudicated representative descriptions and queries, stage-one runs match production, exclusions are checked against evidence, and VDI latency/memory remain acceptable under expected concurrency. SciFact and synthetic results are supporting diagnostics. They cannot substitute for the company decision set.

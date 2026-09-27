# OrbitProof: code for "OrbitProof: Certified LLM Querying with Time-Expanded Property Graphs"

Python 3.10+, one CPU core is enough. `pip install -r requirements.txt`

## Files

| File | Purpose |
|---|---|
| `constellation.py` | Walker-star constellation, ground stations, slotted contact-plan generator (regimes `isl_full`, `isl_intra`, `no_isl`) |
| `teg.py` | TEG engine: earliest arrival (plain and hop-bounded), journey witnesses, loop elimination, journey and labeling checkers, replay and minimality checks, time-agnostic semantics |
| `sqlview.py` | The TEG property-graph view materialised in DuckDB; Kleene-star reachability as a recursive CTE |
| `enum_baseline.py` | Trail enumeration with post-filter or incremental pruning (the path-predicate baseline) |
| `agent.py` | Journey IR (JSON schema), parsing and grounding, certified execution, IR-to-SQL compiler, answer renderer |
| `make_benchmark.py` | Generates the NL question benchmark with gold IR and gold answers |
| `harness.py` | LLM study: `sql`, `sql_hint`, `ir` systems; providers `anthropic`, `openai` (any OpenAI-compatible endpoint), `mock-gold`, `mock-naive` |
| `experiments.py` | Runs every experiment in the paper|
| `mutation.py` | Mutation tests: 20 classes of corrupted witnesses, labelings and answers (incl. forged, unattained lower bounds for plain/VIA/HOPS earliest answers) that the checker must reject |
| `check_review_regressions.py` | Regression tests for every counterexample in the external pre-submission review |
| `check_small_graph_oracle.py` | Independent brute-force oracle on random small plans (answers and certificates) |
| `check_via_hops_attack.py` | Regression test for the EARLIEST VIA/HOPS attainment gap found in the internal audit |
| `check_engine_vs_sql.py`, `check_certificates.py`, `check_tau.py` | Correctness checks (engine vs DuckDB for tau in {0,1}, hop-bounded vs enumeration, loop elimination, replay and minimality) |
| `bench.jsonl` | The 500-instance benchmark used in the paper (seed 11, one paraphrase per instance) |
| `bench_all_paraphrases.jsonl` | Same 500 instances as `bench.jsonl`, all three paraphrases each (1,500 questions; fixed in revision 2.5) |
| `run_llm_study.sh` | One command for the whole LLM study: 4 systems x 2 LLMs, then summary and paper rebuild |
| `summarize.py` | Logs to Table V (`paper/gen/tab_llm.tex`), with Wilson CIs, McNemar tests, per-template accuracy and error taxonomy (`llm_summary.md`) |
| `fault_injection.py` | Fault-injection simulation of LLM errors (Sec. VII-E, Tables V-VI): intent errors through the real pipeline, single-error queries through the reference engine, and expected behaviour under a stated error model |
| `alt_slot_width.py` | Headline numbers under 60-s vs 10-s slots (`alt_slot_width.md`, `paper/gen/alt_macros.tex`) |


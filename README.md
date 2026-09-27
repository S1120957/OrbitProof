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
| `experiments.py` | Runs every experiment in the paper; writes `../paper/figs/fig_scal.pdf`, `../paper/gen/*.tex` and `results.json` |
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

## Reproduce the paper's numbers

```bash
python3 experiments.py --only scal,enum,ws   # ~45 s
python3 experiments.py --only hor            # ~2 min (24-h horizon)
python3 experiments.py --only sens,mut       # ~20 s; writes figure, tables and macros once all parts exist
python3 fault_injection.py                   # ~1 min; Tables V-VI and fi_macros.tex
python3 harness.py --bench bench.jsonl --system sql --provider mock-naive --emit-macro RNaiveAcc
python3 harness.py --bench bench.jsonl --system ir  --provider mock-gold  --emit-macro RGoldIRAcc
cd ../paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

All counts and error rates are deterministic (fixed seeds). Timings vary slightly between runs and machines; the macros keep the text in sync.

## Run the LLM study (not yet done)

```bash
export ANTHROPIC_API_KEY=...  MODEL_A=<exact dated model id>
export MODEL_B=<open-weights model id>  BASE_URL_B=http://<vllm-host>:8000/v1   # OPENAI_API_KEY if needed
LIMIT=20 ./run_llm_study.sh    # stratified dry run: 2 questions per (regime, type); report only
./run_llm_study.sh             # full run; re-run the same command to resume after an interruption
```

The runner calls `harness.py` for each system and LLM. Generation is parallel (`WORKERS`, default 4), with retries and back-off on rate limits. It is cached in `logs/<system>_<A|B>.jsonl.raw.jsonl` and resumes automatically. Each record stores the model id, prompt version hash, temperature and timestamp.

Afterwards `summarize.py --write-paper` fills Table V, and the script rebuilds the PDF.

Report the exact model ids and the prompt hash (printed by the harness) in the paper. The prompts are in `harness.py` (`SCHEMA_TXT`, `HINT_TXT`, `VIEW_TXT`, `SQL_TASK_TXT`, `IR_TASK_TXT`); changing them changes the hash and invalidates cached generations.

To test the pipeline without keys: `PROVIDER_A=mock-gold MODEL_A=x PROVIDER_B=mock-naive MODEL_B=y LIMIT=10 ./run_llm_study.sh`.
Mock runs are pipeline tests, not LLM results. The runner never writes Table V for them, `summarize.py --write-paper` refuses mock logs, and `python3 summarize.py --reset-paper` restores the TBD table.

**OpenAI model as LLM A** (e.g. GPT-4o): `PROVIDER_A=openai MODEL_A=gpt-4o-2024-11-20 OPENAI_API_KEY=... ./run_llm_study.sh`.
**vLLM server as LLM B**: `MODEL_B=meta-llama/Llama-3.1-70B-Instruct BASE_URL_B=http://<host>:8000/v1 ./run_llm_study.sh`.

**No API key?** Any OpenAI-compatible server works as LLM B, and either model may be run alone (set only `MODEL_A` or only `MODEL_B`). For example, a local open-weights model with Ollama (CPU is enough, just slow):

```bash
ollama pull qwen2.5-coder:7b          # or another instruct model; check the exact tag
MODEL_B=qwen2.5-coder:7b BASE_URL_B=http://localhost:11434/v1 WORKERS=2 ./run_llm_study.sh
```

Report the exact model tag and quantisation in the paper.

Known gaps:
- The harness prompts for JSON and validates it afterwards (as Sec. VI states). Enabling a provider's structured-output mode is optional; if you do, say so in the paper.
- In the intra-plane regime every pair is reachable, so `T1_exists` has only "yes" answers there. The no-ISL regime is balanced.

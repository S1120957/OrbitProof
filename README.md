# OrbitProof

Certified natural-language querying of satellite contact plans (LEO and delay-tolerant networks).

An LLM maps a question to a small typed *journey IR*. A deterministic compiler and engine answer it over a time-expanded property-graph view of the contact plan, in which a node is a (node, time-slot) pair. Every answer is released only with a certificate that an independent checker validates against the base contact plan:

- **"yes" and earliest-arrival answers:** a journey witness, which must attain the claimed value;
- **"no" answers and "nothing earlier" claims:** a closed arrival-time labeling.

## Install

```bash
pip install -r requirements.txt        # Python >= 3.10
```

## Quick start

```bash
python make_benchmark.py --out bench.jsonl      # 500-instance question benchmark (deterministic)
python check_certificates.py && python check_tau.py && python check_via_hops_attack.py
python experiments.py --quick                   # smoke test of all experiments
python fault_injection.py                       # simulated LLM intent and query errors
```

Outputs (results, figures, tables) are written to `out/` (override with `ORBITPROOF_OUT`) and are not tracked.

## LLM study

```bash
PROVIDER_A=openai MODEL_A=<model-id> OPENAI_API_KEY=... ./run_llm_study.sh
```

`LIMIT=20` gives a stratified dry run. `MODEL_B` and `BASE_URL_B` add an OpenAI-compatible server such as vLLM or Ollama.

## Files

| File | Purpose |
|---|---|
| `constellation.py` | Walker constellations and slotted contact plans |
| `teg.py` | Time-expanded engine, witnesses, labelings and checkers |
| `sqlview.py` | Time-expanded view in DuckDB (recursive SQL) |
| `agent.py` | Journey IR, grounding, certified execution, IR-to-SQL compiler |
| `make_benchmark.py` | Benchmark generator |
| `harness.py`, `run_llm_study.sh`, `summarize.py` | LLM study |
| `experiments.py`, `enum_baseline.py`, `mutation.py`, `fault_injection.py`, `alt_slot_width.py` | Experiments |
| `check_*.py` | Correctness checks |

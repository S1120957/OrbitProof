#!/usr/bin/env bash
# Runs the full LLM study (4 systems x 2 LLMs), summarises it and rebuilds the paper.
# Interrupted runs resume: re-run the same command.
#
# Environment (either model may be run alone):
#   MODEL_A      exact, dated model id for LLM A; PROVIDER_A anthropic (default) or openai
#   ANTHROPIC_API_KEY or OPENAI_API_KEY; BASE_URL_A for a non-default OpenAI-compatible endpoint
#   MODEL_B      model id for LLM B served by an OpenAI-compatible endpoint (e.g. vLLM)
#   BASE_URL_B   e.g. http://localhost:8000/v1   (OPENAI_API_KEY if the server needs one)
# Optional:
#   BENCH (default bench.jsonl), WORKERS (default 4), LIMIT (e.g. 20 for a dry run)
#   PROVIDER_A / PROVIDER_B (default anthropic / openai; mock-gold or mock-naive to test the pipeline)
set -euo pipefail
cd "$(dirname "$0")"
BENCH=${BENCH:-bench.jsonl}
WORKERS=${WORKERS:-4}
LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARG=(--limit "$LIMIT"); fi
PROVIDER_A=${PROVIDER_A:-anthropic}
PROVIDER_B=${PROVIDER_B:-openai}
mkdir -p logs

for SYS in sql sql_hint view_sql ir; do
  if [[ -n "${MODEL_A:-}" ]]; then
    BASE_A_ARG=()
    if [[ -n "${BASE_URL_A:-}" ]]; then BASE_A_ARG=(--base-url "$BASE_URL_A"); fi
    python3 harness.py --bench "$BENCH" --system "$SYS" --provider "$PROVIDER_A" --model "$MODEL_A" \
      --label A --workers "$WORKERS" --out "logs/${SYS}_A.jsonl" "${LIMIT_ARG[@]}" "${BASE_A_ARG[@]}"
  fi
  if [[ -n "${MODEL_B:-}" ]]; then
    python3 harness.py --bench "$BENCH" --system "$SYS" --provider "$PROVIDER_B" --model "$MODEL_B" \
      --base-url "${BASE_URL_B:-http://localhost:8000/v1}" --label B --workers "$WORKERS" \
      --out "logs/${SYS}_B.jsonl" "${LIMIT_ARG[@]}"
  fi
done

if [[ -n "${LIMIT:-}" ]]; then
  python3 summarize.py "logs/*.jsonl"                 # dry run: report only, paper untouched
  echo "Dry run done (LIMIT=$LIMIT). Unset LIMIT for the full run."
  exit 0
fi

if [[ "$PROVIDER_A" == mock* || "$PROVIDER_B" == mock* ]]; then
  python3 summarize.py "logs/*.jsonl"                 # mock providers: report only, never the paper
  echo "Mock providers: pipeline test only. Table V was NOT written."
  exit 0
fi

python3 summarize.py "logs/*.jsonl" --write-paper
cd ../paper
if command -v pdflatex >/dev/null && command -v bibtex >/dev/null; then
  pdflatex -interaction=nonstopmode main.tex >/dev/null
  bibtex main >/dev/null
  pdflatex -interaction=nonstopmode main.tex >/dev/null
  pdflatex -interaction=nonstopmode main.tex >/dev/null
  echo "Paper rebuilt: paper/main.pdf."
else
  echo "pdflatex/bibtex not found: Table V was written to paper/gen/tab_llm.tex; compile the paper elsewhere (e.g. Overleaf)."
fi
echo "Report: code/llm_summary.md. Record MODEL_A/MODEL_B and the prompt hash in the paper."

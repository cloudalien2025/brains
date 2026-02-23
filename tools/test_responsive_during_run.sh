#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8081}"
API_KEY="${WORKER_API_KEY:-${API_KEY:-}}"
BRAIN_ID="${BRAIN_ID:-}"
KEYWORD="${KEYWORD:-ai news}"
DURATION_SECONDS="${DURATION_SECONDS:-60}"

if [[ -z "$API_KEY" ]]; then
  echo "ERROR: set WORKER_API_KEY or API_KEY" >&2
  exit 1
fi

if [[ -z "$BRAIN_ID" ]]; then
  echo "ERROR: set BRAIN_ID" >&2
  exit 1
fi

INGEST_PAYLOAD="${INGEST_PAYLOAD:-}"
if [[ -z "$INGEST_PAYLOAD" ]]; then
  INGEST_PAYLOAD=$(cat <<JSON
{"keyword":"${KEYWORD}","selected_new":2}
JSON
)
fi

echo "Triggering run for brain=${BRAIN_ID} at ${BASE_URL}"
RUN_JSON=$(curl -sS --max-time 2 -X POST "${BASE_URL}/v1/brains/${BRAIN_ID}/ingest" \
  -H "X-Api-Key: ${API_KEY}" -H "Content-Type: application/json" \
  -d "${INGEST_PAYLOAD}")
RUN_ID=$(python - <<'PY' "$RUN_JSON"
import json,sys
print(json.loads(sys.argv[1])["run_id"])
PY
)

echo "Run started: ${RUN_ID}"
END=$((SECONDS + DURATION_SECONDS))
ITER=0
while (( SECONDS < END )); do
  ITER=$((ITER + 1))
  for PATH_SUFFIX in "/v1/health" "/v1/runs/${RUN_ID}" "/v1/runs/${RUN_ID}/report"; do
    CODE=$(curl -sS -o /tmp/brains_resp_body.txt -w '%{http_code}' --max-time 2 \
      -H "X-Api-Key: ${API_KEY}" "${BASE_URL}${PATH_SUFFIX}" || true)
    if [[ "$CODE" == "000" ]]; then
      echo "FAIL timeout path=${PATH_SUFFIX} iter=${ITER}" >&2
      exit 1
    fi
    if [[ "$PATH_SUFFIX" == "/v1/runs/${RUN_ID}/report" && "$CODE" == "202" ]]; then
      :
    elif [[ "$CODE" -ge 400 ]]; then
      echo "FAIL code=${CODE} path=${PATH_SUFFIX} iter=${ITER}" >&2
      cat /tmp/brains_resp_body.txt >&2 || true
      exit 1
    fi
  done
  sleep 1
done

echo "PASS responsive for ${DURATION_SECONDS}s run_id=${RUN_ID}"

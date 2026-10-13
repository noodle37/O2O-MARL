#!/bin/bash
# Fetch the og-marl 2halfcheetah vault (8.3 GiB) through hf-mirror.
# The downloader resumes from sparse-file holes, so a crash only costs the
# in-flight chunks; this loop restarts it until the zip passes integrity + unzip.
# Usage: nohup bash scripts/run_fetch_2halfcheetah.sh > logs/fetch_2halfcheetah.out 2>&1 &
set -uo pipefail

cd /home/jxy/multiAgentRL
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
SCENARIO="${1:-2halfcheetah}"
ENV_NAME="${2:-mamujoco}"
SOURCE="${3:-og_marl}"

for attempt in $(seq 1 60); do
  echo "=== [$(date '+%F %T')] fetch attempt $attempt ${SOURCE}/${ENV_NAME}/${SCENARIO} ==="
  "$PY" scripts/fetch_vault.py \
    --source "$SOURCE" --env "$ENV_NAME" --scenario "$SCENARIO" \
    --mirror https://hf-mirror.com \
    --workers 6 --chunk-mb 8 --read-timeout 45 --unzip
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== [$(date '+%F %T')] fetch OK ==="
    exit 0
  fi
  echo "=== [$(date '+%F %T')] fetch attempt $attempt failed rc=$rc, resuming in 15s ==="
  sleep 15
done

echo "=== [$(date '+%F %T')] fetch gave up after 60 attempts ==="
exit 1

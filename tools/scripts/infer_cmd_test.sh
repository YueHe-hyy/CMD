#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <num-gpus> <checkpoint.pth> [data-root] [eval-tag]" >&2
  exit 2
fi

GPUS=$1
CKPT=$2
DATA_ROOT=${3:-$(cd "$(dirname "$0")/../.." && pwd)}
EVAL_TAG=${4:-competition_test}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

cd "${REPO_ROOT}/tools"
bash scripts/dist_test.sh "${GPUS}" \
  --cfg_file cfgs/xmu_da_models/centerpoint_robosense_test.yaml \
  --ckpt "${CKPT}" \
  --eval_tag "${EVAL_TAG}" \
  --set DATA_CONFIG.DATA_PATH "${DATA_ROOT}"

RESULT=$(find "${REPO_ROOT}/output/xmu_da_models/centerpoint_robosense_test" -path "*/${EVAL_TAG}/result.pkl" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
if [[ -z "${RESULT}" ]]; then
  echo "Inference finished but result.pkl was not found." >&2
  exit 1
fi

python cmd_competition.py package \
  --result "${RESULT}" \
  --output "${REPO_ROOT}/submission.zip" \
  --data-root "${DATA_ROOT}"

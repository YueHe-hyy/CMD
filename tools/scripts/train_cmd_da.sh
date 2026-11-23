#!/usr/bin/env bash
set -euo pipefail

GPUS=${1:-8}
DATA_ROOT=${2:-$(cd "$(dirname "$0")/../.." && pwd)}
EXTRA_TAG=${3:-source_mix_sparse}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

python "${REPO_ROOT}/tools/cmd_competition.py" check-data \
  --data-root "${DATA_ROOT}" \
  --sensors ouster hesai \
  --strict

for INFO_FILE in xmu_infos_train_ouster.pkl xmu_infos_train_hesai.pkl; do
  if [[ ! -s "${DATA_ROOT}/${INFO_FILE}" ]]; then
    echo "Missing ${DATA_ROOT}/${INFO_FILE}" >&2
    echo "Generate infos first: python tools/cmd_competition.py make-infos --data-root ${DATA_ROOT} --sensors ouster hesai --workers 8" >&2
    exit 1
  fi
done

cd "${REPO_ROOT}/tools"
bash scripts/dist_train.sh "${GPUS}" \
  --cfg_file cfgs/xmu_da_models/centerpoint_source_mix.yaml \
  --extra_tag "${EXTRA_TAG}" \
  --set DATA_CONFIG.DATA_PATH "${DATA_ROOT}"

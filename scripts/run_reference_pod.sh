#!/usr/bin/env bash
# Forecast a round's exported contexts with the reference models on a rented
# Lium GPU, bring the forecasts back, and always delete the pod.
#
#   scripts/run_reference_pod.sh CONTEXTS_NPZ OUT_DIR [MODELS]
#
# The pod receives only scripts/reference_forecasts.py and the contexts npz
# (histories and horizons, never the truth) -- no credentials of any kind.
# Needs the `lium` CLI configured on the runner. Env: REF_GPUS (comma list of
# acceptable single-GPU types, default RTX4090,A6000,L40,L40S,RTX6000,A100),
# REF_TTL (default 1h, the backstop if this script dies before its trap runs).
set -euo pipefail

CTX=$1
OUT=$2
MODELS=${3:-timesfm3,toto2-2.5b,toto2-22m,toto2-4m}
GPUS=${REF_GPUS:-RTX4090,A6000,L40,L40S,RTX6000,A100}
POD="forge-ref-${GITHUB_RUN_ID:-local}-${RANDOM}"
STAGE=$(mktemp -d)
export COLUMNS=250

cleanup() {
  yes y | lium rm "$POD" >/dev/null 2>&1 || true
  rm -rf "$STAGE"
}
trap cleanup EXIT

cp "$(dirname "$0")/reference_forecasts.py" "$CTX" "$STAGE/"
CTX_NAME=$(basename "$CTX")

# The market moves between listing and renting, so try the cheapest few
# single-GPU nodes in turn until one is acquired. `lium up` resolves nodes
# by UUID only (a HUID gives "not found" despite its help text).
NODES=$(lium ls --count 1 --format json 2>/dev/null | python3 -c '
import json, sys
ok = set(sys.argv[1].split(","))
raw = sys.stdin.read()  # a fresh install prints a banner before the JSON
nodes = [n for n in json.loads(raw[raw.index("["):]) if n.get("gpu_type") in ok and (n.get("vram_gb") or 0) >= 24]
for n in sorted(nodes, key=lambda n: n["price_per_hour"])[:6]:
    print(n["id"], n["huid"], n["gpu_type"], n["price_per_hour"])' "$GPUS")
[ -n "$NODES" ] || { echo "no $GPUS node listed" >&2; exit 1; }
while read -r NODE HUID TYPE PRICE; do
  echo "renting $TYPE $HUID (\$$PRICE/h) as $POD"
  UP=$(yes y | timeout 900 lium up "$NODE" --name "$POD" --ttl "${REF_TTL:-1h}" 2>&1 || true)
  lium ps </dev/null 2>/dev/null | grep -q "$POD" && break
  printf '%s\n' "$UP" | tail -2 >&2
done <<< "$NODES"
lium ps 2>/dev/null | grep -q "$POD" || { echo "could not rent any of: $(awk '{print $2}' <<< "$NODES" | xargs)" >&2; exit 1; }
for _ in $(seq 1 60); do
  lium ps 2>/dev/null | grep "$POD" | grep -q RUNNING && break
  sleep 20
done
lium ps 2>/dev/null | grep "$POD" | grep -q RUNNING || { echo "pod never reached RUNNING" >&2; exit 1; }

lium rsync "$POD" "$STAGE/" /root/ref/ >/dev/null
# Install into the image's own torch: the image's pip config pins torch and
# permits system installs, so no second CUDA wheel is downloaded.
lium exec "$POD" "set -e; cd /root/ref; \
  pip -q install 'timesfm[torch]>=3' 'toto-models>=1.0' >/dev/null 2>&1; \
  for m in \$(echo $MODELS | tr ',' ' '); do \
    python3 reference_forecasts.py --contexts $CTX_NAME --out out --models \$m \
      || python3 reference_forecasts.py --contexts $CTX_NAME --out out --models \$m || true; \
  done" 2>&1 | grep -v -i -E 'userwarning|warnings.warn|unauthenticated requests' || true

mkdir -p "$OUT"
IFS=',' read -ra NAMES <<< "$MODELS"
for m in "${NAMES[@]}"; do
  lium scp "$POD" "/root/ref/out/$m.npz" "$OUT/" -d >/dev/null 2>&1 \
    && echo "fetched $m" || echo "no forecasts for $m" >&2
done

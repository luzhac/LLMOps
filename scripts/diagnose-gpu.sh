#!/usr/bin/env bash
# GPU node scale-up diagnostics.
# Answers three questions in one pass:
#   1. Is the node pool actually on-demand now, or still Spot?
#   2. Is the on-demand L4 quota zero? (different quota from Spot)
#   3. What is the raw GCE error code behind "GCE out of resources"?
#
# Usage: bash scripts/diagnose-gpu.sh
# Safe to run any time. Read-only: it never changes cluster or infra state.
#
# Output goes to the terminal AND to .diag/latest.txt (plus a timestamped copy),
# so an assistant with access to this repo folder can read the result directly
# instead of asking you to copy-paste terminal output.
# Set DIAG_DIR= to change the location, DIAG_TEE=0 to disable file output.

set -uo pipefail

diag_dir="${DIAG_DIR:-.diag}"
if [[ "${DIAG_TEE:-1}" == "1" && -z "${_DIAG_TEEING:-}" ]]; then
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  [[ "${diag_dir}" = /* ]] || diag_dir="${repo_root}/${diag_dir}"
  mkdir -p "${diag_dir}"
  stamp="$(date +%Y%m%d-%H%M%S)"
  export _DIAG_TEEING=1
  # Re-run self, mirroring all output into the files.
  bash "${BASH_SOURCE[0]}" "$@" 2>&1 \
    | tee "${diag_dir}/latest.txt" "${diag_dir}/${stamp}.txt"
  echo
  echo "Saved to ${diag_dir}/latest.txt and ${diag_dir}/${stamp}.txt"
  exit 0
fi

namespace="${NAMESPACE:-llm-platform}"
cluster="${CLUSTER_NAME:-trade-balance-llm}"
zone="${ZONE:-europe-west4-a}"
region="${REGION:-europe-west4}"
pool="${POOL_NAME:-l4-spot}"

hr() { printf '\n=== %s %s\n' "$1" "$(printf '=%.0s' {1..50})"; }

hr "0. Context"
echo "cluster=${cluster}  zone=${zone}  pool=${pool}  namespace=${namespace}"
kubectl config current-context 2>/dev/null || echo "kubectl context: UNAVAILABLE"

hr "1. Node pool: on-demand or still Spot?"
echo "Look at SPOT. False/empty = on-demand. True = terraform apply did not take effect."
gcloud container node-pools describe "${pool}" \
  --cluster="${cluster}" --zone="${zone}" \
  --format="table[box](name,config.machineType,config.spot:label=SPOT)" \
  2>&1 || echo "describe failed"

echo
echo "Autoscaling block. An EMPTY block means autoscaling is OFF, and the"
echo "cluster autoscaler will emit a blank 'Pod didn't trigger scale-up:'."
echo "Either min/maxNodeCount (per zone) or totalMin/totalMaxNodeCount must be present."
gcloud container node-pools describe "${pool}" \
  --cluster="${cluster}" --zone="${zone}" \
  --format="yaml(autoscaling)" 2>&1 || true

echo
echo "Zones this pool may place nodes in (single zone = no fallback if that zone is empty):"
gcloud container node-pools describe "${pool}" \
  --cluster="${cluster}" --zone="${zone}" \
  --format="value(locations)" 2>&1 || true

hr "2. L4 quota in ${region}"
echo "NVIDIA_L4_GPUS            = on-demand quota"
echo "PREEMPTIBLE_NVIDIA_L4_GPUS = Spot quota (a DIFFERENT bucket)"
echo "If the on-demand limit is 0, this is a QUOTA problem, not a stockout."
# NOTE: `regions describe` returns a single resource, so it rejects --filter.
# Flatten the quota list, emit CSV, then grep.
{ echo "METRIC,LIMIT,USAGE"
  gcloud compute regions describe "${region}" \
    --flatten="quotas[]" \
    --format="csv[no-heading](quotas.metric,quotas.limit,quotas.usage)" \
    2>/dev/null | grep -i "nvidia_l4"
} | column -t -s, || echo "quota lookup failed"

hr "3. Which zones actually offer L4"
gcloud compute accelerator-types list \
  --filter="name~nvidia-l4 AND zone~${region}" \
  --format="table(zone,name)" 2>&1 || true

hr "4. Raw GCE errors behind the failed scale-up"
echo "ZONE_RESOURCE_POOL_EXHAUSTED = genuine stockout"
echo "QUOTA_EXCEEDED               = quota, k8s Events can mislabel this as 'out of resources'"
gcloud compute operations list \
  --filter="targetLink~gke-${cluster}-${pool} AND error.errors[0].code:*" \
  --sort-by=~endTime --limit=8 \
  --format="table(endTime,operationType,error.errors[0].code,error.errors[0].message)" \
  2>&1 || echo "no matching operations"

hr "5. Cluster-side state"
kubectl -n "${namespace}" get pods,pvc -o wide 2>&1 || true
echo
kubectl get nodes -L cloud.google.com/gke-accelerator,cloud.google.com/gke-spot 2>&1 || true

hr "6. PVC zone pin (blocks any multi-zone plan)"
echo "standard-rwo is a ZONAL persistent disk. Once Bound, it pins the pod to that zone."
pv="$(kubectl -n "${namespace}" get pvc vllm-model-cache -o jsonpath='{.spec.volumeName}' 2>/dev/null || true)"
if [[ -n "${pv}" ]]; then
  echo "PV: ${pv}"
  kubectl get pv "${pv}" \
    -o jsonpath='{range .spec.nodeAffinity.required.nodeSelectorTerms[*].matchExpressions[*]}{.key}={.values[*]}{"\n"}{end}' \
    2>/dev/null || true
else
  echo "PVC vllm-model-cache not bound (or absent) - no zone pin."
fi
echo
echo "Binding mode of standard-rwo (WaitForFirstConsumer lets a new PVC follow the pod):"
kubectl get storageclass standard-rwo -o jsonpath='{.volumeBindingMode}{"\n"}' 2>&1 || true

hr "7. Recent scheduling events"
kubectl -n "${namespace}" describe pod -l app.kubernetes.io/name=vllm 2>/dev/null \
  | sed -n '/^Events:/,$p' | tail -25 || echo "no vllm pod"

hr "Done"
cat <<'EOF'
How to read this:

  Section 1 SPOT=True        -> terraform apply did not land. Re-check state.
  Section 1 autoscaling EMPTY-> autoscaling is OFF. The autoscaler will not even
                               try, and the Event reason is blank. Re-enable with:
                                 gcloud container node-pools update l4-spot \
                                   --cluster=trade-balance-llm --zone=europe-west4-a \
                                   --enable-autoscaling --total-min-nodes=0 \
                                   --total-max-nodes=1 --location-policy=ANY
  Section 2 on-demand limit 0-> QUOTA problem. Request NVIDIA_L4_GPUS quota.
  Section 4 QUOTA_EXCEEDED   -> same as above; the k8s Event was misleading.
  Section 4 ZONE_RESOURCE_POOL_EXHAUSTED -> real stockout. Options:
      a) add node_locations across zones listed in section 3
         (but first delete the PVC from section 6, or the zonal disk re-pins you)
      b) try g2-standard-4 instead of g2-standard-8
      c) fall back to n1-standard-8 + nvidia-tesla-t4 (16 GB still fits the AWQ model)
      d) scale to zero and retry in a few hours - costs nothing while waiting

Stop the autoscaler retry loop at any time with:
  bash scripts/model-session.sh down
EOF

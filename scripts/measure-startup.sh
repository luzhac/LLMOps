#!/usr/bin/env bash
# Measure vLLM cold-start, phase by phase.
#
# Run it AFTER the vLLM pod reaches Ready. It reads timestamps that Kubernetes
# and vLLM already record, so it does not need to be running during startup.
#
# Usage: bash scripts/measure-startup.sh

set -uo pipefail

namespace="${NAMESPACE:-llm-platform}"

pod="$(kubectl -n "${namespace}" get pod -l app.kubernetes.io/name=vllm \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)"

if [[ -z "${pod}" ]]; then
  echo "No vLLM pod found in namespace ${namespace}. Is replicaCount still 0?" >&2
  exit 1
fi

echo "Pod: ${pod}"
echo

# --- Kubernetes-level phases -------------------------------------------------

created="$(kubectl -n "${namespace}" get pod "${pod}" -o jsonpath='{.metadata.creationTimestamp}')"
scheduled="$(kubectl -n "${namespace}" get pod "${pod}" \
  -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].lastTransitionTime}')"
img_ready="$(kubectl -n "${namespace}" get pod "${pod}" \
  -o jsonpath='{.status.conditions[?(@.type=="PodReadyToStartContainers")].lastTransitionTime}')"
ready="$(kubectl -n "${namespace}" get pod "${pod}" \
  -o jsonpath='{.status.conditions[?(@.type=="Ready")].lastTransitionTime}')"

secs_between() { # $1 earlier ISO8601, $2 later ISO8601
  local a b
  a="$(date -u -d "$1" +%s 2>/dev/null)" || return 1
  b="$(date -u -d "$2" +%s 2>/dev/null)" || return 1
  echo $(( b - a ))
}

fmt() { # seconds -> "Xm Ys"
  printf '%dm %02ds' $(( $1 / 60 )) $(( $1 % 60 ))
}

echo "=== Kubernetes phases ==="
printf '%-38s %s\n' "Pod created" "${created}"
if [[ -n "${scheduled}" ]]; then
  printf '%-38s %s\n' "  -> scheduled (waiting for GPU node)" "$(fmt "$(secs_between "${created}" "${scheduled}")")"
fi
if [[ -n "${img_ready}" && -n "${scheduled}" ]]; then
  printf '%-38s %s\n' "  -> image pulled" "$(fmt "$(secs_between "${scheduled}" "${img_ready}")")"
fi
if [[ -n "${ready}" && -n "${img_ready}" ]]; then
  printf '%-38s %s\n' "  -> container start + model load" "$(fmt "$(secs_between "${img_ready}" "${ready}")")"
fi
if [[ -n "${ready}" ]]; then
  printf '%-38s %s\n' "TOTAL cold start" "$(fmt "$(secs_between "${created}" "${ready}")")"
else
  echo "Pod is not Ready yet — rerun once it is."
fi

# Kubernetes reports the pull duration directly in the Pulled event; this is the
# most trustworthy single number for image-pull cost.
echo
echo "=== Image pull, as reported by kubelet ==="
kubectl -n "${namespace}" describe pod "${pod}" 2>/dev/null \
  | grep -E "Successfully pulled image|Pulling image" || echo "(no pull events — image was already cached on the node)"

# --- vLLM-internal phases ----------------------------------------------------

echo
echo "=== vLLM engine phases (from its own logs) ==="
kubectl -n "${namespace}" logs "${pod}" 2>/dev/null \
  | grep -oE "Loading weights took [0-9.]+ seconds|Model loading took [0-9.]+ GiB and [0-9.]+ seconds|torch.compile took [0-9.]+ s in total|init engine \(.*\) took [0-9.]+ s|Graph capturing finished in [0-9]+ secs" \
  | sed 's/^/  /' || echo "(no matching log lines)"

echo
echo "Note: image pull is usually the largest phase. GKE Image Streaming"
echo "(gcfs_config in infra/modules/platform/gke.tf) targets exactly that phase;"
echo "model load and torch.compile are unaffected by it."

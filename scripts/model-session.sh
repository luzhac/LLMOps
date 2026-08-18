#!/usr/bin/env bash
set -euo pipefail

namespace="${NAMESPACE:-llm-platform}"
action="${1:-}"

show_startup_diagnostics() {
  echo >&2
  echo "vLLM did not become Ready before the timeout." >&2
  echo "Current workload and node state:" >&2
  kubectl -n "${namespace}" get pods,pvc -o wide >&2 || true
  kubectl get nodes -L cloud.google.com/gke-accelerator,cloud.google.com/gke-spot >&2 || true
  echo >&2
  echo "Recent vLLM scheduling/startup events:" >&2
  kubectl -n "${namespace}" describe pod -l app.kubernetes.io/name=vllm >&2 || true
  echo >&2
  echo "If Events show 'GCE out of resources', Spot L4 inventory is unavailable." >&2
  echo "The autoscaler will retry while replicas remain at 1. Run this to stop retrying:" >&2
  echo "  bash scripts/model-session.sh down" >&2
}

case "${action}" in
  up)
    kubectl -n "${namespace}" scale deployment/vllm --replicas=1
    if ! kubectl -n "${namespace}" rollout status deployment/vllm --timeout=30m; then
      show_startup_diagnostics
      exit 1
    fi
    ;;
  down)
    kubectl -n "${namespace}" scale deployment/vllm --replicas=0
    kubectl -n "${namespace}" wait --for=delete pod -l app.kubernetes.io/name=vllm --timeout=10m || true
    echo "vLLM stopped. The cluster autoscaler may take several minutes to remove the L4 node."
    ;;
  *)
    echo "usage: $0 up|down" >&2
    exit 2
    ;;
esac

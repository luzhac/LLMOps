# Operations Runbook

How to bring this platform up, run a session against it, measure it, and take it back down.

Everything here has been executed against a real GCP project. Commands assume a POSIX shell with
`gcloud`, `kubectl`, `helm` and `terraform` installed and a GCP project with billing enabled.

**The GPU is the entire cost of this platform.** It bills whenever a GPU node exists, whether or
not the model is serving. Section 6 is not optional.

---

## 0. Verify local tooling

```bash
bash scripts/preflight.sh
```

Checks for required binaries (including `gke-gcloud-auth-plugin`, which `kubectl` needs for GKE and
which is not installed with `gcloud` by default), confirms an authenticated `gcloud` session, and
prints the active project.

---

## 1. Bootstrap (once per project)

`infra/bootstrap` creates the things that must exist *before* Terraform can manage state remotely:
the state bucket itself, the Workload Identity Federation pool that lets GitHub Actions
authenticate without service account keys, the CI service account, and the billing budget.

```bash
terraform -chdir=infra/bootstrap init
terraform -chdir=infra/bootstrap apply -var="project_id=YOUR_PROJECT_ID"
```

Note the state bucket name in the output — the next step needs it.

**Why the budget lives here, not in the environment stack:** a billing account is a separate IAM
resource from a project. The CI service account has project-level permissions but not billing-level
ones, so a budget defined in the environment stack fails in CI with a 403 while working fine
locally under your own credentials.

---

## 2. Provision infrastructure

Backends use **partial configuration** — the bucket name is supplied at `init` time rather than
committed. Create a local `infra/environments/production/backend.hcl` (gitignored):

```hcl
bucket = "YOUR_STATE_BUCKET"
prefix = "production"
```

Then:

```bash
cd infra/environments/production
terraform init -backend-config=backend.hcl
terraform plan  -input=false -var="project_id=YOUR_PROJECT_ID" -var="admin_cidr=YOUR_IP/32"
terraform apply -input=false -var="project_id=YOUR_PROJECT_ID" -var="admin_cidr=YOUR_IP/32"
```

Always pass `-input=false`. Without it, a missing variable makes Terraform wait silently on an
interactive prompt — in CI that looks identical to a hang.

`admin_cidr` is added to the cluster's master authorized networks. Set it to your own public IP.

Get cluster credentials:

```bash
gcloud container clusters get-credentials trade-balance-llm --region=asia-southeast1
kubectl config current-context
```

The GPU node pool is created with autoscaling `0..1` and **starts with zero nodes**. This is
intentional — no GPU cost is incurred until a workload requests one.

---

## 3. Deploy workloads

Argo CD reconciles both the application stack and the monitoring stack from Git.

```bash
kubectl apply -f platform/argocd/application.yaml
kubectl apply -f platform/argocd/application-monitoring.yaml

kubectl -n argocd get applications -w
```

Both Applications have `selfHeal: true`. **Anything changed with `kubectl apply` or through the
Grafana UI will be reverted on the next sync.** To change configuration, commit to Git.

If an Application will not leave `OutOfSync`, read the operation message rather than guessing:

```bash
kubectl -n argocd get application monitoring -o jsonpath='{.status.operationState.message}{"\n"}'
```

<details>
<summary>Alternative: deploy with Helm directly (no Argo CD)</summary>

```bash
helm upgrade --install trade-balance-llm platform/helm/trade-balance-llm \
  --namespace llm-platform --create-namespace

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  -f platform/monitoring/kube-prometheus-stack-values.yaml
```

Use this to debug the chart in isolation. Do not mix the two paths against the same cluster —
Argo CD will fight your manual changes.
</details>

---

## 4. Run a session

Scale the model up. This triggers GPU node provisioning, so expect several minutes on a cold start:

```bash
bash scripts/model-session.sh up
```

Watch it happen:

```bash
kubectl get nodes -L cloud.google.com/gke-accelerator --watch
kubectl -n llm-platform get pods -w
kubectl -n llm-platform logs -f deployment/vllm
```

Once `READY` shows `1/1`, expose the gateway and send a request:

```bash
kubectl -n llm-platform port-forward svc/api-gateway 8080:8080
```

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "authorization: Bearer $LLM_API_KEY" \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": "Explain KV cache in two sentences."}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

`stream_options.include_usage` makes vLLM emit a final token-count event. Without it there is no
authoritative token count and any tokens/second figure is a client-side estimate.

---

## 5. Measure

### Cold start, decomposed by phase

```bash
bash scripts/measure-startup.sh
```

Reports node provisioning, image pull, container start, model weight load, and CUDA graph capture
separately. A single total is not actionable — you cannot optimise what you have not attributed.

### Throughput and latency

```bash
python loadtest/benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen/Qwen3-8B-AWQ \
  --concurrency 1,2,4,8 \
  --requests 32
```

Results are written to `benchmark-results/`. The benchmark is streaming-aware: it measures time to
first token separately from inter-token latency, because they have different bottlenecks.

Watch the server side at the same time — client-side numbers alone cannot distinguish "the server
is slow" from "the server is queueing":

```bash
kubectl -n llm-platform logs -f deployment/vllm | grep -E "Running|Waiting|GPU KV cache usage"
```

`Waiting > 0` means requests are queued. `Preemption` messages mean the KV cache ran out and vLLM
evicted in-flight sequences — throughput numbers taken during preemption are not comparable.

### Dashboards

```bash
kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80
kubectl -n monitoring get secret monitoring-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d; echo
```

---

## 6. Scale back to zero

**Do this at the end of every session.**

```bash
bash scripts/model-session.sh down
kubectl get nodes -L cloud.google.com/gke-accelerator --watch
```

The GPU VM bills until the **node** disappears, not until the pod does. The autoscaler removes an
idle node after roughly 10 minutes. Watch until it is gone.

The cluster, PVC, NAT gateway and system node pool still cost money after this. For anything longer
than a day of idleness, destroy fully:

```bash
cd infra/environments/production
terraform destroy -input=false -var="project_id=YOUR_PROJECT_ID" -var="admin_cidr=YOUR_IP/32"
```

If `destroy` hangs on a GPU node pool stuck in `PROVISIONING` (which happens when the pool was
created during a regional capacity shortage), delete the pool out of band and re-run:

```bash
gcloud container node-pools delete gpu-pool \
  --cluster=trade-balance-llm --region=asia-southeast1
```

Afterwards, confirm nothing was orphaned:

```bash
gcloud compute instances list
gcloud container clusters list
gcloud storage buckets list
```

---

## 7. Local checks (no cloud resources created)

```bash
python -m pytest app/gateway/tests -q
python -m ruff check app loadtest

terraform -chdir=infra/environments/production fmt -check -recursive
terraform -chdir=infra/environments/production init -backend=false
terraform -chdir=infra/environments/production validate

helm lint platform/helm/trade-balance-llm
helm template trade-balance-llm platform/helm/trade-balance-llm --namespace llm-platform >/dev/null
```

`terraform init -backend=false` validates syntax without touching remote state or credentials,
which is what CI runs on pull requests.

---

## 8. When something is wrong

Work down the stack; do not reinstall everything because a pod says `Pending`.

```text
local shell/tooling → GCP API/IAM/quota → Terraform → Kubernetes scheduling
→ container start → gateway → vLLM → Prometheus scrape → Grafana query
```

| Symptom | Most likely cause | First command |
|---|---|---|
| Every `kubectl` command times out | kubeconfig points at a destroyed cluster | `kubectl config current-context` |
| GPU pod `Pending`, pool won't scale | Read the message — three different root causes (see below) | `kubectl -n llm-platform describe pod <POD>` |
| Argo CD stuck `OutOfSync` | Self-deleting hook Jobs, or CRDs too large for client-side apply | `kubectl -n argocd get application <APP> -o jsonpath='{.status.operationState.message}'` |
| Dashboards empty | CRDs failed to install, or `ServiceMonitor` label selector mismatch | Prometheus UI → Status → Targets |
| Config change had no effect | Argo CD `selfHeal` reverted it | `git log -- <the file you changed>` |
| Grafana login fails | Env vars do not hot-reload; pod predates the current Secret | `kubectl -n monitoring rollout restart deployment/monitoring-grafana` |

**A GPU pod stuck `Pending` has three distinct causes that look identical.** Read the scheduler
message before acting:

| Message contains | Root cause | Action |
|---|---|---|
| `didn't match Pod's node affinity/selector` | Config drift — the pod asks for a GPU type this pool does not have | Fix the `nodeSelector`. **Do not migrate regions.** |
| `ZONE_RESOURCE_POOL_EXHAUSTED` | Regional stockout | Change zone/region, or try Spot |
| `QUOTA_EXCEEDED` | Account quota | Request a quota increase |

Diagnosing capacity specifically:

```bash
bash scripts/diagnose-gpu.sh
```

---

## 9. Changing region or GPU type

The region is written down in more places than Terraform. Before migrating, find them all:

```bash
grep -rn "asia-southeast1\|nvidia-l4\|g2-standard" \
  --include=*.tf --include=*.yaml --include=*.yml .
gh variable list
gh secret list
```

At minimum, all of these must move together: the Terraform environment stack, the Helm
`nodeSelector`, the Artifact Registry location, and the CI region variable. Missing the
`nodeSelector` produces a pod that will never schedule despite the GPU being available; missing the
CI variable pushes images to a registry that no longer exists.

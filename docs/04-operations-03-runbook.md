# GCP 从零部署与运维 Runbook

本手册只保留可执行主流程。真实错误的完整“现象 → 原因 → 修复 → 验证”在 [`04-operations-05-troubleshooting.md`](04-operations-05-troubleshooting.md)；API、参数和指标原理在 [`03-project-engineering-05-api-and-observability.md`](03-project-engineering-05-api-and-observability.md)；性能容量在 [`04-operations-04-benchmarking.md`](04-operations-04-benchmarking.md)；学习标准在 [`03-project-engineering-01-project-learning-objectives.md`](03-project-engineering-01-project-learning-objectives.md)。这样避免同一故障在多个章节重复且内容不一致。

目标环境：WSL、GCP `europe-central2/europe-central2-c`、GKE Standard、常驻 `e2-standard-2` 系统节点、可缩到 0 的 `n1-standard-8 + 1×T4` 按需（非 Spot）节点。第一次部署不需要 CI、GitHub Actions 或 Argo CD。

## 0. 当前项目状态

**2026-08-17 迁移记录**：`europe-west4` 从 2026-08-15 起持续 L4/T4 区域性 stockout（探测了 west1/west2/west3/west4/west6 全部欧洲 region 均无货），已将整个 cluster（非仅 node pool）从 `europe-west4-a` 迁移到 `europe-central2-c`（华沙），GPU 机型从 `g2-standard-8 + L4` 换成 `n1-standard-8 + T4`（华沙不提供 L4）。迁移过程中 `l4-spot` node pool 曾卡在 `PROVISIONING` 状态导致 Terraform destroy 超时失败，需要手动 `gcloud container node-pools delete` 解除后重新 apply；期间 Gateway 有约1小时真实中断。详见故障手册。T4 相比 L4 的真实影响：不支持 bfloat16（自动降级 float16）、不支持 FlashAttention v2（自动切换 TRITON_ATTN backend，推理更慢）。

截至本次记录，Terraform/GKE、Gateway、Qwen3-8B-AWQ/vLLM、认证 SSE 请求均已真实跑通（在华沙 T4 上重新验证过）；模型已可 scale 到 0。容量 benchmark 和 GKE-managed DCGM GPU 指标仍需验收，不能把尚未测得的性能数字写进简历。

## 1. WSL 工具与 Shell 约定

安装 gcloud、`gke-gcloud-auth-plugin`、Terraform、kubectl、Helm、Docker、Python 的步骤见 [`04-operations-01-wsl-tooling-setup.md`](04-operations-01-wsl-tooling-setup.md)。在仓库根目录：

```bash
bash scripts/preflight.sh
```

脚本名是 `preflight.sh`。`>` 是 Bash 等待未闭合输入的续行提示，不是部署仍在运行；误复制 Markdown 反引号时按 `Ctrl+C`。

每个新终端重新设置非敏感环境变量：

```bash
export GCP_PROJECT_ID='project-2759a06c-a804-420c-a8b'
export GCP_REGION='europe-central2'
export GCP_ZONE='europe-central2-c'
```

## 2. 一次性登录、API 与 Billing

CLI 凭据和 Terraform ADC 是两份登录：

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project "$GCP_PROJECT_ID"
gcloud config set billing/quota_project "$GCP_PROJECT_ID"
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"

gcloud services enable serviceusage.googleapis.com compute.googleapis.com \
  container.googleapis.com artifactregistry.googleapis.com iam.googleapis.com \
  monitoring.googleapis.com storage.googleapis.com billingbudgets.googleapis.com \
  cloudbilling.googleapis.com cloudresourcemanager.googleapis.com \
  --project "$GCP_PROJECT_ID"
```

在 Billing Console 确认账户状态。Free Trial 未升级时不能用 GPU；升级后超过欢迎额度会向付款方式收费，不会在 $300 自动停机。Terraform Budget 只报警，不是硬上限。账户使用 GBP 时 `budget_amount=30` 表示 £30。

## 3. GPU quota 与容量

至少确认：

- `GPUS_ALL_REGIONS >= 1`
- `europe-central2` 的 `NVIDIA_T4_GPUS >= 1`（华沙不提供 L4，`accelerator-types list` 会证实这一点）
- Spot 使用 `PREEMPTIBLE_NVIDIA_T4_GPUS >= 1`（当前 `gpu_spot=false`，实际按需计费，不吃这个 quota）
- N1 CPU 至少 8 vCPU

`accelerator-types describe` 返回 T4 只说明型号存在，不说明 quota/库存可用。准确查询命令见故障手册第 3 条。quota 获批也不保证当时有库存；实测方法是直接 `gcloud compute instances create` 一台探测机，成功立刻删除。

## 4. Terraform 变量与基础设施

```bash
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
curl -4 https://ifconfig.me
gcloud billing accounts list
```

在 `terraform.tfvars` 填：

```hcl
project_id         = "project-2759a06c-a804-420c-a8b"
admin_cidr         = "YOUR_CURRENT_PUBLIC_IPV4/32"
region             = "europe-central2"
zone               = "europe-central2-c"
billing_account_id = "YOUR-BILLING-ACCOUNT-ID"
budget_amount       = 30
gpu_spot            = false
```

`203.0.113.10/32` 是文档保留地址，不能照抄；不要使用 `0.0.0.0/0`。tfvars、state、plan 不提交 Git。

```bash
terraform -chdir=infra/terraform fmt -recursive
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform validate
terraform -chdir=infra/terraform plan -out=trade-balance-llm.tfplan
terraform -chdir=infra/terraform apply trade-balance-llm.tfplan
```

典型 15–30 分钟。Terraform 创建：VPC/subnet、Cloud Router/NAT、单区 GKE、系统 node pool、`0..1` 按需 T4 pool（历史上曾用 Spot L4，2026-08-17 因区域性 stockout 迁移到华沙 T4 按需）、Artifact Registry、GCS、node service account/IAM、可选 Billing Budget。它不创建公网 Load Balancer、Ingress、DNS、证书或 GCP API Gateway。

跨 region 迁移（而不是同 region 换 zone）会强制替换 cluster、subnet、router、NAT、Artifact Registry 这几个资源（`terraform plan` 会显示 `delete, create`），不是普通的 in-place update；GKE node pool 的 `location` 和 machine type 都不支持原地改，改这些字段必然触发替换。

Apply 部分失败时保留 state，修复后重新 plan/apply；不要 destroy，也不要重用失败前的旧 plan。Budget 403/400 的本次真实处理见故障手册第 4、5 条。

```bash
gcloud container clusters get-credentials trade-balance-llm \
  --zone "$GCP_ZONE" --project "$GCP_PROJECT_ID"
kubectl get nodes
```

## 5. 构建 Gateway 镜像

```bash
bash scripts/build-gateway.sh
export GATEWAY_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/trade-balance-llm/gateway"
```

可选扫描：

```bash
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image "${GATEWAY_IMAGE}:0.1.0"
```

这是本项目 FastAPI Gateway，不是收费的 GCP API Gateway。

## 6. 安装 Prometheus/Grafana

`helm repo add/update` 只更新本地 Chart 索引；`helm upgrade --install` 才在 GKE 创建资源。

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values platform/monitoring/kube-prometheus-stack-values.yaml

kubectl apply -f platform/monitoring/dashboard-configmap.yaml
```

不要再安装本项目早期使用的通用 `nvidia/dcgm-exporter` Helm release。真实环境已证明其 PriorityClass 和 GKE COS/NVML 路径不兼容；为它开放 privileged/unconfined 权限不是可接受修复。Gateway/vLLM 指标保存在本地 Prometheus/Grafana；GPU 指标后续按 GKE 官方 managed DCGM + Cloud Monitoring 验收，见故障手册第 14 条。

## 7. 安装应用，GPU 先保持关闭

```bash
export LLM_API_KEY="$(openssl rand -hex 32)"
kubectl create namespace llm-platform --dry-run=client -o yaml | kubectl apply -f -
kubectl -n llm-platform create secret generic llm-api-keys \
  --from-literal=gateway-api-key="$LLM_API_KEY"

helm upgrade --install trade-balance-llm platform/helm/trade-balance-llm \
  --namespace llm-platform \
  --set gateway.image.repository="$GATEWAY_IMAGE" \
  --set gateway.image.tag=0.1.0 \
  --set vllm.replicaCount=0

kubectl -n llm-platform get pods,svc,pvc
```

预期：Gateway 最终 `1/1 Running`；两个 Service 都是 ClusterIP/无 External IP；vLLM Service 无 Endpoint；PVC 可能因 `WaitForFirstConsumer` 为 Pending。这时 GPU node pool 为 0。

## 8. 启动模型会话

```bash
date -u
bash scripts/model-session.sh up
kubectl get nodes -L cloud.google.com/gke-accelerator,cloud.google.com/gke-spot
kubectl -n llm-platform rollout status deployment/vllm --timeout=30m
kubectl -n llm-platform get pods,pvc
kubectl -n llm-platform logs deployment/vllm --tail=200
```

首次常需 10–30 分钟：Spot node、镜像、PVC、模型、权重加载、kernel compilation。重点验证日志没有参数解析错误，PVC 为 Bound，vLLM `1/1 Ready`。

## 9. API 冒烟测试

终端 A：

```bash
kubectl -n llm-platform port-forward svc/api-gateway 8080:8080
```

终端 B（新终端必须重新从 Secret 加载 Key）：

```bash
export LLM_API_KEY="$(kubectl -n llm-platform get secret llm-api-keys \
  -o jsonpath='{.data.gateway-api-key}' | base64 --decode)"

curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d '{
    "model":"Qwen/Qwen3-8B-AWQ",
    "messages":[{"role":"user","content":"Explain KV cache in plain English."}],
    "temperature":0,
    "max_tokens":256,
    "stream":true,
    "stream_options":{"include_usage":true}
  }'
```

成功标准：连续 `data:` SSE、usage、最后 `[DONE]`。`finish_reason=length` 是达到输出上限，不是错误。完整参数和 SDK 调用见 API 指南。

## 10. Prometheus 与 Grafana

Grafana：

```bash
kubectl -n monitoring get secret monitoring-grafana \
  -o jsonpath='{.data.admin-password}' | base64 --decode
echo
kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80
```

打开 `http://127.0.0.1:3000`，用户名 `admin`。如果 3000 被占用，使用 `3001:80` 并打开 3001。

Prometheus 排障：

```bash
kubectl -n monitoring port-forward svc/monitoring-prometheus 9090:9090
```

打开 `http://127.0.0.1:9090`：Query/Graph 查 PromQL，Status → Targets 查抓取状态，Alerts 查规则。port-forward 终端必须保持运行；`Ctrl+C` 只关隧道，不停 Pod。

常用查询：

```promql
up{namespace="llm-platform"}
sum(rate(llm_gateway_requests_total[5m]))
sum(rate(vllm:generation_tokens_total[5m]))
sum(vllm:num_requests_running)
sum(vllm:num_requests_waiting)
avg(vllm:kv_cache_usage_perc) * 100
```

## 11. 性能测试

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r loadtest/requirements.txt

python loadtest/benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --api-key "$LLM_API_KEY" \
  --requests 40 \
  --concurrency 1,2,4,8 \
  --max-tokens 256
```

怎样读单请求/aggregate tokens/s、并发和 SLO，见 [`04-operations-04-benchmarking.md`](04-operations-04-benchmarking.md)。未经真实运行，不填写“最多用户数”或 tokens/s。

## 12. 每次结束先停 GPU

```bash
bash scripts/model-session.sh down
kubectl -n llm-platform get deployment vllm
kubectl get nodes -L cloud.google.com/gke-accelerator --watch
```

vLLM 应为 `0/0`；直到 L4 node 消失前 GPU VM 仍计费。系统节点、GKE、NAT 和 PVC 继续收费。超过一天不用建议完整销毁。

## 13. 完整销毁

先保存脱敏 benchmark、dashboard 和 Billing 证据，再执行：

```bash
helm uninstall trade-balance-llm --namespace llm-platform || true
helm uninstall monitoring --namespace monitoring || true
helm uninstall argocd --namespace argocd || true

terraform -chdir=infra/terraform plan -destroy -out=destroy.tfplan
terraform -chdir=infra/terraform apply destroy.tfplan
```

确认无遗留：

```bash
gcloud compute instances list --project "$GCP_PROJECT_ID"
gcloud compute disks list --project "$GCP_PROJECT_ID"
gcloud compute forwarding-rules list --project "$GCP_PROJECT_ID"
gcloud compute addresses list --project "$GCP_PROJECT_ID"
gcloud compute routers list --project "$GCP_PROJECT_ID"
gcloud container clusters list --project "$GCP_PROJECT_ID"
```

API 保持 enabled 本身不收费。不要删除整个 GCP project，除非确认里面没有其他资源。

## 14. 两种部署路径：手工 Helm vs ArgoCD GitOps

两天内首次验收不需要 Argo CD，第7节的手工 `helm upgrade --install` 已经够用。这两种路径**互斥，不能混用**：ArgoCD 开启 `selfHeal` 后，任何手工 `kubectl scale`/`helm upgrade` 改动都会被它在下个 sync 周期自动revert 回 Git 里记录的状态；要切到 GitOps 模式，必须先确认不再手工改集群状态，副本数这类改动一律通过改 Git 里的值 + push 生效。

### 14.1 安装 ArgoCD 本身

```bash
helm repo add argo https://argoproj.github.io/argo-helm
helm repo update argo
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install argocd argo/argo-cd \
  --namespace argocd \
  --values platform/argocd/values.yaml
```

`platform/argocd/values.yaml` 把 controller/server/repo-server/redis 都限制了资源、绑定在 system 节点池，不占 GPU 节点；关闭了 `notifications` 和 `dex`（单人门户不需要 SSO）。

登录信息：

```bash
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo
kubectl -n argocd port-forward svc/argocd-server 8080:443
```

打开 `https://127.0.0.1:8080`，用户名 `admin`。首次登录后按官方建议删除 `argocd-initial-admin-secret`。

### 14.2 接入这个仓库

`platform/argocd/application.yaml` 里两处需要替换成真实值：

- `spec.source.repoURL`：推到 GitHub 后的仓库地址（`CHANGEME_GIT_REPOSITORY_URL`）
- `spec.source.helm.values` 里的 `gateway.image.repository`：已经填成当前实际用的 `europe-central2-docker.pkg.dev/project-2759a06c-a804-420c-a8b/trade-balance-llm/gateway`，跨 region 迁移后记得同步更新这里

```bash
kubectl apply -f platform/argocd/application.yaml
kubectl -n argocd get application trade-balance-llm
```

`syncPolicy.automated` 已经开了 `prune: true, selfHeal: true`——sync 成功后，Git 就是唯一真相来源（single source of truth）。改副本数、镜像 tag 之类，都改 `platform/argocd/application.yaml` 或 Helm values 再 push，不要再手工 `kubectl scale`。

## 故障快速索引

| 现象 | 详细条目 |
|---|---|
| 脚本找不到 / auth plugin 缺失 | 故障手册 1–2 |
| L4 类型存在但不能创建 | 3 |
| Terraform Budget 403/400 | 4–5 |
| Docker 后出现 `>` / Trivy 失败 | 6–7 |
| PVC Pending | 8 |
| vLLM CLI、rollout、`VLLM_PORT` | 9–10 |
| invalid API key | 11 |
| browser/server 时间漂移 | 12 |
| vLLM/Grafana 面板空 | 13 |
| GPU/DCGM 面板空 | 14 |

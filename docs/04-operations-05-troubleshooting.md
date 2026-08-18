# 实际故障手册

本页只记录本项目真实出现过的故障。每条都按“现象 → 原因 → 修复 → 验证”组织。部署顺序在 [`04-operations-03-runbook.md`](04-operations-03-runbook.md)，原理在 [`03-project-engineering-05-api-and-observability.md`](03-project-engineering-05-api-and-observability.md)。

## 诊断总原则

先判断故障在哪一层，不要一看到 `Pending` 或 timeout 就重装全部系统：

```text
本地 Shell/工具 → GCP API/IAM/Quota → Terraform → Kubernetes 调度
→ 容器启动 → Gateway → vLLM → Prometheus 抓取 → Grafana 查询
```

每次保留四类证据：原始错误、相关资源状态、实施的最小修改、修复后的验证命令。

## 1. `preflight.sh: command not found` / `No such file or directory`

**现象**

```text
preflight.sh: command not found
bash: preslight.sh: No such file or directory
```

**原因**：文件名拼错（`preslight`、`prelight`），或当前目录不对。当前目录默认也不在 Linux 的 `PATH` 中。

**修复**

```bash
# 仓库根目录
bash scripts/preflight.sh

# 已经在 scripts 目录
bash preflight.sh
```

**验证**：脚本继续检查命令；不要用 `sudo` 解决路径或拼写问题。

## 2. `missing required command: gke-gcloud-auth-plugin`

**原因**：kubectl 连接 GKE 所需的认证插件未安装。

**修复**：按 [`04-operations-01-wsl-tooling-setup.md`](04-operations-01-wsl-tooling-setup.md) 安装 Google Cloud CLI 的 GKE auth plugin，然后重新打开 Shell 或刷新 `PATH`。

**验证**

```bash
gke-gcloud-auth-plugin --version
bash scripts/preflight.sh
```

## 3. L4 类型存在，但项目不能创建 GPU

**现象**：`accelerator-types describe` 能返回 `nvidia-l4`，但全局查询显示：

```text
GPUS_ALL_REGIONS,0.0,0.0
NVIDIA_L4_GPUS,1.0,0.0
```

**原因**：型号存在只代表该 zone 提供 L4；实际创建同时受全局 GPU quota、区域 L4 quota、Spot quota 和当时库存约束。区域为 1、全局为 0 时仍不能使用。

**修复**：在 IAM & Admin → Quotas 中按 Compute Engine API 和准确 quota 名称申请；Console 的通用搜索框不是每次都能用 `gpu` 找到。CLI 用 `--flatten` 后再过滤，不能把 `--filter` 直接传给旧版 `project-info describe`。

**验证**

```bash
gcloud compute project-info describe --project "$GCP_PROJECT_ID" \
  --flatten='quotas[]' \
  --format='csv[no-heading](quotas.metric,quotas.limit,quotas.usage)' \
  | grep -E 'GPUS_ALL_REGIONS'

gcloud compute regions describe europe-west4 --project "$GCP_PROJECT_ID" \
  --flatten='quotas[]' \
  --format='csv[no-heading](quotas.metric,quotas.limit,quotas.usage)' \
  | grep 'NVIDIA_L4_GPUS'
```

## 4. Terraform 已建好 GKE，最后 Budget API 403

**现象**

```text
The billingbudgets.googleapis.com API requires a quota project
reason: SERVICE_DISABLED
```

**原因**：Terraform 使用 Application Default Credentials（ADC）；gcloud CLI 的 project 设置不会自动成为 ADC quota project，或者 Billing Budgets API 未启用。

**修复**

```bash
gcloud config set project "$GCP_PROJECT_ID"
gcloud config set billing/quota_project "$GCP_PROJECT_ID"
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
gcloud services enable serviceusage.googleapis.com billingbudgets.googleapis.com \
  cloudbilling.googleapis.com cloudresourcemanager.googleapis.com \
  --project "$GCP_PROJECT_ID"

terraform -chdir=infra/terraform plan -out=trade-balance-llm.tfplan
terraform -chdir=infra/terraform apply trade-balance-llm.tfplan
```

API 提示 `Would you like to enable and retry (y/N)?` 时输入 `y`，表示同一条操作先启用缺失 API，再重试；不是重新创建整个集群。

**验证**：新的 plan 通常只剩 Budget；已经成功的 GKE 和 node pool 会从 state 复用。不要 destroy，也不要继续使用失败前的旧 plan。

## 5. Budget API 400 `invalid argument`

**原因**：Budget 金额货币与 Billing Account 货币不一致。这个账户以 GBP 结算，硬编码 USD 会被 API 拒绝。

**修复**：Terraform 通过 `data.google_billing_account` 读取账户货币，并让 Budget 使用相同币种；`budget_amount = 30` 在本账户表示 £30。

**验证**

```bash
terraform -chdir=infra/terraform plan
```

检查计划中的 Budget currency 与 Billing Account 一致。Budget 只报警，不会自动停机或封顶消费。

## 6. Docker push 已成功，但终端只显示 `>`

**原因**：`>` 是 Bash 续行提示符，通常因为复制了 Markdown 的三个反引号、未闭合引号或行尾 `\\`；不是 Docker 仍在运行。

**修复**：按 `Ctrl+C` 取消未完成输入，然后只复制代码框内部命令。

**验证**：push 输出已经有 digest 时镜像已上传；`Ctrl+C` 不会撤销已经完成的 build/push。

## 7. Trivy 找不到镜像、Docker socket 和远程认证

**现象**

```text
failed to connect to the docker API at /var/run/docker.sock
containerd socket not found
remote ... Unauthenticated request
```

**原因**：Trivy 临时容器没有看到宿主 Docker socket；本地运行时失败后又尝试匿名从私有 Artifact Registry 拉取。

**修复**

```bash
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image "${GATEWAY_IMAGE}:0.1.0"
```

**验证**：日志进入 vulnerability scan，而不是 `unable to find the specified image`。Docker socket 权限很高，只挂载给可信扫描器。

## 8. PVC 显示 `Pending`

**现象**：`vllm-model-cache` 在 vLLM 副本为 0 时 Pending。

**原因**：`standard-rwo` 使用 `WaitForFirstConsumer`。没有 vLLM Pod 时，GKE 不知道磁盘应放在哪个 zone，因此暂不创建 Persistent Disk。这是等待状态，不等于故障。

**诊断与验证**

```bash
kubectl -n llm-platform describe pvc vllm-model-cache
```

- `waiting for first consumer` 且 vLLM 为 0：正常。
- vLLM 启动后先 `Provisioning`、随后 `Bound`：正常。
- 反复 `ProvisioningFailed`、quota、permission 或 CSI 错误：才是故障。

## 9. vLLM rollout timeout，CLI 参数不兼容

**现象**

```text
vllm: error: unrecognized arguments: --disable-log-requests
```

并且 `--model` 出现即将移除的 warning。

**原因**：Chart 参数来自旧 vLLM 版本，而镜像固定为 v0.26.0。

**修复**：模型 ID 改为 `vllm serve` 的位置参数；日志开关改为 v0.26 支持的 `--no-enable-log-requests`。

**验证**

```bash
helm template trade-balance-llm platform/helm/trade-balance-llm --namespace llm-platform
kubectl -n llm-platform rollout status deployment/vllm --timeout=30m
kubectl -n llm-platform logs deployment/vllm --tail=200
```

warning 是弃用提醒，不是当前失败；`unrecognized arguments` 才会让进程退出。

## 10. vLLM 把 `VLLM_PORT` 读成 `tcp://...`

**原因**：Kubernetes 自动注入同名 Service 环境变量，`vllm` Service 生成的 `VLLM_PORT=tcp://...` 与 vLLM 自己期望的数值变量冲突。

**修复**：vLLM Pod 设置 `enableServiceLinks: false`，仍通过 DNS `http://vllm:8000` 访问 Service。

**验证**：Pod 不再因端口解析退出，`/health` 和 `/metrics` 返回 200。

## 11. curl 返回 `{"detail":"invalid API key"}`

**原因**：新终端没有继承旧终端的 `export LLM_API_KEY=...`，或 Secret 已轮换而 Gateway 仍持有旧环境变量。

**修复**

```bash
export LLM_API_KEY="$(kubectl -n llm-platform get secret llm-api-keys \
  -o jsonpath='{.data.gateway-api-key}' | base64 --decode)"
test -n "$LLM_API_KEY" && echo 'LLM_API_KEY is set'
```

只有 Secret 本身被更新时才需要：

```bash
kubectl -n llm-platform rollout restart deployment/api-gateway
kubectl -n llm-platform rollout status deployment/api-gateway
```

**验证**：相同 curl 返回 SSE 数据和最后的 `[DONE]`。不要打印、截图或提交真实 Key。

## 12. Grafana/Prometheus 提示 browser 与 server 时间差 1 分钟

**原因**：Windows/WSL 本地时钟与 GKE 节点 UTC 时间漂移，不是 Prometheus 数据库损坏。

**修复**：先同步 Windows 时间，再执行 `wsl --shutdown` 重新进入 WSL；同时确认 `date -u`。页面重新加载后再判断查询时间窗口。

**验证**：告警消失；最近 5 分钟的查询不再整体错位。

## 13. vLLM、TTFT、tokens/s 面板为空

按顺序检查：

```bash
kubectl -n llm-platform get deployment,svc,endpoints
kubectl -n llm-platform get servicemonitor llm-platform
kubectl -n monitoring port-forward svc/monitoring-prometheus 9090:9090
```

在 Prometheus 查询：

```promql
up{namespace="llm-platform"}
vllm:num_requests_running
rate(vllm:generation_tokens_total[5m])
```

**常见原因**：vLLM 已 scale 到 0、最近 5 分钟没有请求、Service 没 Endpoint、target DOWN，或 dashboard 用了与当前 vLLM 版本不一致的指标名。vLLM 为 0 时这些面板为空是正确行为，不应填成 0。

## 14. GPU utilization / framebuffer 面板为空

本次真实诊断得到两个连续错误：

```text
Desired Number of Nodes Scheduled: 1
Current Number of Nodes Scheduled: 0
Error creating: insufficient quota to match these scopes:
[{PriorityClass In [system-node-critical system-cluster-critical]}]
```

去掉 Chart 默认的 `system-node-critical` 后，Pod 可以创建，但立即：

```text
Starting dcgm-exporter
exit status 1
CrashLoopBackOff
```

**根因**：通用 NVIDIA Chart 的默认 PriorityClass 不适用于这个 GKE namespace；进一步地，自管 Exporter 在该 GKE COS 节点上未获得受支持的 NVML/驱动接口。Prometheus 没有任何 `DCGM_FI_DEV_*` 序列，所以 Grafana 当然为空。

**安全处置**：已卸载这个 CrashLooping release。不要为了显示指标给 Exporter 加 `privileged: true`、`seccomp: Unconfined` 和任意权限。GKE 1.30.1+ 的推荐路径是启用 GKE-managed DCGM 与 Managed Service for Prometheus，再在 Cloud Monitoring 查看 GPU 指标。当前本地 Grafana dashboard 只展示 Gateway/vLLM 指标；GPU 指标属于后续官方集成验收项。

**验证成本停止**

```bash
kubectl -n llm-platform get deployment vllm
kubectl get nodes -L cloud.google.com/gke-accelerator --watch
```

vLLM 应为 `0/0`，随后 L4 节点应由 autoscaler 删除；直到节点消失前仍有 GPU VM 费用。

## 15. `loadtest/benchmark.py` 报 `ModuleNotFoundError: No module named 'httpx'`

**现象**

```text
Traceback (most recent call last):
  File "loadtest/benchmark.py", line 16, in <module>
    import httpx
ModuleNotFoundError: No module named 'httpx'
```

**原因**：直接用系统 `python3` 跑的，没激活项目的 `.venv`；`httpx` 只装在 `.venv` 里。

**修复**

```bash
python3 -m venv .venv        # 如果还没建过
. .venv/bin/activate
pip install -r loadtest/requirements.txt
python loadtest/benchmark.py ...
```

**验证**：`which python` 应该指向 `.venv/bin/python`，不是系统 Python；`pip show httpx` 能查到版本。每开一个新终端都要重新 `. .venv/bin/activate`，这个不会持久生效。

## 16. Prometheus 抓不到 Gateway/vLLM，`ServiceMonitor` 存在但 Targets 里没有

**现象**：`kubectl get servicemonitor` 能看到对象，但 Prometheus 的 `Status → Targets`（或 `/api/v1/targets`）里没有 `api-gateway`/`vllm` 这两个 job，Grafana 全部面板空白。

**原因**：kube-prometheus-stack 的 Prometheus CR 默认只扫描带 `release: <helm release名>` label 的 ServiceMonitor（`serviceMonitorSelector: {matchLabels: {release: xxx}}`）。本项目 Chart 的 [`platform/helm/trade-balance-llm/templates/servicemonitors.yaml`](../platform/helm/trade-balance-llm/templates/servicemonitors.yaml) 里这个 label 写死是 `release: monitoring`；如果 `helm install` kube-prometheus-stack 时用了别的 release 名字（比如 `kube-prometheus-stack`），两边对不上，ServiceMonitor 对象存在但永远不会被发现。

**修复**：kube-prometheus-stack 的 release 名必须叫 `monitoring`，跟 Chart 里写死的 label 对齐：

```bash
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values platform/monitoring/kube-prometheus-stack-values.yaml
```

**验证**

```bash
kubectl -n monitoring port-forward svc/monitoring-prometheus 9090:9090
curl -s http://127.0.0.1:9090/api/v1/targets | grep -A2 '"job":"vllm"'
```

`health` 字段应为 `up`。

## 17. Terraform 跨 region 迁移时，GPU node pool 卡在 `PROVISIONING`，destroy 超时失败

**现象**：`terraform apply` 在销毁旧 node pool 时卡住不动，最终报错：

```text
Error: NodePool "...nodePools/l4-spot" has state "PROVISIONING" with message ""
```

Terraform state 里这个资源处于"正在被替换但没完成"的中间态；此时对应的 `system` node pool 可能已经被删掉但新的还没建，Gateway 会真的中断。

**原因**：region 迁移会强制替换整个 cluster/node pool（location 字段不可变）；如果旧的 GPU node pool 当时正因为 autoscaler 反复重试 scale-up（比如遇到 stockout）而处于活跃的 in-flight 状态，GKE API 会拒绝在这个时候删除它。完整背景见 [`04-operations-06-incident-spot-l4-capacity.md`](04-operations-06-incident-spot-l4-capacity.md)。

**修复**：跳出 Terraform，直接用 gcloud 强制删除卡住的 node pool，再重新 `terraform plan`/`apply` 一次，Terraform 会检测到资源已经不存在，正常继续完成剩余的创建：

```bash
gcloud container node-pools delete l4-spot --cluster=trade-balance-llm --zone=<旧zone> --quiet
terraform -chdir=infra/terraform plan -out=fix.tfplan
terraform -chdir=infra/terraform apply fix.tfplan
```

**验证**：`gcloud container node-pools describe l4-spot ...` 报 404（说明真的删干净了）之后再重新 apply；apply 完成后检查 `kubectl -n llm-platform get pods` 里 Gateway 恢复 `1/1 Running`。

## 18. Windows/WSL 下 git 索引大小写和磁盘实际大小写不一致，`git status` 看不到真实改动

**现象**：明明改了文件（比如整段重写了 `docs/04-operations-03-runbook.md`），`git status`/`git diff` 按磁盘上真实的文件夹名（`TradeBalanceLlm/...`）查询却显示"无变化"；但换成全小写路径（`tradebalancellm/...`）查询能看到真实 diff。

**原因**：Windows/NTFS 默认大小写不敏感但保留原始大小写显示——如果曾经有一次 `git add`/脚本用了跟磁盘真实文件夹名大小写不一致的路径（比如打成全小写），Windows 会当作"同一个文件夹"直接放行、不报错，但 git 索引里记录的是当时敲的那个精确字符串。此后任何按磁盘真实大小写发起的 git 查询，都会因为索引里存的是另一套大小写而查不到对应记录。纯 Linux（大小写敏感）不会出现这个问题，这是 Windows 文件系统特有的坑。

**修复**：不要直接 `git mv 大写路径 小写路径`（如果当前有进程的工作目录正卡在这个文件夹里，两次 rename 都会因为"目录被占用"失败）。改用只操作索引、不涉及文件系统 rename 的方式：

```bash
git rm -r --cached tradebalancellm      # 只删索引记录，不动磁盘文件
git add TradeBalanceLlm                  # 用磁盘上真实的大小写重新加入索引
git status --short TradeBalanceLlm/      # 应该能看到所有真实改动了
```

**验证**：`git ls-files | grep -c '^tradebalancellm/'` 应该是 0（没有小写残留）；`git status` 按真实大小写路径查询能看到预期的全部改动文件数。

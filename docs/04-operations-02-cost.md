# GCP 成本模型与重要警告 / GCP cost model and hard warnings

价格会随区域、货币和结算协议变化；执行 Terraform apply 前必须核对控制台估价，公式比虚假的精确数字更重要。

Prices change by region, currency and billing agreement. The numbers below are
planning estimates as of 14 August 2026; verify the GCP estimate shown before
Terraform apply. The formulas matter more than false precision.

## 免费试用 GPU 限制 / Free-trial GPU restriction

非付费 Free Trial 账户不能添加 GPU 或申请 GPU 配额；升级后剩余额度仍保留到原到期日，但超过额度会扣款。

GCP's non-billable Free Trial account cannot add GPUs and cannot request quota
increases. To use an L4 you normally must click **Activate** and convert the
billing account to paid. Google states that the unused $300 welcome credit
continues until the original 90-day expiry, but the payment method is charged
if spend exceeds remaining credit. This is why the repository defaults to zero
GPU replicas and creates budget alerts.

## 选定机器 / Selected machine

GCP 不提供 RTX 3090 VM；g2-standard-8 包含一张 24 GB NVIDIA L4、8 vCPU 和 32 GiB 内存。

GCP does not offer an RTX 3090 VM. `g2-standard-8` includes:

- one NVIDIA L4, 24 GB VRAM;
- eight vCPUs;
- 32 GiB system memory.

Public price trackers and Google's base price page put it at approximately:

- europe-west4 on-demand: **about $0.90/hour**;
- europe-west4 Spot: **about $0.47/hour**.

Spot is suitable for a portfolio test but can be interrupted and may be
temporarily unavailable. Always confirm the console estimate.

## 其他持续计费资源 / Other recurring resources

除 GPU 外，GKE 管理费、系统节点、Cloud NAT、磁盘、PVC、Artifact Registry 和 GCS 也可能持续计费。

| Resource | Rough planning rate | Control |
|---|---:|---|
| GKE zonal cluster management | $0.10/hour | Monthly $74.40 GKE free-tier credit often offsets one zonal cluster |
| `e2-standard-2` system node | roughly $0.07-$0.10/hour in Europe | Destroy cluster between work periods if not needed |
| Cloud NAT gateway | roughly $0.045/hour plus processed data | Exists while infrastructure exists |
| 50 GB system + 100 GB GPU disks | region-dependent, several dollars/month | GPU disk/node disappears with pool; model PVC remains |
| 30 GiB model-cache PVC | roughly low single digits/month | Delete PVC/cluster after evidence captured |
| Artifact Registry/GCS | pennies at this scale | 30-day bucket lifecycle |
| External load balancer | avoided | ClusterIP plus port-forward by default |

The system node and NAT are the easy costs to forget because they continue when
the model is scaled to zero.

## 作品集会话示例 / Example portfolio sessions

安全计划是约 30-60 个有测量记录的 GPU 小时，而不是让端点永久在线。

Assume Spot L4 `$0.47/h`, system/NAT/storage overhead `$0.13/h` while the cluster
exists, and GKE management fee offset by the monthly credit.

| Usage pattern | GPU | Platform overhead | Approximate total |
|---|---:|---:|---:|
| One 6-hour test day | $2.82 | $0.78 | $3.60 |
| Five 8-hour test days, cluster destroyed between days | $18.80 | $5.20 | $24.00 |
| 50 GPU hours across one continuously running week | $23.50 | $21.84 | $45.34 |
| Spot L4 running 24/7 for 30 days | about $350 | $90+ | **Over the $300 credit** |
| On-demand L4 running 24/7 for 30 days | about $670 | $90+ | **Far over budget** |

Therefore the safe plan is approximately 30-60 measured GPU hours, not a
permanent endpoint.

## 每百万输出 Token 成本 / Cost per million output tokens

用每小时平台成本、总输出 Token/秒和有效利用率计算单位成本，必须使用实测数据替换示例值。

Let:

- `H` = total platform dollars per active hour;
- `T` = measured aggregate output tokens/second at the chosen concurrency;
- `U` = useful utilisation fraction, such as 0.8.

Then:

```text
cost_per_1M_output_tokens = H / (T * 3600 * U) * 1,000,000
```

Example only: if active cost is `$0.60/h`, measured aggregate throughput is 80
output tokens/s, and useful utilisation is 80%, the compute cost is about
`$2.60 per 1M output tokens`. This excludes input-prefill cost, failed requests,
network, engineering time and idle warm capacity. Replace 80 with the measured
benchmark; never put this example in the CV as a result.

## 已包含的成本控制 / Cost controls included

项目包含预算阈值、Spot GPU、0..1 节点池、默认零副本、无公网负载均衡、保留策略和明确销毁步骤；预算只告警，不会停止资源。

- Billing budget thresholds at 25%, 50%, 80% and 100%.
- Spot L4 and GPU pool `min=0/max=1`.
- Model Deployment default `replicas=0`.
- No external load balancer.
- Model cache and result retention policies.
- Explicit model shutdown and Terraform destruction steps.

Budgets send alerts; they do **not** stop resources. A hard-stop automation would
need Pub/Sub + function/service logic and carries the risk of deleting or
interrupting workloads. It is not enabled automatically.

## 权威参考资料 / Authoritative references

以下官方链接用于核对免费计划、GKE 费用、G2/L4 规格和最新价格。

- GCP Free Program and GPU restriction:
  https://docs.cloud.google.com/free/docs/free-cloud-features
- GKE fee and $74.40 monthly credit:
  https://cloud.google.com/kubernetes-engine/pricing
- G2/L4 machine specification:
  https://docs.cloud.google.com/compute/docs/gpus
- Current pricing calculator:
  https://cloud.google.com/products/calculator


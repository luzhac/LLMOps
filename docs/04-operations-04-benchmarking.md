# 性能基准与验收

## 要回答的问题

这轮测试不是为了得到一个孤立的“每秒多少 token”，而是回答：

1. 单个用户在热态能得到多少输出 tokens/s？
2. 整个平台在不同并发下能产生多少 aggregate tokens/s？
3. 在给定 SLO 下每秒能完成多少请求、同时支持多少活跃请求？
4. 从哪个并发开始 TTFT、排队、429 或失败率明显恶化？
5. 这些数字对应什么模型、量化、context、GPU 和输出长度？

“最多支持多少用户”没有脱离业务模型的固定答案。一个用户每分钟问一次和持续生成 2,000 token 完全不同。项目应报告：**在固定 prompt/output、并发和 SLO 下的请求吞吐与活跃并发**。

## 当前模型精度/量化

模型 ID 是 `Qwen/Qwen3-8B-AWQ`：AWQ 通常表示 4-bit weight-only quantization。它主要把模型权重压到约 4 bit；activation、accumulation 和 KV cache 会使用其他 dtype。因此正确说法是“AWQ 4-bit 权重量化模型”，不要说“所有计算和显存都是 INT4”。

运行时确认：

```bash
kubectl -n llm-platform logs deployment/vllm --tail=300 \
  | grep -Ei 'model|quant|dtype|weight|kv cache|gpu memory'
```

把实际日志与 Helm values 一起保存；仅凭模型名不能替代运行证据。

## 指标口径

| 指标 | 计算/来源 | 回答什么问题 |
|---|---|---|
| TTFT | 请求开始到第一个内容 token | 用户多久看到开始回答；受 queue + prefill 影响。 |
| E2E latency | 请求开始到 `[DONE]` | 一个请求总共多久完成。 |
| per-request output tokens/s | completion tokens / 首 token 后生成时间 | 单个用户的 decode 速度。 |
| aggregate output tokens/s | 全部 completion tokens / 场景 wall time | 整个平台总输出吞吐。 |
| request throughput/s | 成功请求数 / 场景 wall time | 在固定长度下每秒完成多少请求。 |
| vLLM generation tokens/s | `rate(vllm:generation_tokens_total[5m])` | 服务器侧总 token 速率，用来交叉验证。 |
| running/waiting | vLLM gauge | 是否开始排队、调度是否饱和。 |
| KV cache usage | vLLM gauge | context/concurrency 的 KV 容量压力。 |

## 每次测试前固定变量

记录：zone、`g2-standard-8`、L4 数量、Spot/按需、模型 ID/revision、vLLM image、AWQ、`maxModelLen`、`maxNumSeqs`、GPU memory utilization、prompt、输出上限、冷/热缓存。不同输出长度的结果不能直接横向比较。

## 0. 成本与状态检查

```bash
bash scripts/model-session.sh up
kubectl -n llm-platform rollout status deployment/vllm --timeout=30m
kubectl -n llm-platform get pods,pvc
kubectl get nodes -L cloud.google.com/gke-accelerator,cloud.google.com/gke-spot
```

测试结束必须执行文末 down。首次启动或模型缓存为空时不要把下载/编译时间算进热态推理性能。

## 1. 冒烟与预热

先验证 `/v1/models`、一次非流式、一次流式、错误 API Key=401。然后用相同 prompt 发 2–5 个预热请求，排除首次 kernel/cache 行为。

## 2. 单用户基线

终端 A：

```bash
kubectl -n llm-platform port-forward svc/api-gateway 8080:8080
```

终端 B：

```bash
export LLM_API_KEY="$(kubectl -n llm-platform get secret llm-api-keys \
  -o jsonpath='{.data.gateway-api-key}' | base64 --decode)"

python3 -m venv .venv
. .venv/bin/activate
pip install -r loadtest/requirements.txt

python loadtest/benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --api-key "$LLM_API_KEY" \
  --requests 30 \
  --concurrency 1 \
  --max-tokens 256
```

保存 JSON 中 median/p95 TTFT、latency、per-request output tokens/s。

## 3. 并发阶梯

当前 Gateway 和 vLLM 都将并发上限设为 8，所以正式容量阶梯先测 `1,2,4,8`：

```bash
python loadtest/benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --api-key "$LLM_API_KEY" \
  --requests 40 \
  --concurrency 1,2,4,8 \
  --max-tokens 256
```

脚本为 closed-loop、批次式（batch）workload，不是滑动窗口：`concurrency=N, requests=M` 会分成 `M/N` 批，**每批同时发出 N 个请求，等这一批全部完成才发下一批**，不是"跑完一个立刻补一个"。这跟按固定速率持续发送的 open-loop 压测（比如几分钟内发几百个请求那种）关注点不同：这里衡量的是"给定并发数下单个请求的真实体验"，不是"系统整体能扛多大流量"。

当前 `concurrency=8` 这个上限同时来自两处独立配置（现在恰好都是8，不是巧合以外的强制关联）：Gateway 自己的信号量（`gateway.maxConcurrentRequests`，超过直接 429，没有排队）和 vLLM 引擎自己的 `vllm.model.maxNumSeqs`。调整并发上限必须两边一起改，只改一边不会生效。

脚本输出：

- `request_throughput_per_second`
- `aggregate_output_tokens_per_second`
- TTFT median/p95
- latency median/p95
- per-request tokens/s median/p95
- success/failure 和每个原始请求结果

并发 8 是配置上限，不自动等于“8 个用户就是容量”。容量点应是同时满足：失败率为 0、p95 TTFT/E2E 在你声明的 SLO 内、waiting 不持续积压、KV cache 有余量。

## 4. 过载行为

要证明 Gateway 的保护工作，可单独用并发 12/16 做短测试。因为 `maxConcurrentRequests=8`，预期一部分请求得到受控 429，而不是把 GPU/Pod 打崩。不要把这轮 429 混入正常容量数字。

```bash
python loadtest/benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --api-key "$LLM_API_KEY" \
  --requests 32 \
  --concurrency 12,16 \
  --max-tokens 256
```

### 4.1 找真实饱和点（临时调高配置上限）

`--concurrency 1,2,4,8` 测出来的曲线如果到8还在接近线性上升（aggregate tokens/s 没有走平），说明还没测到硬件真实上限，只是测到了配置上限。要找真实拐点，两处上限要一起临时调高再重测：

```bash
helm upgrade trade-balance-llm platform/helm/trade-balance-llm \
  --namespace llm-platform --reuse-values \
  --set gateway.maxConcurrentRequests=16 \
  --set vllm.model.maxNumSeqs=16

kubectl -n llm-platform rollout status deployment/vllm --timeout=10m
kubectl -n llm-platform rollout status deployment/api-gateway --timeout=5m

python loadtest/benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --api-key "$LLM_API_KEY" \
  --requests 48 \
  --concurrency 8,16,24 \
  --max-tokens 256
```

判断"真实饱和"的标志：aggregate tokens/s 开始走平甚至下降、`vllm:num_requests_waiting` 持续不为零、`vllm:kv_cache_usage_perc` 逼近 100%、或者 p95 TTFT 明显恶化（比如从 0.3s 跳到 2s+）。测完记得把两个值改回 8（或你验证过的稳定值），不要让临时调试配置留在生产。

## 5. 同时看运行指标

Grafana 看 Gateway RPS/p95、vLLM generation tokens/s、TTFT/TPOT、running/waiting、KV cache。Prometheus 可直接查询：

```promql
sum(rate(vllm:generation_tokens_total[5m]))
sum(vllm:num_requests_running)
sum(vllm:num_requests_waiting)
avg(vllm:kv_cache_usage_perc) * 100
```

负载脚本的 aggregate tokens/s 是从客户端 usage 计算的整个场景平均值；Prometheus 是滑动 5 分钟的服务端 rate，两者窗口不同，不要求小数完全相等，但趋势应一致。

GPU utilization/framebuffer 使用 GKE-managed DCGM 的 Cloud Monitoring 视图。本项目不再依赖已验证不兼容的自管 DCGM Helm Chart。

## 6. 怎么找到“最大值”

先画表，不要只挑最高数字：

| concurrency | success % | p95 TTFT | p95 E2E | req/s | aggregate tok/s | waiting | KV % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 4 | | | | | | | |
| 8 | | | | | | | |

典型曲线是 aggregate tokens/s 先上升再趋平，而 TTFT/单请求速度变差。选“最大可用并发”时必须先定义例如 p95 TTFT < 2 s、失败率 < 1% 的 SLO；没有 SLO，只能说测试到的最大配置，不能说支持的最大用户数。

## 7. 后续单变量实验

一次只改一个变量并重新跑同一套 prompt：

- `maxNumSeqs` 4 vs 8；
- `maxModelLen` 4096 vs 8192；
- prefix caching on/off（重复长前缀请求）；
- 128 vs 256 vs 512 output tokens；
- AWQ vs BF16/FP8，前提是显存、安全余量和硬件支持允许。

同时保留质量样本。量化对比如果只有速度没有回答质量检查，是不完整结论。

## 8. 冷启动和恢复

分别记录：发出 scale up、GPU node Ready、Pod scheduled、PVC bound、模型权重加载、kernel compilation 完成、readiness 通过。冷缓存和热 PVC 缓存至少各一次。

再验证：Pod 重建、vLLM=0 时 Gateway 503、并发超限 429、Spot 驱逐（若自然发生）和重新就绪时间。

## 验收清单

- [x] Terraform/GKE/Helm 已真实部署。
- [x] 认证流式请求成功并返回 usage。
- [x] vLLM scale-to-zero 已执行。
- [ ] 固定 revision 与实际 dtype/quantization 日志已保存。
- [ ] 冷/热启动时间已记录。
- [ ] 并发 1/2/4/8、每档至少 30 个请求已保存。
- [ ] 客户端与 Prometheus tokens/s 趋势已交叉验证。
- [ ] Gateway/vLLM dashboard 截图已保存。
- [ ] GKE-managed DCGM GPU 指标证据已保存。
- [ ] 401、429、上游不可用和一次恢复测试已保存。
- [ ] Billing 实际 session cost 已保存。
- [ ] 销毁后无 orphan resources。

## 结束测试

```bash
bash scripts/model-session.sh down
kubectl get nodes -L cloud.google.com/gke-accelerator --watch
```

直到 L4 节点消失前 GPU VM 仍计费。模型为 0 后 PVC、系统节点、NAT 和集群仍可能收费；超过一天不用执行完整 destroy。

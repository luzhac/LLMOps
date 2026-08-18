# Incident: Spot L4 scale-up failed with `GCE out of resources`

## 现象

关闭模型后再次执行：

```bash
bash scripts/model-session.sh up
```

等待 30 分钟后得到：

```text
Waiting for deployment "vllm" rollout to finish...
error: timed out waiting for the condition
```

## 诊断

```bash
kubectl -n llm-platform get pods,pvc -o wide
kubectl -n llm-platform describe pod -l app.kubernetes.io/name=vllm
kubectl get nodes -L cloud.google.com/gke-accelerator,cloud.google.com/gke-spot
```

本次实际状态：PVC 为 `Bound`，vLLM Pod 为 `Pending`、`Node: <none>`，只有系统节点。Events 明确显示：

```text
TriggeredScaleUp: ... 0->1
FailedScaleUp: GCE out of resources
NotTriggerScaleUp: 1 in backoff after failed scale-up
```

## 根因

`europe-west4-a` 当时没有可分配的 Spot `g2-standard-8 + NVIDIA L4` 库存。Quota 已获批只代表允许使用，不保证 Spot 实时容量。rollout timeout 是等待结果，不是根因。

这个故障发生在节点创建之前，所以与 warm model cache、PVC、vLLM 参数、模型加载和 readiness probe 无关。

## 安全处理

如果可以等待，保持副本为 1，cluster autoscaler 会在 backoff 后重试。必须保持关注，因为库存出现后节点会自动创建并开始收费。

如果暂时不再等待：

```bash
bash scripts/model-session.sh down
```

确认：

```bash
kubectl -n llm-platform get deployment vllm
kubectl get nodes -L cloud.google.com/gke-accelerator
```

vLLM 应为 `0/0`，并且没有 L4 节点。

不要为了立即成功直接把 Spot 改成按需；按需更贵，而且仍不保证库存。如果确实要改按需，必须先重新核算费用、审查 Terraform plan，并接受 node pool 可能被替换。迁移 zone 还涉及 zonal GKE 和已绑定 zonal PVC，不是一次简单的参数修改。

## 预防和改进

- `model-session.sh` 在 rollout timeout 后自动打印 Pod/PVC/node 和 describe Events。
- benchmark 时预留 Spot 获取时间，不把节点库存等待算作模型加载时间。
- 记录四个时间点：请求 scale-up、首次 FailedScaleUp、节点 Ready、vLLM Ready。
- 不使用时立即 scale to zero，避免副本在后台等待并突然产生 GPU 费用。

## 后续更新（2026-08-17）：区域性 stockout 升级为跨 region 迁移

上面这次是单点、可等待的 stockout。2026-08-17 遇到的是**持续多天、全 region 无货**（探测了 europe-west1/2/3/4/6 全部欧洲 region 均无 L4/T4 库存），"等 backoff 后重试"这个策略失效，最终执行了整个 cluster 从 `europe-west4` 迁移到 `europe-central2`（华沙）+ GPU 型号从 L4 换成 T4。完整过程和踩的坑见 runbook 第0节。

**这次事件暴露的问题，是"GPU 是否随时可扩容"这个假设本身不成立**——不是配置或代码问题，是云厂商当时手上有没有货，这个变量不受我们控制。真实企业应对紧俏 GPU 型号容量问题的常见手段（不是本项目已实现，是行业实践记录）：

1. **预留容量（Committed Use Discount / Capacity Reservation）**：提前签约保证一批容量，不跟公共 on-demand 池抢；L4 一年期 commitment 比 on-demand 便宜约37%，代价是失去按需灵活性。
2. **多 region/多云预置**：本次迁移是事后被动应对；成熟做法是提前在多个 region（甚至多个云厂商）保留小额兜底容量，出问题时分钟级切换而不是像这次一样现场排查+新建整套基础设施。
3. **CPU/内存类资源的水平自动扩缩容，对稀缺 GPU SKU 不直接适用**——常规 HPA/cluster-autoscaler 解决"要不要多开一台"，解决不了"这个型号全球没货"；这次的 `total_max_node_count` autoscaler 全程都在正确工作，只是申请不到资源。
4. **过载时优雅降级优先于硬扩容**：Gateway 现有的 `maxConcurrentRequests` 429 快速失败机制，就是这个思路的体现——容量不够时明确拒绝，而不是让请求堆积拖垮整个系统。

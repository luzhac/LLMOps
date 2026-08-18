# 项目目的与面试掌握标准

## 这个项目真正要证明什么

目标不是“两天学完所有机器学习”，而是把已有的分布式系统/云平台能力延伸到 **LLM inference infrastructure**：能把一个开源模型部署成受控 API，解释其性能和显存约束，用数据找容量拐点，能关停成本，并能从真实故障恢复。

完成一次 AI 帮助下的部署只是起点。面试标准是：离开命令记录后，你仍能画出链路、解释关键配置、根据现象定位层级，并明确哪些数字是实测、哪些只是理论。

## 六个学习模块

### 1. GCP 与 Terraform

你应能解释并操作：

- 为什么这是 GKE 而不是 EKS；Terraform 与 Helm 的边界。
- VPC、subnet、private node、Cloud Router/NAT、authorized network 各自解决什么问题。
- system node pool 与 `0..1` Spot L4 pool 为什么分开。
- 全局 GPU quota、区域 L4 quota、Spot quota 和库存的区别。
- Terraform state、partial apply、重新 plan，以及为什么失败后不必 destroy。
- Billing Budget 只是告警；GPU node、GKE、NAT、disk、Artifact Registry 分别怎样计费。

### 2. Kubernetes 与 Helm

你应能从 `kubectl get/describe/logs/events` 定位：

- Deployment、Pod、Service、Endpoint、Secret、PVC/StorageClass 的关系。
- `Pending` 是状态而不是根因；`WaitForFirstConsumer` 为什么正常。
- node selector、taint/toleration、GPU resource request 怎样让 vLLM 去 L4 节点。
- readiness/liveness、rollout timeout、CrashLoopBackOff、ImagePull 的区别。
- Helm values 如何生成 manifest；为什么不能让 Helm、手工 scale 和 Argo CD 同时争夺副本数。
- scale-to-zero 时哪些资源消失、哪些费用仍保留。

### 3. vLLM 与 LLM 推理

这是你当前最需要补的部分。至少能白板解释：

- **prefill**：一次处理全部 prompt，计算密集，主要影响 TTFT。
- **decode**：逐 token 生成，反复读取权重/KV，常受显存带宽限制，主要影响 TPOT。
- **continuous batching**：不同请求在每个 decode step 动态进出 batch，提高 GPU 利用率和总吞吐。
- **KV cache**：保存每层历史 token 的 K/V，避免重复计算；context 越长、并发越高，显存需求越大。
- **AWQ**：本项目权重是 4-bit 权重量化；它减少权重显存，但计算中的 activation/KV cache 不是全部 4-bit，不能说“整个模型精度都是 INT4”。
- BF16、FP8、AWQ 在质量、显存、硬件支持和 kernel 性能上的取舍。
- `maxModelLen`、`maxNumSeqs`、GPU memory utilization 与容量/稳定性的关系。
- 单 GPU 副本扩容与 tensor parallelism 的差别：前者复制整个模型增大请求容量；后者把一个模型跨 GPU 切分以容纳/加速单实例。
- 冷启动包括节点创建、拉镜像、下载模型、加载权重、kernel compilation 和 readiness。

### 4. API 与 Gateway

你应能解释：

- Qwen（模型）、vLLM（服务器）、Gateway（平台入口）不是同一个东西。
- OpenAI-compatible `chat/completions`、messages、max_tokens、stream/SSE 和 usage。
- 为什么 concurrency 不是一个模型请求字段，而存在客户端、Gateway、vLLM 三层。
- API Key、429 背压、503 上游不可用、超时和 request ID。
- 为什么本地 port-forward HTTP 可以用于实验，公网生产入口必须 HTTPS 和正式身份系统。

### 5. 性能与可观测性

你应能说清并实测：

- TTFT、TPOT/inter-token latency、E2E latency、RPS、per-request tokens/s、aggregate tokens/s。
- 为什么吞吐最大点通常不是最佳用户体验点；并发上升时 TTFT/排队会上升。
- Gateway、vLLM、GPU 三类指标各自从哪里产生，ServiceMonitor/Prometheus/Grafana 怎样连接。
- Counter、Gauge、Histogram 和 `rate`、p95 的基本含义。
- 用并发 1/2/4/8 逐级压测，按 SLO 和失败率确定“可支持容量”，而不是只报理论 `maxNumSeqs=8`。
- 为什么负载工具测得的 aggregate tokens/s 是容量证据，Grafana tokens/s 是运行态交叉验证。

### 6. 可靠性与成本

你应能讨论下一阶段设计，而不是假装单机样例已经拥有：

- queue-aware autoscaling 比只看 GPU utilization 更接近用户等待压力。
- Spot 驱逐、上游不可用和模型冷启动时的 fallback/降级路径。
- min replica 0 与 1 在成本和冷启动 SLO 间的取舍。
- 多模型/多团队如何做 token、GPU-second 和请求维度的 cost attribution。
- 模型 revision、镜像 digest、配置和 benchmark 环境怎样形成可复现证据。

## 项目完成的四个级别

| 级别 | 标准 | 当前含义 |
|---|---|---|
| L1 跑通 | Terraform/GKE/Helm 成功，认证流式回答成功，能关闭 GPU | 已有真实证据。 |
| L2 解释 | 能不用照抄文档解释每层资源、关键参数和出现过的故障 | 现在重点学习。 |
| L3 测量 | 保存并发阶梯 benchmark、Grafana/vLLM 指标、冷启动和实际费用 | 尚需执行验收。 |
| L4 设计 | 能说明生产扩容、fallback、HTTPS、多副本/TP 和成本归属方案 | 面试前打磨。 |

## 两天目标与面试前目标

两天内合理目标：完成 L1，读懂核心路径，跑一轮受控 benchmark，形成真实 incident notes。不要试图补完 Transformer 训练、所有传统 ML 模型或 CUDA kernel 开发。

拿到面试后再深入：

- 亲手重复一次无 AI 提示的 model up/down、curl、Prometheus 查询和故障定位。
- 用白板讲 prefill/decode/KV cache/continuous batching。
- 比较并发 1/2/4/8 的 TTFT、TPOT、aggregate tokens/s、waiting/KV usage。
- 准备一个 3 分钟项目介绍和两个真实故障故事。

## 如何诚实讲 AI 帮你修故障

可以说使用 AI 加速了文档和诊断，但你必须掌握决策链。例如：

> vLLM rollout timeout 最初只是表象。我先看 Pod logs，发现 v0.26 不再接受旧 CLI flag；修正参数后又发现 Kubernetes ServiceLinks 注入的 `VLLM_PORT` 与应用变量冲突。关闭 service links 后 readiness、模型列表和流式请求都验证通过。我保留了错误、最小修改和验证步骤。

这比声称“所有代码都独立完成”更可信，也能证明你真正理解了排障过程。

## 面试前自测

不看文档回答下面问题，答不出来就回到相应章节：

1. 为什么同一模型并发上升时 aggregate tokens/s 可能增加，但单用户 tokens/s 和 TTFT 变差？
2. 8B AWQ 权重约 6 GiB，为什么 24 GiB L4 仍不能把剩余空间全当并发容量？
3. PVC Pending 什么时候正常，什么时候故障？
4. Prometheus 中 `generation_tokens_total` 为什么要用 `rate` 才是 tokens/s？
5. Gateway 429 与 vLLM waiting queue 分别说明哪一层饱和？
6. 把 replica 从 1 改为 2 与 tensor parallel size 从 1 改为 2 有什么不同？
7. 为什么 Budget 到 £30 不会自动停 GPU？
8. 当前 HTTP 为什么不直接暴露公网，生产应该加什么？

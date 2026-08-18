# Deliveroo GenAI Infrastructure 面试准备手册

Transformer与KV cache概念的前置教程见 [Qwen推理运维视角](02-llm-inference-00-qwen-inference-operations-walkthrough.md)。先用一次Qwen请求理解shape、QKV、logits、prefill/decode和显存公式，再使用本手册练习岗位问答。

本文把项目原理和面试题合在一条学习路径中：先理解模型怎样启动和回答，再理解cache、参数和多人并发，最后进入系统设计、GPU优化、Agent和行为面试。

## 1. Deliveroo面试形式

根据Deliveroo当前官方工程面试指南，通常包括：

1. Recruiter prescreen，约20–45分钟。
2. Technical screen，多数使用HackerRank，约60–90分钟。
3. Full loop，通常2–3轮、每轮约60分钟。
4. Hiring committee与team allocation。

Backend官方指南说明Full loop常覆盖coding、architecture和behavioural。这个岗位还会在这些通用环节上增加LLM inference、GPU、MLOps和GenAI平台深挖。实际安排以recruiter发来的邀请为准。

- Deliveroo工程面试指南：https://careers.deliveroo.co.uk/interview-guide-for-engineers/
- Backend面试指南：https://careers.deliveroo.co.uk/our-interview-guide-backend-engineering/
- Architecture准备指南：https://deliveroo.engineering/2022/01/14/preparing-for-the-deliveroo-architecture-interview.html

Deliveroo明确要求live coding时不使用AI工具。项目可以由AI辅助完成，但面试前必须能在没有AI的情况下解释、编码和排障。

---

## 2. 先建立最基础的四层模型

| 层 | 本项目 | 作用 |
|---|---|---|
| 硬件 | NVIDIA L4 24GB | 执行矩阵计算并保存权重、activation和KV cache。 |
| 容器镜像 | `vllm/vllm-openai:v0.26.0` | 提供Python、PyTorch、CUDA用户库、vLLM和OpenAI-compatible server。 |
| 模型 | `Qwen/Qwen3-8B-AWQ` | tokenizer、config和量化权重，决定语言能力。 |
| 平台入口 | FastAPI Gateway | API Key、并发保护、日志、指标和转发。 |

vLLM镜像不是Facebook/Meta提供的，也不包含Qwen权重。Qwen由Qwen团队发布，Hugging Face负责托管模型文件。Meta提供的是Llama等模型。

Hugging Face类似模型仓库，不代表所有模型都免费商用。每个模型必须单独检查license、gated access、使用条款、模型大小和vLLM兼容性。模型文件即使允许免费下载，GPU、网络和存储仍然收费。

---

## 3. 从`model-session.sh up`到API Ready发生什么

```text
Deployment从0扩到1
  ↓
GPU Pod出现，但没有合适节点
  ↓
Cluster Autoscaler请求g2-standard-8 + L4
  ↓
节点启动，GKE安装/准备GPU驱动和device plugin
  ↓
Pod调度，挂载30GiB PVC
  ↓
拉取vLLM容器镜像
  ↓
执行vLLM serve Qwen/Qwen3-8B-AWQ
  ↓
读取model config、tokenizer和固定revision
  ↓
PVC有模型文件：复用；没有：从Hugging Face下载
  ↓
加载AWQ权重到GPU显存
  ↓
选择/编译CUDA、Triton、Torch kernels
  ↓
根据剩余显存建立KV cache block pool
  ↓
启动HTTP server和/metrics
  ↓
/health通过，Kubernetes readiness变为Ready
```

因此“镜像自动运行”只说对了一部分。镜像封装了运行环境，但仍需要配置模型ID、revision、量化、上下文、并发、显存比例、GPU、PVC和健康检查。

### Cold start与warm-cache start

- **Cold-cache start**：新节点拉镜像，PVC没有模型，下载模型、加载权重、编译kernel。
- **Warm model cache start**：PVC已有模型，不再下载权重，但新节点仍可能重新拉镜像，模型仍要重新加载进GPU，kernel也可能重新编译。
- **Warm request**：vLLM已经Ready后发送预热请求，让运行路径和cache进入稳定状态。

本项目PVC设置的是`HF_HOME=/model-cache`，主要持久化Hugging Face模型文件。当前没有单独持久化`VLLM_CACHE_ROOT`，所以scale-from-zero后不能假设Torch/Triton compilation cache一定复用。

---

## 4. 一个请求怎样生成回答

```text
JSON messages
  ↓ tokenizer
token IDs
  ↓ prefill
处理全部prompt，生成每层历史token的K/V
  ↓ KV cache
保存这些K/V
  ↓ decode循环
每次根据当前状态生成下一个token
  ↓ detokenize + SSE
把文本片段流给用户
  ↓
遇到EOS/stop或max_tokens，释放该请求的KV blocks
```

### Tokenization

Tokenizer把文本转换成模型使用的token ID。Token不等于英文单词或汉字；一个词可能拆成多个token。

### Prefill

Prefill一次处理整个输入prompt，可以高度并行，通常计算密集。它建立初始KV cache，主要影响TTFT。

### Decode

Decode通常一次生成一个新token。每一步都读取模型权重和历史KV，通常更受显存带宽限制，主要影响inter-token latency/TPOT和输出tokens/s。

### Streaming

`stream=true`不会让模型算得更快；它只是让用户不必等完整答案，首token产生后就通过SSE逐步显示，因此改善感知体验。

---

## 5. 本项目涉及的所有cache

不要把所有cache都叫“KV cache”。它们保存的内容、位置和生命周期不同。

| Cache | 保存什么 | 在哪里 | 生命周期 | 解决什么问题 |
|---|---|---|---|---|
| Container image cache | vLLM镜像layer | GPU节点本地磁盘 | 节点删除后通常消失 | 避免重复拉取大镜像。 |
| Hugging Face/model cache | config、tokenizer、模型权重 | 30GiB PVC `/model-cache` | Pod/节点删除后仍保留 | 避免重复下载约数GB模型。 |
| Compilation cache | Torch/Triton/AOT编译产物 | 默认vLLM本地cache目录 | 当前未独立持久化 | 避免重复kernel compilation。 |
| KV cache | 活跃序列每层历史token的K/V张量 | GPU显存 | 请求完成后释放/复用 | decode时避免重新计算全部历史。 |
| Prefix cache | 可复用的相同prompt前缀KV blocks | vLLM进程的KV block pool | 进程重启后通常丢失，也会被淘汰 | 重复system prompt/文档前缀时减少prefill。 |
| Application response cache | 完整请求到完整回答 | 本项目没有实现 | 取决于设计 | 完全相同且允许复用的请求可不调用模型。 |

### KV cache为什么吃显存

模型需要为每层、每个历史token保存key和value。粗略关系：

```text
KV显存 ∝ 层数 × KV heads × head dimension × 2(K和V)
       × token数量 × 并发序列数 × KV dtype字节数
```

所以context越长、并发越高，KV cache容量压力越大。AWQ主要压缩模型权重，不代表KV cache也是4-bit。

### Prefix caching什么时候有效

例如100个请求都以同一个长system prompt和同一份产品文档开头，vLLM可以复用相同前缀的KV blocks，减少重复prefill。若每个prompt从开头就不同，命中率低。Prefix cache通常改善重复前缀的TTFT，不会消除后续decode成本。

---

## 6. 当前参数逐项解释

### 模型启动参数

| 当前值 | 含义 | 调大/改变的影响 |
|---|---|---|
| `Qwen/Qwen3-8B-AWQ` | 模型仓库ID | 换模型后必须重做显存、质量和性能验证。 |
| 固定revision SHA | 固定模型文件版本 | 避免仓库更新导致不可复现。 |
| `quantization=awq` | 4-bit weight-only量化 | 降低权重显存；可能有质量/kernel取舍。 |
| `maxModelLen=8192` | 单序列输入+输出最大token数 | 越大，最坏KV容量越高。 |
| `maxNumSeqs=8` | 一个engine iteration最多调度的序列数 | 越大可能提高吞吐，也增加KV/延迟压力。不是“总用户数”。 |
| `gpuMemoryUtilization=0.90` | vLLM规划使用约90% GPU显存 | 高了可能增加KV容量，也更容易OOM；低了更安全但容量少。 |
| prefix caching enabled | 开启重复前缀KV复用 | 只对相同前缀有收益，也占用cache blocks。 |

### Gateway参数

| 当前值 | 含义 |
|---|---|
| `maxConcurrentRequests=8` | 最多8个请求同时通过Gateway。 |
| semaphore等待约50ms | 拿不到并发槽位就返回429，不做长时间排队。 |
| request timeout 300s | 上游请求最长等待约5分钟。 |
| API Key | 入口身份检查；不是完整企业身份系统。 |

流式请求会一直占用Gateway semaphore，直到SSE结束或连接终止。因此一个生成很长内容的请求比短请求占槽位更久。

### 每个API请求的参数

| 参数 | 含义 |
|---|---|
| `messages` | system/user/assistant对话历史。 |
| `max_tokens` | 这次最多生成多少输出token。 |
| `temperature` | 随机性；性能/回归测试通常设0。 |
| `top_p` | nucleus sampling范围。 |
| `stream` | 是否逐片返回SSE。 |
| `stop` | 遇到指定停止条件时结束。 |

`max_tokens=256`不等于context length 256。请求必须满足：

```text
prompt tokens + generated tokens ≤ maxModelLen
```

---

## 7. 多个用户同时请求时到底发生什么

当前路径有两道容量控制：

```text
用户请求
  ↓
Gateway semaphore：最多8个；拿不到很快429
  ↓
vLLM scheduler：running + waiting
  ↓
continuous batching
  ↓
单张L4
```

### Gateway有没有队列

严格说只有一个很短的semaphore等待，不是业务队列。第9个请求等待约50ms仍拿不到槽位，就返回：

```text
HTTP 429 gateway concurrency limit reached
```

这样做是fail fast，防止无限排队把内存和延迟拖垮。生产环境可以在前面增加有上限的queue，但必须设置queue timeout、最大深度和过载策略。

### vLLM有没有队列

有。vLLM scheduler会维护running和waiting请求。当GPU调度容量、KV blocks或其他资源不足时，请求可能进入waiting。对应指标：

```promql
vllm:num_requests_running
vllm:num_requests_waiting
```

### Continuous batching怎样工作

传统静态batch要等一批请求一起开始、一起结束。Continuous batching按推理iteration动态变化：

```text
Step 1: A B C生成token
Step 2: A B C生成token
Step 3: A结束，D加入；B C D生成token
Step 4: B结束，E加入；C D E生成token
```

GPU不是简单地一次只服务一个用户再切换，而是把多个序列的工作组成批次做矩阵计算。因此8个用户不一定使每个人严格慢8倍；aggregate tokens/s通常会上升。但GPU、显存带宽和KV容量仍是有限的，单人体验可能变差。

### 八个人同时请求会不会慢

可能会，主要表现为：

- 更多请求同时prefill，TTFT上升。
- decode batch变大，aggregate吞吐提高，但单请求TPOT可能变差。
- 长context消耗更多KV blocks，使其他请求等待。
- 长输出请求长期占Gateway槽位。
- KV接近满或GPU饱和后，waiting增加。

但“八个人”本身还不能决定速度。还要看：

- 每个prompt多长；
- 每个输出多长；
- 请求是否同时到达；
- prefix是否相同；
- 模型大小和量化；
- GPU和vLLM配置；
- 允许的TTFT/TPOT SLO。

### 怎样保证用户体验

不能通过一句“支持8并发”保证速度。需要：

1. 定义SLO，例如p95 TTFT < 2秒、p95 TPOT < 80ms。
2. 用固定prompt/output在并发1/2/4/8下测量。
3. 在SLO恶化前设置Gateway admission limit。
4. 根据waiting/queue depth而不只是GPU utilization扩容。
5. 增加vLLM replicas和GPU容量。
6. 对交互、batch、付费/内部任务设置不同priority。
7. 设置max output、context、timeout和per-tenant quota。
8. 过载时429、降级到小模型或fallback，而不是无限排队。

### OpenAI是不是也会“人多变慢”

任何有限算力的推理服务都有排队和容量竞争，商业提供商会通过大型GPU池、路由、自动扩容、admission control、rate limits和不同服务等级降低波动。但OpenAI没有公开其内部是否对某个模型使用与vLLM完全相同的continuous batching实现，不能把本项目架构当作其内部事实。

OpenAI官方API公开了requests/tokens rate limits及剩余额度header；请求也可以处于`queued`或`in_progress`状态，并有default、flex、priority等service tier。由此能确定的是：商业API同样做容量管理和限额，不代表每次请求都保证完全相同延迟。具体性能或SLA应以客户合同、所选tier和当时官方文档为准。

- OpenAI API request/rate-limit headers：https://platform.openai.com/docs/api-reference/backward-compatibility
- OpenAI API service tier/status：https://platform.openai.com/docs/api-reference/responses

---

## 8. 扩容和并行的四种方式

### 8.1 Continuous batching：一个实例内提高利用率

同一张GPU同时调度多个序列。解决单实例吞吐问题，不增加物理GPU。

### 8.2 多副本/data parallel：提高请求容量

```text
Gateway/Service
  ├─ vLLM replica 1 → GPU 1（完整模型）
  ├─ vLLM replica 2 → GPU 2（完整模型）
  └─ vLLM replica 3 → GPU 3（完整模型）
```

适合模型能放进单GPU、目标是支持更多独立请求。每个副本都要重新加载完整权重。

### 8.3 Tensor parallel：一个模型跨多GPU

```text
一个vLLM实例
  ├─ GPU 1：部分权重
  └─ GPU 2：部分权重
```

通过`--tensor-parallel-size 2`等参数配置。适合单GPU放不下模型，或单实例需要多GPU计算。代价是每层计算存在GPU通信，不等于性能线性翻倍。

### 8.4 Pipeline parallel：按模型层切分

不同GPU/节点负责不同层，适合更大模型或跨节点场景。pipeline bubble和跨节点网络会增加复杂度。

面试核心区别：

- **多副本**复制完整模型，主要增加请求吞吐和可用性。
- **Tensor parallel**切分一个模型，主要解决单模型显存/单实例计算。
- **Continuous batching**在一个实例内动态组合请求。

---

## 9. 基础面试题：必须先会

### 项目与组件

1. Qwen、vLLM、Hugging Face、Docker image和Gateway分别是什么？
2. vLLM镜像里为什么没有Qwen权重？
3. 模型从哪里下载？为什么要固定revision？
4. Hugging Face上的模型是否都能免费商用？
5. 为什么用AWQ？“4-bit模型”准确指什么？
6. 为什么用L4而不是CPU？
7. 为什么需要Gateway，不能直接公开vLLM？
8. 为什么使用ClusterIP和port-forward？
9. `curl -N`、Bearer Key和SSE分别是什么？
10. 模型回答一次成功能证明什么，不能证明什么？

### Kubernetes/GCP

11. Deployment、Pod、Service和Endpoint有什么关系？
12. PVC Pending什么时候正常？
13. 为什么system pool和GPU pool分开？
14. node selector、taint/toleration和GPU resource request各做什么？
15. scale-to-zero后哪些费用停止，哪些继续？
16. quota和实时GPU capacity有什么区别？
17. Spot和on-demand的成本/可靠性取舍？
18. Terraform partial apply后为什么不必destroy？
19. readiness、liveness和startup probe有什么区别？
20. 为什么本次warm-cache启动仍会失败？答案：Spot节点都没创建，尚未进入模型加载阶段。

---

## 10. 中级面试题：本岗位核心

1. 从文本输入到第一个token经历哪些步骤？
2. Prefill和decode为什么性能特征不同？
3. KV cache保存什么？为什么随context/concurrency增长？
4. AWQ为什么不能让全部显存都变成4-bit？
5. Continuous batching为什么提高aggregate throughput？
6. 为什么并发提高后单用户体验可能下降？
7. `max_tokens`、`maxModelLen`、`maxNumSeqs`和Gateway concurrency的区别？
8. TTFT、TPOT、E2E latency、RPS和tokens/s分别衡量什么？
9. Per-request tokens/s和aggregate tokens/s为什么不同？
10. 为什么GPU utilization 100%不一定是好事？
11. Prefix cache与普通KV cache有什么关系和区别？
12. 为什么模型PVC缓存了，kernel仍可能重新编译？
13. 第9个请求在当前系统会发生什么？
14. vLLM running/waiting怎样反映饱和？
15. 怎样通过并发1/2/4/8找到容量拐点？
16. 为什么容量必须和SLO一起报告？
17. 长prompt和长output分别主要影响哪部分？
18. 流式返回为什么改善体验却不减少总计算量？
19. Gateway p95为什么不能完全替代客户端流式E2E测量？
20. 如何识别是Gateway、vLLM、GPU还是云库存造成延迟？

---

## 11. 高级面试题：系统设计和取舍

1. 设计一个同时支持Qwen、DeepSeek和外部OpenAI API的内部LLM平台。
2. 如何在Spot低成本和交互式SLO间选择容量？
3. 如何设计queue-aware autoscaling？使用哪些信号？
4. Scale-to-zero、minimum replica 1和预热池怎样取舍？
5. 多副本和tensor parallel什么时候分别使用？
6. 如何在多zone做GPU serving，同时处理模型cache和数据局部性？
7. Spot驱逐时如何fallback到on-demand或外部模型？
8. 如何给团队/用户做request、token和GPU-second成本归属？
9. 如何防止一个租户的长context请求拖慢全部用户？
10. 如何做模型版本canary、rollback和质量release gate？
11. 如何构建offline eval和online SLO，避免只优化吞吐？
12. 如何安全记录prompt/response而不泄漏个人数据？
13. 如何处理retry storm、timeout budget和circuit breaker？
14. 如何支持高吞吐batch inference而不影响实时请求？
15. 如何比较vLLM、SGLang和TensorRT-LLM？应使用什么benchmark和质量基线？
16. 什么时候使用SFT、LoRA、DPO，而不是prompt/RAG？
17. 如何设计LLM Gateway的model routing和vendor abstraction？
18. 如何设计Agent Gateway的tool authorization、budget和audit？

---

## 12. Python/Backend Coding准备

优先练习：

- `asyncio.Semaphore`并发控制；
- bounded producer/consumer queue；
- timeout和exponential backoff；
- rate limiter；
- SSE事件解析；
- streaming proxy；
- idempotency key；
- circuit breaker；
- 聚合滑动窗口指标；
- 用pytest覆盖成功、失败、timeout和并发边界。

Live coding时要边写边说明：输入、边界、复杂度、失败模式和测试，而不是沉默写完。

---

## 13. Behavioural准备

至少准备三个STAR故事：

1. **vLLM启动故障**：旧CLI flag与`VLLM_PORT` ServiceLinks冲突。
2. **Spot容量故障**：rollout timeout只是表象，Events显示`GCE out of resources`。
3. **成本/架构取舍**：为什么选择Spot、scale-to-zero、AWQ和单GPU；什么时候改on-demand。

每个故事按以下顺序：

```text
目标和背景
→ 观察到的现象
→ 收集的日志/Events/指标
→ 根因
→ 最小修复
→ 验证
→ 后续预防
```

不要只说“AI帮我修好了”。要能解释为什么错误发生、哪个证据排除了其他可能、修改带来什么新风险。

---

## 14. 推荐学习顺序

### 第一轮：基础

1. 四层模型：GPU、镜像、模型、Gateway。
2. 完整启动流程。
3. 完整请求流程。
4. 所有cache的区别。
5. 当前参数表。

### 第二轮：性能

1. Prefill/decode。
2. KV cache与prefix cache。
3. Continuous batching。
4. TTFT/TPOT/tokens/s。
5. 并发1/2/4/8 benchmark。

### 第三轮：生产设计

1. Queue、admission control和429。
2. Replica、tensor parallel和pipeline parallel。
3. Spot/on-demand/fallback。
4. Observability、SLO、cost attribution。
5. 多模型Gateway、Agent和eval。

完成标准不是“看过”，而是能关掉文档，用白板画出启动链路、请求链路和多人并发链路，并回答面试官连续三个“为什么”。

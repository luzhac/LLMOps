# Transformer推理、KV Cache与GPU扩容基础

本文回答这个项目中最容易混淆的一组问题：Transformer为什么产生Q/K/V，KV cache为什么占显存，token和context是什么关系，长上下文怎样估算容量，多张GPU怎样并行，以及GPU服务怎样自动扩容。

若下面的概念仍显得抽象，先读 [Qwen推理运维视角](02-llm-inference-00-qwen-inference-operations-walkthrough.md)：它从`B=2,T=100`的一次真实推理路线开始，逐步手算QKV、dot product、logits与KV cache。

阅读顺序是：模型原理 → 单请求推理 → 显存计算 → 多用户调度 → 多GPU → 自动扩容。

## 1. 先把概念分层

| 层次 | 概念 | 回答的问题 |
|---|---|---|
| 模型结构 | Transformer、Self-Attention、Q/K/V、MLP | 模型怎样理解上下文和计算下一个token。 |
| 推理过程 | tokenization、prefill、decode、KV cache | 一个请求怎样生成回答。 |
| 推理引擎 | vLLM、PagedAttention、continuous batching、prefix caching | 怎样高效管理多请求和GPU显存。 |
| 服务平台 | Gateway、queue、replicas、autoscaling、fallback、SLO | 怎样让多个用户稳定使用模型。 |

KV cache来自Transformer自回归推理；continuous batching和queue来自推理服务系统。它们相关，但不是同一层概念。

## 2. Transformer的最小知识链

```text
文本
  ↓ tokenizer
token IDs
  ↓ embedding
每个token变成向量
  ↓
多层Transformer
  ├─ Self-Attention：从上下文找相关信息
  └─ MLP：对信息进行非线性变换
  ↓
logits：下一个token的概率
  ↓ sampling
选出下一个token
```

本项目的Qwen3-8B有36层Transformer。每生成一个token，都要经过这36层。

### 自回归生成

LLM不是一次直接写出整篇回答，而是重复预测下一个token：

```text
输入：The capital of France is
预测：Paris

新输入：The capital of France is Paris
预测：.
```

因此decode具有天然的顺序依赖：第N+1个输出token必须等第N个token产生后才能计算。

## 3. Attention中的Q、K、V

可以把Self-Attention理解为查资料：

- Query（Q）：当前token想查找什么。
- Key（K）：每个历史token的索引或特征标签。
- Value（V）：每个历史token提供的实际信息。

概念流程：

```text
当前token的Q
  ↓ 与历史token的K计算相关性
attention scores
  ↓ 按相关性组合历史token的V
当前token得到的上下文信息
```

实际计算常写作：

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

面试不必先背公式，必须能解释：Q负责查询，K用于匹配，V是被取回的信息。

## 4. KV cache为什么存在

在自回归decode中，历史token的K和V已经算过，并且不会改变。

没有KV cache：

```text
生成第1个token：重算全部prompt的K/V
生成第2个token：重算prompt和第1个token的K/V
生成第3个token：再次重算全部历史
```

有KV cache：

```text
prefill：一次计算并保存prompt的K/V
decode第1步：读取历史K/V，只新增一个token的K/V
decode第2步：继续复用历史，只新增一个token的K/V
```

Transformer在数学上可以不使用KV cache，但在线自回归推理会因为反复重算历史而非常低效。因此KV cache是推理优化，不是模型权重的一部分。

## 5. Prefill和Decode

### Prefill

Prefill一次处理全部输入token并建立初始KV cache。它并行度较高，通常计算密集，主要影响TTFT（time to first token）。

### Decode

Decode每轮通常为每条序列生成一个token，读取权重和已有KV cache。它反复执行、顺序依赖明显，常受显存带宽影响，主要影响TPOT/inter-token latency和输出tokens/s。

### Streaming

`stream=true`不让模型计算更快，只是在首token产生后立即通过SSE发送，使用户不必等待完整回答。

## 6. Token、当前token数和Context

- token是模型处理文字的基本单位，不等于单词或汉字。
- context是当前请求中模型能够看到的token集合。
- context window是允许容纳的最大token数量。

一个请求的当前token数通常是：

```text
system prompt
+ 历史messages
+ 当前用户输入
+ 工具结果
+ 当前已经生成的输出
```

例如：

```text
system prompt       100 tokens
历史对话            500 tokens
当前问题            200 tokens
已经生成的回答      224 tokens
────────────────────────────
当前总数          1,024 tokens
```

本项目`maxModelLen=8192`表示单条序列的输入与输出总token数不能超过8192，不表示每个用户都会固定占用8192。

## 7. 当前Qwen模型每token的KV cache怎样计算

固定revision的模型配置是：

```text
num_hidden_layers    = 36
num_key_value_heads  = 8
head_dim             = 128
KV dtype             ≈ BF16，2 bytes
```

一个token在一层的Key有：

```text
8 KV heads × 128 = 1,024个数值
```

Value同样有1,024个数值。K和V总计：

```text
1,024 + 1,024 = 2,048个数值
2,048 × 2 bytes = 4,096 bytes = 4 KiB/层
```

36层合计：

```text
4 KiB × 36 = 144 KiB/token
```

因此：

| 单个用户当前token数 | 单用户KV cache | 8个相同用户 |
|---:|---:|---:|
| 1,024 | 144 MiB | 1.125 GiB |
| 2,048 | 288 MiB | 2.25 GiB |
| 4,096 | 576 MiB | 4.5 GiB |
| 8,192 | 1.125 GiB | 9 GiB |

换算依据：`1 MiB = 1024 KiB`，所以`1024 tokens × 144 KiB/token = 144 MiB`。

这是架构估算。vLLM还存在block对齐和metadata等开销。若显式采用FP8 KV cache，每个数值可能约1 byte，理论KV容量约减半，但必须重新验证质量、kernel支持和性能；当前项目没有设置FP8 KV cache。

## 8. 24 GB显存不只是存6.1 GB权重

当前AWQ权重文件约6.10 GB（5.68 GiB）。GPU运行时还需要：

```text
GPU显存
  = 模型权重
  + CUDA/PyTorch/vLLM runtime
  + activation
  + CUDA graph buffers
  + kernel workspace
  + KV cache block pool
  + 安全余量
```

### Runtime

CUDA context、PyTorch allocator、GPU库、stream、kernel代码和vLLM内部管理数据。模型文件相当于数据，runtime相当于执行这些数据的程序。

### Activation

模型每层计算过程中的中间张量。它通常比权重和KV更短命，用完即可覆盖，但prefill和大batch会造成峰值。

### CUDA Graph

把重复GPU操作录成可复用执行图，减少decode过程中CPU逐个发起kernel的开销。为不同batch shape捕获graph时通常需要固定buffer，因此占用显存。

### Workspace

矩阵乘法、attention、归约和量化kernel计算时使用的临时“草稿纸”。

### KV block pool

vLLM预先把一部分显存划成固定blocks，类似停车场。请求按实际token数量占用blocks，请求结束后释放复用。`nvidia-smi`可能从启动开始就显示较高显存，而`vllm:kv_cache_usage_perc`只表示停车位实际使用比例。

完整说明见[`02-llm-inference-02-vllm-memory-guide.md`](02-llm-inference-02-vllm-memory-guide.md)。

## 9. 500K和1M上下文怎样估算

如果仅假设仍使用当前Qwen3-8B结构和BF16 KV：

```text
500,000 × 144 KiB ≈ 68.7 GiB KV cache
1,000,000 × 144 KiB ≈ 137.3 GiB KV cache
1,048,576 × 144 KiB = 144 GiB KV cache
```

这还没有加模型权重、runtime和activation。

但不能用这个公式声称GPT模型实际占用同样显存。OpenAI没有公开对应GPT模型的层数、KV heads、head dimension、KV dtype、稀疏attention、分片和offload实现，所以只能计算“假设Qwen支持1M”的大小。

截至本文核对时，OpenAI官方API页面显示：

- `chat-latest`（最新ChatGPT Instant模型的API对应项）为400,000 context window、128,000 max output。
- GPT-5.6 Sol API为1,050,000 context window、128,000 max output。

ChatGPT界面实际可用上下文还可能受模型选择、计划、工具和产品策略影响，不能把API规格直接当成任意ChatGPT会话的保证。

显存足够也不代表模型自动支持1M；模型训练长度、位置编码/RoPE、attention实现和服务配置都必须支持。

## 10. Continuous batching怎样服务多个用户

传统静态batch要等待整批请求全部结束，长请求会拖住短请求。Continuous batching在每个调度迭代动态组成batch：

```text
迭代1：A、B、C各生成一个token
迭代2：A完成退出，B、C继续，D加入
迭代3：B、C、D继续
```

这样提高GPU利用率和aggregate tokens/s。但更多活跃序列仍会竞争GPU计算和显存带宽，所以单用户TTFT或TPOT可能变差。

本项目的两层限制：

- Gateway最多允许8个请求通过，约50ms拿不到semaphore就返回429。
- vLLM的`maxNumSeqs=8`限制一次调度迭代最多处理的活跃序列数。

它们都不等于“保证8个用户每人8192 tokens且速度不下降”。

## 11. 多GPU并行必须分清

### Tensor Parallel（TP）

一个模型的每层权重拆到多张GPU。一个请求同时使用这组GPU。主要用于单卡装不下模型或需要降低单次计算延迟。

```text
一个模型实例 = GPU1部分权重 + GPU2部分权重
```

KV通常也按attention heads等维度随模型分片，不是“GPU1满了临时把某个用户随便移到GPU2”。

### Pipeline Parallel（PP）

不同层放在不同GPU，请求依次经过各stage：

```text
GPU1：1–12层 → GPU2：13–24层 → GPU3：25–36层
```

每张GPU保存自己负责层的权重和相应KV。

### Data Parallel / Replicas

每个GPU或TP组保存完整模型副本，不同请求路由到不同副本：

```text
Replica1：GPU1，完整Qwen
Replica2：GPU2，完整Qwen
```

对于当前能在单张L4上运行的8B AWQ模型，用户容量不足时通常优先增加replicas，而不是为了并发强行做Tensor Parallel。

### Context/Sequence Parallel

极长上下文系统可能进一步把一个序列的attention/KV计算跨GPU分片。这是解决单个超长请求的高级方案，不能和普通多副本扩容混为一谈。

## 12. 80 GB显卡能放多少用户

不能用`剩余显存 ÷ 单用户KV`作为唯一答案。容量有两个独立瓶颈：

```text
显存瓶颈：KV blocks是否还能容纳更多活跃token
计算瓶颈：GPU是否还能在目标延迟内完成prefill/decode
```

80 GB显卡可能还有KV空间，但GPU计算已经饱和。继续加入用户会让waiting queue、TTFT和TPOT上升。

生产中要先定义负载和SLO：平均/高分位输入长度、输出长度、到达率、streaming比例、p95 TTFT、p95 TPOT和错误率。然后按并发1、2、4、8、16逐级压测。单副本容量取“仍满足SLO的最高稳定并发”，并保留容量余量，而不是取理论最大可塞请求数。

## 13. GPU自动扩容怎样设计

可以自动扩容，但不建议只使用`GPU utilization > 50%`：

- 高GPU利用率可能只是引擎高效运行，不一定已经排队。
- KV cache接近满时，瞬时GPU利用率可能不高。
- prefill形成短尖峰，不值得为每个尖峰启动新节点。
- GPU节点与模型冷启动需要数分钟，必须提前扩容或排队/fallback。

建议的信号优先级：

1. waiting requests或queue delay；
2. pending token backlog；
3. p95 TTFT；
4. KV cache usage；
5. running requests；
6. GPU utilization作为辅助信号。

扩容链路：

```text
Gateway / external queue产生指标
  ↓
KEDA或HPA增加vLLM Deployment replicas
  ↓
新Pod申请nvidia.com/gpu: 1并进入Pending
  ↓
GKE Cluster Autoscaler增加GPU节点
  ↓
节点、驱动、镜像、模型加载、kernel编译
  ↓
readiness通过，Service开始分流
```

scale-from-zero不能只依赖vLLM Pod自身指标，因为0个Pod时没有指标源。应使用Gateway/external queue指标、KEDA，或至少保留一个warm replica。

scale-down必须先停止向副本分配新请求，并等待已有stream完成；还需要cooldown、滞回、最大GPU数、预算告警和fallback，避免抖动与成本失控。

## 14. 当前项目的真实边界

已经实现：

- 一个Qwen/vLLM副本独占一张L4；
- GPU node pool可以从0扩到1；
- 模型Deployment可通过脚本手动从0扩到1；
- Gateway并发保护、vLLM内部调度和监控指标。

尚未实现：

- 根据请求量自动把vLLM replicas从1扩到2以上；
- GPU node pool当前`max_node_count=1`，不能增加第二张GPU；
- 多副本共享/分发模型cache方案；
- 外部queue或KEDA；
- 生产级draining、fallback和预测性扩容。

面试时应表达为“当前实现单副本scale-to-zero；下一阶段设计queue-aware多副本autoscaling”，不能说已经完成请求驱动的GPU横向扩容。

## 15. 容量计算与验证命令

理论KV公式：

```text
KV bytes/token
= layers × KV heads × head_dim × 2(K和V) × dtype bytes
```

实际容量以启动日志和benchmark为准：

```bash
kubectl -n llm-platform logs deployment/vllm --tail=500 | \
  grep -Ei 'GPU KV cache size|Maximum concurrency|KV cache|memory'

kubectl -n llm-platform exec deployment/vllm -- \
  nvidia-smi \
  --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv
```

Grafana重点观察：

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vLLM TTFT p95
vLLM inter-token latency p95
generation tokens/s
GPU utilization
GPU framebuffer used
```

容量验收步骤见[`04-operations-04-benchmarking.md`](04-operations-04-benchmarking.md)。

## 16. 面试必须能回答的问题

1. 为什么Transformer decode需要KV cache？
2. Q、K和V分别代表什么？
3. Prefill和decode为什么性能特征不同？
4. AWQ为什么减小权重但没有把KV cache变成4-bit？
5. 为什么同一个模型的KV bytes/token固定，而不同模型不同？
6. `maxModelLen=8192`和当前token数有什么区别？
7. `maxNumSeqs=8`为什么不等于保证8个用户的SLO？
8. Continuous batching怎样提高吞吐，为什么仍可能增加单用户延迟？
9. Tensor Parallel和模型replica解决的问题有什么不同？
10. 为什么不能只按GPU utilization扩容？
11. scale-from-zero时HPA为什么可能没有指标？
12. 如何通过benchmark决定一个L4副本支持多少用户？

## 参考资料

- 当前Qwen固定revision配置：https://huggingface.co/Qwen/Qwen3-8B-AWQ/blob/cb7d6a337aadb4d2082ed0dcef1032e4f8645194/config.json
- 当前Qwen权重索引：https://huggingface.co/Qwen/Qwen3-8B-AWQ/blob/cb7d6a337aadb4d2082ed0dcef1032e4f8645194/model.safetensors.index.json
- vLLM文档：https://docs.vllm.ai/
- OpenAI `chat-latest`模型页：https://developers.openai.com/api/docs/models/chat-latest
- OpenAI GPT-5.6 Sol模型页：https://developers.openai.com/api/docs/models/gpt-5.6-sol

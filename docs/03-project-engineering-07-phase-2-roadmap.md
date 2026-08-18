# Phase 2 中文路线图：从单GPU推理样例到GenAI平台能力

## 1. Phase 2是什么意思

正确写法是 **Phase 2**，不是Phrase 2。

Phase 2不只是“把难的东西以后再做”，而是有意控制项目范围：

- Phase 1证明核心能力：把开源权重LLM真实部署到GPU，提供受控API，能够监控、排障、关停和说明成本。
- Phase 2证明平台能力：多个模型/租户怎样路由、评估、归因、扩容、回退，以及异步批处理和可选微调怎样进入受控生命周期。

Phase 1是申请岗位前最有价值的最小闭环；Phase 2对应岗位中的部分日常工作和多数Nice-to-have。并不是Phase 2全部完成后才能投递。

## 2. 招聘要求应该怎样分类

### 核心要求，不应全部推迟到Phase 2

招聘原文要求：

- Python和分布式系统后端能力；
- 生产服务、API、数据管道或ML基础设施；
- 可观测性、排障、可靠性、incident response和成本/性能优化；
- 开源模型推理和/或微调的实际经验；
- 使用Claude Code、Codex、Cursor等AI编码工具贯穿完整SDLC。

Phase 1主要补“开源权重实时GPU推理”缺口。AI-assisted SDLC也是核心要求，应该从现在开始保留证据，不能等到Phase 2最后才补。

招聘写的是inference **and/or** fine-tuning，因此一个有数据证据的推理项目可以覆盖技术方向，不需要为了投递假装完成SFT、DPO或LoRA。但作品集实验不能冒充商业生产经验。

### Nice-to-have，但确实属于团队工作范围

- vLLM、SGLang、TensorRT-LLM生产经验；
- 多节点推理/训练与GPU性能优化；
- SFT、DPO/RLHF、LoRA和数据准备/评估；
- Kubernetes、云GPU、serverless GPU和高吞吐batch；
- LLM Gateway、路由、fallback和cost attribution；
- internal developer platform/self-service；
- AI Agent或MCP server生产经验；
- eval、guardrail、LLM tracing、RAG、搜索和vector database。

“Nice-to-have”不表示工作中不用，而表示招聘不会要求每个候选人入职前全部具备。强候选人一般会在其中若干项有深度。

## 3. 进入Phase 2前必须先完成的Phase 1验收

以下内容没有完成时，不应急着增加新组件：

- 并发1/2/4/8 benchmark，保存TTFT、TPOT、aggregate tokens/s和错误率；
- 保存vLLM running/waiting、KV cache和GPU指标证据；
- 记录冷启动时间：节点、镜像、模型加载、kernel compilation；
- 记录一次实际实验费用；
- 清除Git中的secret、tfstate、plan和本地凭据；
- 把项目推到GitHub，并确保CI通过；
- 能不看AI解释一次完整请求链路和至少两个真实故障。

原因：没有基线就不能证明Phase 2改善了吞吐、可靠性或成本。

## 4. 推荐的Phase 2范围

Phase 2分成2A、2B和2C。先做高证据/低成本部分，不需要一次全部实现。

## Phase 2A：平台控制面，优先级最高

### 2A-1 固定评估集和模型发布门禁

目标：模型或配置更新不能只靠“curl能回答”。

实现：

- 建立20–50条与项目任务相关的固定测试样例；
- 保存输入、期望条件和评分方法；
- 检查回答有效性、JSON格式、拒答、安全规则和基础质量；
- 同时记录TTFT、tokens/s、显存和估算成本；
- candidate只有在质量、性能和成本阈值通过后才能promotion；
- 保留上一个revision用于rollback。

完成证据：

- 版本化eval数据集；
- 可重复运行的eval命令；
- candidate与baseline对比报告；
- 一个故意失败的release gate演示。

对应岗位：eval infrastructure、guardrails、model lifecycle、operational excellence。

### 2A-2 多租户usage与cost attribution

目标：回答“哪个团队、哪个模型、哪个请求用了多少资源和钱”。

实现：

- API key映射tenant/team；
- 每个请求记录tenant、model、prompt tokens、completion tokens、状态和延迟；
- Prometheus按tenant/model聚合请求和token；
- 估算GPU-second、token成本或session成本；
- 给tenant设置quota和budget policy；
- 控制label cardinality，request ID只进日志，不作为Prometheus label。

完成证据：

- 两个测试tenant产生独立usage报表；
- Grafana或离线报告能解释费用归属；
- 超额tenant得到明确429/配额响应。

对应岗位：LLM Gateway、cost attribution、self-service platform。

### 2A-3 多后端路由与fallback

目标：Gateway不再只知道一个固定vLLM地址。

实现：

- 后端注册配置：model name、endpoint、健康状态、成本等级和能力；
- 按model、tenant或policy路由；
- timeout、有限重试、circuit breaker；
- vLLM不可用时回退到较小模型或受控managed API；
- 记录实际选择的backend和fallback原因；
- streaming开始输出后不能随意透明重试，避免重复/拼接回答。

为了控制费用，可以先用mock backend演示路由和故障，不必立即购买第二张GPU。

完成证据：

- primary成功路径；
- primary超时/503后fallback路径；
- 两条路径的测试、日志和指标；
- 明确哪些错误允许重试。

对应岗位：LLM Gateway、vendor abstraction、reliability和fallback。

### 2A-4 AI-assisted SDLC证据

目标：证明AI编码工具参与设计、开发、测试、审查、发布和运行，而不是只生成一段代码。

实现与证据见[`03-project-engineering-06-ai-assisted-sdlc.md`](03-project-engineering-06-ai-assisted-sdlc.md)。

## Phase 2B：GPU弹性与批处理，需要受控实验

### 2B-1 Queue-aware多GPU副本自动扩容

目标：请求压力上升时增加完整模型replica，压力下降时安全缩容。

建议架构：

```text
Gateway/external queue metrics
  ↓
KEDA或HPA调整vLLM replicas
  ↓
GPU Pod Pending
  ↓
GKE Cluster Autoscaler增加GPU node
  ↓
模型Ready后Service开始分流
```

主要指标：

1. waiting requests或queue delay；
2. pending token backlog；
3. p95 TTFT；
4. KV cache usage；
5. running requests；
6. GPU utilization仅作辅助。

必须修改/解决：

- GPU node pool从`0..1`调整为允许2个或更多节点；
- 获得相应GPU quota和实际库存；
- 当前`standard-rwo`模型PVC不能被假定为跨节点共享方案；应选择每节点cache、对象存储下载、镜像预热或RWX方案；
- 0 replicas时没有vLLM Pod指标，scale-from-zero要使用Gateway/external metric；
- 新副本冷启动期间需要bounded queue、fallback或最小warm replica；
- scale-down前drain streaming请求；
- 加cooldown、hysteresis、最大GPU数和预算保护。

完成证据：

- 持续负载触发1→2 replicas和GPU nodes；
- 新副本Ready后waiting/TTFT恢复；
- 空闲后安全2→1或1→0；
- 没有中断正在进行的stream；
- 记录扩容时间和额外费用。

当前项目只有1张L4 quota/最大节点1时，可以先提交设计、manifest和模拟指标，但不能声称实际验证了多GPU扩容。

### 2B-2 高吞吐batch inference

目标：异步任务优先追求总吞吐和单位成本，而不是单请求低TTFT。

实现：

- 输入采用JSONL或对象存储manifest；
- Kubernetes Job或Argo Workflows编排分片、重试和合并；
- 固定模型revision、prompt版本和输入版本；
- 保存逐条状态，支持失败重跑而不重复全部任务；
- 比较online逐请求和batch的tokens/s、GPU利用率和单位成本；
- 设置优先级，避免batch挤占实时SLO。

完成证据：

- 可重跑的100/1000条小型batch；
- 部分失败恢复；
- 吞吐和成本对比报告。

对应岗位：high-throughput batch inference、data pipelines、jobs from days to hours。

## Phase 2C：选做加分项，不建议同时展开

### 2C-1 LoRA/SFT小实验

只有存在明确任务、训练数据和baseline eval时才做。推荐先LoRA/SFT，不从DPO/RLHF开始。

最小闭环：数据准备 → train/validation split → LoRA训练 → adapter artefact → 固定eval → 与base比较 → promotion/rollback。

不能只凭训练loss下降宣称模型更好，也不能把一次notebook训练称为分布式生产pipeline。

### 2C-2 Agent与MCP

实现一个范围很小的Agent，例如读取只读运维状态并生成诊断建议。重点不是聊天页面，而是：

- tool allowlist和参数schema；
- tenant/user授权；
- timeout、重试和最大步骤数；
- token/tool调用预算；
- prompt injection和敏感数据边界；
- 每次tool call审计；
- 高风险动作人工批准；
- agent eval和失败案例。

MCP server只是把工具/资源以标准协议暴露给Agent，不等于Agent本身。

### 2C-3 VLM或第二推理引擎

VLM可以展示图像输入和更复杂显存特征；SGLang/TensorRT-LLM对比可以展示engine选型。但必须使用相同模型、revision、prompt、并发和硬件做公平benchmark，否则没有结论价值。

## 5. 推荐执行顺序

如果目标是尽快投递并准备面试：

```text
先完成Phase 1 benchmark和GitHub发布
  ↓
Phase 2A-1 eval/release gate
  ↓
Phase 2A-2 cost attribution
  ↓
Phase 2A-3 routing/fallback
  ↓
根据面试反馈选择：autoscaling / batch / Agent / LoRA其中一个
```

不要同时建设完整Agent平台、RAG、vector DB、LoRA、DPO和多GPU集群。组件数量多不等于岗位证据强。

## 6. 简历声明门禁

| 能力 | 只有设计/文档 | 本地或mock验证 | 云上真实验证 | 可采用的说法 |
|---|---|---|---|---|
| Queue-aware autoscaling | 是 | 否 | 否 | Designed queue-aware GPU autoscaling. |
| Queue-aware autoscaling | 是 | 是 | 否 | Implemented and simulated custom-metric scaling policy. |
| 多GPU扩容 | 是 | 是 | 是 | Validated 1→2 GPU replica scale-out under measured load. |
| Eval gate | 是 | 是 | 不一定需要GPU常驻 | Built versioned eval and release-gate workflow. |
| LoRA | 是 | notebook成功 | 有eval/artefact | Fine-tuned and evaluated a LoRA adapter；不能说distributed training。 |
| Agent/MCP | 是 | tool/auth/eval运行 | 有部署证据 | Built and deployed a bounded agent/MCP service；是否production必须诚实。 |

## 7. Phase 2完成标准

推荐不要把“全部2A–2C完成”作为目标。对于本项目，合理的Phase 2完成定义是：

- 完成eval release gate；
- 完成tenant usage/cost attribution；
- 完成primary/fallback路由；
- 选择并真实验证autoscaling、batch、Agent或LoRA中的一个；
- 所有改变通过CI、人工diff review和可回滚发布；
- 保存基线与改动后的质量、性能、成本和故障证据。

达到这个范围，已经比单纯堆叠很多未验证组件更像真实GenAI平台工程。

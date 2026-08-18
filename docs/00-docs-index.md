# Documentation Index：文档总索引

顶层文档使用“组号 + 主题 + 组内顺序”命名。按文件名排序时，同一学习或工作流程会自动聚在一起。

## 01 · AI Learning：AI核心学习主线

建议按顺序学习。

1. [Machine Learning from First Principles](01-ai-learning-01-machine-learning-from-first-principles.md)
2. [Transformer from First Principles](01-ai-learning-02-transformer-from-first-principles.md)
3. [Generative AI and RAG from First Principles](01-ai-learning-03-generative-ai-and-rag-from-first-principles.md)
4. [AI Agents from First Principles](01-ai-learning-04-ai-agents-from-first-principles.md)

## 02 · LLM Inference：LLM推理与性能

0. [Qwen推理运维视角：QKV、Logits与KV Cache](02-llm-inference-00-qwen-inference-operations-walkthrough.md)
1. [Transformer Inference Fundamentals](02-llm-inference-01-transformer-inference-fundamentals.md)
2. [vLLM Memory Guide](02-llm-inference-02-vllm-memory-guide.md)
3. [王木头Transformer视频学习稿](02-llm-inference-03-wangmutou-transformer-video-guide.md)
   - [完整渲染HTML版本](02-llm-inference-03-wangmutou-transformer-video-guide.html)

## 03 · Project Engineering：项目设计与工程实现

1. [Project Learning Objectives](03-project-engineering-01-project-learning-objectives.md)
2. [Architecture](03-project-engineering-02-architecture.md)
3. [Model Lifecycle](03-project-engineering-03-model-lifecycle.md)
4. [API Design and Gateway Hardening](03-project-engineering-04-api-design-and-gateway-hardening.md)
5. [API and Observability](03-project-engineering-05-api-and-observability.md)
6. [AI-Assisted SDLC](03-project-engineering-06-ai-assisted-sdlc.md)
7. [Phase 2 Roadmap](03-project-engineering-07-phase-2-roadmap.md)

## 04 · Operations：部署、成本与故障处理

1. [WSL Tooling Setup](04-operations-01-wsl-tooling-setup.md)
2. [Cost and Safety](04-operations-02-cost.md)
3. [Deployment Runbook](04-operations-03-runbook.md)
4. [Benchmarking](04-operations-04-benchmarking.md)
5. [Troubleshooting](04-operations-05-troubleshooting.md)
6. [Spot L4 Capacity Incident](04-operations-06-incident-spot-l4-capacity.md)

架构决策记录继续保留在 [decisions/](decisions/)；它们已经使用ADR编号，不需要再改名。

## 05 · Career：岗位、面试与学习管理

1. [Job Requirements Map](05-career-01-job-requirements-map.md)
2. [Deliveroo Interview Guide](05-career-02-deliveroo-interview-guide.md)
3. [CV After Project](05-career-03-cv-after-project.md)
4. [Learning Retention and Knowledge Management](05-career-04-learning-retention-and-knowledge-management.md)

## 06 · Reference：来源与审计

1. [GitBook Content Map and Public Repository Audit](06-reference-01-gitbook-content-map-and-public-audit.md)

## 命名规则

```text
组号-组名-组内顺序-具体主题.md
01-ai-learning-02-transformer-from-first-principles.md
04-operations-05-troubleshooting.md
```

新增文档时优先放进现有组，并分配下一个组内序号。不要因为临时阅读顺序改变而频繁重命名已有文件；需要更灵活的学习顺序时，只修改本索引。

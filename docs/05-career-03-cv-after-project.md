# 项目完成后的草稿——暂勿提交 / DRAFT AFTER PROJECT COMPLETION — DO NOT SUBMIT YET

只有 04-operations-04-benchmarking.md 中的适用项目都有真实证据后才能删除此提示；当前 GCP/vLLM 成果仍是计划。

Remove this banner only after every applicable item in `04-operations-04-benchmarking.md` has
real evidence. The CV below is written as the post-completion version requested;
today the GCP/vLLM claims are planned, not yet earned.

---

# ZHUO LU

London, UK | No visa sponsorship required  
luzhac@gmail.com | linkedin.com/in/luzhac | github.com/luzhac

## 软件工程师——GENAI / ML 基础设施 / SOFTWARE ENGINEER — GENAI / ML INFRASTRUCTURE

Python 与 ML 平台工程师，拥有六年独立负责生产交易与数据平台的经验，覆盖异步服务、实时流水线、Kubernetes、基础设施即代码、可观测性、事故响应和成本/性能权衡。

Python and ML-platform engineer with six years of hands-on ownership of an
independent production trading and data platform, spanning asynchronous backend
services, real-time pipelines, Kubernetes delivery, infrastructure as code,
observability, incident response and performance/cost trade-offs.

开源权重模型服务经验包括在 GCP GKE L4 上用 vLLM 部署 Qwen、构建带身份验证的 OpenAI 兼容流式网关，并测量 TTFT、吞吐量、并发、GPU 内存、冷启动和成本。

Hands-on open-weight model serving experience includes deploying Qwen3-8B-AWQ
with vLLM on an autoscaling GCP GKE NVIDIA L4 node pool; building an authenticated
OpenAI-compatible streaming gateway; and benchmarking TTFT, token throughput,
concurrency, GPU memory, cold start and cost. Broader cloud-native experience
includes AWS EKS, GCP GKE, Terraform, Docker, Helm, Argo CD, Prometheus, Grafana,
MLflow and Argo Workflows.

## 核心技能 / CORE SKILLS

核心技能覆盖 GenAI 推理、Python 后端、ML 平台、云交付、可靠性/性能和 AI 辅助工程。

**GenAI 推理：**vLLM、开源权重服务、Qwen、OpenAI 兼容 API、SSE、连续批处理、KV Cache、AWQ、基准测试、版本控制和成本建模。

**GenAI inference:** vLLM, open-weight LLM serving, Qwen, OpenAI-compatible APIs,
HTTP/SSE streaming, continuous batching, KV-cache and context management, AWQ
quantisation, inference benchmarking, model revision control and cost modelling.

**Python 与后端：**Python、asyncio、FastAPI/Flask、REST/WebSocket、Redis、pandas、SQL、日志、身份验证、并发控制和延迟敏感服务。

**Python and backend:** Python, asyncio, FastAPI/Flask, REST and WebSocket APIs,
Redis pub/sub, pandas, SQL, structured logging, authentication, concurrency
controls, external API integration and latency-sensitive services.

**ML 平台：**容器化推理、MLflow、Argo Workflows、训练任务、模型晋级、监控、漂移分析和可复现评估。

**ML platform:** Containerised inference, MLflow experiment tracking/model
registry, Argo Workflows, scheduled data/training jobs, model promotion,
prediction monitoring, drift analysis and reproducible evaluation.

**云与交付：**GCP、AWS、Kubernetes、Terraform、Docker、Helm、Argo CD 和 GitHub Actions。

**Cloud and delivery:** GCP GKE, NVIDIA L4, Artifact Registry, Cloud Storage,
Cloud NAT and IAM; AWS EKS, EC2, ECS, ECR, S3 and CloudWatch; Kubernetes,
Terraform, Docker, Helm, Argo CD and GitHub Actions.

**可靠性与性能：**Prometheus、Grafana、DCGM、Loki、CloudWatch、健康探针、SLO、告警、运行手册、事故响应和容量/性能优化。

**Reliability and performance:** Prometheus, Grafana, DCGM GPU metrics, Loki,
CloudWatch, health probes, SLI/SLOs, alerting, runbooks, incident response,
root-cause analysis, capacity planning and latency/throughput/cost optimisation.

**AI 辅助工程：**用 Coding Agent 辅助设计、实现、重构、测试、基础设施审查、文档和发布检查，并保留 diff、验证和人工批准。

**AI-assisted engineering:** Coding-agent workflows for design exploration,
implementation, refactoring, test scaffolding, Terraform/Kubernetes review,
documentation and release checks, with scoped context, diff review, automated
validation and human approval.

## 工作经历 / PROFESSIONAL EXPERIENCE

### 独立 Python / ML 平台工程师——MarsCompute.com（自雇）/ Independent Python / ML Platform Engineer — MarsCompute.com (self-employed)

自 2020 年独立负责 AWS 交易和数据平台的 Python 服务、基础设施、部署、监控、事故响应，以及可靠性与成本决策。

London, United Kingdom | 2020–Present  
github.com/luzhac

Self-funded AWS trading and data-processing platform, live since 2020. Sole
owner of Python services, infrastructure, deployment, monitoring, incident
response and reliability/cost decisions.

- Built asynchronous Python services for market-data ingestion, validation,
  signal generation and execution using REST/WebSocket APIs, Redis pub/sub and
  structured logging.
- Replaced repeated REST polling with WebSocket aggregation and vectorised
  processing, reducing signal-to-execution latency to below 200 ms.
- Progressed services from EC2 through ECS to EKS; packaged workloads with
  Docker and Helm and used health probes, resource controls and autoscaling
  patterns.
- Built reusable Terraform for networking, IAM, EKS, ALB, DNS and supporting
  services, with remote state and reviewable plans.
- Implemented GitHub Actions for Python tests, Terraform/Helm validation, image
  builds, Trivy scanning and ECR publishing; used Argo CD for declarative
  deployment and rollback.
- Built Prometheus, Grafana, Loki and CloudWatch dashboards and alerts; owned
  incidents from diagnosis and mitigation through code fix and preventive
  monitoring.
- Made explicit cost/capacity decisions, including pausing a reproducible
  four-node EKS environment when its continued cost was not proportionate.
- Trained and evaluated LightGBM gradient-boosted tree models over market-data
  datasets of up to approximately 36 million rows, with lag/rolling features,
  walk-forward validation and MLflow-tracked parameters and metrics.
- Compared LightGBM and XGBoost under clean out-of-sample, transaction-cost and
  stability gates; rejected candidates that did not justify promotion to the
  live decision path.
- Built a reusable FastAPI/MLflow model-serving project and Argo Workflows for
  data preparation, training and model registration, with latency,
  prediction-distribution and drift monitoring patterns.
- Built a focused Python RAG backend with LLM APIs, document ingestion,
  embeddings, retrieval, prompt construction and response validation.

### 自由职业 AI / LLM 工程审查员——DataAnnotation.tech（自雇）/ Freelance AI / LLM Engineering Reviewer — DataAnnotation.tech (self-employed)

两年有偿评估 LLM 生成的 Python、后端、DevOps、Kubernetes、Terraform 和 ML 输出，并设计对抗性 Prompt、边界案例和结构化反馈。

Remote, United Kingdom | January 2024–December 2025

- Completed two years of paid work evaluating LLM-generated Python, backend,
  DevOps, Kubernetes, Terraform and ML outputs against defined correctness and
  quality criteria.
- Designed adversarial prompts and edge cases exposing hallucinated APIs,
  unsafe code, unsupported assumptions, weak tests and incomplete reasoning.
- Produced structured feedback and peer-reviewed contributor submissions in a
  distributed asynchronous environment.
- Developed practical judgement about model failure modes and the testing,
  monitoring and human controls needed before AI output can be trusted.

## 精选项目 / SELECTED PROJECTS

### GCP 开源权重 LLM 推理平台 / Open-Weight LLM Inference Platform on GCP

使用 Terraform、GKE、Spot L4、vLLM、FastAPI、Prometheus/Grafana、DCGM 和 Argo CD 构建成本受控、可观测且支持流式输出的推理平台。

- Provisioned a zonal GKE platform with Terraform, separating a small system
  pool from an autoscaling `0..1` Spot `g2-standard-8` pool with one 24 GB NVIDIA
  L4, GPU drivers, taints/tolerations, private nodes and Workload Identity.
- Served a pinned Qwen3-8B-AWQ checkpoint through vLLM and an authenticated
  FastAPI gateway supporting OpenAI-compatible HTTP/SSE streaming, request IDs,
  bounded concurrency and controlled overload responses.
- Measured cold/warm startup, TTFT, latency, per-request and aggregate output
  tokens/second, error rate, GPU utilisation, memory and cost across concurrency
  levels 1, 2, 4 and 8; retained benchmark artefacts and explicit release gates.
- Added Prometheus/Grafana and DCGM GPU observability, alerts, operational
  runbooks, failure/recovery tests, GitHub Actions validation and Argo CD GitOps.
- Kept MLflow and Argo Workflows outside the online critical path; documented
  their justified use for later LoRA/evaluation and batch-inference workflows.

### ML 平台工程 / ML Platform Engineering | github.com/luzhac/ml-engineering

整合 FastAPI、MLflow、Argo Workflows 和 Argo CD，构建可复现模型生命周期与受监控服务。

- Built EKS platform components integrating FastAPI inference, MLflow tracking
  and registry, Argo Workflows and Argo CD delivery for reproducible model
  lifecycle and monitored serving.

### AWS 云平台 / AWS Cloud Platform | github.com/luzhac/cloud-platform

用 Terraform、EKS、Helm、GitHub Actions、Argo CD 和完整监控栈构建 AWS 云平台。

- Built a Terraform-managed EKS platform with Helm, GitHub Actions, Argo CD,
  Prometheus, Grafana, Loki, CloudWatch and least-privilege IAM patterns.

## 认证 / CERTIFICATIONS

以下为 AWS、Kubernetes、Terraform 和 Azure 相关专业认证。

- AWS Certified Machine Learning — Associate (2024)
- AWS Certified DevOps Engineer — Professional (2025)
- Certified Kubernetes Administrator (CKA) (2025)
- HashiCorp Certified: Terraform Associate (2025)
- AWS Certified Solutions Architect — Associate (2025)
- Microsoft Certified: Azure Data Engineer Associate (2024)

## 教育经历 / EDUCATION

软件工程硕士——哈尔滨工业大学，中国。

Education details follow in English.

Master's degree in Software Engineering — Harbin Institute of Technology, China


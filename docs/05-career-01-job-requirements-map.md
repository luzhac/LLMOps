# Deliveroo 中级 GenAI 基础设施职位要求映射 / Deliveroo mid-level GenAI infrastructure requirement map

本文把招聘描述转化为工程证据；Terraform 成功或模型回答一次都不代表项目已经完成。

This document translates the vacancy language into engineering evidence. The
project is not considered complete merely because Terraform applies or a model
returns one answer.

| Vacancy concept | Meaning in practice | Evidence in this repository | Completion evidence required |
|---|---|---|---|
| Open-weight real-time serving | Run weights you control behind an online API | Qwen3-8B-AWQ + vLLM + gateway | Successful streamed requests and pinned version |
| Inference engine | GPU runtime that schedules and batches generation | vLLM Helm deployment | Startup log, `/metrics`, benchmark report |
| Backend/API fundamentals | Auth, timeouts, overload, streaming and stable contract | FastAPI gateway | Unit tests and live curl test |
| GPU autoscaling | Add/remove GPU nodes based on pending GPU Pods | GKE node pool `0..1` | Timestamps for scale-up and scale-down |
| Latency/throughput optimisation | Measure TTFT, decode rate, aggregate throughput | Streaming benchmark | Results at concurrency 1/2/4/8 |
| Quantisation/memory | Trade model quality/precision for memory and capacity | AWQ checkpoint, KV/context limits | GPU memory and output-quality comparison |
| Observability | Metrics, logs, dashboards and actionable alerts | Prometheus, Grafana, DCGM, rules | Dashboard screenshot plus alert test |
| Reliability | Health probes, bounded load, graceful shutdown, retry plan | Probes, 429 gate, runbook | Failure injection and recovery notes |
| Cost control | Attribution, budget, scale-to-zero and documented unit cost | Budget, Spot pool, cost model | Billing export/console evidence and session cost |
| Kubernetes/GCP | Reproducible GPU platform | Terraform GKE and Helm | Clean create and destroy from runbook |
| GitOps | Desired workload state reconciled from Git | Argo CD Application | Sync/rollback demonstration |
| Evaluation/guardrails | Detect quality regression and unsafe output | Phase-2 design | Fixed eval set and release gate before claiming |
| Fine-tuning/LoRA | Adapt model weights for a measured task | Explicitly not in phase 1 | Only add after dataset and baseline exist |
| Batch inference | Efficient asynchronous processing | Phase-2 Argo Workflow design | Not claimed in phase 1 CV |
| LLM gateway/routing | Stable tenant-facing API across model backends | Gateway boundary | Auth, model routing or fallback evidence |
| AI-assisted SDLC | Use agents with review/testing controls | Repository workflow and CI | Retain diffs, tests and human approval |

## 必需项与加分项 / Essential versus nice-to-have

扎实的开源权重推理项目可以覆盖核心要求，无需假装完成 SFT、DPO 或 LoRA；加分项只有在有亲手实践证据后才能写入简历。

The role asks for production experience with open-weight inference **and/or**
fine-tuning. A strong inference project can therefore address the essential
technical gap without pretending to have completed SFT, DPO or LoRA. Those are
nice-to-have areas and should only enter the CV after hands-on evidence exists.

## 面试说明 / Interview explanation

中文速读：我已有 Python、Kubernetes、Terraform、MLflow、编排和可观测性经验；本项目专门补齐 GCP GPU 推理、vLLM、流式网关、性能与成本测量能力。

> I already had transferable production experience in Python services,
> Kubernetes, Terraform, MLflow, workflow orchestration and observability. I
> built this project to close the specific GPU-serving gap. I used vLLM with a
> pinned open-weight Qwen model on an autoscaling GKE L4 node, put a streaming
> gateway in front, and measured TTFT, token throughput, concurrency, GPU memory,
> cold start and cost. I deliberately kept MLflow and Argo Workflows out of the
> online critical path until there was a genuine fine-tuning or batch-evaluation
> requirement.

## 第一阶段完成后仍不能做出的宣称 / Claims that must not be made after phase 1

不能声称运行过高可用多 GPU 生产集群、完成分布式微调、达到生产规模流量、实现语义漂移检测，或在无前后基准时声称延迟降低 X%。

- "Operated a highly available multi-GPU production fleet."
- "Performed distributed fine-tuning/RLHF/DPO."
- "Achieved production-scale traffic" without publishing load assumptions.
- "Implemented model drift detection" if only infrastructure metrics exist.
- "Reduced latency by X%" without a saved before/after benchmark.


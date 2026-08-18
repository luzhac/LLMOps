# 模型生命周期、MLflow、Argo 与漂移 / Model lifecycle, MLflow, Argo and drift

## vLLM 需要 MLflow 吗？/ Does vLLM need MLflow?

不需要。vLLM 可直接加载固定 revision 的 Hugging Face 检查点；在线路径加入 MLflow 会增加数据库、对象存储和故障依赖。

No. vLLM can load a pinned Hugging Face checkpoint directly. Adding MLflow to
the online serving path would add database/object-store operations and another
failure dependency without improving inference.

Phase 1 identifies the deployed artefact with:

- Hugging Face repository ID;
- immutable revision SHA;
- quantisation format;
- vLLM image tag;
- Helm release/Git commit;
- benchmark report.

That is enough for a reproducible inference-only project.

## 什么时候 MLflow 才有用？/ When MLflow becomes useful

当出现 LoRA/QLoRA、微调参数、跨版本评估、发布指标和模型晋级需求时，再引入 MLflow。

Add MLflow when there are experiments to compare or artefacts to promote:

- LoRA/QLoRA adapters;
- fine-tuning hyperparameters and datasets;
- evaluation runs across model revisions;
- quality, latency and cost metrics tied to a release candidate;
- controlled aliases such as `candidate` and `production`.

Even then, resolve a model version before deployment. Do not make every live
request depend on a registry lookup.

## Argo CD 与 Argo Workflows / Argo CD versus Argo Workflows

Argo CD 检查集群是否与 Git 一致；Argo Workflows 编排多步骤任务。推理 Deployment 和微调流水线应由不同控制器负责。

| Tool | Question answered | Use here |
|---|---|---|
| Argo CD | "Does the cluster match Git?" | Deploy/rollback Helm workloads |
| Argo Workflows | "Which multi-step job should run next?" | Future batch evaluation, cache warm-up or LoRA pipeline |

An inference Deployment is not a workflow. A fine-tuning pipeline is not a
long-running Deployment. Using the correct controller is part of the design.

## LLM 中的漂移是什么？/ What “drift” means for an LLM

冻结权重不会自行漂移，但输入、输出行为、性能和 RAG 数据源都会变化；Prometheus 只能发现性能漂移，语义质量还需要版本化评估集和人工审核。

A frozen model's weights do not silently drift. Several surrounding things can:

1. **Input drift:** language, token length, topics, image sizes or tenant mix
   change.
2. **Output/behaviour drift:** refusal rate, JSON validity, tool-call success,
   toxicity or factual quality changes after prompts, adapters or model versions.
3. **Performance drift:** TTFT, tokens/second, queue depth or error rate changes
   after load/runtime/infrastructure changes.
4. **Data-source drift:** a RAG index or upstream business data becomes stale.

Prometheus detects performance drift, not semantic quality by itself. Semantic
quality requires a versioned evaluation set, task-specific scoring, sampled
human review and release gates.

## 新模型 revision 的发布门禁 / Release gate for a new model revision

固定候选版本后，依次执行正确性与安全评估、相同性能场景、质量与成本比较、可选金丝雀发布，只有阈值通过后才能晋级。

1. Pin candidate model/runtime revisions.
2. Run offline correctness and safety evaluation.
3. Run identical warm performance scenarios.
4. Compare quality, TTFT, throughput, GPU memory and estimated cost.
5. Canary a small traffic share if real traffic exists.
6. Promote only when explicit thresholds pass; retain rollback revision.

This is the LLM equivalent of the controlled model-promotion discipline already
used in the TradeBalance LightGBM research: a model that fits is not necessarily
a model that should be promoted.


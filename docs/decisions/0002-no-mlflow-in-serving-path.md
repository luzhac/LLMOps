# ADR 0002：第一阶段服务路径不使用 MLflow 或 Argo Workflows / No MLflow or Argo Workflows in the phase-1 serving path

状态：已接受。

Status: accepted.

## 决策 / Decision

第一阶段使用固定的 Hugging Face revision 与 Git/Helm revision 标识模型，并由 Argo CD 协调部署；没有真实实验或工作流需求前不引入 MLflow/Argo Workflows。

Use an immutable Hugging Face revision plus Git/Helm revision for phase-1 model
identity. Use Argo CD for deployment reconciliation. Do not deploy MLflow or
Argo Workflows until evaluation, batch inference or fine-tuning creates a real
workflow/experiment requirement.

## 影响 / Consequences

这样成本更低、故障模式更少、所有权更清晰；以后加入 LoRA 或多模型评估时，再由 MLflow 跟踪实验、Argo Workflows 编排任务。

- Lower cost, fewer failure modes and clearer ownership.
- The model can start without a registry/database dependency.
- Evaluation reports are files in phase 1.
- When LoRA or multi-model evaluation is added, MLflow tracks experiments and
  adapters, while Argo Workflows orchestrates the jobs. They still remain off
  the per-request inference path.


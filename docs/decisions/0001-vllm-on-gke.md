# ADR 0001：在 GKE Standard 上运行 vLLM / vLLM on GKE Standard

状态：第一阶段已接受。

Status: accepted for phase 1.

## 决策 / Decision

决定在 GKE Standard 的 L4 节点池上使用 vLLM 官方 OpenAI 兼容镜像。

Use vLLM's official OpenAI-compatible image on a GKE Standard L4 node pool.

## 原因 / Reasons

这样可以直接展示开源权重推理、连续批处理、KV Cache、流式输出、指标和 GKE GPU 运维能力。

- Directly addresses the target role's open-weight serving and vLLM areas.
- Continuous batching, KV-cache management, streaming and metrics are built in.
- GKE Standard exposes node pools, taints, autoscaling and GPU operations clearly.
- The existing Kubernetes/Terraform experience transfers while adding genuine
  GCP/GPU evidence.

## 备选方案 / Alternatives

备选方案包括 Ollama、TensorRT-LLM、Vertex AI 托管端点和单台 Compute Engine VM；各自在简单性、性能和基础设施证据方面有不同取舍。

- **Ollama:** fastest local demo but weaker evidence for high-throughput serving.
- **TensorRT-LLM:** potentially faster but adds engine/build complexity before a
  baseline exists; compare later with measurement.
- **Vertex AI managed endpoint:** less infrastructure ownership and may not use
  the welcome credit/model path in the intended way.
- **Single Compute Engine VM:** cheaper/simpler, but omits the Kubernetes GPU
  scheduling and autoscaling evidence the project is intended to demonstrate.


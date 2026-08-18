# API、参数与可观测性指南

## 先分清三个东西

- **Qwen/Qwen3-8B-AWQ**：模型权重和 tokenizer，负责“回答什么”。
- **vLLM**：GPU 推理服务器，加载模型、管理 KV cache、continuous batching，并提供 OpenAI-compatible HTTP API。
- **FastAPI Gateway**：本项目自己写的入口，负责 API Key、并发保护、错误转换、日志和 Gateway 指标，再把请求转发给 vLLM。

Qwen 不是一个网页产品，vLLM 也不是聊天网页。Ollama 常配 CLI 或第三方聊天 UI，所以看起来像“有输入页面”；这个项目面向生产服务，交付物首先是 API。`curl` 是最小冒烟测试，真实业务可用 Python/JavaScript OpenAI SDK、后端服务或另建 Web UI 调用。没有聊天页面不代表部署不完整。

## curl 测试到底做了什么

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d '{
    "model":"Qwen/Qwen3-8B-AWQ",
    "messages":[{"role":"user","content":"Explain KV cache."}],
    "temperature":0,
    "max_tokens":256,
    "stream":true,
    "stream_options":{"include_usage":true}
  }'
```

- `curl`：HTTP 客户端，不是模型工具。
- `-N`：关闭 curl 输出缓冲，让 SSE token 片段马上显示。
- URL：`kubectl port-forward` 暴露在本机的 Gateway。
- `Authorization: Bearer ...`：Gateway 的 API Key，不是 Hugging Face token。
- `Content-Type`：请求体是 JSON。
- `-d`：发送 OpenAI Chat Completions 风格的请求体。

响应里的 `data: {...}` 是 SSE 事件；把各个 `choices[0].delta.content` 依次拼接就是完整文本。`finish_reason=length` 只表示达到 `max_tokens`，不是错误。最后的 `data: [DONE]` 表示流结束。

## OpenAI-compatible 是什么意思

它表示 vLLM 实现了 OpenAI API 的常用路径和 JSON/SSE 结构，现有 OpenAI SDK 可以只替换 `base_url`、API Key 和 model 名称来调用。它不表示请求会发给 OpenAI，也不表示 vLLM 支持 OpenAI 平台的每一个 endpoint/参数。

本项目实际使用：

| Endpoint / 字段 | 含义 |
|---|---|
| `GET /v1/models` | 查看当前服务的模型 ID。 |
| `POST /v1/chat/completions` | 发送多轮消息并生成回复。 |
| `model` | 必须与 vLLM 服务的模型名匹配。 |
| `messages` | 有序对话；常用 role 是 `system`、`user`、`assistant`。 |
| `temperature` | 采样随机性；0 更适合可重复性能测试。 |
| `max_tokens` | 这一次最多生成多少输出 token；不是上下文总长度。 |
| `stream` | `true` 时通过 SSE 边生成边返回。 |
| `stream_options.include_usage` | 流结束前返回 prompt/completion token 统计。 |

三个容易混淆的上限：

| 配置 | 所属层 | 含义 |
|---|---|---|
| 请求 `max_tokens=256` | API 请求 | 单个回答最多生成 256 token。 |
| vLLM `maxModelLen=8192` | 推理引擎 | 单条序列输入加输出的最大上下文长度。 |
| vLLM `maxNumSeqs=8` | 推理引擎 | 一次调度迭代最多处理的序列数上限。 |
| Gateway `maxConcurrentRequests=8` | 入口保护 | 同时允许通过 Gateway 的请求数；超出快速返回 429。 |

因此，**concurrency 不是 Qwen API 的字段**。它可能指负载测试客户端同时发多少请求、Gateway 同时接收多少请求，或 vLLM 同时调度多少序列。面试时必须先说清是哪一层。

Python SDK 示例：

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key=os.environ["LLM_API_KEY"],
)

stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B-AWQ",
    messages=[{"role": "user", "content": "Explain KV cache."}],
    max_tokens=256,
    temperature=0,
    stream=True,
)
for event in stream:
    print(event.choices[0].delta.content or "", end="", flush=True)
```

## HTTP 安全边界

目前浏览器/客户端访问的是 `http://127.0.0.1:8080`，但它不是公网裸 HTTP：

```text
本地客户端 --HTTP--> 本地 port-forward
             --经 kubectl 到 GKE API 的认证 TLS 通道-->
Gateway ClusterIP --集群内 HTTP--> vLLM ClusterIP
```

这适合个人测试。`ClusterIP` 没有公网 IP，关闭 port-forward 后外部无法访问。生产对公网或跨不可信网络开放时必须加 HTTPS 的 Ingress/Gateway/Load Balancer、托管证书、正式身份认证、速率限制和网络策略；不能把当前 Gateway 直接改成 public LoadBalancer 后继续传明文 Key。集群内 Gateway→vLLM 是否需要 mTLS，要依据威胁模型，而不是看到 HTTP 就一律判断泄漏。

## 企业里通常怎么用

```text
Web / mobile / internal service
        │ HTTPS + identity
        ▼
Ingress / API gateway / auth / rate limit
        ▼
本项目 FastAPI Gateway
        ▼ OpenAI-compatible request + SSE
vLLM replicas → GPU
```

前端通常不直接持有可长期使用的模型 API Key，而是调用业务后端；业务后端做用户权限、prompt 组装、审计、配额和数据治理。模型平台主要交付稳定 API，不一定交付聊天 UI。

## 指标是怎样到 Grafana 的

```text
Gateway /metrics ─┐
vLLM /metrics ────┼─ Service + ServiceMonitor → Prometheus → PromQL → Grafana panel
GKE managed DCGM ─┘                                      └→ Cloud Monitoring GPU view
```

`ServiceMonitor` 不是采集器本身；它告诉 Prometheus Operator 应该发现哪个 Service、访问哪个命名端口和 `/metrics` 路径。Prometheus 每 15 秒抓取并保存时间序列，Grafana 只负责查询和画图。

### Gateway 指标在哪里实现

代码在 `app/gateway/main.py`，使用 Python `prometheus_client`：

| 指标 | 类型 | 代码何时更新 | Grafana 含义 |
|---|---|---|---|
| `llm_gateway_requests_total` | Counter | 每个 HTTP 请求结束后按 method/path/status 加 1 | `rate(...[5m])` 得 requests/s。 |
| `llm_gateway_request_duration_seconds` | Histogram | middleware 用单调时钟记录请求处理时间 | bucket 计算 p95。 |
| `llm_gateway_in_flight_requests` | Gauge | 请求进入 +1，退出 -1 | 当前 Gateway 内请求数。 |
| `llm_gateway_rejections_total` | Counter | API Key 失败或并发 semaphore 超时 | 401/429 过载证据。 |

注意：对普通非流式请求，Gateway duration 接近端到端处理时间；对 `StreamingResponse`，FastAPI middleware 的完成边界可能早于最后一个 SSE token，因此它不能替代客户端测得的完整流式总延迟。流式 SLO 应以负载测试的 TTFT/E2E 和 vLLM 原生指标为准。

### vLLM 指标

vLLM 自己在 8000 端口 `/metrics` 暴露：

| Panel / PromQL | 含义 |
|---|---|
| `sum(rate(vllm:generation_tokens_total[5m]))` | 整个实例最近 5 分钟平均输出 token/s。无请求时为 0；实例关闭时无序列。 |
| TTFT p95 histogram | 从请求到首 token 的 95 分位，受排队和 prefill 影响。 |
| TPOT/inter-token p95 | 首 token 后 token 间隔，主要反映 decode 体验。 |
| `vllm:num_requests_running` | 正在 execution batch 中的请求。 |
| `vllm:num_requests_waiting` | 因容量等待调度的请求；持续大于 0 说明饱和。 |
| `vllm:kv_cache_usage_perc * 100` | KV cache block 使用百分比；接近 100% 会限制并发/上下文。 |

### GPU 指标为什么曾经为空

`DCGM_FI_DEV_GPU_UTIL` 和 `DCGM_FI_DEV_FB_USED` 不由 vLLM 或 Gateway 产生，它们必须来自 DCGM Exporter。本次真实检查发现自管 DaemonSet 先因 GKE 保留 PriorityClass 无法创建，修正后又因 GKE COS/NVML 接口不兼容而 CrashLoop，所以 Prometheus 从未收到 `DCGM_FI_DEV_*`。

项目不采用高权限绕过。GKE 的受支持路径是 GKE-managed DCGM + Managed Service for Prometheus，并在 Cloud Monitoring 看 GPU utilization、framebuffer、功耗和温度。本地 Grafana dashboard 已移除会永久为空的 DCGM panel；以后完成官方数据源接入后再加回，而不是先画一个没有数据来源的图。

## 空面板的正确排查顺序

1. 时间范围是否覆盖请求发生时间；浏览器时钟是否同步。
2. vLLM 是否为 `1/1`；scale-to-zero 后空白是预期。
3. Service 是否有 Endpoint。
4. Prometheus `Status → Targets` 是否 UP。
5. 在 Prometheus 先查询原始指标名，再查询 `rate`/histogram。
6. 最近 5 分钟是否真的有请求；Counter 的 rate 需要样本窗口。
7. Dashboard 的指标名是否与锁定的 vLLM 版本一致。

不要用“Grafana 空白”直接推断 GPU、模型或 Prometheus坏了；Grafana 是链路最后一层。

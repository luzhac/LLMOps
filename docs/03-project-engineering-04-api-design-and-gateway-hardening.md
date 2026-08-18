# API设计与LLM Gateway工程指南

本文不是泛泛的REST教程，而是以当前`FastAPI Gateway → vLLM`项目为例，建立一套后端工程师可以实际使用的API设计框架。覆盖契约、认证授权、密钥、限流、并发、幂等、重试、错误、SSE、观测、回归测试、发布和代码结构。

如果问题中的“秘密回归”指密钥管理、幂等/重试或回归测试，本章分别都有专门章节。

## 1. API设计是不是后端工程师的工作

是。后端工程师不只是写HTTP handler，还要决定：

- 谁可以调用；
- 可以调用什么；
- 输入和输出契约；
- 高负载时怎样保护系统；
- 上游失败时返回什么；
- 是否允许重试，怎样避免重复副作用/费用；
- 怎样监控、审计、发布、兼容和回滚；
- 怎样控制安全和成本。

对于Deliveroo这个岗位尤其相关，因为招聘明确包含LLM Gateway、Agent Gateway、可靠性、fallback、SLO、guardrail和cost attribution。

## 2. API设计的完整框架

每个API在编码前依次回答：

```text
1. Consumer：谁调用？浏览器、内部服务、batch还是Agent？
2. Use case：同步查询、长任务还是流式生成？
3. Contract：路径、方法、schema、错误和版本是什么？
4. Identity：怎样认证？
5. Policy：调用者可以访问哪个模型/工具/数据？
6. Protection：rate limit、quota、concurrency和body limit是什么？
7. Reliability：timeout、retry、idempotency、circuit breaker和fallback是什么？
8. Observability：怎样看流量、错误、延迟、费用和审计？
9. Lifecycle：怎样测试、发布、兼容、canary和rollback？
```

先写清这些问题，再选择FastAPI、Redis、Envoy或其他库。库不能替代策略设计。

## 3. 当前项目的API边界

```text
Client
  ↓ HTTP/OpenAI-compatible contract
FastAPI Gateway
  ├─ auth
  ├─ concurrency protection
  ├─ request ID/logs/metrics
  └─ proxy/streaming
  ↓ internal HTTP
vLLM
  ↓
Qwen + GPU
```

Gateway不负责模型推理；vLLM不应该负责企业tenant身份、配额和cost attribution。

当前公开给客户端的主要契约应该收敛为：

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/health/live` | 进程是否活着；通常不依赖上游。 |
| `GET` | `/health/ready` | 是否可以接流量；需要关键依赖可用。 |
| `GET` | `/v1/models` | 返回调用者可访问的模型。 |
| `POST` | `/v1/chat/completions` | 非流式或SSE流式生成。 |
| `GET` | `/metrics` | Prometheus内部抓取，不应公网开放。 |

当前`/v1/{path:path}`是方便Phase 1的catch-all proxy。Phase 2应改成明确allowlist或显式路由，避免客户端借Gateway访问未审查的vLLM endpoint，也能生成更准确的OpenAPI和指标标签。

## 4. Contract：路径、方法和Schema

### 路径与HTTP方法

- `GET`读取，不应改变服务状态；
- `POST`创建计算或提交生成任务；
- `/v1`代表外部契约版本，不代表内部代码版本；
- 名称保持稳定，不把具体vLLM版本写入URL；
- Gateway可以替换后端引擎，但保持客户端契约。

### 输入Schema

LLM请求至少验证：

- `model`是非空字符串且在tenant allowlist；
- `messages`存在、role合法、总项数有限；
- `max_tokens`在允许范围；
- `temperature`、`top_p`范围合法；
- `stream`是布尔值；
- JSON body和单条message有大小上限；
- 不接受任意content type；
- 暂不支持的参数给明确错误，而不是静默忽略。

因为要保持OpenAI compatibility，可以采用“核心字段严格验证、已知扩展字段允许透传”的策略，但必须有endpoint allowlist和总body limit。

### 输出Schema

非流式响应保持稳定JSON。流式响应保持：

```text
Content-Type: text/event-stream
data: {...}
data: [DONE]
```

Gateway不得缓冲整个流。客户端断开时应取消上游请求，并在`finally`中释放并发槽位。

## 5. 统一错误契约

不要让FastAPI错误、httpx错误和vLLM原始错误形成三套随机格式。建议：

```json
{
  "error": {
    "type": "rate_limit_error",
    "code": "rate_limit_exceeded",
    "message": "Tenant request rate exceeded.",
    "request_id": "req_..."
  }
}
```

常用状态码：

| Status | 含义 |
|---:|---|
| `400` | JSON/业务参数不合法。 |
| `401` | 没有凭据或凭据无效。 |
| `403` | 身份有效，但没有模型/操作权限。 |
| `404` | endpoint或对该调用者可见的model不存在。 |
| `409` | 同一idempotency key对应不同请求。 |
| `413` | 请求体过大。 |
| `422` | Schema validation失败；是否对外保留由契约决定。 |
| `429` | rate、quota或concurrency限制；提供`Retry-After`。 |
| `502` | 上游返回无效响应/网关错误。 |
| `503` | 模型未Ready、无容量或fallback不可用。 |
| `504` | 上游超时。 |

错误信息不应泄漏内部URL、stack trace、API key、prompt全文或敏感上游响应。

## 6. Authentication、Authorization和Secret

三者不能混为一谈：

- Authentication：你是谁？
- Authorization：你能访问哪个模型、tenant、tool或操作？
- Secret management：凭据怎样生成、保存、轮换和撤销？

### 当前状态

Phase 1只有一个共享Bearer key，使用`hmac.compare_digest`验证，并通过Kubernetes Secret注入。适合个人演示，不适合多租户生产。

### Phase 2 API key模型

建议每个key关联：

```text
key_id / last_four
tenant_id
scopes或allowed_models
enabled
created_at / expires_at
rate/quota policy
哈希后的secret
```

完整API key只在创建时返回一次；服务端保存安全哈希，不在日志或数据库保存明文。支持两个key并存完成无停机轮换，并能立即撤销。

用户浏览器不应持有长期平台key。用户交互通常使用OIDC/OAuth2登录业务后端；服务到服务可使用短期身份、mTLS或受控API key。不要自己实现密码登录和JWT签发器，优先集成成熟Identity Provider。

生产secret应使用Google Secret Manager配合CSI/External Secrets等受控同步方案。Kubernetes Secret只是分发对象，不应把值提交Git。

FastAPI官方提供APIKey、OAuth2和OpenID Connect的OpenAPI/security工具，但工具只负责协议集成，权限模型仍需要应用定义。

## 7. Rate limit、Concurrency和Quota不是一回事

| 控制 | 例子 | 保护什么 |
|---|---|---|
| Rate limit | 每tenant每分钟60请求 | 防突发/滥用并保证公平。 |
| Token rate | 每tenant每分钟100K tokens | 控制LLM计算与费用。 |
| Concurrency | 每tenant同时2个、实例总计8个 | 保护GPU实时容量。 |
| Quota | 每tenant每天1M tokens或£10 | 控制周期总用量/预算。 |
| Queue bound | 最多等待20个或2秒 | 防止无限排队和内存增长。 |

当前Gateway只有进程内`Semaphore(8)`：拿不到槽位约50ms后返回429。这是并发保护，不是rate limit。

### LLM为什么需要多维限制

两个请求的成本可能相差几百倍：

```text
请求A：100 input + 50 output tokens
请求B：8,000 input + 2,000 output tokens
```

只限制requests/min会让请求B绕过真实资源公平性。建议组合：

- RPM：requests per minute；
- TPM：tokens per minute；
- per-tenant concurrency；
- daily/monthly token或cost quota；
- global GPU concurrency/queue保护。

输出token在请求前未知。可以根据`prompt estimate + max_tokens`预留预算，完成后按真实usage结算/退回；流中断也要记录已生成usage。预留策略必须防止用户把`max_tokens`设很大占满全部预算。

### 常见算法

| 算法 | 优点 | 缺点 | 适合 |
|---|---|---|---|
| Fixed window | 最简单、便宜 | 窗口边界可能双倍突发 | 低风险内部API。 |
| Sliding log | 精确 | 内存和操作成本高 | 低流量严格审计。 |
| Sliding window counter | 准确度/成本折中 | 比fixed复杂 | 通用API。 |
| Token bucket | 允许受控burst，限制平均速率 | 要正确维护时间/原子更新 | 交互式API常用。 |
| Leaky bucket | 输出速率平滑 | burst体验较差 | 严格保护下游。 |

## 8. Rate limit用第三方还是自己写

结论：**策略自己设计，分布式一致性和成熟协议尽量复用。**

### 单进程作品集

可以实现一个小型in-memory token bucket来学习算法和写单元测试。但它只在当前单Gateway进程内有效；重启后状态消失，多副本会各算各的。

### 多副本生产

选择之一：

1. **Redis-backed limiter**：所有Gateway共享Redis，用原子命令或Lua完成check-and-consume。Redis官方说明本地counter在负载均衡后会失效，并给出Redis/Lua token bucket方案。
2. **Envoy/Kong/managed API gateway**：在入口统一做global RPM/IP/basic quota；Envoy global rate limit服务可返回429。
3. **两层组合**：边缘Gateway做粗粒度防攻击，应用Gateway做tenant/model/token/cost等LLM语义策略。

不要自己发明分布式锁、跨副本时钟同步或认证加密。即使使用`limits`等Python库，也要确认后端store、原子语义、故障模式和多副本一致性；“装了rate-limit库”不等于生产限流完成。

当前项目最合理的实现：

- 定义`RateLimiter`接口；
- 单元测试使用确定性的in-memory实现；
- Phase 2多副本使用`redis-py`和原子token bucket/sliding counter；
- 不为作品集立即购买昂贵managed Redis，可先在system node运行非HA Redis验证，同时明确这不是生产HA方案；
- 未来入口采用Envoy/managed Gateway时，把IP/RPM粗限流下沉，token quota仍留在应用层。

## 9. Idempotency、Request ID和Retry

### Request ID

用于关联日志、客户端、Gateway和上游，不防止重复执行。

### Idempotency key

用于告诉服务器“这是同一个业务操作的重试”。服务端通常保存：

```text
tenant + idempotency_key
request fingerprint
状态/响应引用
过期时间
```

同一个key和相同request可返回同一结果；同一个key配不同body返回409。

### LLM生成为什么难重试

- 生成可能非确定；
- 重试会再次消耗GPU和token费用；
- SSE已经向客户端发送部分token后，切换后端会产生重复或拼接回答；
- 客户端断开不代表上游一定立即停止。

推荐规则：

- 建连失败、首byte前503/504可有限重试；
- 使用指数退避、jitter和总时间预算；
- 429遵守`Retry-After`；
- SSE发出首个chunk后默认不透明重试；
- 非流式高价值请求可以实现idempotency结果缓存；
- 每次attempt记录同一个logical request ID和不同attempt ID。

## 10. Timeout、Circuit Breaker、Fallback和Backpressure

Timeout至少区分：

- connect timeout：连接上游；
- pool timeout：等待HTTP连接池；
- first-token/read timeout：等待数据；
- total deadline：请求总预算；
- client disconnect/cancellation。

当前设置connect 5秒、request 300秒，是起点，不是所有场景统一答案。

Circuit breaker在连续失败或高错误率时暂时停止向故障后端发送新请求，避免雪崩。Fallback可以路由到：

- 同模型另一replica；
- 较小/较便宜模型；
- closed-source managed API；
- 异步队列或明确503。

Fallback必须满足数据和能力策略，不能把敏感prompt未经授权发送到外部vendor。

Backpressure优先使用bounded concurrency和bounded queue。无限队列只会把显性429变成长时间超时和内存风险。

## 11. Streaming API特殊问题

- middleware记录到`StreamingResponse`对象返回，不一定代表最后一个token结束；
- TTFT和完整E2E应由客户端或专门stream wrapper测量；
- 客户端断开要取消httpx upstream stream；
- semaphore必须直到流关闭才释放；当前代码已使用`finally`处理这一点；
- 心跳、代理buffering和idle timeout必须协调；
- usage通常在流结尾出现，异常中断时可能不完整；
- 发出首chunk后错误只能结束stream，不能再返回普通JSON状态码。

## 12. Observability与Audit

### RED指标

- Rate：请求数和tokens/s；
- Errors：按稳定error code分类；
- Duration：非流式E2E、TTFT、TPOT和stream总时长。

LLM Gateway还需要：

- tenant/model/backend；
- prompt/completion tokens；
- running/waiting/queue delay；
- rate/concurrency/quota rejection；
- fallback/circuit breaker；
- estimated cost或GPU-second；
- client disconnect和cancelled upstream。

### Cardinality

Prometheus label不能放request ID、API key、完整URL、prompt或用户自由文本。request ID进入结构化日志；指标只使用有界tenant ID、model、route、status/error code。

### Privacy

默认不记录完整prompt和回答。若为了质量审核采样，必须有脱敏、访问控制、保留期限和tenant policy。

## 13. Security检查表

- 公网入口HTTPS和托管证书；
- API key/OIDC，明确authorization；
- body、message、token和header大小限制；
- 只允许明确endpoint和上游host，防SSRF；
- 不信任客户端提供的tenant ID；身份中解析tenant；
- 日志不记录Authorization、secret和敏感prompt；
- CORS只允许真实前端origin；纯服务API不必随意`*`；
- `/metrics`、debug和内部health不对公网；
- 非root、只读文件系统、最小IAM和NetworkPolicy；
- dependency/image扫描和固定版本；
- tool-calling/Agent还要增加tool allowlist、参数验证和人工批准。

Prompt injection不能通过普通HTTP认证自动解决；它属于模型输入和Agent工具信任边界，需要guardrail和tool authorization。

## 14. API回归与兼容性测试

### Unit tests

- API key验证和权限；
- schema边界；
- rate limiter时间推进；
- error mapping；
- retry decision；
- usage/cost计算。

### Contract tests

- 路径、状态码、header和JSON shape；
- OpenAI SDK能否调用；
- SSE chunk格式和`[DONE]`；
- 旧客户端在添加字段后仍工作；
- OpenAPI schema snapshot/diff。

### Integration tests

- Gateway和mock/vLLM；
- timeout、503、慢stream和断线；
- Redis limiter原子行为；
- 两个Gateway replicas共享限制。

### Load tests

- RPM、并发和token长短分布；
- 429是否按预期出现；
- 队列是否有界；
- p95 TTFT/TPOT与错误率；
- limiter/Redis是否成为新瓶颈。

### LLM质量回归

不能只比较完整字符串，因为模型输出可能变化。使用固定prompt、temperature 0、结构/关键条件/任务评分、安全规则和人工抽查；性能与语义质量分别验收。

## 15. Versioning、发布和回滚

- `/v1`内优先做向后兼容的additive change；
- 删除/改字段需要deprecation窗口；
- error code比自由message更适合客户端逻辑；
- 变更前做OpenAPI/contract diff；
- CI通过后canary或小流量发布；
- readiness通过不等于业务SLO正常，发布后看5xx、429、TTFT、waiting和fallback；
- 镜像tag/digest、模型revision、Gateway配置和Git commit必须可关联；
- 定义自动/人工rollback阈值。

## 16. 当前Gateway现状审计

| 能力 | 当前状态 | Phase 2建议 |
|---|---|---|
| Bearer auth | 一个共享明文配置key，constant-time compare | tenant key、hash、scope、rotation/revoke。 |
| Authorization | 无模型/tenant权限 | key→tenant→allowed models/policy。 |
| 路由契约 | `/v1/{path}`通用代理 | 显式models/chat allowlist和schema。 |
| Body validation | 只尝试解析`stream` | Pydantic核心字段、body/token上限。 |
| Concurrency | 全局Semaphore 8，50ms后429 | 保留GPU保护，增加per-tenant并发和Retry-After。 |
| Rate/token quota | 无 | Redis/global RPM+TPM+周期quota。 |
| Request ID | middleware产生并返回 | 存入`request.state`并一致传给所有上游/attempt。 |
| Timeout | connect 5s、request 300s | 分层deadline、disconnect cancellation。 |
| Retry/fallback | 无 | 首byte前有限重试、breaker和policy fallback。 |
| Error contract | FastAPI与上游混合 | 统一稳定error schema/code。 |
| Streaming | SSE raw passthrough，流结束释放slot | 加disconnect、TTFT/E2E/usage处理和测试。 |
| Readiness | 上游`<500`视为可用 | 要求预期成功响应，并区分dependency状态。 |
| Metrics/logs | 基本RED、结构化request日志 | tenant/model/backend/token/cost，控制cardinality。 |
| Secret | Kubernetes Secret | Secret Manager、rotation、无Git明文。 |
| Tests | 3个基础测试 | schema/auth/rate/error/stream/contract/load/failure tests。 |

上述是设计审计，不代表必须一次全部修改。每次改动先固定测试，避免“大重构+新功能”同时发生。

## 17. 推荐代码结构

当前`main.py`约200行仍可读，不应为了架构图立即拆成20个文件。随着Phase 2能力实际增加，可逐步演进为：

```text
app/gateway/
  main.py                 # app装配和lifespan
  config.py               # Settings
  api/
    routes.py             # 显式endpoint
    schemas.py            # request/response/error contract
  auth.py                 # credential→principal/tenant
  policies.py             # model、quota和routing policy
  rate_limit/
    base.py               # interface
    memory.py             # tests/local
    redis.py              # distributed implementation
  upstream.py             # httpx、timeout、retry、stream
  errors.py               # stable error mapping
  observability.py        # metrics/logging/audit
  tests/
    unit/
    contract/
    integration/
```

拆分原则：有独立职责、独立测试或第二实现时再拆。不要仅为了“看起来企业级”制造层级。

## 18. 推荐在当前项目做，不另开普通API项目

最强的作品集叙事是：

```text
Phase 1：Qwen + vLLM + GPU实时推理
Phase 2：把薄Gateway演进为tenant-aware LLM Gateway
```

这样同时展示：

- Python/FastAPI后端；
- API contract和streaming；
- 分布式限流和GPU背压；
- tenant/authorization；
- resilience/fallback；
- observability和cost attribution；
- Kubernetes/GPU autoscaling。

只有当你需要学习与LLM无关的数据库事务、复杂CRUD、支付或event-driven业务时，才值得另开普通API项目。针对这份工作，继续强化当前Gateway更直接。

## 19. 分阶段实现顺序

### API-0：冻结Phase 1行为

- 为现有models、chat、SSE、401、429和upstream error增加contract tests；
- 修正request ID一致传递；
- 明确当前OpenAI-compatible范围。

### API-1：显式契约与错误

- endpoint allowlist；
- Pydantic核心schema和body limits；
- 统一error envelope/code；
- 413/422/502/503/504映射；
- OpenAPI和兼容性测试。

### API-2：Tenant、Rate limit和Quota

- principal/tenant模型；
- per-tenant allowed models；
- in-memory limiter测试实现；
- Redis-backed RPM/TPM；
- per-tenant/global concurrency；
- 429、`Retry-After`和quota metrics。

### API-3：可靠上游

- 分层timeout；
- 首byte前有限retry；
- circuit breaker；
- backend routing/fallback；
- client disconnect cancellation；
- failure injection测试。

### API-4：Usage、Cost和Release

- prompt/completion usage；
- tenant/model/backend attribution；
- estimated cost/GPU-second；
- eval gate、canary和rollback记录。

### API-5：真正公网入口（只有需要时）

- HTTPS Gateway/Ingress；
- OIDC或正式service identity；
- WAF/Cloud Armor和edge rate limit；
- Secret Manager；
- 多副本和Redis HA；
- 隐私、审计和保留策略。

## 20. Phase 2 API完成标准

不以文件数量判断，至少需要证据：

- 两个tenant拥有不同模型/配额策略；
- 多Gateway replica下rate limit仍是全局一致；
- 429带稳定error code和`Retry-After`；
- vLLM慢、503和断stream都有测试；
- request ID贯穿客户端、Gateway和上游日志；
- prompt/completion token和成本按tenant归属；
- OpenAI SDK非流式/流式contract回归通过；
- load test证明queue有界且没有把Redis变成瓶颈；
- 变更可canary、监控和rollback；
- 所有secret不进入Git。

## 参考资料

- FastAPI Security：https://fastapi.tiangolo.com/tutorial/security/
- Redis distributed rate limiting：https://redis.io/docs/latest/develop/use-cases/rate-limiter/
- Envoy global rate limiting：https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/other_features/global_rate_limiting.html
- Envoy Gateway rate limiting：https://gateway.envoyproxy.io/docs/concepts/rate-limiting/

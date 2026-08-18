# AI Assistant、Coding Agent与完整SDLC

## 1. SDLC是什么意思

招聘原文写的是 **SDLC：Software Development Life Cycle，软件开发生命周期**，不是SDL。

完整SDLC通常包括：

```text
需求/问题定义
  ↓
设计
  ↓
实现
  ↓
测试与安全检查
  ↓
代码审查
  ↓
发布/部署
  ↓
监控与故障响应
  ↓
复盘和下一轮改进
```

Deliveroo招聘原文特别要求使用Claude Code、Codex、Cursor等工具参与designing、generating code、testing、monitoring和releasing software。这是岗位核心要求，不是Nice-to-have。

## 2. AI Assistant、Coding Agent和产品Agent不是一回事

### AI Assistant

对话式帮助：解释概念、给代码片段、审查思路。它通常等待用户复制粘贴和执行。

### Coding Agent

例如Codex、Claude Code或Cursor Agent。它可以在授权范围内：

- 查看整个仓库；
- 修改多个文件；
- 运行测试和静态检查；
- 读取日志并定位故障；
- 根据结果继续迭代；
- 产生可供人审查的diff。

你现在使用Codex，让它读取项目、写Markdown/代码、执行Terraform/Helm/Kubernetes检查、根据日志修复问题，属于coding-agent workflow。

### 产品AI Agent

这是提供给业务或用户的Agent：LLM自主决定调用搜索、数据库、订单或其他工具来完成任务。它需要Agent Gateway、tool authorization、audit、budget和eval。

招聘原文同时提到了两类：

- “使用AI coding tools贯穿SDLC”是开发者工作方式，是核心要求；
- “构建AI agents或MCP servers”是平台/产品经验，是Nice-to-have。

## 3. 只写prompt让AI干活算不算

算开始，但不足以证明“贯穿完整SDLC”。

如果流程只是：

```text
给一句prompt → AI生成代码 → 直接运行/提交
```

面试官会担心：

- 你是否理解设计取舍；
- 是否检查了错误、安全和成本；
- 测试失败时能否独立定位；
- AI是否覆盖了用户原有修改；
- 是否知道怎样回滚；
- 离开AI能否解释关键代码。

合格流程应该是：

```text
你定义目标、约束和验收标准
  ↓
Agent检查仓库并提出设计/计划
  ↓
你审查风险和范围
  ↓
Agent实现小而可审查的改动
  ↓
Agent运行测试，报告失败和证据
  ↓
你检查diff、命令、成本和安全
  ↓
CI再次独立验证
  ↓
人工批准发布
  ↓
Agent辅助检查metrics/logs/rollback条件
```

## 4. Prompt和Markdown怎样用才专业

Markdown不是目的，它是把模糊意图变成可审查工程契约的工具。

一个好的coding-agent任务应写清：

```markdown
## 目标
为Gateway增加tenant级token usage指标。

## 范围
- 只修改Gateway和测试。
- 不创建云资源。
- 不改变现有OpenAI-compatible响应。

## 约束
- tenant来自已认证API key映射。
- request_id不能成为Prometheus label。
- 不把API key写入日志。

## 验收标准
- 单元测试覆盖两个tenant。
- ruff和pytest通过。
- 文档说明指标来源和cardinality风险。

## 发布/回滚
- Helm部署后验证metrics。
- 异常时回滚到上一镜像tag。
```

这比“帮我把监控改好”更能控制Agent，也更能证明你的工程判断。

## 5. 在当前项目中怎样形成完整证据

### 需求与设计

- 每个较大改动先有issue/spec或ADR；
- 记录目标、非目标、风险、成本和验收标准；
- AI可以产生候选方案，人负责选择并说明原因。

### 实现

- 一次只让Agent做一个边界明确的改动；
- 查看`git diff`，不要只看Agent总结；
- 不允许AI提交secret、tfstate、plan或本地凭据；
- 保留有意义的commit，而不是一个“AI changed everything”提交。

### 测试

当前仓库已有：

- Gateway pytest；
- Ruff；
- Terraform fmt/init/validate；
- Helm lint/template；
- GitHub Actions CI。

Agent应先运行最相关的小测试，再运行完整CI；失败时保存根因和修复证据，不能只重复运行直到偶然通过。

### Review与人工批准

- 检查diff是否超出需求；
- 检查异常路径、超时、重试、并发和资源释放；
- Terraform必须审查plan；
- Kubernetes变更先看Helm render/diff；
- GPU、外网暴露、IAM和删除操作必须人工批准；
- live coding面试不能依赖AI。

### 发布

- 固定镜像tag/digest、模型revision和Helm/Git revision；
- CI通过后才部署；
- 先测试环境/小流量，再推广；
- 定义readiness和rollback条件；
- 发布记录关联commit、配置和benchmark。

### 监控与incident response

- 发布后检查Gateway 5xx/429、p95 latency、vLLM waiting、KV usage和GPU指标；
- AI可以帮助从日志产生假设，但人必须验证实际层级和命令；
- 把真实故障写成“现象→证据→根因→修复→验证”；
- 不允许AI为了让监控变绿而无依据提高权限或关闭安全控制。

## 6. 当前项目已经有的证据

- Codex参与了架构解释、Terraform/Helm/Gateway修改、文档和故障诊断；
- `.github/workflows/ci.yml`会独立执行Python、Terraform和Helm检查；
- `Makefile`提供相同的本地验证入口；
- `docs/04-operations-05-troubleshooting.md`保留真实错误、根因、修复和验证；
- 模型revision、vLLM image和部署参数固定；
- GPU费用和破坏性操作由用户确认。

仍应补充：

- 把仓库正式提交到Git并推送GitHub；
- 使用feature branch和PR，而不是所有文件一次提交；
- PR中保留需求、测试结果、风险、截图/benchmark和rollback；
- 至少完成一次“AI辅助实现→CI→部署→监控→回滚或复盘”的完整记录；
- 亲自复述AI修复过的核心故障，不依赖聊天记录。

## 7. 推荐的每次改动模板

```text
1. Problem
   用户/系统问题是什么？

2. Constraints
   安全、成本、兼容性和不能修改什么？

3. Plan
   Agent建议哪些方案？为什么选择当前方案？

4. Change
   修改哪些文件？有没有超出范围？

5. Verification
   跑了哪些测试？输出是什么？

6. Human review
   人工检查了哪些风险？批准了什么？

7. Release
   如何部署、canary和rollback？

8. Observe
   发布后看哪些指标和日志？

9. Learn
   发生了什么，下一次怎样改进？
```

## 8. 面试时怎样回答

推荐回答：

> I use coding agents as part of a controlled engineering workflow, not as an unchecked code generator. I define the problem, constraints and acceptance criteria; use Codex to inspect the repository, explore designs and implement scoped changes; then review the diff and run independent tests in CI. Infrastructure plans, GPU cost, IAM and destructive actions retain human approval. For releases I pin the model and image revision, verify health and metrics after rollout, and keep rollback steps. I also preserve incident evidence so I can explain the root cause without relying on the agent transcript.

中文含义：

> 我把Coding Agent作为受控工程流程的一部分，而不是不检查的代码生成器。我负责定义问题、约束和验收标准，让Codex检查仓库、比较方案并完成范围明确的修改；之后人工审查diff，并由CI独立验证。Terraform plan、GPU成本、IAM和破坏性操作保留人工批准。发布时固定模型、镜像和配置版本，检查健康与指标并准备回滚。我保留真实故障证据，因此即使不看Agent聊天，也能解释根因和修复。

## 9. 不能做出的说法

- 不要说“项目全部由我独立手写”，如果实际大量使用Agent。
- 不要说“AI替我完成了整个项目”，这会抹掉你的工程判断。
- 不要把通过一次测试说成完整SDLC。
- 不要把CI文件存在说成已经完成production release。
- 不要把Coding Agent和业务Agent/MCP混为一谈。

准确说法是：你使用AI提高开发速度，同时保留范围控制、diff review、自动测试、人工批准、运行验证和可解释性。

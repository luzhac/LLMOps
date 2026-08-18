# vLLM内存、显存与并发容量说明

本文只讨论当前项目的实际部署：`Qwen/Qwen3-8B-AWQ`、vLLM v0.26.0、一张NVIDIA L4和`g2-standard-8`节点。

## 1. vLLM运行在哪台机器

vLLM不在system管理节点运行。它的Pod通过`nodeSelector`选择L4节点，同时申请`nvidia.com/gpu: 1`，所以它运行在GPU node pool的`g2-standard-8`节点上。

Gateway、Argo CD和轻量监控主要运行在`e2-standard-2` system pool。Gateway通过Kubernetes Service `http://vllm:8000`访问GPU节点上的vLLM Pod。

```text
system node（CPU）                     GPU node（g2-standard-8）
Gateway / monitoring   ──HTTP──>      vLLM Pod ──> NVIDIA L4
```

## 2. 不要把三种存储混在一起

| 资源 | 当前配置 | 用途 |
|---|---:|---|
| L4 GPU显存（VRAM） | 24 GB GDDR6 | 模型权重、KV cache、activation、CUDA graph和kernel workspace。 |
| GPU节点普通内存（RAM） | 32 GB | Linux、kubelet、容器进程、tokenizer、host buffer、page cache和编译过程。它和GPU显存完全独立。 |
| vLLM Pod RAM request | 20 GiB | Kubernetes调度时核算的资源需求；不是启动时预先占满20 GiB。 |
| vLLM Pod RAM limit | 28 GiB | 容器普通内存上限；超过后可能被`OOMKilled`。 |
| `/dev/shm` size limit | 8 GiB | 内存型临时共享空间的最大值；不会启动即占满，实际使用会形成内存压力。 |
| 模型PVC | 30 GiB磁盘 | 保存下载的模型文件；不是RAM，也不是GPU显存。 |
| GPU节点启动盘 | 100 GB `pd-balanced` | 节点OS、容器镜像层和临时文件；节点scale-to-zero后节点本地cache会消失。 |

Google公布的`g2-standard-8`规格是8 vCPU、32 GB普通内存和一张24 GB L4。节点的32 GB不能全给vLLM，因为OS、Kubernetes和DaemonSet也要使用一部分。

## 3. 模型到底占多少

本项目固定revision的`Qwen3-8B-AWQ`权重索引记录`safetensors`权重总大小为：

```text
6,100,576,256 bytes
≈ 6.10 GB（十进制）
≈ 5.68 GiB（二进制）
```

这说明PVC上主要权重文件约6.1 GB，也大致说明量化权重本体的规模。但是“模型文件大小”不等于“vLLM总显存占用”。运行时显存包括：

```text
GPU显存
  = AWQ模型权重和未量化张量
  + CUDA/PyTorch运行时
  + activation和临时workspace
  + CUDA graph
  + vLLM的KV cache block pool
```

AWQ是4-bit weight-only quantization。它主要压缩线性层权重；activation、部分张量和KV cache并不因此全部变成4 bit。

项目设置：

```yaml
gpuMemoryUtilization: "0.90"
```

因此vLLM会把24 GB显存的约90%，也就是约21.6 GB，作为该实例的显存规划目标。它先profile权重、运行时和峰值activation，再把余量分配给KV block pool。

所以不能简单地说：

```text
24 - 6.1 = 17.9 GB KV cache
```

中间还必须扣掉运行时、activation、CUDA graph和workspace。准确的权重加载量和KV容量应查看当次vLLM启动日志。

## 4. 用户增加时，显存怎样变化

模型权重只加载一份，不会每来一个用户就再复制约6.1 GB模型。

多个用户共享：

- 同一份模型权重；
- 同一个vLLM engine；
- 同一个预分配KV block pool。

每个活跃序列拥有自己的KV blocks。prompt越长、已经生成的token越多、同时活跃的序列越多，KV使用量越大。请求结束后，其KV blocks会被释放复用；启用prefix caching时，部分可复用前缀可能暂时保留，之后也可以被淘汰。

vLLM通常在启动时预先规划和分配KV block pool。因此：

- `nvidia-smi`可能从启动开始就显示较高显存占用；
- 用户从1增加到8时，整个进程的allocated显存不一定线性上涨；
- 真正明显上涨的是池内已使用KV blocks比例，即`vllm:kv_cache_usage_perc`；
- GPU计算利用率、running/waiting requests、TTFT和inter-token latency也会变化。

到达容量上限后，常见结果不是立刻OOM，而是请求进入vLLM waiting queue、延迟上升，或者先被本项目Gateway的并发保护返回HTTP 429。极端长上下文、配置不合理或运行时峰值仍可能造成显存OOM。

## 5. 普通RAM怎样变化

GPU节点的32 GB普通RAM用于：

- 容器、Python和vLLM进程；
- tokenizer和请求对象；
- 模型加载期间的host-side读取、mmap/page cache或临时buffer；
- HTTP/SSE buffer；
- kernel compilation过程；
- 节点OS、kubelet和GPU相关DaemonSet。

用户并发、长prompt和大量网络buffer会让普通RAM有所增加，但正常推理中，并发容量更常先受GPU计算和GPU KV cache约束。每个请求不会在普通RAM中复制完整模型。

需要区分Kubernetes的request和limit：

- `memory request: 20Gi`用于调度决策，不代表固定占用20 GiB；
- `memory limit: 28Gi`是上限，超过后容器可能被内核杀死；
- 节点虽有32 GB RAM，可分配给Pod的量会小于32 GB，因为系统要预留资源。

## 6. 现场查看真实数值

模型启动后执行以下命令。

```bash
# vLLM Pod调度到了哪台节点
kubectl -n llm-platform get pod \
  -l app.kubernetes.io/name=vllm -o wide

# Pod和节点的普通RAM/CPU；需要metrics-server可用
kubectl top pod -n llm-platform --containers
kubectl top node

# L4总显存、已用显存、空闲显存和GPU利用率
kubectl -n llm-platform exec deployment/vllm -- \
  nvidia-smi \
  --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv

# 启动日志中的权重、memory profile和KV block信息
kubectl -n llm-platform logs deployment/vllm --tail=500 | \
  grep -Ei 'weight|memory|kv cache|gpu blocks|profile'

# 是否发生普通RAM OOM、显存OOM或调度问题
kubectl -n llm-platform describe pod \
  -l app.kubernetes.io/name=vllm
```

若模型当前scale到0，前两条只能看到没有vLLM Pod；先执行`bash scripts/model-session.sh up`并等待Ready。

## 7. Grafana里两个容易混淆的指标

| 指标 | 实际回答的问题 |
|---|---|
| `vllm:kv_cache_usage_perc` | vLLM预分配的KV cache block pool当前用了多少。 |
| DCGM framebuffer used | 整张GPU的显存当前用了多少，包含权重、KV、runtime等。 |

因此“GPU显存已经用了约90%”和“KV cache只用了10%”可以同时成立：大块显存已由vLLM规划/占用，但其中的KV blocks只有一小部分正在承载活跃token。

## 8. 面试时的一分钟回答

> 我们把vLLM Pod固定调度到一台`g2-standard-8` GPU节点。节点有32 GB host RAM，L4有独立的24 GB VRAM。AWQ模型权重文件约6.1 GB，但总显存还包括runtime、activation、CUDA graph和KV block pool。vLLM按`gpu-memory-utilization=0.9`规划约90%的显存，并把扣除非KV部分后的余量用于KV cache。并发用户共享一份模型，每个序列只增加自己的KV blocks；所以并发升高主要提升KV使用率、GPU负载和延迟，不是每个用户复制一份模型。host RAM由Pod设置20 GiB request、28 GiB limit，真实值分别用`kubectl top`和`nvidia-smi`观察。

## 参考

- Google Cloud G2/L4规格：https://docs.cloud.google.com/compute/docs/gpus
- 当前模型权重索引：https://huggingface.co/Qwen/Qwen3-8B-AWQ/blob/cb7d6a337aadb4d2082ed0dcef1032e4f8645194/model.safetensors.index.json
- 当前项目参数：`infra/terraform/variables.tf`、`infra/terraform/gke.tf`和`platform/helm/trade-balance-llm/values.yaml`

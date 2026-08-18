# Machine Learning from First Principles：机器学习第一性原理

这是一份独立的机器学习（Machine Learning, ML）复习教材，来源于个人GitBook仓库 [luzhac/book](https://github.com/luzhac/book) 中的旧笔记，并重新组织、纠错和补充英文面试术语。

本文和 [Transformer from First Principles](01-ai-learning-02-transformer-from-first-principles.md) 分开：

- 本文讲传统机器学习、通用神经网络训练和强化学习概览；
- Transformer文档讲tokenization、embedding、Attention、Transformer Block、语言模型训练和推理；
- 本文不会展开Q/K/V、RoPE、KV cache、vLLM或GPU推理。

## 0. 学完以后应该能做到什么

| 级别 | 能力 |
|---|---|
| L1 术语 | 能用中英文解释sample、feature、label、model、loss、gradient和metric。 |
| L2 任务 | 能判断问题属于回归、分类、聚类、降维还是强化学习。 |
| L3 数据 | 能写出X和y的形状，并正确划分train/validation/test。 |
| L4 模型 | 能解释线性模型、树、随机森林、SVM、K-Means、PCA和神经网络。 |
| L5 评估 | 能根据业务代价选择metric、threshold和validation方法。 |
| L6 实践 | 能建立无数据泄漏的scikit-learn训练与评估pipeline。 |

### 0.1 英文面试核心术语（Core Interview Terminology）

| 中文 | 英文 | 简短英文解释 |
|---|---|---|
| 样本 | sample / observation | A sample is one row of data. |
| 特征 | feature / input variable | A feature is an input used to make a prediction. |
| 标签 | label / target | The label is the value the model tries to predict. |
| 特征矩阵 | feature matrix | The feature matrix X contains samples in rows and features in columns. |
| 参数 | parameter | Parameters are learned from training data. |
| 超参数 | hyperparameter | Hyperparameters control training or model capacity. |
| 训练集 | training set | The training set is used to fit model parameters. |
| 验证集 | validation set | The validation set is used for model selection and tuning. |
| 测试集 | test set | The test set estimates final generalization performance. |
| 监督学习 | supervised learning | Supervised learning uses labeled examples. |
| 无监督学习 | unsupervised learning | Unsupervised learning looks for structure without labels. |
| 强化学习 | reinforcement learning | An agent learns a policy from interaction and rewards. |
| 回归 | regression | Regression predicts a continuous value. |
| 分类 | classification | Classification predicts a class or class probability. |
| 聚类 | clustering | Clustering groups similar samples without labels. |
| 降维 | dimensionality reduction | Dimensionality reduction maps data to fewer dimensions. |
| 损失函数 | loss function | The loss is the objective minimized during training. |
| 评估指标 | evaluation metric | A metric measures model performance. |
| 梯度 | gradient | A gradient measures how the loss changes with a parameter. |
| 梯度下降 | gradient descent | Gradient descent updates parameters in the negative-gradient direction. |
| 正则化 | regularization | Regularization discourages unnecessary complexity. |
| 过拟合 | overfitting | The model learns training noise and fails to generalize. |
| 欠拟合 | underfitting | The model is too simple to capture the useful pattern. |
| 数据泄漏 | data leakage | Training uses information unavailable at prediction time. |
| 交叉验证 | cross-validation | A model is evaluated across multiple data splits. |
| 混淆矩阵 | confusion matrix | It counts true and false predictions by class. |
| 精确率 | precision | Among predicted positives, how many are correct? |
| 召回率 | recall / sensitivity | Among actual positives, how many did we find? |
| 模型产物 | model artifact | A model artifact is the saved output of training, such as weights and preprocessing metadata. |
| 模型注册表 | model registry | A model registry versions approved model artifacts and their metadata. |
| 在线端点 | online endpoint | An endpoint serves low-latency predictions for incoming requests. |
| 批量推理 | batch inference | Batch inference predicts a stored dataset instead of serving one live request at a time. |
| 特征存储 | feature store | A feature store keeps reusable training and serving features consistent. |
| 数据漂移 | data drift | The input distribution changes over time. |
| 概念漂移 | concept drift | The relationship between inputs and the target changes over time. |
| 机器学习运维 | MLOps | MLOps makes training, deployment, monitoring and governance repeatable. |

## 1. 机器学习到底在做什么（What Machine Learning Does）

传统程序由人编写规则：

```text
输入 + 人工规则 → 输出
```

机器学习从历史数据中学习参数：

```text
训练数据 + 学习算法 → 模型参数
新输入 + 模型参数 → 预测
```

设：

- `X`：输入特征（features）；
- `y`：真实目标（target）；
- `f_θ`：参数为`θ`的模型；
- `ŷ = f_θ(X)`：模型预测；
- `L(y, ŷ)`：损失函数。

训练寻找使损失较小的参数：

$$
\theta^*=\arg\min_\theta L\left(y,f_\theta(X)\right)
$$

真正目标不是记住训练数据，而是在没见过的数据上仍然有效，这叫泛化（generalization）。

## 2. 学习问题的类别（Learning Paradigms）

| 类型 | 训练数据 | 学习目标 | 典型任务 |
|---|---|---|---|
| 监督学习（Supervised Learning） | `X + y` | 学习输入到目标的映射 | 分类、回归 |
| 无监督学习（Unsupervised Learning） | 只有`X` | 发现结构或表示 | 聚类、PCA、异常检测 |
| 强化学习（Reinforcement Learning） | 状态、动作、奖励、下一状态 | 学习长期奖励最大的策略 | 游戏、机器人、动态决策 |

还常见：

- 半监督学习（semi-supervised learning）：少量标注数据加大量未标注数据；
- 自监督学习（self-supervised learning）：从数据本身构造训练目标；
- 在线学习（online learning）：数据到达时持续更新模型。

重要纠正：LDA（Linear Discriminant Analysis）需要类别标签，因此是有监督方法，不属于无监督学习。

## 3. 数据怎样表示（Data Representation and Shapes）

传统表格机器学习通常使用：

```text
X: [N, D]
y: [N] 或 [N, K]
```

其中：

- `N`：样本数（number of samples）；
- `D`：特征数（number of features）；
- `K`：输出目标数或类别数。

例如4位客户，每位客户有年龄、收入和交易次数3个特征：

```text
X = [
  [25, 30000,  5],
  [40, 70000, 20],
  [31, 45000, 11],
  [52, 90000, 28]
]
shape(X) = [4, 3]
```

如果目标是“是否流失”：

```text
y = [1, 0, 1, 0]
shape(y) = [4]
```

在这里：

- 一行（row）是一个sample；
- 一列（column）通常是一个feature；
- `X[2,1]`是第3个样本的第2个特征；
- 表格feature通常有明确业务含义，不等于embedding dimension。

## 4. 从原始数据到可信模型（End-to-End Workflow）

```mermaid
flowchart LR
    A["定义问题和业务代价"] --> B["收集与理解数据"]
    B --> C["划分train/validation/test"]
    C --> D["只在train上拟合预处理"]
    D --> E["训练候选模型"]
    E --> F["在validation/CV上选择"]
    F --> G["在test上做一次最终评估"]
    G --> H["部署、监控、再训练"]
```

### 4.1 先划分，再预处理

正确顺序：

1. 先划分训练、验证和测试数据；
2. 只用训练集计算均值、标准差、填充值或PCA方向；
3. 把训练得到的预处理器应用到验证集和测试集。

错误顺序：

```text
全部数据标准化 → 再拆分
```

这让测试集统计信息进入训练流程，属于数据泄漏（data leakage）。

### 4.2 怎样划分

- 随机划分（random split）：样本近似独立同分布时；
- 分层划分（stratified split）：分类任务中保持类别比例；
- 按组划分（group split）：同一用户、病人或设备不能跨集合；
- 时间划分（time-based split）：用过去训练、未来验证。

交易和时间序列任务通常不能随便shuffle，否则未来信息可能泄漏到过去。

### 4.3 常见预处理

| 问题 | 常见方法 | 注意事项 |
|---|---|---|
| 缺失值 | median/mode imputation | 填充值只从训练集计算。 |
| 尺度不同 | standardization / normalization | SVM、K-Means、PCA通常较敏感。 |
| 类别变量 | one-hot、ordinal、target encoding | target encoding特别容易泄漏。 |
| 异常值 | 检查、变换、clip/Winsorize | 真实极端事件不能机械删除。 |
| 长尾分布 | log transform、robust scaling | 变换要符合数据含义。 |
| 类别不平衡 | class weight、resampling、threshold | 不要只看accuracy。 |

Winsorization只适合确认极端值主要是噪声或测量误差等情况。金融尾部事件可能正是重要信号，不能因为“极端”就自动截尾。

## 5. 监督学习：回归（Regression）

### 5.1 线性回归（Linear Regression）

单样本形式：

$$
\hat y=\mathbf w^T\mathbf x+b
$$

批量形式：

$$
\hat{\mathbf y}=X\mathbf w+b
$$

常用均方误差（Mean Squared Error, MSE）：

$$
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
$$

理解：

- `w_j`表示其他条件不变时，第`j`个特征增加1单位带来的预测变化；
- 平方使较大的误差受到更强惩罚；
- 线性回归是“对参数线性”，输入可以包含`x²`或交互项。

经典统计推断会讨论残差独立、同方差和正态性。不能简单说“输入数据必须正态分布”；正态假设主要影响置信区间和显著性检验。

### 5.2 回归指标（Regression Metrics）

| 指标 | 含义 | 特点 |
|---|---|---|
| MAE | 平均绝对误差 | 与目标同单位，对大误差相对稳健。 |
| MSE | 平均平方误差 | 强烈惩罚大误差。 |
| RMSE | MSE平方根 | 与目标同单位。 |
| R² | 相对均值基准解释的方差比例 | 可能为负，不是accuracy。 |
| MAPE | 平均绝对百分比误差 | 目标接近0时不稳定。 |

## 6. 监督学习：分类（Classification）

### 6.1 逻辑回归（Logistic Regression）

逻辑回归虽然名字包含regression，但解决分类问题。

先计算线性分数（logit）：

$$
z=\mathbf w^T\mathbf x+b
$$

再通过Sigmoid得到正类概率：

$$
p(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}}
$$

反过来：

$$
\log\frac{p}{1-p}=z
$$

左边叫对数几率（log-odds）或logit。逻辑回归假设log-odds与特征线性，而不是“目标y与特征线性”。

二分类交叉熵（binary cross-entropy / log loss）：

$$
L=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$

### 6.2 概率、阈值和标签

```text
模型分数/logit → probability → threshold → predicted label
```

默认阈值0.5并非天然正确：

- 漏诊代价高：优先提高recall；
- 误报代价高：优先提高precision；
- 人工审核容量有限：选择符合容量的阈值。

### 6.3 混淆矩阵与指标

| | 实际正类 | 实际负类 |
|---|---:|---:|
| 预测正类 | TP | FP |
| 预测负类 | FN | TN |

$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$

$$
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

$$
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
$$

- Accuracy适合类别较平衡且错误代价接近的情况；
- ROC-AUC衡量排序能力；
- PR-AUC在正类很少时通常更有解释力；
- AUC不告诉你某个阈值下的具体业务结果；
- 概率还需要检查校准（calibration）。

## 7. 树模型与集成学习（Trees and Ensembles）

### 7.1 决策树（Decision Tree）

决策树反复选择特征和切分点，把数据划分为更纯的子集。

分类树常用Gini impurity或entropy；回归树常用MSE下降。

优点：

- 学习非线性和特征交互；
- 通常不需要标准化；
- 规则可以可视化。

缺点：

- 单棵树高方差，容易过拟合；
- 小的数据变化可能产生不同的树；
- 贪心切分不保证全局最优。

常用控制参数：`max_depth`、`min_samples_split`、`min_samples_leaf`。

### 7.2 随机森林（Random Forest）

随机森林组合多棵不同的决策树：

1. 对样本做bootstrap sampling；
2. 每次切分只考虑随机特征子集；
3. 分类多数投票，回归取平均。

这叫Bagging（Bootstrap Aggregating），主要降低单棵树的variance。

注意：

- feature importance不等于因果影响；
- impurity importance可能偏爱高基数或连续特征；
- 是否原生处理缺失值取决于具体实现；
- 更多树通常更稳定，但计算成本也会上升。

Boosting与Bagging不同：Boosting按顺序训练弱学习器，后续模型修正之前的误差。常见实现包括Gradient Boosting、XGBoost、LightGBM和CatBoost。

## 8. 支持向量机（Support Vector Machine, SVM）

线性SVM寻找最大间隔（maximum-margin）的决策边界。离边界最近并决定边界位置的样本叫支持向量（support vectors）。

关键概念：

- margin：分类边界到最近样本的距离；
- `C`：间隔宽度与训练错误之间的权衡；
- kernel trick：计算高维特征空间中的相似性；
- `gamma`：RBF核中单个样本的作用范围。

适合中小规模、高维或边界清晰的问题。限制是：

- 核SVM在大样本下可能很慢；
- 对特征尺度敏感；
- `C`和`gamma`需要验证；
- SVM并非天然对异常值鲁棒，过大的`C`可能迫使边界拟合异常样本。

## 9. 无监督学习（Unsupervised Learning）

### 9.1 K-Means

K-Means最小化样本到所属质心的平方距离：

$$
\sum_{i=1}^{N}\|x_i-\mu_{c_i}\|^2
$$

步骤：

1. 初始化K个质心；
2. 样本分给最近质心；
3. 重新计算每个簇的均值；
4. 重复直到收敛。

它适合近似球形、尺度和密度相近的簇。需要预先选择K，并且对尺度和异常值敏感。

### 9.2 DBSCAN

DBSCAN根据局部密度形成簇：

- `eps`：邻域半径；
- `min_samples`：形成核心点所需邻居数；
- 标签`-1`通常表示噪声点。

它能发现非球形簇且不必预设簇数，但不同密度的簇较难使用同一参数，高维距离也可能失去区分度。

聚类标签没有天然业务意义。得到cluster 0、1、2后，仍需结合特征分布和业务知识解释。

## 10. 降维（Dimensionality Reduction）

| 方法 | 使用标签吗 | 目标 | 常见用途 |
|---|---|---|---|
| PCA | 否 | 保留最大方差方向 | 压缩、去噪、建模前处理 |
| LDA | 是 | 增大类间分离、减小类内分散 | 有标签分类降维 |
| t-SNE | 否 | 尽量保留局部邻域 | 探索性二维/三维可视化 |
| Autoencoder | 通常不需人工标签 | 学习可重构输入的潜在表示 | 非线性压缩、去噪、异常检测 |

### 10.1 PCA

PCA寻找相互正交的主成分（principal components）。第一主成分解释最大方差，后续主成分在与前面正交的条件下解释剩余方差。

PCA对尺度敏感，通常需要标准化。主成分是原始特征的线性组合，不一定有直接业务含义。

### 10.2 LDA与t-SNE的边界

- LDA使用标签，是supervised dimensionality reduction；
- t-SNE图中的簇间距离和簇面积不宜作强解释；
- t-SNE受随机种子、perplexity和预处理影响；
- t-SNE主要用于探索性可视化，漂亮分群不是模型有效性的证明。

## 11. 泛化、正则化与模型选择（Generalization）

### 11.1 欠拟合与过拟合

| 现象 | 训练表现 | 验证表现 | 常见原因 |
|---|---|---|---|
| 欠拟合 | 差 | 差 | 模型太简单、特征不足、训练不足 |
| 合理拟合 | 好 | 接近训练表现 | 容量和数据匹配 |
| 过拟合 | 很好 | 明显较差 | 模型太复杂、数据少、泄漏或噪声 |

### 11.2 L1与L2正则化

$$
L_{total}=L_{data}+\lambda R(\theta)
$$

- L1：`R(w)=Σ|w_j|`，倾向得到稀疏权重；
- L2：`R(w)=Σw_j²`，平滑压缩大权重；
- `λ`越大，正则化越强。

正则化可能增加训练误差，却降低验证误差。这是用少量bias换取较低variance。

### 11.3 参数与超参数

参数（parameters）由训练学习，例如线性系数、神经网络权重和决策树切分。

超参数（hyperparameters）由人或搜索过程选择，例如：

- learning rate；
- tree depth；
- number of trees；
- SVM的`C`和`gamma`；
- regularization strength。

测试集不能参与超参数选择，否则它不再是独立的最终评估。

## 12. 神经网络训练基础（Neural Network Training）

本节只讲通用神经网络，不展开Transformer架构。

### 12.1 线性层与激活函数

一层神经网络：

$$
z=xW+b
$$

$$
h=\phi(z)
$$

其中`W,b`是参数，`z`是pre-activation，`φ`是activation function，`h`是输出。

如果只堆叠线性层而没有非线性激活，多层仍可合并成一个线性变换，无法学习复杂非线性边界。

| 激活函数 | 范围/形式 | 用途和限制 |
|---|---|---|
| ReLU | `max(0,z)` | 简单高效；可能出现dead ReLU。 |
| Sigmoid | `(0,1)` | 二分类输出；饱和区梯度很小。 |
| Tanh | `(-1,1)` | 零中心，但也会饱和。 |
| GELU/SiLU | 平滑门控 | 现代深层网络常用。 |

纠正：ReLU缓解了部分梯度消失，但不能说它完全消除了梯度消失。

### 12.2 Forward、Loss、Backpropagation与Optimizer

```mermaid
flowchart LR
    X["输入batch"] --> F["Forward pass"]
    F --> P["预测/logits"]
    P --> L["Loss"]
    L --> B["Backpropagation"]
    B --> G["Gradients"]
    G --> O["Optimizer step"]
    O --> U["更新parameters"]
```

前向传播（forward pass）产生预测并计算loss。反向传播（backpropagation）使用链式法则从loss向后计算每个参数的梯度。

梯度下降更新：

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L
$$

其中`η`是学习率（learning rate）。

### 12.3 Batch、Step与Epoch

- batch：一次forward/backward使用的一组样本；
- batch size：每个batch的样本数；
- iteration/step：一次参数更新；
- epoch：完整遍历训练集一次。

若10,000个样本、batch size为100，则每个epoch有100个steps。

### 12.4 Logits、Softmax与Cross-Entropy

多分类模型输出`K`个logits：

```text
logits = [2.0, 3.0, 1.0]
```

Softmax转换为概率：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

正确类别索引为`c`时，单样本交叉熵为：

$$
L=-\log p_c
$$

深度学习框架通常直接接收logits，在内部组合log-softmax和negative log-likelihood以保证数值稳定。不要在`CrossEntropyLoss`前手动Softmax，除非API明确要求概率。

Logits是一次forward得到的结果，不是模型参数。参数与输入共同决定logits。

### 12.5 Optimizer

- SGD：按当前梯度更新；
- Momentum：积累平滑的更新方向；
- Adam/AdamW：结合动量和自适应尺度；
- loss决定优化目标，optimizer决定如何使用梯度。

### 12.6 CNN、LSTM、Autoencoder与Embedding

- CNN（Convolutional Neural Network）：用局部连接和权重共享处理网格数据；
- LSTM（Long Short-Term Memory）：用门控状态处理序列依赖；
- Autoencoder：encoder压缩输入，decoder重构输入；
- Embedding：把离散类别或ID映射为可训练的稠密向量。

Embedding不是Transformer专属概念。推荐系统、文本分类和实体表示都可以使用embedding。

## 13. 强化学习概览（Reinforcement Learning Overview）

强化学习不是固定的`X → y`数据集，而是智能体与环境的循环：

```text
state → agent chooses action → environment returns reward and next state
```

| 中文 | 英文 | 含义 |
|---|---|---|
| 智能体 | agent | 作出动作的学习者 |
| 环境 | environment | 接收动作并返回状态和奖励 |
| 状态 | state | 当前决策所需信息 |
| 动作 | action | 智能体可选择的行为 |
| 奖励 | reward | 环境给出的即时反馈 |
| 策略 | policy | 从状态到动作分布的规则 |
| 价值函数 | value function | 未来累计奖励的期望 |
| 探索/利用 | exploration/exploitation | 尝试新动作与使用已知好动作的权衡 |

算法关系：

- Q-learning：学习state-action value；
- DQN：用神经网络逼近Q值；
- Policy Gradient：直接优化policy；
- Actor-Critic：actor学习策略，critic估计价值；
- PPO：限制策略更新幅度，提高训练稳定性。

交易RL要警惕非平稳市场、回测泄漏、交易成本、滑点和奖励函数投机。高回测reward不等于可部署策略。

## 14. 机器学习应用领域地图（Machine Learning Application Map）

先分清两个维度：

- 学习范式（learning paradigm）说明模型怎样获得学习信号，例如supervised、unsupervised、self-supervised和reinforcement learning；
- 应用领域（application domain）说明模型解决什么现实问题，例如recommendation、computer vision和fraud detection。

同一个应用领域可以使用多种学习范式。例如推荐系统既可以使用监督学习预测点击，也可以使用self-supervised learning学习用户和商品embedding，还可以用强化学习优化长期互动。

| 应用领域 | 常见输入 | 常见输出 | 常用模型 | 关键指标与风险 |
|---|---|---|---|---|
| 预测分析（predictive analytics） | 表格特征 | 数值、类别或风险分数 | linear/logistic regression、trees、boosting | RMSE、F1、AUC、calibration；警惕data leakage |
| 时间序列（time-series forecasting） | 按时间排列的观测 | 未来数值或事件 | baseline、ARIMA、trees、RNN/Transformer | MAE、RMSE、MAPE；必须按时间验证 |
| 推荐系统（recommender system） | 用户、商品、互动历史 | 排名或点击概率 | collaborative filtering、matrix factorization、two-tower | Recall@K、NDCG@K；冷启动与反馈回路 |
| 自然语言处理（NLP） | token序列 | 类别、实体、文本或embedding | linear model、RNN、Transformer | F1、BLEU/ROUGE或任务指标；偏见与幻觉 |
| 计算机视觉（computer vision） | 图像或视频 | 类别、框、mask或生成图像 | CNN、Vision Transformer | accuracy、mAP、IoU；分布偏移 |
| 语音识别（speech recognition） | 音频波形或spectrogram | 文本序列 | CTC、RNN、Transformer | WER（word error rate）；口音与噪声 |
| 自主系统（autonomous systems） | 传感器与状态 | 动作或轨迹 | perception model、planning、RL | safety、latency、failure rate；尾部事故 |
| 生成模型（generative models） | prompt、条件或噪声 | 文本、图像、音频 | autoregressive model、diffusion model | quality、factuality、safety；评估不能只靠单一分数 |
| 金融建模与算法交易（financial modeling and algorithmic trading） | 市场、基本面与另类数据 | 回报、风险、仓位或动作 | regression、trees、time-series model、RL | walk-forward结果、Sharpe、drawdown；成本、滑点和泄漏 |

### 14.1 时间数据为什么不能普通随机切分

若要预测明天，训练时只能看到明天以前的信息。普通random split可能把未来样本分进训练集，再用过去样本验证，造成look-ahead bias。

应按时间使用：

```text
train: [过去────────] validation: [较近──] test: [最新──]
```

模型调优可使用rolling validation或walk-forward validation。任何feature、scaler和target encoding也必须只用当时可获得的数据拟合。

### 14.2 应用名称不等于算法名称

“推荐系统”不是单一算法，“深度学习”也不是一个应用领域。面试时应按以下顺序回答：

1. 业务决策是什么；
2. 输入和目标是什么；
3. 学习范式与模型候选是什么；
4. 离线指标和业务指标是什么；
5. 哪些数据、部署与安全风险会让结果失效。

## 15. 从模型到生产：MLOps与AWS映射（MLOps and AWS Mapping）

训练出高分模型只是中间步骤。生产系统需要一条可重复、可审计的生命周期：

```text
data → processing → training → model artifact → registry
                                            ↓
                         online endpoint / batch inference
                                            ↓
                              monitoring → retraining
```

| 生命周期环节 | 要解决的问题 | AWS笔记中的对应服务 |
|---|---|---|
| 数据与处理（data and processing） | 清洗、变换、数据版本和可复现性 | Amazon S3、SageMaker Processing、Data Wrangler |
| 训练（training） | 隔离训练环境、记录参数与指标、扩展算力 | SageMaker Training Jobs、Debugger、Profiler、distributed training、Spot |
| 模型产物与审批（artifact and approval） | 保存weights、preprocessing和metadata；版本化与审批 | S3 model artifact、SageMaker Model Registry |
| 在线推理（online inference） | 低延迟、按请求预测、扩缩容 | SageMaker real-time endpoint、autoscaling |
| 批量推理（batch inference） | 对S3中的大批数据异步预测，不维持常驻端点 | SageMaker Batch Transform |
| 编排与自动化（orchestration） | 把处理、训练、评估、审批和部署组成DAG/CI-CD | SageMaker Pipelines、Step Functions、EventBridge、Lambda |
| 特征管理（feature management） | 复用特征并减少training-serving skew | SageMaker Feature Store |
| 监控与治理（monitoring and governance） | 监控系统、数据、模型效果、公平性和解释性 | CloudWatch、Model Monitor、Clarify |
| 安全（security） | 最小权限、网络隔离、加密和审计 | IAM、VPC、KMS、CloudTrail |

### 15.1 四个容易混淆的概念

- model artifact是训练保存下来的文件；endpoint是加载某个artifact并对外提供推理的运行服务。文件本身不会自动接收请求。
- online endpoint适合低延迟、逐请求预测；batch inference适合已经存储的大批数据，不要求即时返回。
- data drift是输入分布$P(X)$变化；concept drift是输入到目标的关系$P(y\mid X)$变化。前者可在没有即时标签时检测，后者通常需要延迟到来的真实标签。
- feature store的核心价值不是“多存一份数据”，而是特征复用、版本管理和online/offline一致性，减少training-serving skew。

### 15.2 生产监控至少分四层

| 层 | 例子 |
|---|---|
| Operational monitoring | latency、error rate、CPU/GPU、endpoint availability |
| Data quality and drift | schema、missing values、range、distribution shift |
| Model performance | accuracy、precision/recall、calibration、business KPI |
| Bias and explainability | subgroup performance、feature attribution、approval evidence |

只监控endpoint是否存活不够；只监控data drift也不能证明模型效果已经下降。

### 15.3 安全与成本不是最后再补

- IAM使用least privilege，不把access key、password或token写入Git；
- 敏感数据按需使用VPC、encryption at rest/in transit和审计日志；
- dev/test endpoint使用后及时关闭，生产根据流量autoscale；
- Spot training能降低成本，但训练程序应支持checkpoint和中断恢复；
- 自动重训练后仍应通过验证、审批与回滚条件，不能见到漂移就盲目上线。

## 16. 一个无数据泄漏的scikit-learn骨架

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_features = ["age", "income", "trade_count"]
categorical_features = ["country", "account_type"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]
prediction = (probability >= 0.5).astype(int)

print("ROC-AUC:", roc_auc_score(y_test, probability))
print(classification_report(y_test, prediction))
```

Pipeline的价值：

- imputer和scaler只在训练数据上fit；
- 验证或测试数据只transform；
- 交叉验证时每个fold独立拟合预处理；
- 部署时复用相同的预处理逻辑。

正式建模还应增加cross-validation、threshold选择、校准、数据漂移和业务指标监控。

## 17. 旧GitBook笔记需要纠正的地方

| 旧表述或结构 | 更准确的理解 |
|---|---|
| 把LDA列为无监督学习 | LDA使用标签，是监督降维/分类方法。 |
| ReLU不存在梯度消失 | ReLU缓解部分梯度消失，但有dead ReLU等问题。 |
| SVM对噪声和异常值鲁棒 | 程度取决于C和数据；异常值仍可能影响边界。 |
| 随机森林自动处理缺失值 | 取决于库和版本，应明确设计缺失值策略。 |
| t-SNE不可用于任何特征工程 | 它主要用于探索性可视化，不适合作为通用可复现变换。 |
| 逻辑回归假设y与X线性 | 它假设log-odds与特征线性相关。 |
| AUC越高模型一定可用 | 还要看阈值、校准、业务代价和数据漂移。 |
| 测试集可以反复调参 | 这会让测试集泄漏进模型选择。 |
| 极端值应该Winsorize | 先判断它是错误、噪声还是真实尾部事件。 |
| feature importance代表因果 | 模型关联不自动具有因果解释。 |

## 18. 英文面试回答模板（Interview Answer Templates）

### What is the difference between a parameter and a hyperparameter?

> A parameter is learned from training data, such as a linear coefficient or a neural-network weight. A hyperparameter controls training or model capacity, such as the learning rate, tree depth, or regularization strength.

### What is overfitting?

> Overfitting happens when a model learns training-specific noise instead of the underlying pattern. It performs well on training data but significantly worse on unseen data.

### Why do we need a validation set?

> We use the validation set for model selection, hyperparameter tuning, and threshold selection. The test set should remain untouched until the final evaluation.

### What is data leakage?

> Data leakage occurs when training uses information that would not be available at prediction time. It produces overly optimistic offline metrics.

### Precision or recall: which matters more?

> It depends on the cost of each error. Recall matters when false negatives are expensive, while precision matters when false positives are expensive.

### What is the difference between loss and metric?

> The loss is the objective optimized during training. A metric evaluates model behavior and may reflect a business goal without being directly optimized.

### Why put preprocessing inside a pipeline?

> A pipeline fits preprocessing only on the training fold, prevents leakage, and keeps training and inference transformations consistent.

### What is backpropagation?

> Backpropagation applies the chain rule from the loss through the computation graph to calculate the gradient of every trainable parameter.

## 19. 主动回忆题（Active Recall）

1. `X: [N,D]`中的N和D分别是什么？
2. 回归与分类的输出有什么区别？
3. 为什么要先划分数据再标准化？
4. validation set和test set分别做什么？
5. 什么情况下不能随机划分？
6. MSE为什么对大误差敏感？
7. 为什么逻辑回归是分类算法？
8. logit、probability和predicted label有什么区别？
9. precision和recall分别回答什么问题？
10. 为什么AUC高不保证生产效果好？
11. 决策树为什么容易过拟合？
12. Random Forest怎样降低variance？
13. SVM中的margin、C和kernel分别做什么？
14. K-Means和DBSCAN适合什么数据？
15. PCA、LDA和t-SNE是否使用标签？
16. parameter与hyperparameter有什么区别？
17. L1与L2正则化有什么区别？
18. forward、loss、backprop和optimizer的顺序是什么？
19. batch、step和epoch有什么区别？
20. logits为什么不是模型参数？
21. CrossEntropyLoss为什么通常直接接收logits？
22. Q-learning、DQN、Actor-Critic和PPO有什么关系？
23. 举出三个data leakage例子。
24. 为什么feature importance不等于causality？
25. learning paradigm与application domain有什么区别？
26. 为什么普通random split不适合时间序列？
27. online endpoint与batch inference分别适合什么场景？
28. data drift与concept drift有什么区别？
29. feature store怎样减少training-serving skew？
30. model artifact、model registry和deployed endpoint是什么关系？

## 20. 七天复习计划

| 天 | 内容 | 必须输出 |
|---:|---|---|
| 1 | ML类型、X/y形状、数据划分 | 为三个业务问题选择任务并写形状。 |
| 2 | 线性回归、逻辑回归 | 手算线性预测、Sigmoid和loss。 |
| 3 | 分类指标与threshold | 从混淆矩阵计算precision、recall和F1。 |
| 4 | 决策树、随机森林、SVM | 用英文比较三个模型。 |
| 5 | 聚类和降维 | 为不同数据分布选择方法。 |
| 6 | 神经网络训练 | 画forward→loss→backprop→optimizer。 |
| 7 | Pipeline、MLOps与面试回答 | 写无泄漏pipeline，画出artifact→registry→deployment→monitoring，并口述英文回答。 |

第1、7、30天重新完成主动回忆题。先答题，再查文档，不要用反复重读代替知识提取。

## 21. 原始笔记映射

本教材按目录重新整理了GitBook仓库中的：

- `mlbasic/`：学习类型、数据处理、经典模型、神经网络基础与强化学习，构成第1—13节；
- `ml-appl/`：预测、时间序列、推荐、NLP、视觉、语音、自主系统和金融应用，整理为第14节；
- `mle-aws/`：训练、部署、编排、feature store、monitoring、安全与成本，抽象成vendor-neutral MLOps概念，并在第15节保留AWS服务映射。

以下目录有自己的学习目标，因此不塞进机器学习基础正文：

- `trade/`与`ai-trading/`：属于交易研究和AI trading应用；
- `gai/`：属于Generative AI、LLM和RAG；
- `agent/`：属于prompting、agentic workflow和multi-agent systems；
- Transformer基础继续保留在独立的 [Transformer from First Principles](01-ai-learning-02-transformer-from-first-principles.md)。

仓库根目录还保留了一份与`mlbasic/`重叠的旧导出。后续维护应选`mlbasic/`或本教材作为canonical source，避免两份内容继续分叉。原始GitBook仓库作为历史资料保留；本文是用于系统复习和英文面试的纠错重构版。

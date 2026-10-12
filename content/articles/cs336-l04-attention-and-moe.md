---
title: "CS336 L04：注意力替代方案与混合专家"
date: "2026-08-04"
description: "线性注意力、门控状态模型、稀疏注意力与 MoE：从计算形式到路由、负载均衡和通信成本。"
tags: ["CS336","Attention","MoE"]
category: "笔记"
---

## 1. 完整注意力的成本

### 1.1 标准注意力回顾

省略 batch 和 head 维度，设序列长度为 $n$，query/key 维度为 $d_k$，value 维度为 $d_v$：

$$
Q,K\in\mathbb{R}^{n\times d_k},
\qquad
V\in\mathbb{R}^{n\times d_v}.
$$

标准缩放点积注意力为：

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}\!\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M
\right)V,
$$

其中 $M$ 是因果 mask。中间的注意力分数矩阵：

$$
QK^{\mathsf T}\in\mathbb{R}^{n\times n}.
$$

这带来两个随 $n^2$ 增长的主要矩阵乘法：

$$
QK^{\mathsf T}:\quad O(n^2d_k),
$$

$$
AV:\quad O(n^2d_v).
$$

所以 attention 核心计算近似为：

$$
O\!\left(n^2(d_k+d_v)\right).
$$

完整 Transformer 还包含 $Q/K/V/O$ 投影和 FFN，它们通常随 $n$ 线性增长。因此上下文很长时，attention 的 $n^2$ 项会越来越突出。

### 1.2 解码时还有 KV cache 成本

自回归生成第 $t$ 个 token 时，不需要重新计算所有历史 $K,V$，而是把它们保存在 KV cache 中。但新 query 仍需读取并匹配前 $t$ 个位置：

$$
\text{单步 attention 成本}=O(td),
$$

生成完整长度 $n$ 的序列累计仍近似为 $O(n^2d)$。KV cache 本身也随序列长度线性增长。

### 1.3 不同的优化路径

在替换注意力之前，常见的“基础工具”包括：

- **局部/滑动窗口注意力（local/SWA）**：每个 token 只看固定窗口；
- **局部与全局层混合**：多数层便宜，少数层负责全局传播；
- **GQA/MQA**：减少 KV head 和 KV cache；
- **FlashAttention 与系统优化**：不改变数学结果，减少 HBM 读写和中间张量。

这里进一步讨论三类更激进的改变：

1. 通过结合律和核技巧得到线性注意力；
2. 将序列历史压缩为可递归更新的状态；
3. 用学习到的索引器只选择少数历史 token。

> GPU 擅长大而规则的矩阵乘法。递归、稀疏 gather/scatter、较小的矩阵和复杂门控可能降低硬件利用率。必须同时看理论 FLOPs、并行度、内存访问、kernel 成熟度和端到端吞吐。

## 2. 线性注意力与矩阵结合律

### 2.1 最简形式

先暂时令注意力中的非线性归一化 $\rho$ 为恒等映射：

$$
\operatorname{Attn}(Q,K,V)=QK^{\mathsf T}V.
$$

矩阵乘法满足结合律，因此：

$$
(QK^{\mathsf T})V
=
Q(K^{\mathsf T}V).
$$

两种计算顺序的中间张量不同。

#### 先计算 token-token 关系

$$
QK^{\mathsf T}\in\mathbb{R}^{n\times n},
$$

成本包含 $O(n^2d_k+n^2d_v)$。

#### 先聚合 key-value

$$
K^{\mathsf T}V
\in
\mathbb{R}^{d_k\times d_v},
$$

再计算：

$$
Q(K^{\mathsf T}V)
\in
\mathbb{R}^{n\times d_v}.
$$

成本约为：

$$
O(nd_kd_v),
$$

对固定的 $d_k,d_v$ 来说，它对序列长度 $n$ 是线性的，而且不显式构造 $n\times n$ 注意力矩阵。

### 2.2 为什么不能直接把 softmax 也交换？

标准注意力为：

$$
\operatorname{softmax}(QK^{\mathsf T})V.
$$

softmax 是非线性操作，一般不存在：

$$
\operatorname{softmax}(QK^{\mathsf T})V
=
Q\operatorname{softmax}(K^{\mathsf T}V).
$$

因此，上面的结合律只直接适用于没有 softmax 的线性形式。若想近似 softmax attention，可使用特征映射（feature map）或核技巧：

$$
\operatorname{sim}(q,k)
\approx
\phi(q)^{\mathsf T}\phi(k).
$$

则非因果形式可以写成：

$$
y_i
=
\frac{
\phi(q_i)^{\mathsf T}
\left(\sum_j\phi(k_j)v_j^{\mathsf T}\right)
}{
\phi(q_i)^{\mathsf T}
\left(\sum_j\phi(k_j)\right)
}.
$$

分子状态存储 key-value 聚合，分母状态提供归一化。不同线性注意力方法的差别之一，就是如何选择 $\phi$、如何做归一化、如何加入位置/衰减，以及如何高效实现。

> 这里通常指对序列长度 $n$ 的计算或状态更新为线性复杂度，不是说整个网络没有非线性。

### 2.3 因果线性注意力的递归形式

对于语言模型，第 $t$ 个位置只能使用 $1,\ldots,t$ 的信息。定义状态：

$$
S_t
=
S_{t-1}+k_tv_t^{\mathsf T},
$$

再读取：

$$
y_t=q_t^{\mathsf T}S_t.
$$

其中：

$$
k_t\in\mathbb{R}^{d_k},
\quad
v_t\in\mathbb{R}^{d_v},
\quad
S_t\in\mathbb{R}^{d_k\times d_v},
\quad
y_t\in\mathbb{R}^{d_v}.
$$

$k_tv_t^{\mathsf T}$ 是一个外积，可以理解为把“key $k_t$ 应映射到 value $v_t$”写入一个快速更新的记忆矩阵；$q_t^{\mathsf T}S_t$ 则从状态中读取与 query 相关的信息。

### 2.4 训练—推理对偶性

同一个运算有两种等价视角：

- **并行/矩阵视角**：对整段序列进行批量矩阵运算，适合训练；
- **递归/状态视角**：每一步更新 $S_t$，适合逐 token 解码。

递归推理只需保存固定 shape 的状态 $S_t$，而不是长度不断增长的全部 KV cache。每生成一个 token 的状态更新成本不直接随历史长度增长。

> 矩阵结合律展示了这两种计算形式的等价关系。现代实现还会使用 prefix scan、分块（chunkwise）算法或专用 kernel，在保持并行性的同时避免显式的完整 $n\times n$ 矩阵。核心思想是：**数学上既能作为序列矩阵运算，也能作为递归状态更新。**

### 2.5 线性注意力的表达能力代价

标准 attention 可以为每个 query 显式形成对所有历史 token 的不同权重；线性注意力把历史压进固定大小状态：

$$
S_t\in\mathbb{R}^{d_k\times d_v}.
$$

当上下文越来越长时，许多 key-value 关联必须共享这块有限容量，容易出现：

- 不同记忆相互干扰；
- 精确检索能力下降；
- 旧信息无法被有选择地遗忘或覆盖；
- 训练时表现不错，但长上下文 retrieval 不一定等价于完整 attention。

这解释了后续方法为什么不断加入**衰减、门控和定向擦除**。

## 3. 从线性注意力到门控状态模型

### 3.1 RetNet 式衰减：让旧状态逐渐遗忘

最简单的改造是在旧状态前乘衰减系数：

$$
S_t
=
\gamma S_{t-1}+k_tv_t^{\mathsf T},
\qquad 0\le\gamma\le 1.
$$

- $\gamma=1$：永久累加，接近基础线性注意力；
- $\gamma<1$：越旧的信息权重越小；
- 如果不同通道或位置有不同 $\gamma$，模型可以学习多种记忆时间尺度。

固定衰减比完全不遗忘更灵活，但对所有输入使用同一种遗忘策略仍然受限。

### 3.2 Mamba‑2：输入相关的选择性门控

Mamba‑2 的选择性状态更新可简化为：

$$
S_t
=
\gamma_tS_{t-1}+k_tv_t^{\mathsf T},
$$

$$
y_t
=
q_t^{\mathsf T}S_t+Dv_t,
$$

$$
\gamma_t=f(x_t).
$$

与固定衰减相比，$\gamma_t$ 由当前输入决定，因此模型可以根据内容决定保留或遗忘多少历史。$Dv_t$ 是直接通路（skip/direct term），允许当前输入不必全部经过状态记忆再到输出。

基础线性注意力像一本只会不断追加的笔记本；Mamba‑2 增加了由输入控制的“保留比例”，不同 token 可以触发不同程度的遗忘。

Mamba‑2 的理论背景是状态空间对偶性（state space duality，SSD）：某类结构化状态空间模型和某类受限注意力矩阵可以用统一形式表示。这让模型既有递归视角，也能用并行/分块算法训练。

> 这个简化形式展示了线性注意力与选择性状态更新的联系。完整 Mamba‑2 还包含参数化方式、分块算法、归一化和硬件友好实现等设计；这里公式用于理解核心状态更新，不是完整实现规范。

### 3.3 Gated DeltaNet：选择性擦除再写入

仅靠整体衰减 $\gamma_t$ 有一个问题：若要修改与当前 key 相关的记忆，可能不得不一起缩小其他方向的记忆。Delta rule 尝试更精确地更新与当前 key 对齐的方向。

Gated DeltaNet 的状态更新为：

$$
S_t
=
\gamma_t
\left(I-\beta_tk_tk_t^{\mathsf T}\right)S_{t-1}
+
\beta_tk_tv_t^{\mathsf T},
$$

$$
y_t=q_t^{\mathsf T}S_t,
$$

$$
\gamma_t=f_\gamma(x_t),
\qquad
\beta_t=f_\beta(x_t).
$$

逐项理解：

1. $\gamma_t$ 控制整体历史保留；
2. $I-\beta_tk_tk_t^{\mathsf T}$ 主要擦除与当前 key 方向相关的旧内容；
3. $\beta_tk_tv_t^{\mathsf T}$ 把新的 key-value 关系写入状态；
4. 当 $\beta_t=0$ 时，可以执行“本步不写入”的操作。

它不只是让整本笔记一起褪色，而是先找到与当前 key 相同的条目，擦掉旧答案，再写入新答案。这与快速权重编程（fast weight programming）和测试时训练（test-time training）有紧密联系。

### 3.4 三种状态更新的对比

| 方法 | 状态更新核心 | 能否遗忘 | 能否定向覆盖 | 主要限制 |
|---|---|---:|---:|---|
| 基础线性注意力 | $S_t=S_{t-1}+k_tv_t^{\mathsf T}$ | 否 | 否 | 状态干扰、旧记忆累积 |
| 衰减/Mamba‑2 式 | $S_t=\gamma_tS_{t-1}+k_tv_t^{\mathsf T}$ | 是 | 较弱 | 整体衰减可能误伤其他记忆 |
| Gated DeltaNet | 擦除 key 方向后再写入 | 是 | 是 | 算法和 kernel 更复杂 |

### 3.5 为什么现代模型常采用 hybrid？

固定大小的递归状态效率高，但完整 attention 的内容寻址和精确检索能力仍然很强。实际模型常混合两者：

- 多数层使用线性/状态层，控制长上下文成本；
- 少数层使用完整或局部 attention，恢复更灵活的 token-token 交互。

课程材料列举的 2026 案例包括：

- **MiniMax M1 / text‑01**：约 $7:1$ 的线性层与完整注意力层混合；
- **Nemotron 3**：大致采用 Mamba 与 attention 的混合；
- **Qwen 3.5 / Qwen Next**：约 $3:1$ 的 Gated DeltaNet 与 attention 混合。

这些案例用于说明趋势，不代表该比例对所有模型最优。课程材料也明确提醒：不同论文之间训练数据、预算和配方不完全可控，严格的同预算消融仍然有限。

## 4. 稀疏注意力：不压缩全部历史，而是只选相关位置

### 4.1 与线性状态方法的差别

- **线性/状态方法**：把所有历史压缩到固定大小状态；
- **稀疏注意力**：保留历史 KV，但每个 query 只访问其中一小部分。

设历史长度为 $n$，每个 query 只选择 $k\ll n$ 个位置，则核心 attention 可从：

$$
O(n^2d)
$$

降为近似：

$$
O(nkd).
$$

关键问题变成：**怎样便宜而准确地找到这 $k$ 个位置？**

### 4.2 DeepSeek Sparse Attention（DSA）

DSA 使用学习式索引器选择历史位置：

1. 轻量索引器（indexer）为 query 与历史候选位置估计相关性；
2. 选出 top-$k$ 位置；
3. 主 attention 只在选中位置计算。

这保留了显式检索历史 token 的能力，同时减少主 attention 的计算。但必须注意：

- 索引器本身也有计算成本；若仍对所有 query-key 对打分，理论上可能保留 $O(n^2)$ 的轻量项；
- top-$k$ 选择与不规则 KV 访问会增加 kernel 和内存系统难度；
- 训练和推理的选择结果应保持一致，否则会出现行为偏差；
- 课程材料指出，这类稀疏注意力可以在短上下文稠密预训练后进行 post-hoc adaptation，但具体训练方法应以对应模型论文为准。

> SWA 的稀疏模式由距离固定决定；DSA 的稀疏模式由内容相关的索引器动态决定。前者规则、容易高效实现，后者更灵活但系统复杂度更高。

## 5. 注意力方案对比

| 方法 | 历史如何表示 | 序列复杂度直觉 | 推理解码状态 | 优势 | 主要风险 |
|---|---|---:|---|---|---|
| Full attention | 全部 KV，任意 token-token 关系 | $O(n^2d)$ | 随 $n$ 增长的 KV cache | 表达力与检索强 | 长上下文昂贵 |
| SWA | 固定局部窗口 | $O(nwd)$ | 固定窗口 KV | 规则、kernel 友好 | 远距离传播需跨层 |
| Linear attention | 固定大小聚合状态 | $O(nd_kd_v)$ | 固定大小状态 | 线性扩展、递归解码 | 状态容量和干扰 |
| Mamba‑2 | 选择性衰减状态 | 近似线性 | 固定大小状态 | 输入相关遗忘 | 完整实现较复杂 |
| Gated DeltaNet | 可擦除/写入的状态 | 近似线性 | 固定大小状态 | 更精确的记忆更新 | kernel 与训练复杂 |
| DSA | top-$k$ 历史 KV | 主 attention 约 $O(nkd)$ | 仍保存可检索 KV | 显式稀疏检索 | 索引与不规则访存成本 |
| Hybrid | 状态层 + 少量 attention | 取决于比例 | 两类状态并存 | 兼顾效率和质量 | 架构与系统更复杂 |

> 不要只问“复杂度是多少”，还要问：状态大小是否固定？是否需要保存 KV？训练能否沿序列并行？解码每步读多少内存？是否存在成熟 kernel？长上下文评测是在测试困惑度，还是测试精确 retrieval？

## 6. MoE：总参数与激活计算

### 6.1 稠密 FFN 回顾

标准 Transformer block 中，每个 token 都经过同一个 FFN：

$$
\operatorname{FFN}(x)
=
W_2\sigma(W_1x).
$$

若模型维度为 $d$、中间维度为 $d_{ff}$，忽略 bias，则一个两层 FFN 的参数量约为：

$$
N_{\text{FFN}}\approx 2dd_{ff}.
$$

每个 token 都使用全部参数，所以增加 FFN 宽度会同时增加：

- 总参数量；
- 每 token 激活参数量；
- 每 token FLOPs。

### 6.2 MoE 层是什么？

MoE 通常用 $E$ 个独立 FFN 专家替换一个稠密 FFN，再增加一个路由器（router/gating network）。对 token 表示 $x$：

$$
r(x)=W_rx\in\mathbb{R}^{E},
$$

路由器从 $E$ 个专家中选择 top-$k$ 集合 $\mathcal{T}(x)$。输出为：

$$
y
=
\sum_{e\in\mathcal{T}(x)}
g_e(x)\operatorname{FFN}_e(x).
$$

其中 $g_e(x)$ 是选中专家的门控权重。

稠密 FFN 像所有问题都交给同一个部门；MoE 有许多部门，调度员对每个 token 只叫来少数专家。

### 6.3 总参数与激活参数

若每个专家大小都与原 FFN 相同，则 MoE 专家总参数约为：

$$
N_{\text{total,experts}}
\approx
E\cdot 2dd_{ff}.
$$

但每个 token 只激活 $k$ 个专家：

$$
N_{\text{active,experts/token}}
\approx
k\cdot 2dd_{ff}.
$$

当 $k$ 固定时，增加 $E$ 可以大幅增加模型总容量，而理想化的专家计算量不随 $E$ 增加。这就是“同样 FLOPs，更多参数”的核心。

> 路由计算、dispatch/combine、跨设备 all-to-all、负载不均、padding、token dropping 和专家权重读取都有成本。尤其在小 batch、单机或低并发推理时，理论激活 FLOPs 可能无法转化为等比例的延迟优势。

### 6.4 为什么 MoE 受欢迎？

MoE 的主要收益是：

1. **参数效率**：相同激活计算下，更多参数通常能带来更低 loss 或更强能力；
2. **训练效率**：在合适系统上，达到某一质量所需的训练时间可能低于稠密模型；
3. **竞争力**：多种开放模型显示 MoE 可与更昂贵的稠密模型竞争；
4. **可扩展到多设备**：不同专家可放在不同设备上，形成专家并行（expert parallelism）。

但 MoE 没有更早成为默认方案，也有明确原因：

- 基础设施和通信复杂；
- 优势通常在多设备/多节点规模更明显；
- 路由目标较启发式，可能不稳定；
- 模型总权重仍要存储，部署内存需求并不等于激活参数量；
- 稀疏访问对单 token、低 batch 推理未必友好。

### 6.5 MoE 通常替换哪里？

最常见的是替换 Transformer block 的 FFN/MLP 子层：

```text
x → Norm → Attention → residual
  → Norm → Router → selected FFN experts → weighted sum → residual
```

原因是 FFN 本身占据大量参数，而且每个 token 独立计算，天然适合按 token 路由。也存在对 attention head 做 MoE 的方法，但课程材料指出它们相对少见。

## 7. 专家路由

### 7.1 三类路由视角

路由可以分为三类：

1. **Token chooses expert**：每个 token 选择分数最高的专家；
2. **Expert chooses token**：每个专家选择最想处理的 token；
3. **Global routing**：把整个 batch 的 token-expert 分配视为匹配/优化问题。

现代 LLM 中最常见的是第一类：token-choice top-$k$。

### 7.2 标准 top-k 路由

路由 logits：

$$
z_e(x)=w_e^{\mathsf T}x,
\qquad e=1,\ldots,E.
$$

先计算概率的一种常见形式：

$$
p_e(x)
=
\frac{\exp z_e(x)}{\sum_{j=1}^{E}\exp z_j(x)}.
$$

再选出：

$$
\mathcal{T}(x)
=
\operatorname{TopK}(p(x),k).
$$

MoE 输出为：

$$
y
=
\sum_{e\in\mathcal{T}(x)}
\tilde p_e(x)\operatorname{FFN}_e(x).
$$

$\tilde p_e$ 的定义存在变体：

- 先对所有专家 softmax，再取 top-$k$；
- 先取 top-$k$ logits，再只在选中专家上 softmax；
- 使用 sigmoid 得分做选择，再对选中权重归一化。

这些写法会影响门控权重、梯度和数值尺度，读论文或代码时不能只看到“top-$k$”就认为完全相同。

原始材料中的历史例子包括 Switch Transformer 的 $k=1$、GShard/Mixtral/Grok 的较小 $k$，以及 Qwen、DBRX、DeepSeek 等采用更大的激活专家数。具体数字属于模型配置，不是理论规律。

### 7.3 离散选择与梯度

离散集合：

$$
\mathcal{T}(x)=\operatorname{TopK}(p(x),k)
$$

会在专家排名交换时突然改变，因此对“是否选中”这一离散决策不可微。实践中通常：

- 只对已选中专家和门控权重反向传播；
- 把 top-$k$ 选择视作分段固定；
- 通过辅助损失、噪声或偏置调节路由行为。

因此不是“MoE 完全无法反向传播”，而是**离散选择本身没有普通的平滑梯度**。

### 7.4 其他路由方法

其他路由方法包括：

- **哈希路由（hashing）**：用固定规则分配，常作简单基线；
- **强化学习（REINFORCE）**：把路由视作策略，但梯度方差和系统复杂度较高；
- **随机近似**：给路由 logits 加 Gaussian 噪声，或给输入/路由分数加乘性 jitter，减少路由脆弱性并鼓励探索；
- **全局匹配/线性分配**：在整个 token 集合和专家容量间求较平衡的分配，但求解成本更高。

现代大模型最常见的仍是 top-$k$ 加启发式平衡机制，因为它简单、可扩展且经验上有效。

## 8. 专家粒度：更多小专家与共享专家

### 8.1 Fine-grained experts

将一个大专家切成多个更小专家，并相应增加选中的专家数，可让模型组合出更细粒度的专家集合。

假设原来有 $N$ 个专家、每个 token 选 $K$ 个。将每个专家切成 $m$ 个较小专家后，可以变成：

$$
mN\ \text{个专家，激活}\ mK\ \text{个专家}.
$$

若切分比例设计得当，激活计算量可以近似保持不变，但组合数量与路由灵活性增加。

### 8.2 Shared experts

共享专家对所有 token 始终激活：

$$
y
=
\sum_{e\in\mathcal{T}(x)}g_e(x)E_e(x)
+
\sum_{s=1}^{E_s}E^{\text{shared}}_s(x).
$$

设计动机是：

- 让共享专家学习通用知识；
- 让 routed experts 更专注于差异化模式；
- 减少不同 routed experts 重复学习相同基础能力。

但共享专家会对所有 token 增加固定 FLOPs。课程材料对比的消融结果也并不完全一致：DeepSeek 的实验支持细粒度和共享专家，OLMoE 的实验更支持细粒度专家，却没有观察到共享专家的同等收益。因此应把它看成需要验证的架构选择，而不是必然更好。

### 8.3 专家配置的比较口径

课程材料列出了 GShard、Switch、ST‑MoE、Mixtral、DBRX、Grok、DeepSeek、Qwen、OLMoE、MiniMax、Llama 4 等模型的 routed experts、active experts、shared experts 和 fine-grained ratio。

比较专家配置时，需要区分四个量：

| 量 | 回答的问题 |
|---|---|
| Routed experts | 一共有多少个可选专家？决定总容量和权重存储 |
| Active experts | 每个 token 实际计算几个？决定主要激活 FLOPs |
| Shared experts | 是否有每个 token 必经的通用 FFN？增加固定计算 |
| Expert size/ratio | 每个专家有多大？决定总参数与激活参数的真实口径 |

只说“256 个专家、激活 8 个”仍不足以比较模型，因为专家尺寸可能不同。

## 9. 专门化与负载均衡

### 9.1 Expert collapse

如果路由器早期偏爱少数专家，这些专家会得到更多 token 和梯度，因而学得更快；学得更快又让路由器更偏爱它们，形成正反馈：

```text
轻微偏好 → 更多 token → 更多更新 → 表现更好 → 更强偏好
```

结果可能是：

- 少数专家过载；
- 大量专家几乎不训练；
- token 被丢弃；
- 设备间负载不均，整体速度由最忙的设备决定；
- 名义上的大参数量没有真正被利用。

### 9.2 Switch Transformer 式辅助平衡损失

设 batch 中共有 $T$ 个 token、$N$ 个专家。定义专家 $i$ 实际接收的 token 比例：

$$
f_i
=
\frac{1}{T}
\sum_{x\in\mathcal{B}}
\mathbf{1}
\left[
\operatorname{argmax}_j p_j(x)=i
\right],
$$

以及路由器分给专家 $i$ 的平均概率：

$$
P_i
=
\frac{1}{T}
\sum_{x\in\mathcal{B}}p_i(x).
$$

辅助损失为：

$$
L_{\text{balance}}
=
\alpha N
\sum_{i=1}^{N}f_iP_i.
$$

理想均衡时：

$$
f_i\approx P_i\approx\frac{1}{N}.
$$

如果某专家已经接收很多 token，$f_i$ 较大，那么继续增大 $P_i$ 会受到更强惩罚，从而推动路由概率转向其他专家。

> $f_i$ 反映离散分配结果，但包含不可微的 argmax；$P_i$ 是可微的软概率。把二者相乘，可以用真实负载调节可微概率的梯度。

### 9.3 专家、设备与通信三层平衡

只保证每个专家平均接收相同 token 还不够，因为多个专家可能放在同一设备上。课程材料以 DeepSeek v1–v2 为例介绍：

- **Expert-level balance**：专家之间 token 数量均衡；
- **Device-level balance**：不同设备承载的总专家计算均衡；
- **Communication balance**：不同设备发送与接收的 token 流量均衡。

系统性能由最慢的设备和最拥塞的通信链路决定。因此模型层面的“专家均匀”不必然等于系统层面的“运行均匀”。

### 9.4 辅助损失的副作用

语言模型主损失希望把 token 送到最合适的专家；平衡损失希望把 token 送得均匀。两者可能冲突：

$$
L
=
L_{\text{LM}}
+
\lambda L_{\text{balance}}.
$$

$\lambda$ 太小，可能发生专家坍缩；$\lambda$ 太大，路由器为了均匀而牺牲专家专门化与主任务质量。这也是平衡目标被称为“heuristic”的原因。

### 9.5 DeepSeek V3 的 per-expert bias

一种减少干扰梯度的方法，是给每个专家维护路由偏置 $b_i$：

$$
s_i(x)=p_i(x)+b_i.
$$

top-$k$ 选择使用调整后的 $s_i$。训练过程中：

- 专家过载时降低其偏置；
- 专家欠载时提高其偏置。

这相当于用在线控制器调节负载，而不把主要平衡压力直接作为梯度加入语言模型目标。课程材料提醒：这被称为“auxiliary-loss-free balancing”，但 DeepSeek V3 仍保留较弱的 sequence-wise 辅助项，因此不应理解成完全不存在任何辅助平衡约束。

## 10. Capacity、token dropping 与额外随机性

### 10.1 Expert capacity

实际系统通常为每个专家分配有限容量。若平均每个专家应收到 $Tk/N$ 个 token，可定义：

$$
C
=
\left\lceil
\text{capacity factor}
\cdot
\frac{Tk}{N}
\right\rceil.
$$

容量因子大于 $1$ 可容纳一定不均衡，但会增加 padding、预留 buffer 或计算浪费。

### 10.2 Token dropping

某专家收到超过容量的 token 时，系统可能：

- 丢弃超额 token 的专家分支；
- 退回 residual path；
- 发送到次优专家；
- 使用无丢弃的动态 shape / block-sparse 实现。

不同处理会改变数值行为、吞吐和内存。

### 10.3 为什么 MoE 可能有额外随机性？

若容量和 dropping 在 batch 级决定，那么一个 token 是否被专家接收，可能取决于同一个 batch 中还有哪些其他 token 在竞争容量。于是：

- 同一个输入放入不同 batch，结果可能不同；
- 在线服务中，其他用户请求可能间接改变容量竞争；
- 分布式排序、并发和实现细节也可能引入差异。

这不是所有 MoE 实现必然存在的问题，但当系统使用有限容量和 token dropping 时必须警惕。

## 11. MoE 的系统实现与通信

### 11.1 Expert parallelism

将不同专家放到不同设备：

1. 每个设备先计算本地 token 的路由结果；
2. 按目标专家把 token 发送到对应设备；
3. 各设备对收到的 token 执行本地专家 FFN；
4. 把专家输出发回 token 原来的设备；
5. 按门控权重组合输出。

这通常需要两次 all-to-all 风格通信：一次 dispatch，一次 combine。

### 11.2 主要系统瓶颈

- **负载不均**：某设备计算更多，其他设备等待；
- **小矩阵问题**：token 被切到很多专家后，每个 GEMM 变小，GPU 利用率降低；
- **跨节点通信**：网络带宽与延迟可能吞掉稀疏计算收益；
- **动态 shape**：不同 batch 的每个专家 token 数不同；
- **内存容量**：总专家权重必须放在整个设备组中；
- **权重带宽**：低 batch 推理时，每个专家只服务少量 token，权重重用不足。

### 11.3 Grouped GEMM 与 block-sparse kernel

朴素实现会循环调用每个专家的小矩阵乘法，kernel launch 多且利用率低。MegaBlocks 等系统将 token-expert 分配组织成块稀疏或 grouped GEMM：

- 将同一专家的 token 聚合；
- 一次调度多个专家矩阵乘法；
- 减少 padding 与无效计算；
- 更好利用 GPU tensor cores。

课程材料还以 Nemotron 3 为例提到：可以在跨设备发送前先将 activation 降维，以减少通信量；代价是增加投影计算并可能损失信息。这项设计需要同时衡量通信节省与投影开销。

### 11.4 训练与推理并不具有相同收益

训练有大量 token，可以把每个专家的 token 聚合成较大的矩阵，较容易获得并行效率。逐 token 或低并发推理时：

- 每步 token 少；
- 专家权重读取难以摊薄；
- 路由和通信占比变大；
- 总参数内存仍然很大。

所以“37B active parameters”更接近计算口径，不代表它在所有硬件上的延迟、显存和成本都等同于 37B 稠密模型。

## 12. Router 的数值稳定性

### 12.1 路由器对数值误差敏感

top-$k$ 由专家分数的相对排序决定。若两个专家分数接近，低精度舍入就可能改变选择，造成离散路径突变。常见的处理方式是：

- 主模型大部分计算使用 bf16/fp16；
- router logits 或 router softmax 使用 fp32；
- 必要时加入 router z-loss。

### 12.2 Router z-loss

设 token $x$ 的 router logits 为 $z(x)\in\mathbb{R}^{E}$，定义：

$$
L_z
=
\frac{1}{T}
\sum_{x\in\mathcal{B}}
\left(
\log\sum_{e=1}^{E}\exp z_e(x)
\right)^2.
$$

训练目标加入：

$$
L
=
L_{\text{LM}}
+
\lambda_{\text{balance}}L_{\text{balance}}
+
\lambda_zL_z.
$$

z-loss 抑制 router logits 整体漂移到过大幅值。因为 softmax 对所有 logits 加同一个常数时概率不变，主任务本身不一定约束这种共同漂移；但过大 logits 会恶化低精度数值稳定性。课程材料引用的 ST‑MoE 实验显示，移除 z-loss 可能带来明显训练 spike。

> - z-loss 控制 router logits 的数值尺度和稳定性；
> - balancing loss 控制 token 在专家之间的分配。

### 12.3 小数据微调中的过拟合

稀疏 MoE 总参数很多，但每个专家在小型微调数据上只能看到路由给自己的子集，可能更容易过拟合或更新不足。微调时可以尝试两种策略：

- ST‑MoE 的方案：微调时重点更新非 MoE/稠密组件；
- DeepSeek 的方案：使用规模很大的 SFT 数据。

工程上还可考虑较小学习率、冻结部分专家、对 router 单独设学习率、增加正则和监控专家使用率，但这些属于应通过实验验证的扩展策略。

## 13. Upcycling：从稠密模型初始化 MoE

### 13.1 为什么要 upcycle？

从零训练大型 MoE 成本高且路由早期不稳定。Upcycling 的目标是利用已有稠密模型：

1. 复制或切分预训练稠密 FFN，初始化多个专家；
2. 新增并初始化 router；
3. 继续预训练，让专家逐渐分化。

最简单的复制初始化会让专家一开始完全相同。路由噪声、数据差异和梯度更新随后打破对称性。

### 13.2 收益与限制

潜在收益：

- 继承已有语言能力；
- 避免完全从零训练；
- 用继续训练预算换取更大的稀疏容量。

限制：

- 专家可能长期同质化；
- 新 router 仍需学习；
- 仍需大量继续训练 token 才能稳定获得收益；
- 结构变更过大时，旧权重不一定是理想初始化。

课程材料以 MiniCPM MoE 和 Qwen MoE 为成功案例。它们说明 upcycling 可行，但并不是“一复制就免费得到更强模型”。

## 14. DeepSeek MoE 演化案例

DeepSeek 系列的配置展示了细粒度专家、共享专家和负载均衡策略的变化。下面的参数量沿用原笔记记录的报告口径。

### 14.1 DeepSeek MoE V1

课程材料概括：约 16B 总参数、2.8B 激活参数，采用：

- 细粒度 routed experts；
- 始终激活的 shared experts；
- 标准 top-$k$ 路由；
- expert-level 与 device-level 辅助平衡损失。

核心目标是让共享专家吸收通用知识，让更多小 routed experts 学习更专门的组合。

### 14.2 DeepSeek V2

课程材料概括：约 236B 总参数、21B 激活参数，并新增：

- **Top-$M$ device routing**：限制 token 优先访问少数设备上的专家，减少跨设备通信；
- **communication balance loss**：同时平衡发送与接收通信负载；
- 更大规模的细粒度 routed experts 与 shared experts。

路由既影响模型选择，也决定 token 在设备之间如何传输。

### 14.3 DeepSeek V3

官方报告给出的主模型口径是 671B 总参数、每 token 激活 37B 参数。课程材料强调的变化包括：

- 更多细粒度专家，激活少量 routed experts 加 shared expert；
- sigmoid 分数与 top-$k$ 后归一化的路由变化；
- top-$M$ 设备限制；
- per-expert bias 的辅助损失弱化/近似无辅助损失平衡；
- 仍保留 sequence-wise 辅助约束。

> 总参数是否包含额外训练模块、active parameters 是否包含 attention/embedding/shared experts，可能因报告表格而不同。比较模型时应读取同一来源中的定义，不要只比较宣传数字。

## 15. MLA 与 MTP

MLA 和 MTP 分别改进注意力缓存与预测方式，与 MoE 一起组成 DeepSeek V3 的效率方案。

### 15.1 Multi-head Latent Attention（MLA）

标准 attention 解码需要缓存每个历史 token 的 $K,V$。MLA 的基本想法是先把隐藏状态压缩成低维 latent：

$$
c_t^{KV}=W^{DKV}h_t,
$$

再从 latent 恢复或等价吸收出 key/value 计算：

$$
k_t=W^{UK}c_t^{KV},
\qquad
v_t=W^{UV}c_t^{KV}.
$$

若线性投影可以代数吸收到 query/output 侧，推理时主要缓存较小的 $c_t^{KV}$，从而降低 KV cache。

#### RoPE 为什么带来困难？

没有 RoPE 时，线性矩阵可利用结合律合并。加入位置相关旋转矩阵后：

$$
\langle q_tR_t,\,k_sR_s\rangle
$$

包含与位置 $t,s$ 有关的旋转，某些投影不再能简单预先合并。课程材料给出的解决思路是：保留少量不经过 latent 压缩、专门承担 RoPE 的 key/query 维度，其余维度采用 latent cache。

> MLA 主要压缩 KV 表示；MQA/GQA 共享 KV heads；线性注意力将历史聚合成递归状态。三者都可减少推理成本，但数学机制不同。

### 15.2 Multi-Token Prediction（MTP）

标准语言模型在位置 $t$ 只预测下一个 token：

$$
p(x_{t+1}\mid x_{\le t}).
$$

MTP 增加轻量预测模块，让同一隐藏状态还预测更远的未来 token：

$$
p(x_{t+j}\mid x_{\le t}),
\qquad j=1,2,\ldots.
$$

训练时可提供更密集的未来监督；推理时，额外预测可作为 speculative decoding 的候选，再由主模型验证。额外预测深度由具体配置决定，候选 token 仍需主模型验证后才能接受。

## 16. 张量形状与数据流

### 16.1 线性注意力

| 张量 | Shape | 含义 |
|---|---:|---|
| $Q$ | $(B,H,S,d_k)$ | query |
| $K$ | $(B,H,S,d_k)$ | key |
| $V$ | $(B,H,S,d_v)$ | value |
| $QK^{\mathsf T}$ | $(B,H,S,S)$ | 标准 attention 分数 |
| $S_t$ | $(B,H,d_k,d_v)$ | 递归 key-value 状态 |
| $y_t$ | $(B,H,d_v)$ | 当前 token 的 head 输出 |

### 16.2 Token-choice MoE

令 batch 内 token 数 $T=B\times S$。

| 张量 | Shape | 含义 |
|---|---:|---|
| $X$ | $(B,S,d)$ 或 $(T,d)$ | 输入 token 表示 |
| Router logits | $(T,E)$ | 每个 token 对每个专家的分数 |
| Top-$k$ indices | $(T,k)$ | 每个 token 选中的专家编号 |
| Top-$k$ weights | $(T,k)$ | 组合专家输出的权重 |
| Dispatched tokens | 动态，逻辑上约 $(Tk,d)$ | 按专家重排后的 token |
| Expert hidden | 动态，约 $(Tk,d_{ff,e})$ | 专家内部 activation |
| Combined output | $(T,d)$ | 聚合后的 MoE 输出 |

MoE 中最难追踪的通常不是线性层 shape，而是 token 重排：

```text
(T, d)
  → router (T, E)
  → top-k indices (T, k)
  → sort / dispatch by expert
  → grouped expert FFNs
  → inverse permutation / combine
  → (T, d)
```

## 17. 资源核算

### 17.1 线性注意力与完整注意力

忽略投影层和常数：

$$
\text{Full attention FLOPs}
\sim
O(BH S^2(d_k+d_v)),
$$

$$
\text{Linear attention FLOPs}
\sim
O(BH S d_kd_v).
$$

两者的理论交叉点大致取决于 $S$ 与状态维度。短序列时，完整 attention 的成熟 kernel 可能更快；长序列时，线性/状态方法的渐近优势才更可能显现。

### 17.2 MoE 参数与计算

若每个专家是普通两层 FFN：

$$
N_{\text{expert}}
\approx
2dd_{ff,e}.
$$

总专家参数：

$$
N_{\text{expert,total}}
\approx
E\cdot 2dd_{ff,e}.
$$

每 token 激活专家参数：

$$
N_{\text{expert,active/token}}
\approx
k\cdot 2dd_{ff,e}.
$$

router 参数约为：

$$
N_{\text{router}}=dE.
$$

router FLOPs 会随 $E$ 增长，虽然通常小于专家 FFN，但在专家非常多或专家非常小时不能忽略。

### 17.3 一个手算例子

设 $d=4096$，每个专家 $d_{ff,e}=4096$，共有 $E=64$ 个专家，每个 token 选择 $k=8$。

单个专家参数约为：

$$
2\times4096\times4096
\approx
33.6\text{M}.
$$

专家总参数约为：

$$
64\times33.6\text{M}
\approx
2.15\text{B}.
$$

每 token 激活的专家参数约为：

$$
8\times33.6\text{M}
\approx
268\text{M}.
$$

总专家参数与每 token 激活专家参数相差 $E/k=8$ 倍。但权重存储仍按约 2.15B 参数计算，不能按 268M 计算。

### 17.4 系统指标

- tokens/s 与端到端 step time；
- 每个专家接收 token 的直方图；
- 最大/平均 expert load；
- dropped token 比例；
- dispatch 与 combine 的通信时间；
- grouped GEMM 时间和矩阵尺寸；
- router entropy、最大 logit、z-loss；
- 每设备 HBM 占用与网络带宽；
- 总参数、激活参数和实际 FLOPs/token。

## 18. 数据流伪代码

### 18.1 因果线性注意力递归

```python
# 单个 head 的教学伪代码，不是高性能训练实现
state = zeros(d_k, d_v)
outputs = []

for t in range(sequence_length):
    state = state + outer(k[t], v[t])
    y_t = q[t] @ state
    outputs.append(y_t)
```

这个版本有明确的串行依赖，适合理解推理；训练时应使用并行 scan、分块算法或成熟实现。

### 18.2 Top-k MoE

```python
# 教学伪代码：重点是数据流，不是完整的分布式实现
router_logits = router(x)                 # (T, E)
router_prob = softmax(router_logits, -1)  # (T, E)
weights, expert_ids = topk(router_prob, k=k)

output = zeros_like(x)
for expert_id in range(num_experts):
    token_id, slot_id = where(expert_ids == expert_id)
    expert_input = x[token_id]
    expert_output = experts[expert_id](expert_input)
    output[token_id] += weights[token_id, slot_id, None] * expert_output
```

真正实现还要处理：

- 对 token 按 expert 排序；
- expert capacity；
- 跨设备 all-to-all；
- grouped GEMM；
- 恢复原 token 顺序；
- load-balancing loss 与 router z-loss；
- 分布式环境下的统计量聚合。

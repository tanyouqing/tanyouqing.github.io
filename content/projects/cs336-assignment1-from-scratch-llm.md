---
title: "CS336 Assignment 1：从零构建与预训练语言模型"
date: "2026-10-16"
description: "从字节级 BPE 到 Decoder-only Transformer，梳理 RMSNorm、Pre-Norm、SwiGLU、RoPE，以及语言模型的预训练、评估与生成流程。"
tags: ["CS336", "LLM", "预训练"]
tech: ["Python", "PyTorch", "BPE", "Transformer"]
github: "https://github.com/tanyouqing/CS336-Assigment1"
image: "🔤"
---

## 项目介绍

学完 Stanford CS336 后，我在 Assignment 1 中把语言模型的主要组件串成了一条完整链路：训练字节级 BPE tokenizer，将文本编码成整数序列，从零实现 Decoder-only Transformer，再用 next-token prediction 预训练，并通过验证集困惑度和自回归生成观察模型。

项目代码：[CS336-Assigment1](https://github.com/tanyouqing/CS336-Assigment1)。本文侧重架构与训练原理，后续的奖励学习与偏好优化见 [Assignment 5 项目笔记](/projects/cs336-assignment5-rl-posttraining/)。

```text
原始文本 → UTF-8 bytes → BPE token IDs → 随机连续窗口
                                          ↓
Embedding → 多层 Pre-Norm Transformer → RMSNorm → LM Head
                                          ↓
                     next-token 交叉熵 → backward → AdamW
                                          ↓
                            验证 / checkpoint / 文本生成
```

仓库实现包括 tokenizer、模型、优化器、训练循环、生成与消融配置。本文中的形状和数值是结构示例，不作为训练成绩或吞吐量报告。

## 从文本到训练样本

### 字节级 BPE：在词表预算内压缩序列

Tokenizer 输出的 token ID 是整数索引，Embedding 再把索引映射成可学习的向量。Token 不必对应一个字或一个完整单词。

字节级 BPE 从全部 256 个单字节符号出发，先用正则划分 pre-token，再反复合并高频相邻符号对。例如 `t + h → th`、`th + e → the`。完整字节表使未见过的 Unicode 文本仍能退化成 UTF-8 字节表示；合并则让常见文本使用更少 token。

这里有三个实现细节：

- **特殊 token 是边界**：`<|endoftext|>` 等标记单独识别，不参与普通 pair 的跨界合并。
- **编码遵循 merge rank**：推理时使用训练得到的合并次序，不能重新根据当前句子的频率选择 pair。
- **还原时先拼 bytes，再解码**：单个 token 可能只是一个 UTF-8 字符的部分字节，逐 token 解码再拼字符串会出错。

词表扩大通常能缩短序列，但会增大 Embedding 和输出投影；词表大小是序列长度、计算和参数量之间的取舍。代码位于 [tokenizer.py](https://github.com/tanyouqing/CS336-Assigment1/blob/main/cs336_basics/tokenizer.py)。

### 右移一位，得到所有位置的监督

预训练不需要人工给每段文本标注答案。原始文本的后一个 token 就是前缀的训练标签。取长度为 $T+1$ 的窗口：

```text
原始窗口：A B C D E
input：   A B C D
target：  B C D E
```

一次前向并行预测四个位置，因果 mask 保证预测 `C` 时只能看到 `A B`。这里的并行训练与推理时逐 token 生成并不矛盾。

本实现用 memory map 读取 token 文件，再随机采样连续窗口。模型输入和标签都是 `[B,T]` 的整数张量；磁盘可用 `uint16` 节省空间，送入 Embedding 前转为 `torch.long`。词表超出该存储类型的范围时必须更换 dtype。

## Decoder-only Transformer 的数据流

设批大小为 $B$，上下文长度为 $T$，词表大小为 $V$，隐藏维度为 $D$，注意力头数为 $H$，每头维度为 $d_h=D/H$。

| 阶段 | 张量形状 | 作用 |
|---|---|---|
| Token IDs | `[B,T]` | 离散词表索引 |
| Embedding | `[B,T,D]` | 得到各位置的初始表示 |
| Q、K、V 分头 | `[B,H,T,d_h]` | 各头从不同投影空间读取上下文 |
| Attention scores | `[B,H,T,T]` | 每个位置对可见位置的打分 |
| 合头与输出投影 | `[B,T,D]` | 回到残差流的宽度 |
| SwiGLU 中间分支 | `[B,T,D_ff]` | 对每个位置执行门控变换 |
| Final norm + LM Head | `[B,T,V]` | 给每个位置的下一个 token 打分 |

这里的 `logits` 是未归一化分数。训练时把它们交给稳定的交叉熵；生成时再转换为采样分布。核心实现见 [model.py](https://github.com/tanyouqing/CS336-Assigment1/blob/main/cs336_basics/model.py)。

### RMSNorm：调节向量尺度

对单个位置的隐藏向量 $h\in\mathbb R^D$：

$$
\operatorname{RMSNorm}(h)
=g\odot\frac{h}{\sqrt{\frac1D\sum_{k=1}^{D}h_k^2+\epsilon}}.
$$

它按均方根缩放，并使用可学习的逐维增益 $g$。与 LayerNorm 相比，RMSNorm 不减均值；归一化沿隐藏维进行，不把不同 token 混在一起。

实现先转 FP32 计算平方、均值和缩放，再转回输入 dtype，减少低精度中间计算的数值风险。归一化不是抹去所有信息：方向、各维相对关系和可学习增益仍参与表示。

### Pre-Norm 与残差：保持主干的信息通路

一个默认 block 写成：

$$
u=h+\operatorname{Attention}(\operatorname{RMSNorm}(h)),
$$

$$
h'=u+\operatorname{SwiGLU}(\operatorname{RMSNorm}(u)).
$$

Pre-Norm 指进入子层之前做归一化，RMSNorm 则是具体采用哪种归一化。它们描述两个不同选择。残差支路保留加法直通路径，深层网络的梯度不必完全经过注意力或 FFN 的变换。

Post-Norm 改为先算 `h + sublayer(h)` 再归一化；本项目提供 `pre / post / none` 消融。默认 Pre-Norm 模型在所有 block 之后还有一次 RMSNorm。训练稳定性仍取决于初始化、学习率和深度，不能只靠一处归一化保证。

### 多头因果注意力：决定从哪些位置读信息

$$
Q=XW_Q^\top,\quad K=XW_K^\top,\quad V=XW_V^\top,
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

因果 mask $M$ 在允许关注的位置为零、未来位置为负无穷。除以 $\sqrt{d_h}$ 用来控制点积随维度增大而放大的尺度，减轻 softmax 饱和。每个 head 独立计算，再合并并投影回 $D$ 维。

注意布尔 mask 的语义：本实现 `True` 表示允许关注，`masked_fill(~mask, -inf)` 屏蔽未来。不同库的接口可能采用不同约定。一个查询的所有 key 若都被屏蔽，softmax 会遇到数值问题。

### RoPE：让点积感知相对位置

RoPE 对 Q、K 中相邻的二维坐标施加与位置相关的旋转：

$$
\begin{pmatrix}q'_{2k}\\q'_{2k+1}\end{pmatrix}
=\begin{pmatrix}
\cos(m\omega_k)&-\sin(m\omega_k)\\
\sin(m\omega_k)&\cos(m\omega_k)
\end{pmatrix}
\begin{pmatrix}q_{2k}\\q_{2k+1}\end{pmatrix},
\qquad \omega_k=\theta^{-2k/d_h}.
$$

旋转矩阵满足相对位置关系：

$$
(R_mq)^\top(R_nk)=q^\top R_{n-m}k.
$$

因此点积可以依赖位置差 $n-m$。本实现只旋转 Q、K，V 保留原表示；正弦、余弦预计算为 buffer。成对旋转要求每头维度为偶数。RoPE 提供位置结构，不自动保证超过训练上下文的长文本能力。

### SwiGLU：按内容控制信息通过

$$
\operatorname{SwiGLU}(h)
=W_2\left[\operatorname{SiLU}(W_1h)\odot(W_3h)\right],
\qquad \operatorname{SiLU}(z)=z\sigma(z).
$$

两个分支先扩展到 $D_{ff}$ 维，一个经过 SiLU，另一个提供内容，再逐元素相乘，最后投影回 $D$ 维。FFN 对各 token 独立计算；跨位置的信息交换发生在 attention 中。

普通两层、宽度 $4D$ 的 FFN 约有 $8D^2$ 个权重。SwiGLU 有三个矩阵，为接近相同参数预算可取：

$$
3DD_{ff}\approx8D^2\quad\Longrightarrow\quad D_{ff}\approx\frac83D.
$$

本实现的默认宽度还向上对齐到 64 的倍数，并提供不带门控的 SiLU FFN 做对照。对齐是工程配置，不能在没有测量时当作加速结果。

## LLM 如何预训练

### 自回归似然与交叉熵

$$
p_\theta(z_{1:T})=\prod_{t=1}^{T}p_\theta(z_t\mid z_{<t}).
$$

最大化训练文本的似然，等价于最小化 token 负对数似然：

$$
\mathcal L_{\rm LM}
=-\frac1{BT}\sum_{b,t}\log p_\theta(z_{b,t+1}\mid z_{b,\le t}).
$$

一条训练样本包含多个前缀的监督。模型并不直接优化“回答有帮助”或“数学最终答案正确”；它学习训练数据中后继 token 的分布。这也是后续需要 post-training 的原因。

计算交叉熵时应使用 log-sum-exp 思路，先减去最大 logit，再求指数和。不要先构造很小的概率再取 log，也不要对交叉熵额外套一层 softmax。

### 一次参数更新

```python
# 对应 train.py 的核心步骤；省略日志、验证与保存
inputs, targets = get_batch(data, batch_size, context_length, device)
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)                  # [B,T,V]
loss = cross_entropy(logits, targets)   # scalar
loss.backward()
gradient_clipping(model.parameters(), max_grad_norm)
optimizer.step()
```

`backward()` 将梯度累加到参数的 `.grad`，`step()` 才改变参数。没有清空梯度会把前一步结果也累加进来；梯度累积则需要有意识地控制缩放和清零时机。

训练中的几个关键选择：

| 选择 | 为什么使用 | 需要注意 |
|---|---|---|
| AdamW | 用一、二阶矩调节各参数更新，独立执行 weight decay | weight decay 不等同于把 L2 项塞进 Adam 梯度 |
| Linear warmup | 训练开始时逐步增大学习率 | warmup 步数属于配置，不是固定法则 |
| Cosine decay | 后期逐步减小更新幅度 | 记录最大学习率、最低值和总步数 |
| 全局梯度范数裁剪 | 限制一次更新前的梯度总范数 | 与后训练的概率比 clipping 是两种操作 |
| Checkpoint | 保存模型、优化器和迭代数，支持继续训练 | 仅保存模型不足以恢复 Adam 的状态 |

代码分布在 [optimizer.py](https://github.com/tanyouqing/CS336-Assigment1/blob/main/cs336_basics/optimizer.py) 与 [train.py](https://github.com/tanyouqing/CS336-Assigment1/blob/main/cs336_basics/train.py)。模型主体采用无 bias 线性层，输入 Embedding 与 LM Head 是独立参数。

### 如何判断训练有用

验证集 loss 衡量对未参与更新的文本的预测能力。自然对数下，困惑度为：

$$
\operatorname{PPL}=\exp(\mathcal L_{\rm val}).
$$

PPL 适合在相同 tokenizer、数据与计算口径下比较。更换词表或语料后，每个 token 的含义和难度都变了，不能机械比较数值。

生成时只使用最后位置的 logits，采样一个 token，拼回上下文，再继续前向。Temperature 改变分布尖锐程度，top-p 保留累计概率足够大的候选集合；二者改变采样行为，不改变已训练权重。仓库的基础生成实现会重复计算上下文，不能将它的速度当作带 KV cache 的推理引擎性能。

## 实现索引与学习体会

| 代码 | 对应知识 |
|---|---|
| `tokenizer.py`、`encode_dataset.py` | BPE 训练、序列化、语料编码 |
| `model.py`、`nn_utils.py` | 模型结构、稳定 softmax 和交叉熵 |
| `data.py`、`train.py` | 连续窗口、训练与验证 |
| `optimizer.py`、`checkpoint.py` | 更新规则、梯度裁剪、断点恢复 |
| `generate.py`、`experiments.py` | 采样、学习率扫描、架构消融 |

这份作业让我把“训练一个语言模型”落实为可追踪的数据流：文本提供监督，因果结构限制可见信息，归一化和残差维持尺度与梯度，优化器不断提高真实后继 token 的概率。理解这些组件后，Assignment 5 中“用奖励给同一套 token log-prob 加权”的训练方式就更容易看清。

更多原理可对照站内的 [Tokenization 笔记](/articles/cs336-l01-tokenization/) 与 [模型架构笔记](/articles/cs336-l03-architectures/)。本页整理架构与实现，不附未经核验的完整训练曲线或课程 leaderboard 成绩。

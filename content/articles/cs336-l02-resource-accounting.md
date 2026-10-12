---
title: "CS336 L02：PyTorch 与资源核算"
date: "2026-07-30"
description: "张量、混合精度、自动微分与训练状态，结合 FLOPs、显存和 Roofline 模型估算训练成本。"
tags: ["CS336","PyTorch","训练系统"]
category: "笔记"
---

## 1. Tensor 是训练系统的基本对象

训练中的数据最终几乎都存为张量（tensor）：

- 输入数据；
- 模型参数；
- 中间激活（activations）；
- 梯度（gradients）；
- 优化器状态（optimizer states）。

一个张量至少需要关注：

| 属性 | 含义 | PyTorch |
|---|---|---|
| rank | 维度数量 | `x.ndim` |
| shape | 每个维度的长度 | `x.shape` |
| dtype | 每个元素的数据类型 | `x.dtype` |
| device | 数据位于 CPU 还是某张 GPU | `x.device` |
| numel | 元素总数 | `x.numel()` |
| element size | 每个元素占多少 bytes | `x.element_size()` |

```python
B, S, H, D = 32, 16, 16, 64
x = torch.zeros(B, S, H, D, dtype=torch.bfloat16)

print(x.shape)          # torch.Size([32, 16, 16, 64])
print(x.ndim)           # 4
print(x.numel())        # B * S * H * D
print(x.element_size()) # 2 bytes
```

在 Transformer 中，rank-4 张量很常见：

$$
x \in \mathbb{R}^{B \times S \times H \times D_h},
$$

其中：

- $B$：batch size；
- $S$：sequence length；
- $H$：attention heads；
- $D_h$：每个 head 的维度；
- 通常 $d_{\text{model}} = H D_h$。

如果只看元素值、不追踪 shape 和 dtype，最常见的结果就是维度错位、隐式广播、额外拷贝，以及错误的显存估算。

## 2. 张量内存只由元素数量和 dtype 决定

对一个独立、连续存储的张量，数据本体占用：

$$
M_{\text{tensor}}
=
\operatorname{numel}(x)
\times
\operatorname{element\_size}(x).
$$

例如，一个 shape 为 $(16,32)$ 的 fp32 张量：

$$
16 \times 32 \times 4
=
2048\ \text{bytes}.
$$

PyTorch 对应：

```python
def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()
```

> GPU 厂商通常用十进制单位：$1\ \text{GB}=10^9$ bytes；操作系统有时显示二进制单位：$1\ \text{GiB}=2^{30}$ bytes。做数量级估算时可用 GB，但记录实验时要注明口径。

### 常见浮点格式

| dtype | bytes/元素 | 指数位 | 主要特点 |
|---|---:|---:|---|
| fp32 | 4 | 8 | 范围和精度较高，显存开销大 |
| fp16 | 2 | 5 | 精度和动态范围都较小，容易上溢/下溢 |
| bf16 | 2 | 8 | 与 fp32 动态范围相近，但有效精度更低 |
| fp8 | 1 | 依格式而定 | 更省存储与带宽，需要更谨慎的缩放策略 |

动态范围（range）主要由指数位决定；数值精度（precision）主要由尾数位决定。因此：

- fp16 和 bf16 都占 2 bytes；
- bf16 的指数位与 fp32 相同，因此动态范围更大；
- bf16 用更少的尾数位换取了动态范围，其表示更粗糙。

最小例子：

```python
x_fp16 = torch.tensor([1e-8], dtype=torch.float16)
x_bf16 = torch.tensor([1e-8], dtype=torch.bfloat16)

print(x_fp16)  # 可能下溢为 0
print(x_bf16)  # 仍可表示非零数量级
```

> bf16 只是与 fp32 的动态范围近似相同；它的尾数更短，因此舍入误差更大。

## 3. 混合精度训练（mixed precision）

如果全部使用 fp32，显存与数据搬运量很大；如果全部使用低精度，训练又可能不稳定。因此实际训练通常让不同操作和状态使用不同精度。

一种用于估算的混合精度配置：

- 参数：bf16，2 bytes/parameter；
- 激活：bf16；
- 梯度：bf16，2 bytes/parameter；
- Adam 一阶、二阶矩状态：fp32，共 8 bytes/parameter。

PyTorch 的自动混合精度（AMP）会让适合低精度的算子使用 bf16/fp16，而对数值敏感的操作保留更高精度：

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)
    loss = loss_fn(logits, targets)
```

为什么优化器状态常用 fp32：一阶、二阶矩会跨很多训练步持续累积；如果精度过低，微小更新可能在舍入中消失。

> 某些混合精度方案还会保存一份 fp32 master weights，需要额外 4 bytes/parameter。这里的 $2+2+8=12$ bytes/parameter 估算没有包含这份副本，也没有包含激活和临时 buffer。

## 4. 用 einops 显式表达张量维度

`einops` 的价值不是提供新的数学操作，而是**给维度命名**，让 shape 变化和求和维度直接出现在代码中。

### 4.1 `einsum`：带维度名称的广义矩阵乘法

```python
from einops import einsum

# x: [batch, seq1, hidden]
# y: [batch, seq2, hidden]
scores = einsum(
    x,
    y,
    "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2",
)
```

数学形式：

$$
\text{scores}_{b,i,j}
=
\sum_{d=1}^{D}
x_{b,i,d}y_{b,j,d}.
$$

规则：输入中出现、但输出中没有出现的维度，会被求和消去。`...` 可以代表任意数量、在两侧兼容的批量维度：

```python
scores = einsum(
    x,
    y,
    "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2",
)
```

### 4.2 `reduce`：沿指定维度聚合

```python
from einops import reduce

# x: [batch, seq, hidden]
y = reduce(x, "... hidden -> ...", "sum")
# y: [batch, seq]
```

### 4.3 `rearrange`：拆分、合并或交换维度

```python
from einops import rearrange

# [batch, seq, heads * head_dim]
x = rearrange(
    x,
    "batch seq (heads head_dim) -> batch seq heads head_dim",
    heads=8,
)
```

这正对应多头注意力中：

$$
d_{\text{model}} = H D_h.
$$

> 它主要改善表达和 shape bookkeeping。最终是否生成高效 kernel，仍取决于具体 pattern、输入布局、编译器和后端实现。

## 5. FLOPs、FLOP/s 与 MFU

这三个概念不能混用：

- **FLOP**：一次浮点运算，例如一次加法或乘法；
- **FLOPs**：某个任务总共执行多少浮点运算；
- **FLOP/s**：每秒完成多少浮点运算，用于描述硬件或程序速度；
- **MFU（Model FLOPs Utilization）**：实际吞吐相对于硬件理论峰值的比例。

### 5.1 矩阵乘法的 FLOPs

设：

$$
X \in \mathbb{R}^{B \times D},
\qquad
W \in \mathbb{R}^{D \times K},
\qquad
Y=XW.
$$

每个 $Y_{b,k}$ 需要长度为 $D$ 的点积。按一次乘法和一次加法各算 1 FLOP：

$$
\operatorname{FLOPs}(XW)
\approx
2BDK.
$$

这里：

- $B$ 可理解为数据点或 token 的数量；
- $DK$ 是线性层权重参数量；
- 因而线性层前向近似为 $2 \times \text{数据点数} \times \text{参数量}$。

### 5.2 实际 FLOP/s

$$
\text{actual FLOP/s}
=
\frac{\text{任务 FLOPs}}{\text{实际执行时间}}.
$$

GPU kernel 是异步执行的，计时前后必须同步：

```python
torch.cuda.synchronize()
start = time.perf_counter()
y = x @ w
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

正式 benchmark 还应包含 warm-up 和多次重复，避免把首次编译、缓存建立和随机抖动当作稳定性能。

### 5.3 MFU

$$
\operatorname{MFU}
=
\frac{\text{actual FLOP/s}}
{\text{hardware peak FLOP/s}}.
$$

计算理论峰值时必须匹配：

- 同一种硬件；
- 同一种 dtype；
- dense 或 sparse 口径；
- 单卡或多卡数量。

这里给出的经验判断是：**MFU 达到 $0.5$，即约 50%，通常已经相当不错。**

MFU 低不一定说明代码“写错了”，还可能来自：

- 算子受内存带宽限制；
- 矩阵太小，无法充分占满 GPU；
- kernel launch、同步或 Python 开销；
- 不规则 shape 和额外类型转换；
- 多卡通信；
- 理论峰值本身是理想上限。

## 6. 算术强度（arithmetic intensity）与 Roofline

一个 GPU 算子可以抽象成：

1. 从显存读取输入；
2. 在计算单元执行运算；
3. 把输出写回显存。

设：

- $F$：该算子的 FLOPs；
- $Q$：需要在显存与计算单元之间搬运的 bytes；
- $P_{\text{peak}}$：硬件峰值 FLOP/s；
- $BW_{\text{peak}}$：显存峰值 bytes/s。

理想情况下，计算与数据搬运完全重叠：

$$
T_{\text{compute}}
=
\frac{F}{P_{\text{peak}}},
\qquad
T_{\text{memory}}
=
\frac{Q}{BW_{\text{peak}}},
$$

$$
T_{\text{ideal}}
\approx
\max\left(
T_{\text{compute}},
T_{\text{memory}}
\right).
$$

### 6.1 两个强度

算子的算术强度：

$$
I_{\text{op}}
=
\frac{F}{Q}
\quad [\text{FLOPs/byte}].
$$

硬件的平衡点（这里称 accelerator intensity）：

$$
I_{\text{hw}}
=
\frac{P_{\text{peak}}}{BW_{\text{peak}}}
\quad [\text{FLOPs/byte}].
$$

比较二者：

| 条件 | 瓶颈 | 含义 |
|---|---|---|
| $I_{\text{op}} < I_{\text{hw}}$ | memory-bound | 搬运数据花的时间更多 |
| $I_{\text{op}} > I_{\text{hw}}$ | compute-bound | 计算花的时间更多 |

> 算子每搬 1 byte 所做的计算不够多，即 $I_{\text{op}}$ 小于硬件平衡点时，是**带宽瓶颈**；反之才是**计算瓶颈**。

### 6.2 常见算子的直觉

下面假设 bf16 为 2 bytes，且只做最简化的 HBM 读写核算：

| 算子 | 近似 FLOPs | 近似 bytes | 算术强度 | 常见瓶颈 |
|---|---:|---:|---:|---|
| ReLU，$n$ 元素 | $n$ | $4n$ | $1/4$ | memory-bound |
| GELU，$n$ 元素 | 约 $20n$ | $4n$ | 约 $5$ | 通常仍 memory-bound |
| 点积 | 约 $2n$ | 约 $4n$ | 约 $1/2$ | memory-bound |
| $n \times n$ 矩阵 × 向量 | 约 $2n^2$ | 约 $2n^2$ | 约 $1$ | memory-bound |
| 两个 $n \times n$ 矩阵相乘 | 约 $2n^3$ | 约 $6n^2$ | 约 $n/3$ | $n$ 足够大时 compute-bound |

关键区别是**数据复用**：

- 矩阵乘向量主要是把大矩阵读一遍，每个权重只参与很少计算；
- 矩阵乘矩阵可以让加载进片上存储的数据参与大量乘加，算术强度随矩阵规模增长。

这也解释了一个重要系统现象：

- 大 batch 的训练以大矩阵乘法为主，较容易 compute-bound；
- 自回归解码一次通常只处理很少 token，更像矩阵乘向量，常常 memory-bound。

### 6.3 Roofline 上限

可达到的性能近似受以下上限限制：

$$
P_{\text{attainable}}
\le
\min\left(
P_{\text{peak}},
I_{\text{op}} \cdot BW_{\text{peak}}
\right).
$$

在理想 Roofline 模型下：

$$
\operatorname{MFU}_{\max}
\approx
\min\left(
1,
\frac{I_{\text{op}}}{I_{\text{hw}}}
\right).
$$

> 实际性能还会受缓存、内存访问是否连续、kernel launch、线程占用率、算子融合、通信和软件实现影响。算子融合之所以有效，核心原因之一就是减少中间结果写回和再次读取，从而降低 $Q$、提高有效算术强度。

## 7. 自动微分与前向/反向 FLOPs

```python
x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

pred = x @ w
loss = 0.5 * (pred - 5).pow(2)
loss.backward()

print(w.grad)
```

当 `requires_grad=True` 时，PyTorch 会为相关运算建立计算图。`loss.backward()` 通过链式法则计算叶子参数的梯度，并把结果累积到 `parameter.grad`。

> 多次调用 `backward()` 会把新梯度加到旧梯度上，而不是覆盖。因此普通训练步结束后必须清空梯度；梯度累积则是有意利用这一行为。

对线性层：

$$
H = XW,
$$

反向传播需要计算：

$$
\frac{\partial L}{\partial X}
=
\frac{\partial L}{\partial H} W^\top,
\qquad
\frac{\partial L}{\partial W}
=
X^\top \frac{\partial L}{\partial H}.
$$

这两个操作都是与前向规模相近的矩阵乘法。因此：

| 阶段 | 近似 FLOPs |
|---|---:|
| forward | $2BP$ |
| backward | $4BP$ |
| forward + backward | $6BP$ |

其中 $B$ 是数据点数量，$P$ 是参数量。对语言模型，把数据点理解为本次训练处理的 token 数，可得到著名的粗略估计：

$$
C_{\text{train}}
\approx
6ND,
$$

其中：

- $N$：模型参数量；
- $D$：训练 token 数；
- $C_{\text{train}}$：训练总 FLOPs。

> 它主要近似由参数矩阵乘法主导的训练。当上下文很长时，attention 中与序列长度平方相关的计算会变得不可忽略；embedding、logits、激活函数、归一化、重计算和通信也可能使实际开销偏离该式。

## 8. 完整训练步骤与状态生命周期

一个最小训练步：

```python
for x, y in dataloader:
    logits = model(x)              # forward
    loss = loss_fn(logits, y)
    loss.backward()                # backward
    optimizer.step()               # update parameters
    optimizer.zero_grad(set_to_none=True)
```

数据流：

```text
batch
  → parameters + activations
  → loss
  → gradients
  → optimizer states
  → updated parameters
```

`optimizer.zero_grad(set_to_none=True)` 让 `.grad` 回到 `None`，通常比逐元素写零减少一次内存操作；下一次 backward 会重新创建梯度张量。

### 优化器状态

| 优化器 | 额外状态 | 常见状态内存 |
|---|---|---:|
| SGD | 无或 momentum | 0 或 4 bytes/parameter |
| AdaGrad | 累积平方梯度 | 4 bytes/parameter |
| Adam / AdamW | 一阶矩 + 二阶矩 | 8 bytes/parameter |

AdamW 的一阶矩 $m_t$ 和二阶矩 $v_t$：

$$
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t,
$$

$$
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2.
$$

它们跨训练步存在，所以不会在每个 step 后释放。

## 9. 训练显存由五部分组成

训练峰值显存可以粗略拆为：

$$
M_{\text{train}}
\approx
M_{\text{params}}
+
M_{\text{grads}}
+
M_{\text{optimizer}}
+
M_{\text{activations}}
+
M_{\text{temp}}.
$$

其中：

- **参数**：随参数量 $N$ 线性增长；
- **梯度**：通常与可训练参数量同阶；
- **优化器状态**：由优化器和状态精度决定；
- **激活**：受 micro-batch、序列长度、层数、hidden size 和实现影响；
- **临时空间**：kernel workspace、通信 buffer、allocator 缓存和内存碎片等。

在这里的 bf16 参数/梯度 + fp32 Adam 状态假设下：

$$
M_{\text{model states}}
\approx
(2+2+8)N
=
12N\ \text{bytes}.
$$

这只是模型状态，不包含激活、临时张量和可选的 fp32 master weights。

## 10. 梯度累积：用时间换 micro-batch 显存

设目标有效 batch size 为 $B$，单次能放进显存的 micro-batch size 为 $b$，则累积步数：

$$
K = \frac{B}{b}.
$$

```python
optimizer.zero_grad(set_to_none=True)

for micro_step in range(K):
    loss = model(micro_batch[micro_step]) / K
    loss.backward()

optimizer.step()
```

效果：

- 单次只保存 $b$ 个样本或序列的激活；
- 参数梯度在 $K$ 个 micro-batch 间累积；
- 最后只更新一次参数；
- 有效 batch size 仍约为 $Kb$。

它主要降低的是**激活峰值显存**，不会降低参数、梯度和优化器状态本身的显存，也不会减少处理同样数据所需的主要 FLOPs。

## 11. 激活检查点：用重算换显存

反向传播需要前向时的中间激活。普通训练会保存很多层的激活，深度 $L$ 增大时，激活显存大致随 $L$ 增长。

激活检查点（activation checkpointing），也称：

- gradient checkpointing；
- rematerialization。

核心机制：

```text
forward：只保存部分层的激活
backward：从最近的 checkpoint 重新计算缺失激活
```

权衡：

| 保存策略 | 激活内存 | 重计算 |
|---|---:|---:|
| 保存每层激活 | $O(L)$ | 几乎无 |
| 完全不保存 | $O(1)$ | 最坏可达 $O(L^2)$ |
| 每隔约 $\sqrt{L}$ 层保存 | $O(\sqrt{L})$ | 约 $O(L)$ |

PyTorch 示例：

```python
from torch.utils.checkpoint import checkpoint

x = checkpoint(block, x, use_reentrant=False)
```

当激活而不是参数成为主要显存瓶颈时，只减小参数精度仍然无法容纳更长序列或更大的 micro-batch。

## 12. 张量形状与数据流

| 对象 | Shape / 类型 | 生命周期与作用 |
|---|---|---|
| token ids | $(B,S)$，整数 | 输入 batch |
| embedding / hidden | $(B,S,d_{\text{model}})$ | 逐层产生的激活 |
| Q/K/V | $(B,H,S,D_h)$ | attention 中间张量 |
| attention scores | $(B,H,S,S)$ | 长上下文时可能很大 |
| logits | $(B,S,V)$ | 对词表的预测 |
| parameter | 由模块决定 | 跨所有 step 保存 |
| gradient | 与 parameter 相同 | backward 产生，更新前使用 |
| Adam states | 每个参数两个 fp32 状态 | 跨所有 step 保存 |
| scalar loss | $()$ | backward 的起点 |

## 13. 资源核算示例

### 13.1 训练总 FLOPs 与时间

训练一个 $70$B 参数模型，使用 $15$T tokens：

$$
C
\approx
6ND
=
6 \times 70 \times 10^9 \times 15 \times 10^{12}
=
6.3 \times 10^{24}\ \text{FLOPs}.
$$

若使用 1024 张 H100，按这里采用的 dense bf16 峰值约 $9.895\times10^{14}$ FLOP/s/卡，并假设 MFU 为 50%：

$$
T
\approx
\frac{6.3\times10^{24}}
{1024 \times 9.895\times10^{14} \times 0.5}
\approx
1.25\times10^7\ \text{s}
\approx
144\ \text{days}.
$$

这是理想化的数量级估计，尚未单独计入停机、通信波动和 checkpoint I/O。

### 13.2 显存最多能放多少参数？

8 张 80 GB H100 的总显存：

$$
8 \times 80 \times 10^9
=
640 \times 10^9\ \text{bytes}.
$$

若只考虑 bf16 参数、bf16 梯度和 fp32 Adam 状态：

$$
N_{\max}
\approx
\frac{640\times10^9}{12}
\approx
53.3\times10^9.
$$

所以理论上界约为 53B 参数，但真实可训练规模更小，因为还需要激活、临时张量、通信 buffer 和 allocator 空间。

> 只算 bf16 参数会得到 $640/2=320$B，这完全忽略梯度、优化器状态和激活，不是训练可行规模。

## 14. 线性层的参数与计算量

考虑：

```python
B, D, K = 1024, 256, 64
x = torch.randn(B, D)
w = torch.randn(D, K, requires_grad=True)
y = x @ w
loss = y.square().mean()
loss.backward()
```

### 14.1 张量形状

| 张量 | Shape |
|---|---|
| `x` | $(B,D)$ |
| `w` | $(D,K)$ |
| `y` | $(B,K)$ |
| `w.grad` | $(D,K)$ |

### 14.2 参数量

$$
P = DK.
$$

### 14.3 前向 FLOPs

$$
F_{\text{forward}}
\approx
2BDK
=
2BP.
$$

### 14.4 反向 FLOPs

需要分别计算输入梯度和权重梯度：

$$
F_{\text{backward}}
\approx
4BDK
=
4BP.
$$

### 14.5 总 FLOPs

$$
F_{\text{train}}
\approx
6BDK
=
6BP.
$$

这个小例子正是 $6ND$ 估算的来源。

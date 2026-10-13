---
title: "从 SGD 到 Muon：优化器的演进、公式与计算过程"
date: "2026-10-13"
description: "从随机梯度、动量和自适应步长，到 AdamW、低内存优化器与矩阵几何，逐步理解 SGD、Adam、Shampoo 和 Muon 如何把梯度变成参数更新。"
tags: ["LLM", "CS336"]
category: "笔记"
---

反向传播给出梯度，优化器决定参数怎样移动。两者经常连在一起使用，却解决不同问题：`loss.backward()` 计算局部导数，`optimizer.step()` 综合当前梯度、历史状态和更新规则，改变参数。

[CS336 L02](/articles/cs336-l02-resource-accounting) 已经介绍了优化器状态的显存开销，[L03](/articles/cs336-l03-architectures) 提到 AdamW 与 weight decay。这篇继续解释这些状态为什么存在，以及 SGD、Adam、Muon 分别怎样使用它们。

本文覆盖主要演进路线及相关分支，不是所有优化器的全集。历史、公式与实现资料核对于 2026-10-13；较新的 Muon 资料中，作者博客、技术报告和软件实现分别标明，实验结果只按原文的任务与预算解释。

## 1. 优化器究竟在计算什么

### 1.1 目标、梯度和更新方向

设训练目标为：

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta).
$$

这里的 $N$ 是训练样本数，$\theta\in\mathbb{R}^P$ 是全部可训练参数。第 $t$ 步，用当前参数 $\theta_{t-1}$ 和小批量 $\mathcal{B}_t$ 计算：

$$
g_t=\frac{1}{|\mathcal{B}_t|}
\sum_{i\in\mathcal{B}_t}\nabla\ell_i(\theta_{t-1}).
$$

优化器产生方向 $u_t$，再做：

$$
\theta_t=\theta_{t-1}-\eta_tu_t.
$$

SGD 取 $u_t=g_t$；Momentum 使用历史梯度；Adam 对历史方向逐坐标缩放；Muon 则处理二维更新矩阵的奇异值。

全文向量公式中的平方、开根号和除法都是逐元素运算。矩阵乘法、SVD、矩阵逆根会另行说明。$\epsilon>0$ 是稳定常数，$\eta_t$ 是基础学习率；基础学习率相同，不代表实际更新量相同。

### 1.2 为什么梯度下降会走得慢

用二次函数作一个可手算的例子：

$$
L(x,y)=\frac12(x^2+100y^2),
\qquad \nabla L=(x,100y).
$$

SGD 的确定性版本给出：

$$
x_t=(1-\eta)x_{t-1},
\qquad y_t=(1-100\eta)y_{t-1}.
$$

为了让 $y$ 稳定，需要 $0<\eta<0.02$，但此时 $x$ 的收敛很慢。一个方向很陡、另一个方向很平，统一步长就容易顾此失彼。随机小批量再加入噪声，轨迹还会来回摆动。

这引出三种不同的改进：

- 通过动量稳定方向，利用多步一致的信息。
- 根据每个坐标的历史尺度调整步长。
- 使用矩阵结构或其他几何信息，让不同方向按更合适的尺度移动。

梯度大不一定代表“这个参数更重要”，也可能只是坐标尺度或局部曲率大。优化器处理的正是这些差别。

## 2. 演进路线与历史坐标

| 时间 | 方法与代表资料 | 当时要解决的问题 | 主要变化 |
|---|---|---|---|
| 1951 | Robbins–Monro 随机逼近 | 用带噪观测寻找目标 | SGD 的理论根源之一 |
| 1964 | Polyak heavy-ball | 迭代在狭长谷底收敛慢 | 累积惯性 |
| 1983 | Nesterov 加速 | 改善光滑凸优化的收敛速率 | 在外推位置计算修正 |
| 2010/2011 | AdaGrad | 在线学习中坐标频率与尺度不同 | 累计平方梯度 |
| 2012 | RMSProp、AdaDelta | AdaGrad 后期步长过小 | 忘记旧尺度；利用更新历史 |
| 2014/2015 | Adam、AdaMax | 统一方向平滑与自适应尺度 | 一阶矩、二阶矩及偏差修正 |
| 2016 | Nadam | 把 Nesterov 思路用于 Adam | 修改一阶方向的组合 |
| 2017/2019 | AdamW | 自适应缩放使 L2 与 weight decay 不再等价 | 解耦参数衰减 |
| 2017、2019/2020 | LARS、LAMB | 大批量训练中层间更新尺度不匹配 | 分层 trust ratio |
| 2018 | AMSGrad | Adam 的特定非收敛反例 | 保留二阶尺度的历史最大值 |
| 2018 | Adafactor、Shampoo、signSGD | 状态内存、矩阵结构、通信成本 | 三条不同的路线 |
| 2019/2020 | RAdam | 自适应学习率在早期估计不稳 | 方差整流 |
| 2020 | AdaBelief | 区分梯度幅值与梯度预测偏差 | 用残差平方估计尺度 |
| 2023 | Lion | 搜索更简洁的更新规则 | 单份动量与符号更新 |
| 2024/2025 | Muon 及其 LLM 扩展 | 矩阵更新方向的谱不均衡，以及大模型训练效率 | 动量矩阵的近似极分解 |

表中有些年份是预印本与正式发表年份，如 Adam 是 2014 年预印本、ICLR 2015 论文，AdamW 是 2017 年预印本、ICLR 2019 论文。RMSProp 应引用 Tieleman/Hinton 的课程资料；Muon 的 2024 年作者博客也不应写成正式会议论文。原始资料见文末。

这些方法形成了分支。AdamW 不解决所有几何问题，Muon 不消除所有自适应优化需求，Adafactor 的低状态开销也不等于它具有 Shampoo 的矩阵预条件能力。

```text
小批量梯度
  ├─ 方向累计：SGD → Momentum → Nesterov
  ├─ 坐标尺度：AdaGrad → RMSProp / AdaDelta → Adam
  │                                      ├─ AdamW：解耦衰减
  │                                      ├─ AMSGrad / RAdam / AdaBelief
  │                                      └─ Adafactor：减少状态存储
  ├─ 层间尺度：LARS / LAMB
  ├─ 符号方向：signSGD / Lion
  └─ 矩阵结构：full-matrix AdaGrad → Shampoo
                                   └─ 与 Muon 的极分解方向存在数学联系
```

这里的箭头表示理解上的联系，不意味着每一种方法都严格由前一种方法推导而来。

## 3. SGD：用便宜的随机方向开始训练

### 3.1 从全量梯度到小批量

全量梯度下降每步遍历全部数据；SGD 用单个样本，现代训练通常用 mini-batch。均匀抽样且满足相应条件时，小批量梯度是全量梯度的无偏估计：

$$
\mathbb{E}[g_t\mid\theta_{t-1}]=\nabla L(\theta_{t-1}).
$$

更新就是：

$$
\theta_t=\theta_{t-1}-\eta_tg_t.
$$

计算步骤只有读取梯度、乘学习率、减到参数上；没有必须跨步保存的历史张量。对 $P$ 个参数，优化器更新计算是 $O(P)$。

Robbins 与 Monro 的 1951 年论文研究的是带噪观测的随机求根，是理解 SGD 的理论起点之一，而不是今天神经网络训练代码的直接说明书。[Robbins–Monro 原文](https://www.columbia.edu/~ww2040/8100F16/RM51.pdf)

### 3.2 优点与限制

SGD 简单、状态少、每步开销低。它保留梯度幅值信息，也便于分析学习率与目标函数的关系。缺点是噪声直接进入每次更新，所有坐标共享同一学习率，在病态曲率上容易振荡。

常见随机逼近条件包括：

$$
\sum_t\eta_t=\infty,
\qquad \sum_t\eta_t^2<\infty.
$$

它们表达“不能过早停止移动，但噪声影响要逐渐可控”。这些条件还依赖目标与噪声假设，不能单独保证任意深度网络达到全局最优。实际训练的固定步数、warmup 和衰减策略，也不应与渐近定理混为一谈。

## 4. Momentum 与 Nesterov：让连续方向积累起来

### 4.1 Polyak Momentum

普通 SGD 在陡峭方向来回变号，在平缓方向却可能持续同向。Momentum 保存一份与参数同形的 buffer：

$$
b_t=\beta b_{t-1}+g_t,
\qquad
\theta_t=\theta_{t-1}-\eta_tb_t.
$$

展开可得：

$$
b_t=\sum_{k=1}^{t}\beta^{t-k}g_k.
$$

同向梯度会累积，交替变号的梯度会部分抵消。$\beta$ 越接近 1，记忆越长，但旧方向也越难消退。Heavy-ball 的历史来源是 Polyak 1964 年工作。[原刊信息](https://www.mathnet.ru/php/archive.phtml?jrnid=zvmmf&option_lang=eng&paperid=7713&wshow=paper)

注意两种约定：

$$
\text{累加式：}\quad b_t=\beta b_{t-1}+g_t,
$$

$$
\text{EMA 式：}\quad m_t=\beta m_{t-1}+(1-\beta)g_t.
$$

零初始化、固定 $\beta$ 时，$m_t=(1-\beta)b_t$。若直接交换公式却保持学习率不变，更新尺度也会改变。恒定梯度下，累加式 $b_t$ 趋向 $g/(1-\beta)$；它不只是把梯度“平滑一下”。

还有一种写法把学习率放进速度：

$$
v_t=\beta v_{t-1}-\eta_tg_t,
\qquad \theta_t=\theta_{t-1}+v_t.
$$

学习率变化时，它与把当前学习率乘在整个 buffer 外的写法一般不同。[PyTorch SGD 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)

Momentum 的收益是方向更稳定且只增加一份状态；代价是需要调节 $\eta$ 和 $\beta$，在转弯、梯度分布改变或动量过大时可能过冲。

### 4.2 Nesterov Accelerated Gradient

Nesterov 的思路是先看惯性预计带到的位置，再计算纠正方向。用速度形式表示：

$$
y_t=\theta_{t-1}+\beta v_{t-1},
$$

$$
v_t=\beta v_{t-1}-\eta_t\nabla\ell_{\mathcal{B}_t}(y_t),
\qquad
\theta_t=\theta_{t-1}+v_t.
$$

如果前方斜坡已经变化，这个修正可以更早抵消惯性。显式 lookahead 形式只需要在 $y_t$ 做一次梯度计算，不是必然需要两个 backward。[Sutskever 等，ICML 2013，式 3–4](https://proceedings.mlr.press/v28/sutskever13.pdf)

软件库常采用重参数化的实现。例如 PyTorch 在存储参数处算 $g_t$，再组合：

$$
b_t=\beta b_{t-1}+g_t,
\qquad u_t=g_t+\beta b_t.
$$

不能把这个实现描述成软件偷偷移动参数、再额外计算一个梯度。比较它与显式 lookahead 形式时，要同时检查参数坐标、初始化和学习率安排。

1983 年原方法在相应光滑凸问题下取得加速收敛率；固定 $\beta$ 的深度学习实现不自动继承原定理。[Nesterov 原文](https://hengshuaiyao.github.io/papers/nesterov83.pdf)

## 5. AdaGrad：让每个坐标有自己的历史尺度

### 5.1 背景与公式

在线学习和稀疏特征中，有些坐标频繁出现，有些坐标很少出现。统一学习率难以同时照顾它们。常用的 diagonal AdaGrad 累加平方梯度：

$$
s_t=s_{t-1}+g_t^2,
\qquad
\theta_t=\theta_{t-1}-\eta_t
\frac{g_t}{\sqrt{s_t}+\epsilon}.
$$

第 $i$ 个坐标的有效系数是 $\eta_t/(\sqrt{s_{t,i}}+\epsilon)$。长期累计平方梯度较大的坐标被缩小；累计较少的坐标相对保留较大步长。这既与出现频率有关，也与梯度幅值有关，不能只解释成词频计数。[Duchi、Hazan、Singer，JMLR 2011](https://jmlr.csail.mit.edu/papers/volume12/duchi11a/duchi11a.pdf)

例如第一步 $g_1=(1,10)$，零初始化且暂时忽略 $\epsilon$：

$$
s_1=(1,100),
\qquad \frac{g_1}{\sqrt{s_1}}=(1,1).
$$

原本 10 倍的幅值差被消除了。这里改变的是更新方向的坐标比例，不只是给整个向量乘一个标量。

### 5.2 优点与代价

AdaGrad 自动利用不同坐标的历史信息，适合理解稀疏在线学习，只需一个 $P$ 元素状态。问题在于过去的大梯度永久进入分母，长期训练中有效步长可能越来越小，难以响应新分布。只有当平方和持续增长时才会出现持续缩小，不能无条件说每个坐标的步长一定趋零。

原论文也讨论 full-matrix 预条件。忽略投影与正则项，一种带阻尼的示意形式是：

$$
S_t=\epsilon I+\sum_{k=1}^{t}g_kg_k^\mathsf{T},
\qquad u_t=S_t^{-1/2}g_t.
$$

原文的一种约定是先对累计矩阵开根，再加 $\delta I$：

$$
H_t=\delta I+\left(\sum_{k\le t}g_kg_k^\mathsf{T}\right)^{1/2},
\qquad u_t=H_t^{-1}g_t.
$$

稳定项的位置与上面的示意形式不同。完整矩阵可以表达坐标关联，但 $P\times P$ 的状态需要 $O(P^2)$ 存储。对大模型整体参数做这种运算通常不可承受，因此后面会出现结构化近似。

## 6. RMSProp 与 AdaDelta：旧梯度不应永久支配步长

### 6.1 RMSProp

RMSProp 把 AdaGrad 的无限累加改为指数移动平均：

$$
s_t=\rho s_{t-1}+(1-\rho)g_t^2,
\qquad
\theta_t=\theta_{t-1}-\eta_t
\frac{g_t}{\sqrt{s_t}+\epsilon}.
$$

历史权重随 $\rho^k$ 衰减，分母不必单调增大，因此可以适应非平稳的梯度尺度。$1/(1-\rho)$ 可作为记忆长度的粗略尺度。

RMSProp 通常追溯到 Tieleman/Hinton 2012 年课程资料；Hinton 原课件将其标为 Tieleman 的 unpublished work，引用时应保留这个身份。[Hinton Lecture 6](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)

其优点是能遗忘很久以前的大梯度，状态仍只有一份。缺点是尺度估计依赖 $\rho$：记忆过长会滞后，过短又容易受噪声影响。基础版本没有 Adam 式偏差修正，仍需基础学习率。

$s_t$ 是非中心二阶矩估计，不是方差。Centered RMSProp 另外保存均值 EMA，使用二阶矩减均值平方。软件实现也有 $\sqrt{s_t}+\epsilon$ 与 $\sqrt{s_t+\epsilon}$ 两种分母位置；相同数值的 $\epsilon$ 不意味着相同效果。[PyTorch RMSprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)

### 6.2 AdaDelta

Zeiler 2012 年的 AdaDelta 同时利用梯度尺度与过去的更新尺度。设 $s_t$ 保存平方梯度 EMA，$q_t$ 保存平方更新 EMA：

$$
s_t=\rho s_{t-1}+(1-\rho)g_t^2,
$$

$$
\Delta_t=-\frac{\sqrt{q_{t-1}+\epsilon}}
{\sqrt{s_t+\epsilon}}g_t,
$$

$$
q_t=\rho q_{t-1}+(1-\rho)\Delta_t^2,
\qquad \theta_t=\theta_{t-1}+\Delta_t.
$$

顺序很重要：先用旧的 $q_{t-1}$ 算本次更新，再把更新平方放入 $q_t$，否则会出现循环依赖。$\epsilon$ 也让零初始化时的更新能够启动。[AdaDelta 原论文](https://arxiv.org/abs/1212.5701)

原算法不需要额外全局学习率，通过 RMS 比值协调更新尺度；但仍有 $\rho$ 和 $\epsilon$。它保存两份状态，比 RMSProp 多一份，并不保证在所有训练任务上更快。PyTorch 额外提供 `lr` 乘数，且更新历史累计的是乘该 `lr` 之前的 delta，阅读实现时应区分它与实际参数位移。[PyTorch Adadelta](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adadelta.html)

## 7. Adam：分别估计方向与尺度

### 7.1 计算过程

Adam 把梯度方向的 EMA 和平方梯度的 EMA 放在一起：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

零初始化时，先作偏差修正：

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},
$$

再更新：

$$
\theta_t=\theta_{t-1}-\eta_t
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

$m_t$ 提供平滑的方向；$v_t$ 提供各坐标最近的尺度。方向频繁反转时，$m_t$ 可能相互抵消，而 $v_t$ 仍然较大，更新因此减弱。Adam 是 Kingma 与 Ba 的 2014 年预印本、ICLR 2015 工作。[Adam 原论文](https://arxiv.org/abs/1412.6980)

### 7.2 偏差修正为什么存在

若梯度恒为 $g$，则：

$$
m_t=(1-\beta_1^t)g,
\qquad
v_t=(1-\beta_2^t)g^2.
$$

除掉相应因子，便恢复 $g$ 和 $g^2$。这是消除零初始化带来的 EMA 权重缺口，不是消除小批量梯度的抽样噪声，也不意味着非平稳梯度下能精确估计真实矩。

第一步 $g_1=(2,-4)$、$\beta_1=0.9$、$\beta_2=0.999$ 时：

$$
m_1=(0.2,-0.4),\quad v_1=(0.004,0.016),
$$

$$
\widehat m_1=(2,-4),\quad \widehat v_1=(4,16).
$$

忽略 $\epsilon$，归一化方向是 $(1,-1)$。没有偏差修正时，方向约为 $(3.162,-3.162)$。这里的数值由公式直接算出，显示了初始化约定对实际步长的影响。

### 7.3 优点、限制与状态

Adam 每步仍是逐元素 $O(P)$ 运算，却能同时平滑方向并适应坐标尺度；额外保存 $m,v$ 两份状态。它不会显式计算 Hessian，“二阶矩”也不等于“二阶导数”。

适应梯度尺度不代表无需调学习率。学习率、两个 $\beta$、$\epsilon$、weight decay 和调度都会影响训练。Adam 的特定理论收敛问题后来由 AMSGrad 工作指出，因此不能把原论文的理论讨论转述成任意深度训练都保证收敛。[Reddi 等，On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)

## 8. AdaMax 与 Nadam：Adam 的相邻分支

### 8.1 AdaMax：用无穷范数尺度

Adam 原论文还给出 AdaMax。它保留一阶矩，但将二阶尺度替换为带衰减的逐坐标最大幅值：

$$
a_t=\max(\beta_2a_{t-1},|g_t|),
\qquad
\theta_t=\theta_{t-1}-\eta_t
\frac{m_t/(1-\beta_1^t)}{a_t+\epsilon}.
$$

`max` 和绝对值都是逐元素操作。它避免平方与开根，在偶发大幅梯度后保留一个逐渐衰减的尺度。优点是规则简洁；代价是仍需两份状态，大梯度可以抑制随后多步更新，没有普遍优于 Adam 的保证。原论文的 $a_t$ 不需要与 Adam 二阶矩相同的偏差修正。[Adam §7](https://arxiv.org/pdf/1412.6980)

### 8.2 Nadam：在自适应尺度上加入 Nesterov 组合

Dozat 2016 年工作将 Nesterov 思路与 Adam 结合。用当前梯度与动量预测共同产生分子方向，比只使用一阶 EMA 多一项即时修正。

为避免混淆固定动量的简化公式与真实实现，这里采用 PyTorch 文档的调度形式。令动量衰减参数为 $\psi$：

$$
\mu_t=\beta_1\left(1-\frac12\,0.96^{t\psi}\right),
\qquad
p_t=\prod_{k=1}^{t}\mu_k.
$$

$m_t,v_t$ 仍按 Adam 的 EMA 计算，但更新分子改为：

$$
\widetilde m_t=
\frac{\mu_{t+1}m_t}{1-p_{t+1}}
+\frac{(1-\mu_t)g_t}{1-p_t},
$$

$$
\theta_t=\theta_{t-1}-\eta_t
\frac{\widetilde m_t}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}.
$$

它仍保存两份参数同形状态，另有标量乘积。优势是无需显式第二次梯度计算便引入 Nesterov 风格的方向组合；限制是调度、偏差修正和框架约定更复杂，不能把一行“Adam 加 Nesterov”当作完整实现。[PyTorch NAdam 公式与原文链接](https://docs.pytorch.org/docs/2.14/generated/torch.optim.NAdam.html)

## 9. AdamW：把正则化和自适应梯度缩放分开

### 9.1 L2 与 weight decay 为什么会分叉

在目标中加入 $\lambda\|\theta\|_2^2/2$，会把梯度改成：

$$
g_t^{\mathrm{reg}}=g_t+\lambda\theta_{t-1}.
$$

无动量、无预条件的 SGD 因而得到：

$$
\theta_t=(1-\eta_t\lambda)\theta_{t-1}-\eta_tg_t.
$$

这里 L2 与相应的乘法衰减等价。但在 Adam 中，$\lambda\theta$ 会进入 $m_t,v_t$，随后也被各坐标的自适应分母缩放，既改变方向历史又改变尺度历史。

### 9.2 AdamW 的更新

AdamW 用数据损失的梯度更新 Adam 状态，另行对参数施加衰减：

$$
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

这恢复了独立于逐坐标梯度缩放的衰减规则，不额外增加参数同形状态。Loshchilov 与 Hutter 的论文直接讨论了这种不等价及解耦方案。[AdamW 原论文，ICLR 2019](https://arxiv.org/abs/1711.05101)

“解耦”不等于训练效果与学习率无关：累计缩小量仍取决于 $\prod_t(1-\eta_t\lambda)$，学习率调度和训练长度会影响总衰减。Bias、归一化增益等参数是否衰减，是训练配方与参数分组选择，不能仅由名称自动决定。[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)

## 10. AMSGrad、RAdam 与 AdaBelief：分别修补不同问题

### 10.1 AMSGrad：防止二阶历史被过快忘记

Reddi、Kale、Kumar 的 ICLR 2018 工作给出 Adam 在特定在线凸问题中的反例，并提出保留长期二阶信息。原始 AMSGrad 的主要改动是：

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\qquad
\overline v_t=\max(\overline v_{t-1},v_t).
$$

用 $\overline v_t$ 代替更新分母中的 $v_t$。原算法采用其论文中的学习率与动量安排；在 PyTorch 的 Adam 实现中，还保留偏差修正：

$$
\theta_t=\theta_{t-1}-\eta_t
\frac{\widehat m_t}
{\sqrt{\overline v_t/(1-\beta_2^t)}+\epsilon}.
$$

优点是二阶累积器不再随 EMA 遗忘而变小；代价是多一份同形最大值状态，历史异常值也可能长期限制更新。原始凸分析并不意味着非凸大模型一定训练得更好。[原论文](https://arxiv.org/abs/1904.09237)、[PyTorch Adam 的 AMSGrad 选项](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)

### 10.2 RAdam：处理早期自适应尺度估计的不稳定

RAdam 将 warmup 的一部分作用解释为降低早期自适应学习率的方差，并设计整流系数。这是特定分析视角，不是 warmup 在所有模型中的唯一原因。[Liu 等，ICLR 2020](https://arxiv.org/abs/1908.03265)

定义近似有效窗口长度：

$$
\rho_\infty=\frac{2}{1-\beta_2}-1,
\qquad
\rho_t=\rho_\infty-
\frac{2t\beta_2^t}{1-\beta_2^t}.
$$

原论文算法在 $\rho_t>4$ 时计算：

$$
r_t=\sqrt{
\frac{(\rho_t-4)(\rho_t-2)\rho_\infty}
{(\rho_\infty-4)(\rho_\infty-2)\rho_t}}
$$

并令：

$$
u_t=r_t\widehat m_t
\frac{\sqrt{1-\beta_2^t}}{\sqrt{v_t}+\epsilon}.
$$

在尚不能启用整流的阶段，使用 $u_t=\widehat m_t$，即暂时不做二阶自适应缩放。它仍需两份状态，多出的标量计算很小。

上式采用原文通用自适应形式中对未修正 $v_t$ 的分母加入 $\epsilon$ 的约定；忽略稳定项时就是 $r_t\widehat m_t/\sqrt{\widehat v_t}$。不同实现会用略不同的启用阈值与稳定项位置；例如 PyTorch 文档采用 $\rho_t>5$。复现时不能把阈值从一个版本、分母从另一个版本拼起来。[PyTorch RAdam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.RAdam.html)

RAdam 提供了较有原则的早期缩放策略，但并不能据此取消所有模型的 warmup；归一化、初始化、层间尺度等仍可能需要学习率逐渐升高。

### 10.3 AdaBelief：用梯度预测误差估计尺度

Adam 的分母关心梯度有多大；AdaBelief 关心梯度与平滑预测有多不一致。保留同样的一阶 EMA，改用：

$$
s_t=\beta_2s_{t-1}+(1-\beta_2)(g_t-m_t)^2,
\qquad
\widehat s_t=\frac{s_t}{1-\beta_2^t},
$$

$$
u_t=\frac{\widehat m_t}{\sqrt{\widehat s_t}+\epsilon}.
$$

这里先算本步 $m_t$，再算残差。为展示机制，上式省略了原文算法在 $s_t$ 中额外加入的稳定常数；逐行复现时应恢复该项，并核对是否启用解耦衰减等实现选项。[Zhuang 等，NeurIPS 2020，Algorithm 2](https://proceedings.neurips.cc/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf)

同方向梯度可使残差变小、步长增大；跳动梯度则使分母增大。优点是能区分“幅值大且稳定”和“幅值大但难预测”；限制是稳定常数尤其重要，残差不是严格意义上的无偏方差估计，也不是 Hessian。两份状态的内存量级与 Adam 相同。

## 11. Adafactor：先减少二阶状态的存储

### 11.1 行列分解

Adam 两份 fp32 状态对大模型很昂贵。Adafactor 针对矩阵参数 $W\in\mathbb{R}^{m\times n}$，只保存平方梯度二阶矩的行、列统计。[Shazeer 与 Stern，ICML 2018](https://proceedings.mlr.press/v80/shazeer18a.html)

采用行和、列和约定，并在平方梯度里先加入稳定项：

$$
Q_t=G_t^2+\epsilon_1\mathbf{1}_m\mathbf{1}_n^\mathsf{T}.
$$

$$
r_t=\beta_2r_{t-1}+(1-\beta_2)Q_t\mathbf{1}_n,
$$

$$
c_t=\beta_2c_{t-1}+(1-\beta_2)\mathbf{1}_m^\mathsf{T}Q_t.
$$

$r_t$ 有 $m$ 个元素，$c_t$ 有 $n$ 个元素。近似恢复二阶矩：

$$
\widetilde V_t=\frac{r_tc_t}{\mathbf{1}_m^\mathsf{T}r_t}.
$$

分子是外积，分母是标量；并不是把 $r$ 和 $c$ 按坐标直接相除。先加入 $\epsilon_1$ 可避免全零梯度下重构时发生 $0/0$，仅在开根号后加稳定项无法修复先前的除零。固定 $\beta_2$ 的教学形式还可以对其进行偏差修正；完整 Adafactor 通常使用随时间变化的二阶衰减和相对步长策略，应以具体版本为准。

它近似的是平方梯度的尺度矩阵，不是对权重 $W$ 做低秩压缩。虽然恢复矩阵是 rank-1，但逐元素 $G/\sqrt{\widetilde V}$ 仍是对角式缩放，不会变成 Shampoo 的左右矩阵预条件。

### 11.2 更新裁剪与参数尺度

仅分解状态还不足以处理训练稳定性。令：

$$
U_t=G_t\oslash\sqrt{\widetilde V_t},
\qquad
\operatorname{RMS}(U)=\sqrt{\frac{1}{mn}\sum_{i,j}U_{ij}^2}.
$$

对已经预条件的更新进行裁剪：

$$
\widetilde U_t=
\frac{U_t}{\max(1,\operatorname{RMS}(U_t)/d)}.
$$

相对步长形式再以参数尺度决定实际步长：

$$
\alpha_t=\rho_t\max(\epsilon_2,\operatorname{RMS}(W_{t-1})),
\qquad
W_t=W_{t-1}-\alpha_t\widetilde U_t.
$$

这里 $\rho_t$ 表示相对步长，与 RAdam 中同名的窗口长度不是同一个量。原算法涉及具体调度；这些式子用于把计算环节拆开。

不保存一阶动量时，矩阵的长期状态从 $O(mn)$ 降到 $O(m+n)$；向量参数仍需相应的未分解状态。若重新加入一份完整动量，整体状态就不再是纯粹的 $O(m+n)$。恢复或广播更新也仍需计算和临时空间，不能把状态减少当作全部显存同时减少。

## 12. LARS 与 LAMB：按层控制相对更新量

### 12.1 LARS

大批量训练降低梯度噪声，却可能使不同层的合适学习率相差很大。LARS 为每一层 $l$ 计算 trust ratio。包含 L2 项的一种常见形式是：

$$
q_l=\frac{\tau\|W_l\|_2}
{\|g_l\|_2+\lambda\|W_l\|_2+\epsilon},
\qquad
u_l=q_l(g_l+\lambda W_l).
$$

然后可把 $u_l$ 输入动量更新。这里的范数是将该层参数展平后的欧氏范数，$\tau$ 是 trust coefficient。它限制的是层的相对更新尺度，不是每个坐标的独立学习率。[You、Gitman、Ginsburg，2017 LARS 预印本](https://arxiv.org/abs/1708.03888)

优点是对大批量中的层间尺度差异作直接处理；限制是零范数、参数分组、bias/norm 排除及 trust ratio 裁剪都需要明确。不同实现的 weight decay 放置不同，不能只按算法名称猜测。

### 12.2 LAMB

LAMB 把 Adam 的逐坐标方向与分层 trust ratio 结合。先计算 Adam 状态，再令：

$$
a_l=\frac{\widehat m_l}{\sqrt{\widehat v_l}+\epsilon}
+\lambda W_l,
\qquad
q_l=\frac{\phi(\|W_l\|_2)}{\|a_l\|_2},
$$

$$
W_l\leftarrow W_l-\eta_tq_la_l.
$$

$\phi$ 可限制权重范数的范围；零分母需要 fallback。LAMB 中的衰减项没有进入 Adam 的矩估计，但整个 $a_l$ 又经过 trust ratio，因此它与直接使用 AdamW 的参数收缩仍有差别。[You 等，ICLR 2020 LAMB](https://arxiv.org/abs/1904.00962)

优点是同时处理坐标尺度与层间尺度；代价是保留 Adam 的两份状态，加上按层归约和参数分组复杂度。原论文针对大批量训练，不能因为名字含有“layer-wise”就推断它在小批量微调上一定更好。

## 13. signSGD 与 Lion：只保留符号时发生了什么

### 13.1 signSGD

2018 年的 signSGD 工作以降低分布式梯度通信成本为重要动机。基础更新：

$$
\theta_t=\theta_{t-1}-\eta_t\operatorname{sign}(g_t).
$$

各个非零坐标的更新绝对值相同，幅值信息被丢弃。分布式方案可以传输符号并做 majority vote，但单机使用 `sign()` 本身不自动减少已有通信。[Bernstein 等，ICML 2018](https://proceedings.mlr.press/v80/bernstein18a.html)

这种做法不怕单个坐标的巨大幅值，却更依赖符号可靠性；噪声导致符号错误时，错误方向仍走完整的一步。其理论比较也依赖梯度、噪声与曲率的条件，不能概括为“符号更新和 SGD 总一样好”。

### 13.2 Lion

Lion 来自 Chen 等人 2023 年的优化算法程序搜索。它只保留一份动量，更新方向与长期动量分别使用两个系数：

$$
c_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\operatorname{sign}(c_t),
$$

$$
m_t=\beta_2m_{t-1}+(1-\beta_2)g_t.
$$

顺序不能交换：先用旧 $m_{t-1}$ 计算本次方向，再用 $\beta_2$ 更新长期状态。与 Adam 不同，Lion 的 $\beta_2$ 不是平方梯度系数。[NeurIPS 2023 原论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/9a39b4925e35cf447ccba8757137d84f-Paper-Conference.pdf)、[作者代码](https://github.com/google/automl/blob/master/lion/lion_pytorch.py)

若有 $P$ 个非零坐标，符号方向的欧氏范数是 $\sqrt P$。因此不能直接沿用 Adam 的学习率；原工作也强调 Lion 通常需要较小的学习率。收益是从两份同形状态降到一份，计算简单；限制是丢失幅值、不同任务收益不一致，原论文报告的视觉或扩散实验改进不等于任意 LLM 都更好。

## 14. Shampoo：保留梯度矩阵的行列结构

### 14.1 为什么逐坐标缩放还不够

对 $W\in\mathbb{R}^{m\times n}$，Adam 将各元素看成独立坐标。可是一次矩阵更新会同时影响整组输入与输出方向，关联可能跨越多个元素。

Shampoo 以两个较小的矩阵近似处理这些关联：

$$
L_t=L_{t-1}+G_tG_t^\mathsf{T},
\qquad
R_t=R_{t-1}+G_t^\mathsf{T}G_t,
$$

$$
L_0=\epsilon I_m,\qquad R_0=\epsilon I_n,
$$

$$
W_t=W_{t-1}-\eta_tL_t^{-1/4}G_tR_t^{-1/4}.
$$

这是 Gupta、Koren、Singer 的 ICML 2018 原始矩阵形式。高阶张量会对各个轴构造相应预条件器，实际大规模实现还常采用分块、EMA、较低频率逆根计算等变体。[Shampoo 原论文](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf)

### 14.2 矩阵逆根怎样算

若 $L=Q\Lambda Q^\mathsf{T}$ 是正定矩阵的特征分解：

$$
L^{-1/4}=Q\Lambda^{-1/4}Q^\mathsf{T}.
$$

不是对每个 $L_{ij}$ 作负四分之一次幂。左右预条件器可以旋转并缩放方向，因此超出了 Adam 的独立元素除法。

对一个矩阵，长期状态需要 $m^2+n^2$ 个元素，相比完整 $mn$ 维参数的 full-matrix AdaGrad 状态小得多，但不一定比 Adam 少。直接分解逆根有 $O(m^3+n^3)$ 成本，还要构造 Gram 矩阵和应用左右变换。优势是更丰富的结构信息，限制是额外计算、内存、阻尼与分布式实现复杂度。

### 14.3 与 Muon 的数学联系

先做一个简化：不累计历史，用当前方向矩阵 $D$ 构造 $DD^\mathsf{T}$ 与 $D^\mathsf{T}D$，忽略阻尼，在非零奇异值支撑上取广义逆根。

若紧致 SVD 为 $D=U_r\Sigma_rV_r^\mathsf{T}$，则：

$$
(DD^\mathsf{T})^{-1/4}D(D^\mathsf{T}D)^{-1/4}
=U_r\Sigma_r^{-1/2}\Sigma_r\Sigma_r^{-1/2}V_r^\mathsf{T}
=U_rV_r^\mathsf{T}.
$$

这显示了瞬时左右预条件与极分解方向之间的联系。但原始 Shampoo 累计历史 Gram，Muon 处理动量矩阵；实际阻尼、缩放与实现也不同，二者不是同一个算法。[Bernstein 与 Newhouse，2024 预印本 Old Optimizer, New Norm](https://arxiv.org/html/2409.20325v1)

## 15. Muon：对动量更新矩阵做近似极分解

### 15.1 背景与适用范围

Muon 的名称是 Momentum Orthogonalized by Newton-Schulz。作者观察到神经网络的矩阵更新可能被少数大的奇异方向主导，希望重新分配不同方向的更新尺度。2024 年作者博客介绍算法与小规模训练结果，2025 年 Moonshot 技术报告研究了 LLM 扩展。[Jordan 的原始博客](https://kellerjordan.github.io/posts/muon/)、[Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1)

Muon 通常用于隐藏层的二维权重矩阵。Bias、归一化参数、embedding 和输出 head 通常使用 AdamW 等配套优化器。Embedding 也是二维张量，因此仅按 `p.ndim == 2` 分组并不充分；共享 embedding/head 还要避免重复分组。

这是一种参数分工的训练配方，而不是把模型中每个张量都强行正交化。

### 15.2 先选定动量约定

按当前作者仓库的 EMA 形式：

$$
M_t=\beta M_{t-1}+(1-\beta)G_t.
$$

不使用 Nesterov 时，取 $D_t=M_t$；作者代码的 EMA Nesterov 形式取：

$$
D_t=(1-\beta)G_t+\beta M_t.
$$

随后对 $D_t$ 作矩阵变换。当前 PyTorch 文档采用累加式 $B_t=\beta B_{t-1}+G_t$ 及 $D_t=G_t+\beta B_t$。无 Nesterov 时整体正尺度可被理想极分解消去，但两个 Nesterov 组合的相对权重也有不同，不能当作只差一个学习率。[作者 muon.py](https://raw.githubusercontent.com/KellerJordan/Muon/master/muon.py)、[PyTorch Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html)

### 15.3 理想化的 SVD 更新

对秩为 $r$ 的非零矩阵取紧致 SVD：

$$
D=U_r\Sigma_rV_r^\mathsf{T},
\qquad
\operatorname{Polar}(D)=U_rV_r^\mathsf{T}.
$$

它保留左右奇异向量，把非零奇异值的幅值变成 1。例如：

$$
D=\begin{bmatrix}10&0\\0&1\end{bmatrix}
\quad\Longrightarrow\quad
\operatorname{Polar}(D)=\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$$

若把 $D$ 除以 Frobenius 范数，只得到 $\operatorname{diag}(10,1)/\sqrt{101}$，两个奇异值仍是 10 倍关系。Muon 的理想化变换会改变这种相对比例。

对一般非对角矩阵，Adam 的逐元素缩放也不等价于该操作，因为 SVD 处理的是混合多个坐标的奇异方向。

### 15.4 谱范数下的最陡方向

欧氏范数约束与矩阵谱范数约束，给出的最陡方向不同。考虑一阶近似的局部问题：

$$
\min_{\|\Delta\|_2\le\eta}\langle D,\Delta\rangle,
\qquad
\langle A,B\rangle=\operatorname{tr}(A^\mathsf{T}B).
$$

这里 $\|\Delta\|_2$ 是矩阵谱范数，即最大奇异值。由谱范数与核范数的对偶性：

$$
\langle D,\Delta\rangle\ge-\eta\|D\|_*,
\qquad
\|D\|_* = \sum_i\sigma_i(D).
$$

取 $\Delta=-\eta U_rV_r^\mathsf{T}$ 就达到该下界。这给出 Muon 理想方向的一种几何解释。若 $D$ 是动量而不是当前梯度，它解的是动量方向对应的线性子问题，不能据此断言实际损失每步都下降。

对满秩矩形矩阵，极因子是半正交的：宽矩阵满足 $OO^\mathsf{T}=I$，高矩阵满足 $O^\mathsf{T}O=I$。秩亏时，紧致极因子是 partial isometry，零空间不会自动得到新的单位奇异值；最近的完整半正交补全可能不唯一。这也是“Muon 把所有矩阵变成正交矩阵”不够准确的原因。[Old Optimizer, New Norm](https://arxiv.org/html/2409.20325v1)

### 15.5 Newton–Schulz 怎样避免每步完整 SVD

设必要时已转置 $D$，使行数不大于列数。先归一化：

$$
X_0=\frac{D}{\|D\|_F+\epsilon}.
$$

经典三次迭代的一种形式是：

$$
X_{k+1}=\frac32X_k-\frac12X_kX_k^\mathsf{T}X_k.
$$

若某个奇异值为 $\sigma$，它被映射为：

$$
\sigma\longmapsto\frac32\sigma-\frac12\sigma^3.
$$

归一化将非零奇异值放入合适范围后，反复迭代可使它们趋向 1；零奇异值保持为零。很小的奇异值可能需要很多步，因此固定少量迭代只是近似。

作者实现使用五次多项式与调过的系数：

$$
A_k=X_kX_k^\mathsf{T},
\qquad
X_{k+1}=aX_k+(bA_k+cA_k^2)X_k,
$$

$$
(a,b,c)=(3.4445,-4.775,2.0315).
$$

常用五步，并利用 bf16 矩阵乘法。这里的系数追求少量步骤的实用效果，作者代码明确指出不保证输出精确的 $UV^\mathsf{T}$。不能把经典三次迭代的收敛性质原封不动套到这个五次近似上。[作者实现与注释](https://raw.githubusercontent.com/KellerJordan/Muon/master/muon.py)

转置是为了让 Gram 矩阵尽量小。对 $m\le n$ 的 $m\times n$ 矩阵，每次五次迭代的矩阵乘法约需：

$$
4nm^2+2m^3\quad\text{FLOPs},
$$

这里按一次乘加计 2 FLOPs，忽略线性组合。长期状态少不等于计算免费；小 batch、窄矩阵、通信布局和 kernel 都可能改变实际耗时。

### 15.6 为什么还需要形状缩放

满秩 $m\times n$ 理想极因子有 $\min(m,n)$ 个奇异值 1，因此：

$$
\|O\|_F^2=\min(m,n),
\qquad
\operatorname{RMS}(O)=\frac{1}{\sqrt{\max(m,n)}}.
$$

矩阵越大，每个元素的 RMS 越小。为了让不同形状的更新量更可比较，实现引入缩放 $s(m,n)$。例如 Moonshot 报告采用：

$$
s(m,n)=0.2\sqrt{\max(m,n)},
$$

并加入解耦衰减：

$$
W_t=(1-\eta_t\lambda)W_{t-1}
-\eta_ts(m,n)O_t.
$$

理想满秩下，缩放后更新 RMS 为 0.2；有限步 NS 会引入近似误差。秩亏则是另一件事：即使使用精确的支撑极因子，秩为 $r$ 时，缩放后的 RMS 也只有 $0.2\sqrt{r/\min(m,n)}$，可能明显小于 0.2。这个常数是该配方的经验目标，不是所有 AdamW 更新的恒定 RMS。

原始实现还有 $\sqrt{\max(1,m/n)}$ 一类形状缩放，当前库也提供不同模式。比较学习率之前要先核对缩放；对 weight decay 使用基础 $\eta_t$，不要误把矩阵形状因子也乘进参数衰减。[Moonshot §2.2](https://arxiv.org/html/2502.16982v1)、[PyTorch 的缩放选项](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html)

### 15.7 优点与限制

Muon 主要保存一份动量矩阵，没有 Adam 的逐元素二阶状态；矩阵变换能重新分配奇异方向的幅值。但它丢弃部分幅值信息，引入额外矩阵乘法，需要针对参数角色分组，也需要单独调学习率、缩放与衰减。

2025 年技术报告在 399M–1.5B dense 模型（参数量不含 embedding）的 compute-optimal scaling-law 拟合中报告，相同表现约需 AdamW 的 52% FLOPs，并给出 Moonlight MoE 训练案例。该结果不能改写成所有模型、所有任务或墙钟时间都快两倍。报告还讨论了微调与预训练优化器匹配的问题，不能默认对 AdamW 预训练的公共模型直接换 Muon 就会改善。[Moonshot §3.2、§3.5](https://arxiv.org/html/2502.16982v1)

## 16. 把几种更新放在同一个例子里

### 16.1 相同梯度序列，不同历史处理

取 $g_1=(1,10)$、$g_2=(1,-10)$，所有状态从零开始，暂时忽略 $\epsilon$ 与 weight decay。表中列出第二步方向 $u_2$，实际参数变化还需乘 $-\eta_2$。

| 方法 | 设置 | 第二步方向 | 含义 |
|---|---|---|---|
| SGD | 无状态 | $(1,-10)$ | 只看当前梯度 |
| Momentum | 累加式，$\beta=0.9$ | $(1.9,-1)$ | 第一坐标同向累积，第二坐标部分抵消 |
| AdaGrad | 累加平方 | $(1/\sqrt2,-1/\sqrt2)$ | 幅值差被各坐标平方和消除 |
| RMSProp | $\rho=0.9$，无偏差修正 | $(1/\sqrt{0.19},-1/\sqrt{0.19})$ | 方向约为 $(2.294,-2.294)$ |
| Adam | $\beta_1=0.9$，$\beta_2=0.999$，偏差修正 | $(1,-1/19)$ | 变号方向被动量抑制，同向方向保持 |
| Lion | $\beta_1=0.9$，$\beta_2=0.99$ | $(1,-1)$ | 混合方向最后只留下符号 |

Adam 的第二步分子为：

$$
m_2=(0.19,-0.1),\qquad
\widehat m_2=(1,-10/19),
\qquad
\widehat v_2=(1,100).
$$

所以方向是 $(1,-1/19)$。这比“Adam 等于动量加 RMSProp”更具体：分子抵消符号变化，分母仍然记住幅值。

这些值用来比较规则，不表示所有算法应使用相同学习率。尤其 RMSProp 无偏差修正的早期尺度与 Adam 不同，不能仅凭一步位移排名。

### 16.2 手算 RMS 匹配

对于上一节 $2\times2$ 矩阵 $D=\operatorname{diag}(10,1)$，理想极因子为 $I_2$，其 RMS 是 $1/\sqrt2$。若使用 $s=0.2\sqrt2$，缩放后的两个对角元素为约 $0.282843$，整个矩阵 RMS 恰好为 0.2。

这里的 RMS 是对包括零元素在内的全部四个元素计算，不是只对两个对角元素求平均。矩形形状、有效秩和有限步近似都会影响实际 RMS。

## 17. 两段帮助理解计算顺序的代码

### 17.1 AdamW 的单步计算

下面是单个 dense 参数张量的教学函数。`t` 是从 1 开始的更新次数；`m` 和 `v` 必须跨步保存。真实优化器还需处理参数组、缺失梯度、混合精度、稀疏梯度与分布式状态。

```python
import torch

@torch.no_grad()
def adamw_step(p, g, m, v, t, lr, weight_decay,
              beta1=0.9, beta2=0.999, eps=1e-8):
    m.mul_(beta1).add_(g, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    p.mul_(1 - lr * weight_decay)
    p.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)
```

关键是 `weight_decay` 没有加入 `g`，也就不进入矩估计。这段代码使用可读的非融合写法，会创建临时张量，不能拿它的速度代表高性能 AdamW kernel。

### 17.2 Muon 的矩阵计算

下面明确采用 EMA、无 Nesterov、fp32 和经典三次 NS 迭代，只用于观察方向变换。它不是作者 bf16 五次多项式实现，也不提供完整优化器类。

```python
import torch

def polar_ns_cubic(direction, steps=20, eps=1e-8):
    if direction.ndim != 2:
        raise ValueError("Expected a matrix")
    x = direction.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = 1.5 * x - 0.5 * (gram @ x)
    return x.T if transposed else x

@torch.no_grad()
def muon_teaching_step(w, g, momentum, lr, weight_decay,
                       beta=0.95, ns_steps=20):
    momentum.mul_(beta).add_(g, alpha=1 - beta)
    update = polar_ns_cubic(momentum, steps=ns_steps)
    scale = 0.2 * max(w.shape) ** 0.5
    w.mul_(1 - lr * weight_decay)
    w.add_(update.to(w.dtype), alpha=-lr * scale)
```

读取数据流时，依次追踪 $G\rightarrow M\rightarrow X\rightarrow O\rightarrow\Delta W$，不要把 $W$ 本身做极分解。NS 在变换更新方向，参数矩阵本身不需要变成正交矩阵。

零输入返回零；很小的非零奇异值在有限步下可能仍离 1 很远。可以用 `torch.linalg.svd` 构造理想极因子，比较 Frobenius 误差和奇异值，而不是只看输出 shape。秩亏时，参考值应只保留非零奇异值的支撑，不能任意使用完整 SVD 零空间补全当成唯一正确答案。

这两段代码帮助说明数学顺序；本文的网页构建与公式检查不等于已经完成 GPU 训练验证。

## 18. 状态、计算与显存对照

下表只计优化器额外的长期状态，不含参数、梯度、master weights、激活和临时空间。矩阵行列数为 $m,n$，总参数数为 $P$。

| 方法 | 典型长期状态 | 更新计算 | 主要取舍 |
|---|---|---|---|
| SGD | 无历史张量 | $O(P)$ | 简单、噪声直接进入更新 |
| Momentum / NAG | 一份 buffer | $O(P)$ | 稳定方向，可能过冲 |
| AdaGrad / 基础 RMSProp | 一份平方状态 | $O(P)$ | 坐标尺度适应；累计或遗忘策略不同 |
| AdaDelta | 梯度平方与更新平方 | $O(P)$ | RMS 比值决定步长 |
| Adam / AdamW / AdaMax / Nadam | 两份同形状态 | $O(P)$ | 方向与尺度分离 |
| AMSGrad | 两份矩与一份最大值 | $O(P)$ | 多保存长期二阶信息 |
| RAdam / AdaBelief | 两份同形状态 | $O(P)$ | 修改早期缩放或尺度含义 |
| 无动量 Adafactor | 矩阵约 $m+n$ 个二阶统计 | 仍需遍历矩阵元素 | 节省长期状态，近似尺度 |
| LARS / LAMB | 常见为一份 / 两份，加层标量 | 元素更新与按层归约 | 控制层的相对更新 |
| signSGD / Lion | 无状态 / 一份动量 | $O(P)$ | 丢弃部分幅值信息 |
| 原始矩阵 Shampoo | $m^2+n^2$ | Gram、矩阵逆根及矩阵乘法 | 更丰富的几何结构，开销较大 |
| Muon | 一份动量 | 元素操作与多步矩阵乘法 | 状态较少，需矩阵变换及参数分工 |

若有 1B 个参数，保存一份 fp32 同形状态约需 4 GB，Adam 两份状态约需 8 GB。若参数和梯度都是 bf16，再各加约 2 GB，便得到约 12 GB 模型状态；这还没有激活或可选 fp32 参数副本。Muon 若整份动量使用 fp32，只就该状态估算约 4 GB，但 AdamW 配套参数与 NS 临时矩阵仍然占空间。

实现可能用低精度状态、分片或量化，数字随之变化。算法名称不决定实际 dtype，也不决定优化器状态是否能放进一张 GPU。

## 19. 换优化器时怎样判断收益

### 19.1 先看真正的参数更新

监控每组参数的梯度范数、更新范数及相对更新量：

$$
\frac{\|\Delta W\|_F}{\|W\|_F+\epsilon}.
$$

对 Adam 还可看二阶尺度与有效步长；对 Muon 看更新奇异值、NS 误差与不同形状的 RMS。这些量能区分“学习率数值小”与“实际更新小”。

学习率调度控制基础 $\eta_t$，梯度裁剪改变输入 $g_t$，优化器将梯度变成方向。它们不是同一个开关；对自适应优化器，在梯度上裁剪与在预条件后的更新上裁剪也不等价。

### 19.2 比较预算，而不只比较 step

比较 AdamW 与 Muon 至少固定模型、数据顺序、token budget、有效 batch size 和评测方法，并为各自单独调学习率、衰减与必要的形状缩放。记录验证损失，同时记录每步时间、tokens/s、峰值显存和累计计算量。

相同步数下质量更好，可能值得继续研究；但额外矩阵乘法也可能使墙钟更长。反过来，少一份状态能容纳更大 batch，也可能改变吞吐。这些都需要真实执行，不能从公式中的状态数量直接推出。

### 19.3 理解当前方法缺少什么

| 观察到的问题 | 要检查的机制 | 可用于对照的路线 |
|---|---|---|
| 梯度方向来回变化 | 动量历史与噪声 | Momentum、Adam |
| 坐标尺度差别明显 | 二阶尺度与有效步长 | AdaGrad、RMSProp、Adam |
| 衰减行为难以解释 | 正则项是否进入矩估计 | AdamW 与耦合 L2 对照 |
| 优化器状态占满显存 | 状态 dtype、分片及张量数量 | Adafactor、Lion、Muon |
| 大 batch 的层间更新不协调 | 相对更新量与 trust ratio | LARS、LAMB |
| 更新谱被少数方向支配 | 矩阵结构、奇异值与变换成本 | Shampoo、Muon |

这些是定位问题的检查路径，不是只凭一个症状就能选出最佳优化器。SGD、AdamW、Muon 的优势来自不同信息与不同约束；实验中应把算法、调参预算和系统成本一起报告。

## 20. 原始资料与阅读顺序

优先读 Adam 的 Algorithm 1，再读 AdamW 对 L2 的比较；理解这两篇后，Shampoo 与 Muon 的矩阵路线更容易与逐元素适应区别开。早期历史和特殊分支可以按需查阅。

| 资料 | 作者与发表身份 | 适合查阅的内容 |
|---|---|---|
| [A Stochastic Approximation Method](https://www.columbia.edu/~ww2040/8100F16/RM51.pdf) | Robbins、Monro，1951，Annals of Mathematical Statistics | 随机逼近的历史与假设 |
| [Polyak heavy-ball 原刊记录](https://www.mathnet.ru/php/archive.phtml?jrnid=zvmmf&option_lang=eng&paperid=7713&wshow=paper) | Polyak，1964，期刊论文 | 惯性加速的起点 |
| [Nesterov 1983 原文](https://hengshuaiyao.github.io/papers/nesterov83.pdf) | Nesterov，1983，Doklady 论文扫描 | 凸优化加速的历史 |
| [On the Importance of Initialization and Momentum in Deep Learning](https://proceedings.mlr.press/v28/sutskever13.html) | Sutskever 等，ICML 2013 | 现代 Momentum / NAG 表达 |
| [Adaptive Subgradient Methods](https://jmlr.csail.mit.edu/papers/v12/duchi11a.html) | Duchi、Hazan、Singer，JMLR 2011；前身 COLT 2010 | AdaGrad 与对角/完整矩阵预条件 |
| [Hinton Lecture 6](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf) | Tieleman/Hinton，2012，课程讲义 | RMSProp；不是正式首发论文 |
| [ADADELTA](https://arxiv.org/abs/1212.5701) | Zeiler，2012，arXiv 预印本 | 梯度 RMS 与更新 RMS |
| [Adam](https://arxiv.org/abs/1412.6980) | Kingma、Ba，ICLR 2015；2014 预印本 | 偏差修正与 AdaMax |
| [NAdam 文档及原文链接](https://docs.pytorch.org/docs/2.14/generated/torch.optim.NAdam.html) | Dozat，ICLR 2016 workshop；PyTorch 官方实现资料 | 动量调度与即时梯度项 |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | Loshchilov、Hutter，ICLR 2019；2017 预印本 | AdamW 与耦合 L2 |
| [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237) | Reddi、Kale、Kumar，ICLR 2018；此 arXiv 版本发布于 2019 | 反例、长期记忆与 AMSGrad |
| [RAdam](https://arxiv.org/abs/1908.03265) | Liu 等，ICLR 2020；2019 预印本 | 早期自适应学习率方差 |
| [AdaBelief](https://proceedings.neurips.cc/paper/2020/hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract.html) | Zhuang 等，NeurIPS 2020 | 残差平方代替梯度平方 |
| [Adafactor](https://proceedings.mlr.press/v80/shazeer18a.html) | Shazeer、Stern，ICML 2018 | 行列统计、update clipping 与相对步长 |
| [LARS](https://arxiv.org/abs/1708.03888) | You、Gitman、Ginsburg，2017，arXiv 预印本 | 大批量与层间比例 |
| [LAMB](https://arxiv.org/abs/1904.00962) | You 等，ICLR 2020；2019 预印本 | Adam 与 layer-wise trust ratio |
| [signSGD](https://proceedings.mlr.press/v80/bernstein18a.html) | Bernstein 等，ICML 2018 | 符号更新与压缩通信 |
| [Symbolic Discovery of Optimization Algorithms](https://proceedings.neurips.cc/paper_files/paper/2023/file/9a39b4925e35cf447ccba8757137d84f-Paper-Conference.pdf) | Chen 等，NeurIPS 2023 | Lion 的发现、规则与实验限制 |
| [Shampoo](https://proceedings.mlr.press/v80/gupta18a.html) | Gupta、Koren、Singer，ICML 2018 | 张量结构化预条件 |
| [Old Optimizer, New Norm](https://arxiv.org/html/2409.20325v1) | Bernstein、Newhouse，2024，arXiv 预印本 | 范数、预条件与极分解联系 |
| [Muon 原博客](https://kellerjordan.github.io/posts/muon/) 与 [代码](https://github.com/KellerJordan/Muon) | Jordan 等，2024 作者博客及持续更新的实现 | 原始机制、NS 近似与参数分工 |
| [Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1) | Liu 等，2025，arXiv 技术报告 | RMS 缩放、衰减、scaling-law 与微调实验 |

官方软件文档描述具体实现，原论文描述设计与实验，两者可能采用不同约定。复现一个方法时，应记录版本、参数分组、动量定义、稳定常数位置、学习率调度与状态精度，再对齐公式。

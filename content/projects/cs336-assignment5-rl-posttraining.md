---
title: "CS336 Assignment 5：从策略梯度到 LLM 后训练"
date: "2026-10-16"
description: "以可验证数学奖励为主线，推导 baseline、GRPO 家族、重要性采样、PPO 裁剪与 GSPO，逐项对应 SFT、DPO 和 RL 的实际训练 loss。"
tags: ["CS336", "LLM", "强化学习", "后训练"]
tech: ["Python", "PyTorch", "Transformers", "GRPO", "DPO"]
github: "https://github.com/tanyouqing/CS336-Assignment5"
image: "🎯"
---

## 项目介绍

在 [Assignment 1](/projects/cs336-assignment1-from-scratch-llm/) 中，我实现了从文本到语言模型预训练的链路。Assignment 5 进一步研究：已有 next-token 模型后，怎样通过任务奖励与偏好数据，改变它生成完整回答的方式。

项目代码：[CS336-Assignment5](https://github.com/tanyouqing/CS336-Assignment5)。主线使用 **OLMo-2-0425-1B + GSM8K**，实现 prompting、on-policy GRPO、Dr.GRPO、RFT、课程版 MaxRL，以及旧 rollout 的复用、token reweighting、裁剪与 GSPO。补充路线使用 **Llama 3.1 8B** 完成 SFT 与 DPO 的数据和训练流程。这是两条模型路线，不是把 OLMo 的数学 RL 结果继续交给 Llama 做 DPO。

笔记沿着“基础 RL 对象 → 策略梯度 → 优势与归一化 → 旧数据修正 → 实际 loss”的顺序展开，再回到代码中的张量与训练循环。阅读时始终区分：**严格等价、期望梯度只差常数、目标或估计器发生改变**。

```text
题目分布 → 模型采样回答 → 验证器打分 → baseline / advantage
                                           ↓
                    当前 token log-prob → 代理 loss → 参数更新
                                           ↑
              复用旧回答时：old log-prob → ratio → clipping

离线示范 → SFT 交叉熵
偏好对 + 冻结 reference → DPO 偏好似然
```

本文是完成作业后的原理与实现笔记。所有手算数字均为教学例子；仓库中的单卡配置与训练入口不等于已核验的完整课程实验结果，本文不报告未经验证的准确率、加速比或 GPU 曲线。

## 从基础 RL 到语言生成

### 一个token就是一次动作

给定题目prompt为x，模型回答为y=(y₁,…,y_T)。
选用token粒度时，对应关系如下。

| 基础RL | 单轮LLM任务 | 必须注意的边界 |
|---|---|---|
| 初始状态分布 | 题目分布ρ(x) | 换prompt或题目集也会改变训练问题 |
| 状态s_t | (x,y_<t)，题目加已生成前缀 | 此处采用完整前缀；截断上下文未必仍充分 |
| 动作a_t | 下一个token y_t | 动作空间是词表，也包含终止token |
| 策略πθ(a_t∣s_t) | 模型next-token概率分布 | 模型参数θ就是actor参数 |
| 状态转移 | 把y_t拼接到前缀 | 纯文本生成的转移确定，随机性来自采样 |
| 轨迹τ | 一段完整生成过程 | 包含终止方式，不能忽略EOS/截断约定 |
| 即时奖励r_t | 终局打分或过程打分 | 本作业主要在回答结束时验证最终答案 |
| 回合结束 | EOS、停止规则或任务预算边界 | 终止与采样截断的语义可能不同 |

本文用“选择y_t后收到r_t”的时间下标约定。
若旧笔记使用r_(t+1)，只需整体平移奖励下标。

$$
s_t=(x,y_{<t}),\qquad a_t=y_t,\qquad
s_{t+1}=(x,y_{\le t}).
$$

一般轨迹包含状态转移概率。
在这里，前缀拼接的转移项不依赖θ，参数化概率只来自模型：

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}),
\qquad
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid s_t).
$$

这是从基础RL到LLM的第一条严格等价关系。
整段回答也可看作一次大动作，形成contextual bandit视角。
两个粒度描述同一序列分布，但概率比与优势粒度不能随意混用。

### 即时奖励与回报，在终局验证里有什么区别？

基础RL的reward-to-go是：

$$
\mathcal G_t=\sum_{u=t}^{T}\gamma^{u-t}r_u.
$$

数学验证任务通常只有终局奖励R(x,y)：

$$
r_t=0\ (t<T),\qquad r_T=R(x,y),\qquad \gamma=1.
$$

于是每个位置的回报都等于同一个最终分数：

$$
\boxed{\mathcal G_t=R(x,y)\quad\text{对所有回答位置 }t.}
$$

这解释了为什么同一个最终奖励能用于所有回答token。
前面的即时奖励为零，不等于前面的动作没有训练信号。
它也不表示每个推理步骤都被验证为正确。

若令γ小于1，终局反馈传到早期token时会衰减。
这改变时间偏好；不能不加说明地沿用到数学成功率目标。
长度惩罚若有必要，应明确写入目标或约束。

### Q与V仍然存在，只是不一定学习一个网络

$$
Q^\pi(s_t,a_t)=\mathbb E_\pi[\mathcal G_t\mid s_t,a_t],\qquad
V^\pi(s_t)=\mathbb E_\pi[\mathcal G_t\mid s_t],\qquad
A^\pi=Q^\pi-V^\pi.
$$

在二元终局奖励下，Q表示“选这个token后，最终答对的概率”。
V表示“从这个前缀继续生成，最终答对的概率”。
GRPO不学习critic，不等于这个任务没有价值函数。
它改用整段回答间的奖励统计，近似提供相对学习权重。

## 从期望奖励推导可反向传播的代理目标

### 把一般目标改写成题目条件下的成功率

基础RL希望最大化期望回报。
本作业采用题目分布固定、奖励函数不直接依赖θ的设定：

$$
J(\theta)=\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)].
$$

若R是答对为1、答错为0，J就是平均单次生成成功率。
它不是某份固定示范文本的似然，也不是“推理过程一定正确”的概率。

### 不对离散答案求导，改为对其概率求导

固定x后，逐步展开：

$$
\begin{aligned}
\nabla_\theta J_x
&=\nabla_\theta\sum_y\pi_\theta(y\mid x)R(x,y)\\
&=\sum_y R(x,y)\nabla_\theta\pi_\theta(y\mid x)\\
&=\sum_y\pi_\theta(y\mid x)R(x,y)\nabla_\theta\log\pi_\theta(y\mid x)\\
&=\mathbb E_{y\sim\pi_\theta}\left[R(x,y)\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(y_t\mid s_t)\right].
\end{aligned}
$$

第三行使用∇p=p∇log p，最后一行使用自回归分解。
因此最终公式是：

$$
\boxed{\nabla_\theta J
=\mathbb E_{x,y}\left[\sum_{t=1}^{T}R(x,y)\nabla_\theta\log\pi_\theta(y_t\mid s_t)\right].}
$$

这就是序列REINFORCE在LLM上的具体形式。
验证器可以是不可微的算术判断或测试程序。
梯度来自生成该回答的概率，而不是验证器内部。

### 它与熟悉的Actor-Critic公式怎样连接？

基础更新中有Q(s_t,a_t)或优势A(s_t,a_t)。
终局回报R是Q的一个Monte Carlo样本估计。
再减去不依赖当前动作的baseline，可得：

$$
\nabla_\theta J
=\mathbb E\left[\sum_t(R-b(s_t))\nabla_\theta\log\pi_\theta(y_t\mid s_t)\right].
$$

严格保持期望依赖baseline的条件，下节专门区分。
若用真正的Q−V，便回到优势Actor-Critic。
若每个token都用同一个整段权重，则信用分配更粗。

### 期望目标、采样梯度和训练代理目标是三件事

取B道题、每题G个回答，记N=BG。
设A_ij是本轮算好的固定学习权重：

$$
\hat g=\frac1N\sum_{i=1}^{B}\sum_{j=1}^{G}
A_{ij}\sum_{t=1}^{T_{ij}}\nabla_\theta\log\pi_\theta(y_{ijt}\mid s_{ijt}).
$$

一个产生该梯度的局部代理目标为：

$$
S(\theta;\mathcal D)=\frac1N\sum_{i,j}
\operatorname{sg}(A_{ij})\sum_t\log\pi_\theta(y_{ijt}\mid s_{ijt}),
\qquad \mathcal L=-S.
$$

sg表示停止对权重求导，样本也作为本轮固定数据。
正优势提高该回答token的概率，负优势降低它们。
这说明RL loss为什么看起来像“带正负权重的语言模型loss”。
它是产生更新的代理，不等于原来的期望奖励J。
若样本很陈旧，机械沿用这个代理不再给出当前策略的正确梯度。

SFT在固定示范分布上提高示范似然。
RL通常对自身采样分布更新，权重还来自任务反馈。
但在二元奖励的特殊条件下，RFT与REINFORCE存在精确联系，见后文的 RFT 推导。

### 为什么不能直接对采样奖励调用 backward

数学上可以对原始期望求导。如果只有几个可能回答，甚至可以枚举全部回答并自动微分；语言模型的回答空间随长度指数增长，无法这样计算。

生成一条离散回答后，验证器给出的奖励通常不连接模型的计算图。`loss = -rewards.mean()` 可以估计目标值，却不能用普通自动微分得到参数的正确梯度。REINFORCE 使用概率恒等式绕过离散采样的求导，再把已采样 token 当作固定标签，在当前模型上重算可微 log-prob。

讲义先写带梯度的更新方向，再写不带梯度符号的代理目标，原因就在这里：

$$
\hat g=\frac1N\sum_{i,j}\frac{A_{ij}}{T_{ij}}
\sum_t\nabla\log\pi_\theta(y_{ijt}\mid s_{ijt}),
$$

$$
\mathcal L=-\frac1N\sum_{i,j}\frac{\operatorname{sg}(A_{ij})}{T_{ij}}
\sum_t\log\pi_\theta(y_{ijt}\mid s_{ijt})
\quad\Longrightarrow\quad-\nabla\mathcal L=\hat g.
$$

对固定 batch，最后这个等式精确成立；取采样期望后是否有 $\mathbb E[\hat g]=\nabla J$，还需要检查 baseline、归一化和采样分布。$\nabla\log\pi=(\nabla\pi)/\pi$ 中的概率分母始终包含在自动求导里。

两动作例子中，正确回答概率为 $p$、奖励为 1，错误回答奖励为 0。采到正确回答时，loss 为 $-\log p$，负梯度为 $1-p$；采到错误回答时贡献为零。平均后才得到 $p(1-p)=\nabla J$。单次 loss 的数值和单次估计都不是原始期望目标本身。

## 优势、baseline 与方差

### PPO路线：前缀价值估计加GAE

PPO常用critic预测每个前缀的V_φ(s_t)。
将基础TD误差直接换成token状态：

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),
$$

$$
\hat A_t^{\mathrm{GAE}}
=\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}.
$$

真正终止的V(s_(T+1))为0。
λ=0使用一步TD；λ=1在完整终止轨迹下化为Monte Carlo回报减V。
critic不准确时，GAE的偏差与方差取决于价值误差与λ。
若只是截断未结束的任务，是否bootstrap必须按任务定义判断。
参见[GAE原论文](https://arxiv.org/abs/1506.02438)。

关键区别：PPO可给不同前缀不同优势。
outcome GRPO通常给同一回答所有token相同优势。
两者都借助最终结果，但信用分配粒度不同。

### GRPO路线：同题多个回答互相比

对题目i采样G个回答，奖励为R_i1到R_iG：

$$
\mu_i=\frac1G\sum_jR_{ij},\qquad
\hat A_{ij}=\frac{R_{ij}-\mu_i}{s_i+\varepsilon_{\rm num}}.
$$

这里的s_i是组奖励标准差，不是状态s_t。
本笔记用完整下标区分两者。
连续分布推导使用总体标准差；仓库数值实现采用组内样本标准差（`torch.std` 默认 correction=1），需要保持一致。

$$
s_{i,\mathrm{sample}}
=\sqrt{\frac1{G-1}\sum_j(R_{ij}-\mu_i)^2}.
$$

这个优势是整段回答的相对分数，不是精确的逐token Q−V。
无需训练critic，是节省一个价值估计环节的设计。
不是把Actor-Critic中的V换个名字就完全等价。
方法来源见[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)。

### 为什么减固定baseline保持期望，减组均值却只保持比例？

固定x，记h_j=∇logπ(y_j∣x)，原期望g=E[R_jh_j]。
score恒等式给出E[h_j]=0。
若b不依赖当前回答，则E[bh_j]=0。

但组均值包含自身：

$$
\begin{aligned}
\mathbb E[(R_j-\mu)h_j]
&=\mathbb E[R_jh_j]-\frac1G\mathbb E[R_jh_j]
-\frac1G\sum_{k\ne j}\mathbb E[R_k]\mathbb E[h_j]\\
&=\frac{G-1}{G}g.
\end{aligned}
$$

固定G时只差一个正的统一常数，方向保留，幅度缩小。
其条件是同题独立采样，且这里只讨论减组均值。
如果再除标准差或长度，不能沿用这一结论。

排除自身的leave-one-out baseline为：

$$
b_{-j}=\frac1{G-1}\sum_{k\ne j}R_k,\qquad
R_j-b_{-j}=\frac G{G-1}(R_j-\mu).
$$

在上述条件下，LOO恢复未缩放期望。
它不是创造了新的回答排序信号；仍要区分随机梯度方差。

### 无偏与低方差是两个独立判断

对标量参数的二元例子，动作取 0 或 1，奖励等于动作，成功概率为 p=σ(θ)：

$$
\operatorname{Var}(\hat g_b)
=\frac{p(1-p)(1-p-b)^2}{n}.
$$

最优b是1−p，而非一般意义上的平均奖励p。
取b=p在p小于2/3时降方差，大于2/3时反而增方差。
更一般的向量score最小化总方差时：

$$
b^*(x)=\frac{\mathbb E[R\|h\|^2\mid x]}{\mathbb E[\|h\|^2\mid x]}.
$$

平均奖励忽略了score大小权重，因此不是普适最优baseline。
下面补上这一方差公式的推导。

### 二元策略的方差：从两个取值直接算

令动作 $a\in\{0,1\}$，$p=\sigma(\theta)$，奖励 $R(a)=a$。固定当前参数后，随机的是采样动作，$p$ 是一个确定的数：

$$
J=p,\qquad \nabla_\theta J=p(1-p),\qquad
\nabla_\theta\log\pi_\theta(a)=a-p.
$$

最后一式分别来自 $\nabla\log p=1-p$ 和 $\nabla\log(1-p)=-p$。用固定 baseline $b$，单样本估计为 $Z_b=(a-b)(a-p)$：

| 动作 | 概率 | 梯度估计值 |
|---|---|---|
| $a=1$ | $p$ | $(1-b)(1-p)$ |
| $a=0$ | $1-p$ | $bp$ |

$$
\mathbb E[Z_b]=p(1-b)(1-p)+(1-p)bp=p(1-p),
$$

$$
\mathbb E[Z_b^2]=p(1-b)^2(1-p)^2+(1-p)b^2p^2.
$$

用二阶矩减去均值平方，得到：

$$
\begin{aligned}
\operatorname{Var}(Z_b)
&=p(1-p)\left[(1-p)(1-b)^2+pb^2-p(1-p)\right]\\
&=p(1-p)(1-p-b)^2.
\end{aligned}
$$

对 $n$ 个独立同分布样本取平均，方差是单样本的 $1/n$，因为求和产生 $n$ 个方差、平均系数平方产生 $1/n^2$。组内共享统计的项并不独立，不能照搬这一步。

$$
\operatorname{Var}(\hat g_0)=\frac{p(1-p)^3}{n},\qquad
\operatorname{Var}(\hat g_{b=p})=\frac{p(1-p)(1-2p)^2}{n}.
$$

两者作差：

$$
\operatorname{Var}(\hat g_{b=p})-\operatorname{Var}(\hat g_0)
=\frac{p^2(1-p)(3p-2)}n.
$$

所以均值 baseline 在 $0<p<2/3$ 时降方差，$p=2/3$ 时不变，$2/3<p<1$ 时增方差；在 $p=0$ 或 $p=1$ 的端点极限，两个方差都为零。任意 baseline 能降方差的条件是 $0<b<2(1-p)$。这些结论都要求 $b$ 对当前动作独立。

两个单样本算例说明差别：

| $p$ | 不加 baseline 的方差 | 使用 $b=p$ 的方差 | 解释 |
|---|---|---|---|
| $0.5$ | $0.0625$ | $0$ | 两种动作都产生 $0.25$ 的梯度估计 |
| $0.9$ | $0.0009$ | $0.0576$ | 估计值从 $0.01$ 偶尔跳到 $0.81$，均值仍为 $0.09$ |

在这个特殊二元模型中，最优 $b=1-p$ 使两个动作都产生 $p(1-p)$，方差为零。这不是一般任务都能得到零方差 baseline 的保证。

### 组均值为什么不能当作固定 baseline

在上面的组均值证明中，只有其他回答的奖励能从当前回答的条件期望中提出；自身奖励 $R_j$ 与 score $h_j$ 相关，留下的正是 $g/G$。将均值 `detach()` 只阻止自动求导，不能消除统计依赖。

取 $p=0.5$、$G=2$，四种动作组合等概率；下表是两个更新项的平均：

| 两个动作 | 不减 baseline | 减本组均值 |
|---|---|---|
| $(0,0)$ | $0$ | $0$ |
| $(0,1)$ | $0.25$ | $0.25$ |
| $(1,0)$ | $0.25$ | $0.25$ |
| $(1,1)$ | $0.5$ | $0$ |

原期望是 $0.25$，减组均值后是 $0.125$，符合 $(G-1)/G$。极端地，$G=1$ 时中心化优势全为零，而真实梯度通常不为零。LOO 要求 $G>1$；它与 Dr.GRPO 的区别在这里是统一缩放，不能据此断言方差必然更小。

### 全对组和全错组为什么都没有相对信号？

两者都满足R_ij=μ_i，因此中心化优势为0。
全错可能表示缺乏探索成功；全对可能表示任务太容易。
ε只避免除零，不创造成功样本。
有单独KL或其他辅助目标时，总更新未必为0。

二元独立奖励、成功率η下，一组出现混合奖励的概率为：

$$
P(\text{混合组})=1-\eta^G-(1-\eta)^G.
$$

G影响发现可区分回答的概率，同时增加生成成本。
这比只说“组越大越稳定”更完整。

## 归一化改变了什么：GRPO、Dr.GRPO、RFT 与 MaxRL

### 先把三种修改拆开

一类常见估计器可以统一写成：

$$
\hat g=\frac1Z\sum_{i,j}c_{ij}\sum_{t=1}^{T_{ij}}
\frac{R_{ij}-b_{ij}}{d_i}\nabla_\theta\log\pi_\theta(y_{ijt}\mid s_{ijt}).
$$

b控制baseline，d控制优势缩放，c与Z控制长度和聚合。
先判断这四项，再谈算法名字。
以下表格是本作业的outcome设定，尚未加入off-policy裁剪。

| 方法 | b | d | c | Z |
|---|---|---|---|---|
| 原始REINFORCE | 0 | 1 | 1 | N |
| 标准on-policy GRPO | 组均值μ_i | 组标准差+ε | 1/T_ij | N |
| GRPO_constant | μ_i | 组标准差+ε | 1 | NL_max |
| Dr.GRPO | μ_i | 1 | 1 | NL_max |
| RFT，二元奖励 | 0 | 1 | 1 | NL_max |
| 作业版MaxRL | μ_i | μ_i+ε | 1 | NL_max |

### 标准GRPO怎样从REINFORCE逐步变来？

$$
\boxed{\hat g_{\rm GRPO}
=\frac1N\sum_{i,j}\frac1{T_{ij}}\sum_t
\frac{R_{ij}-\mu_i}{s_i+\varepsilon_{\rm num}}
\nabla_\theta\log\pi_\theta(y_{ijt}\mid s_{ijt}).}
$$

第一步减组均值：有限G时，单独产生(G−1)/G缩放。
第二步除组标准差：不同题目/采样组被不同倍数加权。
第三步除回答长度：长回答中每个token项的系数更小。
后两步通常不再是对原梯度乘一个统一常数。
标准差归一化也不保证各组真实梯度范数完全相同。

### 为什么1/T不是无关紧要的“取平均”？

设同题两个回答长度为2与8，优势相同。
sequence聚合下，每个token的系数相差4倍。
这让短回答的单token更新项权重更大。
实际梯度范数仍取决于token的score方向和大小。

固定Z时，各有效token使用相同聚合系数。
长回答含更多项，因此可能贡献更多总量。
这不等于直接奖励“写得长”，也不保证合成梯度范数更大。

T由采样回答决定，不是全局固定常数。
因此除以T不能简单吸收到统一学习率中。
它也使baseline项乘上动作相关的1/T，原来的抵消条件未必成立。
若转而优化R/T，必须承认那是另一个长度敏感目标。

### Dr.GRPO移除了哪些修改？

$$
\hat g_{\rm Dr}
=\frac1{NL_{\max}}\sum_{i,j}(R_{ij}-\mu_i)
\sum_t\nabla_\theta\log\pi_\theta(y_{ijt}\mid s_{ijt}).
$$

移除奖励标准差与随机回答长度的除法，保留固定分母。
对于本作业的纯on-policy设定，其期望是：

$$
\mathbb E[\hat g_{\rm Dr}]
=\frac{G-1}{G L_{\max}}\nabla_\theta J.
$$

所以应精确地说“与原期望梯度成固定比例”。
不能忽略有限G和L_max后宣称严格数值相等。
讨论来源见[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)。

### RFT只是监督学习吗？本作业里为何又能视作策略梯度？

RFT保留正确的自身回答，提高其序列似然：

$$
\hat g_{\rm RFT}
=\frac1{NL_{\max}}\sum_{i,j}\mathbf1[R_{ij}=1]
\sum_t\nabla_\theta\log\pi_\theta(y_{ijt}\mid s_{ijt}).
$$

二元奖励下，指示函数就是R本身。
若回答来自当前策略，且保留原固定分母：

$$
\mathbb E[\hat g_{\rm RFT}]=\frac1{L_{\max}}\nabla_\theta J.
$$

因此它在此设定下就是无baseline的REINFORCE估计器。
Dr.GRPO与它的期望还差有限G的(G−1)/G。
不能先验断言哪一种方差总是更低。

若只按“留下的正确回答数”重新归一化，分母变为随机数。
若长期训练固定旧回答，也不再是每次fresh on-policy梯度。
这两种常见RFT实践不能直接套用上式的无偏结论。

### MaxRL为什么更重视困难题？

作业版使用：

$$
\hat g_{\rm MaxRL}
=\frac1{NL_{\max}}\sum_{i,j}\frac{R_{ij}-\mu_i}{\mu_i+\varepsilon_{\rm num}}
\sum_t\nabla_\theta\log\pi_\theta(y_{ijt}\mid s_{ijt}).
$$

平均奖励低的组使用更大的除法放大系数。
但全错组分子仍为0，不会自动得到有用方向。
原MaxRL的聚合分母是批次总token数，课程改用固定Z。
本文以课程定义为准，不把该受控变体当成原论文逐项复现。
来源见[Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710)。

### 用大组极限看清“题目重加权”

定义当前模型对题目x的成功率η_θ(x)=E[R∣x]。
假设二元奖励、G趋于无穷，忽略ε与公共常数。
暂时移除长度归一化，避免把两种重加权混在一起。

组均值趋于η，标准差趋于√(η(1−η))。
对每题，减总体均值项的期望为0，所以：

$$
\mathbb E_y\left[\frac{R-\eta}{d(\eta)}\nabla\log\pi_\theta(y\mid x)\right]
=\frac1{d(\eta)}\nabla_\theta\eta_\theta(x).
$$

| 常数聚合下的优势缩放 | 对题目梯度的权重w(x) | 满足F′(η)=w的势函数 |
|---|---|---|
| Dr.GRPO，无除法 | 1 | η |
| GRPO_constant，除标准差 | 1/√(η(1−η)) | 2 arcsin√η |
| MaxRL，除均值 | 1/η | logη |

可写成局部停止梯度权重目标：

$$
J_w=\mathbb E_x[w(x,\operatorname{sg}(\pi_\theta))\eta_\theta(x)],
\qquad
\nabla J_w=\mathbb E_x[w(x)\nabla\eta_\theta(x)].
$$

在上述理想条件下，它也对应E[F(η)]的梯度。
不能一边把w当普通可微函数，又漏掉η∇w这一项。
MaxRL的log成功率联系尤其直观：更关注低成功率题目。
标准差权重在η接近0或1时都大，不是只偏爱困难题。
权重大不保证绝对梯度大，因为∇η本身也可能很小。

有限G、ε、随机分母和长度变化都会破坏上述简单等式。
尤其不能把“比值的期望”换成“期望的比值”。

## 复用旧回答：off-policy 与重要性采样

### 数据分布和当前策略开始分离

记生成本批回答的行为策略为π₀，正在更新的策略为πθ。
第一步更新前二者可相同；更新后回答仍来自旧π₀。
直接沿用旧回答得到：

$$
\mathbb E_{y\sim\pi_0}[R\nabla\log\pi_\theta(y\mid x)],
$$

它一般不等于当前策略下的期望梯度。
采样分布写在哪里，和梯度写在哪里，同样重要。
课程将一批回答中的后续更新称为off-policy。
更广泛文献中，PPO仍常归入近似on-policy方法。
关键事实是“当前策略是否偏离本批行为策略”，而非术语标签。

### 基础重要性采样如何精确作用到LLM？

对于π₀覆盖πθ支持集的设定：

$$
W(y,x)=\frac{\pi_\theta(y\mid x)}{\pi_0(y\mid x)}
=\prod_{t=1}^{T}\underbrace{\frac{\pi_\theta(y_t\mid s_t)}{\pi_0(y_t\mid s_t)}}_{w_t}.
$$

于是：

$$
\boxed{\nabla J_x
=\mathbb E_{y\sim\pi_0}\left[W(y,x)R(x,y)
\sum_t\nabla\log\pi_\theta(y_t\mid s_t)\right].}
$$

证明就是在求和中抵消π₀：

$$
\sum_y\pi_0(y\mid x)\frac{\pi_\theta(y\mid x)}{\pi_0(y\mid x)}f(y)
=\sum_y\pi_\theta(y\mid x)f(y).
$$

它精确修正整段回答分布，而非只修正一个动作。
但大量token概率比连乘可能使方差很大。
例如每项都为1.01，512项乘积约163，而不是1.01。
这是教学极端例子，并非所有轨迹必然如此。

精确性还要求正确记录实际行为分布。
改变temperature、top-p或终止规则后，分母也要对应。
若top-p砍掉了目标策略仍有概率的动作，支持条件可能失败。

### token级概率比为什么只是代理？

PPO/GRPO常使用：

$$
g_{\rm token}
=\mathbb E_{y\sim\pi_0}\left[R\sum_t w_t\nabla\log\pi_\theta(y_t\mid s_t)\right].
$$

这里每一项只乘自己的w_t，没有乘整段W。
因此它不是上一式的代数等价改写。
固定长度L、只用原始奖励R时，可解释为混合策略的代理梯度：

$$
\tilde\pi_t(y\mid x)
=\pi_\theta(y_t\mid s_t)\prod_{u\ne t}\pi_0(y_u\mid s_u),
$$

$$
J_{\rm token}=\mathbb E_x\sum_{t=1}^{L}
\mathbb E_{y\sim\tilde\pi_t}[R(x,y)].
$$

该混合策略只有第t步用当前模型，其余位置都用旧模型。
旧策略的后续动作仍会条件于实际生成的前缀。
所以评估当前token的后果时，假定后续仍由旧策略续写。
这解释了为何局部比率没有完全修正前缀与后缀分布。

在θ等于旧参数时，各w_t=1，该原始奖励梯度与当前策略梯度相同。
离开旧策略后，代理与原目标一般不再一致。
若再加入组优势、长度与裁剪，还会叠加其他修改。

两token一组的pairwise比率，可看作只把该两步换成当前策略。
它处于一个位置和整段位置替换之间，但不能保证方差单调变化。

### 不要把noclip目标写成“ratio再乘log概率”

局部未裁剪代理可写为：

$$
S_{\rm noclip}=\frac1N\sum_{i,j}\frac1{T_{ij}}
\sum_t\operatorname{sg}(\hat A_{ij})w_{ijt}(\theta).
$$

因为：

$$
\nabla w_t=w_t\nabla\log\pi_\theta(y_t\mid s_t).
$$

ratio本身的导数已经产生需要的score项。
再额外乘log概率，会多出另一项，变成不同目标。
对应的可微 loss 写法见后文的 Loss 图鉴。

## PPO 风格裁剪与 GSPO

### 把熟悉的PPO公式换成token形式

定义C(a,w)=min(aw,a·clip(w,1−ε_clip,1+ε_clip))。
本作业off-policy GRPO代理为：

$$
\boxed{S_{\rm clip}
=\frac1N\sum_{i,j}\frac1{T_{ij}}\sum_t
C(\hat A_{ij},w_{ijt}(\theta)).}
$$

它对应基础PPO中的状态、动作与优势替换：
状态变为前缀，动作变为token，优势变为组内回答权重。
若使用critic+GAE，则保留逐token的A_t。
PPO来源见[原论文](https://arxiv.org/abs/1707.06347)。

### min为什么必须和优势符号一起理解？

$$
C(a,w)=
\begin{cases}
a\min(w,1+\epsilon_{\rm clip}),&a\ge0,\\
a\max(w,1-\epsilon_{\rm clip}),&a<0.
\end{cases}
$$

取ε_clip=0.2：

| 优势a | 比率w | aw | 裁剪后的候选项 | 取min | 此项是否仍有梯度 |
|---|---|---|---|---|---|
| +1 | 1.4 | 1.4 | 1.2 | 1.2 | 否，过度增加好回答 |
| +1 | 0.6 | 0.6 | 0.8 | 0.6 | 是，仍纠正降低好回答 |
| −1 | 0.6 | −0.6 | −0.8 | −0.8 | 否，过度压低坏回答 |
| −1 | 1.4 | −1.4 | −1.2 | −1.4 | 是，仍纠正增加坏回答 |

所以不是“超出区间全部没有梯度”。
裁剪是移除某个样本项继续过度改善的激励。
共享参数仍会受其他样本影响，概率比不被硬锁在区间内。
它也不是严格KL约束，更不等于参数梯度范数裁剪。

### GSPO：把整段比率取几何平均

定义每条回答的长度归一化序列比：

$$
u_{ij}(\theta)
=\left[\frac{\pi_\theta(y_{ij}\mid x_i)}{\pi_0(y_{ij}\mid x_i)}\right]^{1/T_{ij}}
=\exp\left[\frac1{T_{ij}}\sum_t\log w_{ijt}(\theta)\right].
$$

序列级代理为：

$$
\boxed{S_{\rm GSPO}=\frac1N\sum_{i,j}C(\hat A_{ij},u_{ij}(\theta)).}
$$

每条回答统一决定裁剪，不是每个token分别决定。
u是几何平均比率，不是token比率的算术平均，也不是原始乘积W。
因为取了1/T次幂，它不再是精确的整段重要性采样权重。
与裁剪一起，这是一种稳定性与偏差之间的取舍。
来源见[GSPO原论文](https://arxiv.org/abs/2507.18071)。

### GSPO中的长度归一化从哪里来？

暂时去掉裁剪，只求u的导数：

$$
\begin{aligned}
\nabla u_{ij}
&=u_{ij}\nabla\left[\frac1{T_{ij}}\sum_t
(\log\pi_\theta(y_{ijt}\mid s_{ijt})-\log\pi_0(y_{ijt}\mid s_{ijt}))\right]\\
&=u_{ij}\frac1{T_{ij}}\sum_t\nabla\log\pi_\theta(y_{ijt}\mid s_{ijt}).
\end{aligned}
$$

所以梯度自然带有1/T，而非另加一个无关平均。
在θ等于旧参数且裁剪未激活时，它与相同优势的on-policy序列平均梯度一致。
离开该点后，不再等于token级比率代理。
若想改成constant-normalized变体，必须重新推导长度系数与指数。
不能仅把目标外面再乘或除T，却仍宣称是同一个GSPO。

### 一个很小的比率例子

若同一回答两个token的比率是2与0.5：
整段比率W=1，几何平均u=1，token级比率却分别远离1。
因此序列级可能不裁剪，而token级会按优势符号裁掉其中一项。
这说明二者聚合信息的方式不同，不代表谁在所有任务都更好。

讲义还提到CISPO：把截断比率作为停止梯度的权重，乘score梯度。
它与PPO“截断后某方向梯度为零”的机制不同。
不是只改同一个clip阈值，就变成了CISPO。

### 旧数据处理方法的比较

| 方法 | 比率 | 裁剪粒度 | 对原当前策略目标的关系 |
|---|---|---|---|
| naive复用 | 不使用 | 无 | 一般有分布偏差 |
| 精确序列IS | ∏_t w_t | 无 | 在支持与可积条件下修正原R梯度；可能高方差 |
| token noclip | 各项用w_t | 无 | 优化局部混合策略代理，非精确序列修正 |
| PPO/GRPO clip | 各项用w_t | token | 再引入裁剪的更新取舍 |
| GSPO | (∏_t w_t)^(1/T) | sequence | 长度归一化与裁剪均改变估计器 |

偏差与方差的具体大小依赖策略距离、回答长度和奖励结构。
不能把这些方法排成适用于所有任务的严格优劣或方差顺序。

## 后训练全景：RLVR、RLHF、KL 与 DPO

### 反馈来源、策略算法、数据形式不是同一维度

| 名称 | 描述的维度 | 与本作业的关系 |
|---|---|---|
| RLVR | 用可验证结果反馈 | 数学答案grader属于该路线 |
| RLHF | 人类偏好等反馈范式 | 经典路线含奖励模型与PPO |
| PPO / GRPO / GSPO | 策略更新方法 | 可与不同奖励来源组合 |
| SFT | 示范数据上的监督目标 | 可用来建立初始行为，不是所有RL的必要前置 |
| DPO | 偏好对上的直接策略优化 | 属于补充材料，不能与GRPO画等号 |

RL不自动创造模型从未探索到的正确行为。
prompt既规定任务，也影响采样探索分布。
few-shot可能提供格式或推理模式，但也占用上下文。
比较算法时应固定prompt与生成条件，否则同时改变了实验问题。

### 奖励模型与critic分别回答什么？

reward model给一段回答打分，或verifier验证任务结果。
critic预测从某个前缀继续生成的未来回报。
二者不是同一个模型，也不是只换输出名字。
GRPO省略critic，不意味着省略奖励来源。

### old policy与reference policy为什么不能混？

| 对象 | 公式中的位置 | 固定多久 |
|---|---|---|
| 当前πθ | 要优化的回答分布 | 随参数更新 |
| 行为π₀ | importance ratio分母 | 本批回答复用期间固定 |
| 参考π_ref | 后训练KL锚点或DPO参考 | 通常整个阶段冻结 |

行为策略解决“数据由谁生成”。
参考策略解决“允许偏离哪个基础行为多少”。
两者即使初始参数相同，职责也不同。

### 常见KL正则目标怎样写？

经典后训练可在任务奖励外，加上参考策略约束：

$$
J_{\rm KL}(\theta)=\mathbb E_x\left[
\mathbb E_{y\sim\pi_\theta}[R(x,y)]
-\beta D_{\rm KL}(\pi_\theta(\cdot\mid x)\Vert\pi_{\rm ref}(\cdot\mid x))\right].
$$

利用自回归分解，序列log-ratio可拆为token项：

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{\rm ref}(y\mid x)}
=\sum_t\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\rm ref}(y_t\mid s_t)}.
$$

它使“每token惩罚偏移、终局给任务分”成为自然构造。
在与定义匹配的分布下取期望才是KL；单个log-ratio可为负。
旧采样分布上的固定统计不自动等于新策略的精确KL。

本reasoning assignment的主线目标是期望可验证奖励。
不能因为某篇GRPO/RLHF笔记写有KL，就默认讲义公式也包含它。
经典流程来源见[InstructGPT](https://arxiv.org/abs/2203.02155)。

### DPO与RL公式的连接，保留到哪里就够？

对固定奖励与参考策略的KL正则问题，理想最优策略满足：

$$
\pi^*(y\mid x)\propto\pi_{\rm ref}(y\mid x)\exp(R(x,y)/\beta).
$$

反解奖励，再代入偏好概率，可得到DPO的分差目标：

$$
\mathcal L_{\rm DPO}
=-\mathbb E\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}\right]\right).
$$

它用chosen/rejected偏好对，不依赖GRPO式在线组内奖励。
两个序列log-prob仍然都是token log-prob的和。
这解释了共享语言模型概率基础，不代表相同训练算法。
来源见[DPO原论文](https://arxiv.org/abs/2305.18290)。

## 手算一组回答与等价性检查

### 同一题的四个回答，怎样得到不同学习权重？

取G=4，奖励为(1,0,0,1)。
均值μ=0.5，样本标准差为√(1/3)≈0.57735。
忽略ε，表中还没加入长度或概率比。

| 回答 | R | RFT权重 | Dr.GRPO权重 | GRPO权重 | MaxRL权重 |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 0.5 | 0.8660 | 1 |
| 2 | 0 | 0 | −0.5 | −0.8660 | −1 |
| 3 | 0 | 0 | −0.5 | −0.8660 | −1 |
| 4 | 1 | 1 | 0.5 | 0.8660 | 1 |

RFT只强化正确回答；中心化方法还压低同组错误回答。
GRPO与MaxRL的数值变化来自不同分母。
不能只比较这组数字的大小就断言算法更强。

### 再把长度加回来

设两个正确回答长度为2与8，N=4、L_max=8。
下表是每个token的系数，不是实际梯度范数。

| 聚合 | 长度2的正确回答 | 长度8的正确回答 |
|---|---|---|
| GRPO序列平均 | 0.866/(4×2)≈0.1083 | 0.866/(4×8)≈0.0271 |
| Dr.GRPO固定分母 | 0.5/(4×8)=0.015625 | 相同0.015625 |

一个方法同时改了优势与长度，不能只归因于baseline。
这就是讲义增加GRPO_constant对照的意义。

### 再把旧策略与裁剪加回来

回答已经由π₀生成，训练时重新评估当前概率。
若正优势token的比率达到1.4、ε_clip=0.2，PPO该项变为常数1.2A。
若负优势token的比率也是1.4，则该项仍会被纠正。
GSPO则先汇总整段log-ratio，再统一判断该回答是否裁剪。

### 一张“等价性账本”

| 修改 | 关系 | 为什么 |
|---|---|---|
| 序列log概率展开为token之和 | 严格等价 | 自回归因子分解 |
| 终局R、γ=1作为各步MC回报 | 对当前任务严格对应 | 前面即时奖励为0，未来同含终局分 |
| 减与当前动作无关的baseline | 保持期望 | score期望为0 |
| 减含自身的组均值 | 固定G时差常数 | 自身贡献留下(G−1)/G |
| 用固定L_max整体缩放 | 差统一常数 | 不依赖本次回答 |
| 除随机组标准差 | 一般改变权重/期望 | 分母依赖题目与采样结果 |
| 每条回答除自身长度 | 一般改变目标权重 | 分母依赖回答 |
| 完整序列IS、未裁剪 | 条件满足时精确修正原R梯度 | 使用整段分布比 |
| token比率替代完整乘积 | 更换代理 | 只替换局部动作策略 |
| PPO裁剪 | 更换代理梯度 | 移除部分过度改善项 |
| GSPO几何平均比率 | 更换估计器 | 1/T幂改变原始分布比 |

“只差常数”说的是期望梯度方向。
有限样本分布、Adam类优化器或其他约束下，训练轨迹未必等同。

### 六个容易混淆的问题

1. 为什么奖励不可微仍能训练？梯度来自回答概率，不来自grader。
2. GRPO优势是否就是真正Q−V？不是，它是组内相对权重。
3. 全错组加ε后能否产生正确方向？不能，零分子仍为零。
4. token概率比是否精确修正整段分布？一般不能。
5. GSPO是否只是把GRPO的clip阈值调小？不是，比率与梯度聚合粒度都变了。
6. old与reference能否互换？不能，一个关联采样数据，一个关联目标锚点。

## Loss 图鉴：从数学目标到 backward

前文推导回答“更新方向从哪里来”，这里逐项写清楚“实际对什么标量调用 `backward()`”。以下片段是核心计算，省略模型加载、优化器和日志，不能代替完整训练循环。

### 统一符号、mask 与梯度边界

此处将回答展平编号为 $n=1,\ldots,N$，$N=BG$；$g(n)$ 表示所属题目组。注意前文的双下标和这里的单下标描述的是同一批回答。

$$
\ell_{nt}=\log\pi_\theta(y_{nt}\mid x_n,y_{n,<t}),\qquad
\ell^0_{nt}=\log\pi_0(y_{nt}\mid x_n,y_{n,<t}),
$$

$$
m_{nt}\in\{0,1\},\quad L_n=\sum_t m_{nt},\quad
q_n=\sum_t m_{nt}\ell_{nt},\quad Z=NL_{\max}.
$$

$m$ 只标记被训练的回答标签位置，排除 prompt 与 padding。下文要求每条回答至少有一个有效 token；`clamp_min(1)` 是防止除零的工程保护，不能代替空回答的数据检查。$Z$ 对一次完整 optimizer batch 固定，不能跟随各 microbatch 的实际 token 数变化。

奖励、优势、old log-prob 和 reference log-prob 在本轮更新中固定；只有当前策略的 log-prob 保留梯度。下列变量约定：`logp`、`old_logp`、`mask` 为 `[N,T]`，`adv` 为 `[N]`，`rewards` 为 `[B,G]`。

```python
# 各 RL 片段共用的聚合；mask 已按 labels 对齐
def sequence_mean(per_token, mask):
    lengths = mask.sum(-1).clamp_min(1)
    return ((per_token * mask).sum(-1) / lengths).mean()

def constant_mean(per_token, mask, Z):
    return (per_token * mask).sum() / Z
```

两种聚合分别是“每条回答内部平均，再平均回答”与“全体有效项求和，统一除固定常数”。它们对变长回答的加权不同。

### SFT：示范序列的条件似然

SFT 在固定示范数据上最大化条件 log-likelihood。补充作业将 Alpaca 模板化文档加 EOS 后拼接、打包，按 next-token 训练，不默认只对回答部分加 mask：

$$
S_{\rm SFT}=\frac1M\sum_{k=1}^{M}\log\pi_\theta(z_k\mid z_{<k}),
\qquad \mathcal L_{\rm SFT}=-S_{\rm SFT}.
$$

若 `labels` 是尚未移位的完整序列，可写成：

```python
logits = model(input_ids).logits
loss = F.cross_entropy(
    logits[:, :-1].contiguous().reshape(-1, logits.size(-1)),
    labels[:, 1:].contiguous().reshape(-1),
)
```

交叉熵已经包含负号与平均，不能再取负。本仓库 `PackedSFTDataset` 返回的 labels **已经右移**，实际 `supplement.sft_loss` 使用同位置评分器：

```python
scored = get_response_log_probs(model, batch["input_ids"], batch["labels"])
loss = -scored["log_probs"].mean()
```

不能把前一个片段的 shift 再套到后一个数据接口，否则会变成预测下下个 token。两种写法区别在输入合同，不在交叉熵原理。

### DPO：相对 reference 的偏好似然

用 Bradley–Terry 模型表示 chosen 比 rejected 更受偏好的概率，并用相对 reference 的 log-prob 重参数化隐式奖励。记：

$$
d_w=\log\pi_\theta(y_w\mid x)-\log\pi_{\rm ref}(y_w\mid x),
\qquad
d_l=\log\pi_\theta(y_l\mid x)-\log\pi_{\rm ref}(y_l\mid x).
$$

$$
S_{\rm DPO}=\mathbb E_{\rm data}\log\sigma\!\left(\beta(d_w-d_l)\right),
\qquad \mathcal L_{\rm DPO}=-S_{\rm DPO}.
$$

```python
# 四个输入都是有效 response token 的 log-prob 求和，形状 [N]
margin = ((policy_chosen_logp - ref_chosen_logp.detach())
          - (policy_rejected_logp - ref_rejected_logp.detach()))
loss = -F.logsigmoid(beta * margin).mean()
```

Chosen 和 rejected 的当前策略分数都要保留梯度；reference 冻结。标准 DPO 使用序列 log-prob 的**和**，换成长度均值会改变目标。这是偏好似然，不是 REINFORCE 的奖励加权交叉熵。

### 标准 on-policy GRPO：组内标准化加序列平均

原始任务关心期望奖励，实际 loss 则构造为能够产生 GRPO 梯度估计器的局部代理：

$$
A_n=\frac{R_n-\mu_{g(n)}}{s_{g(n)}+\epsilon_{\rm num}},\qquad
\mathcal L_{\rm GRPO}=-\frac1N\sum_n\frac{\operatorname{sg}(A_n)}{L_n}q_n.
$$

```python
a = adv.detach()[:, None]         # [N,1] 广播到 [N,T]
loss = sequence_mean(-a * logp, mask)
```

正优势提高这条回答的 token 概率，负优势压低。组均值、随机组标准差和随机长度使它与原始 REINFORCE 的关系需要分别分析；不能把代理的负号移回原始期望目标，声称二者数值相等。

### GRPO_constant：仅替换聚合

优势仍然除以组标准差，去掉每条回答的 $1/L_n$，改用固定 $Z$：

$$
\mathcal L_{\rm constant}=-\frac1Z\sum_n\operatorname{sg}(A_n)q_n.
$$

```python
loss = constant_mean(-adv.detach()[:, None] * logp, mask, Z)
```

它用于将长度归一化的影响与优势标准化的影响分开；不要一边更换两者，一边把差异都归因于其中一个。

### Dr.GRPO：保留组均值，去掉两种随机除法

$$
A_n=R_n-\mu_{g(n)},\qquad
\mathcal L_{\rm Dr}=-\frac1Z\sum_n\operatorname{sg}(R_n-\mu_{g(n)})q_n.
$$

```python
mu = rewards.mean(dim=1, keepdim=True)   # [B,1]
adv = (rewards - mu).reshape(-1).detach()
loss = constant_mean(-adv[:, None] * logp, mask, Z)
```

从 REINFORCE 逐步看：先同题采样并减组均值；标准 GRPO 再除组标准差、再按各回答长度平均；Dr.GRPO 撤销后两个操作，用固定 $Z$ 保留尺度控制。有限组均值仍包含自身奖励，所以纯 on-policy、独立同题采样且奖励不直接依赖参数时：

$$
\mathbb E[-\nabla\mathcal L_{\rm Dr}]
=\frac{G-1}{G L_{\max}}\nabla J.
$$

这与前文的组均值证明一致；仓库的 LOO 是额外 baseline 对照，不是作业 Dr.GRPO 的定义。

### RFT：二元奖励下只保留正确回答

$$
A_n=\mathbf1[R_n=1]=R_n,\qquad
\mathcal L_{\rm RFT}=-\frac1Z\sum_n R_n q_n.
$$

```python
adv = rewards.reshape(-1).detach()      # 二元 reward，不减 baseline
loss = constant_mean(-adv[:, None] * logp, mask, Z)
```

错误样本贡献为零，可以跳过它们的前向。过滤后依然保留原 batch 的 $Z$，不能改为“留下的正确回答数乘当前长度”。当前策略新采样、二元奖励、固定分母时，它对应无 baseline REINFORCE；长期复用旧正确回答或重新随机归一化时不再直接满足该结论。

### MaxRL：用组均值缩放优势

$$
A_n=\frac{R_n-\mu_{g(n)}}{\mu_{g(n)}+\epsilon_{\rm num}},\qquad
\mathcal L_{\rm MaxRL}=-\frac1Z\sum_n\operatorname{sg}(A_n)q_n.
$$

```python
mu = rewards.mean(dim=1, keepdim=True)
adv = ((rewards - mu) / (mu + eps)).reshape(-1).detach()
loss = constant_mean(-adv[:, None] * logp, mask, Z)
```

低均值组的放大系数较大；全错组仍是零分子。$\epsilon$ 在均值接近零时会显著影响权重。这里遵循课程版固定归一化，不能将其直接写成原论文所有实现细节的复现。

### Off-policy naive：相同代码，不同采样分布

$$
\mathcal L_{\rm naive}=-\frac1N\sum_n\frac{\operatorname{sg}(A_n)}{L_n}q_n,
\qquad y_n\sim\pi_0.
$$

```python
loss = sequence_mean(-adv.detach()[:, None] * logp, mask)
```

数值公式与 on-policy GRPO 一样。不同之处是数据来自已经偏离当前策略的 $\pi_0$，没有分布修正；代码相同不代表估计器具有相同统计性质。

### Off-policy no-clip：对 ratio 求导

$$
w_{nt}=\exp(\ell_{nt}-\ell^0_{nt}),\qquad
\mathcal L_{\rm noclip}=-\frac1N\sum_n\frac{\operatorname{sg}(A_n)}{L_n}
\sum_t m_{nt}w_{nt}.
$$

```python
ratio = torch.exp(logp - old_logp.detach())
loss = sequence_mean(-adv.detach()[:, None] * ratio, mask)
```

这里 loss 主体是 $-Aw$，因为 $\nabla w=w\nabla\log\pi_\theta$。写成 `-A * ratio * logp` 且让 ratio 参与求导，会多出导数项；把 ratio 完全 detach 后只优化 `-A * ratio` 又会切断当前策略梯度。Token ratio 仍是局部代理，不是精确的完整序列 IS。

### Off-policy GRPO clip：对两个候选目标取 min

$$
\bar w_{nt}=\operatorname{clip}(w_{nt},1-\epsilon_{\rm clip},1+\epsilon_{\rm clip}),
$$

$$
\mathcal L_{\rm clip}=-\frac1N\sum_n\frac1{L_n}\sum_t m_{nt}
\min(A_n w_{nt},A_n\bar w_{nt}).
$$

```python
ratio = torch.exp(logp - old_logp.detach())
clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
a = adv.detach()[:, None]
loss = sequence_mean(-torch.minimum(a * ratio, a * clipped), mask)
```

必须先分别乘优势，再取 min。只对 ratio 调用 `clamp` 后乘优势，会丢掉正负优势对应的单侧保护机制。Loss 取负后用梯度下降，才与代理目标上升一致。

### GSPO：每条回答一个可裁剪的比率

$$
u_n=\exp\left(\frac1{L_n}\sum_t m_{nt}(\ell_{nt}-\ell^0_{nt})\right),
$$

$$
\mathcal L_{\rm GSPO}=-\frac1N\sum_n
\min\left(A_nu_n,A_n\operatorname{clip}(u_n,1-\epsilon_{\rm clip},1+\epsilon_{\rm clip})\right).
$$

```python
lengths = mask.sum(-1).clamp_min(1)
log_ratio = logp - old_logp.detach()
u = torch.exp((log_ratio * mask).sum(-1) / lengths)
clipped = u.clamp(1 - clip_eps, 1 + clip_eps)
a = adv.detach()
sequence_loss = -torch.minimum(a * u, a * clipped)
loss = sequence_loss.mean()
```

几何平均指数中的 $1/L_n$ 已通过链式法则进入梯度，不能再对这个序列标量除一次长度。仓库为了兼容 `[N,T]` 接口，将 `sequence_loss[:,None]` 广播到 token 维，再经过 `sequence` 聚合恢复序列标量；这个广播没有新增一次长度缩放。

### 各分支的实现选择

| 方法 | 学习权重 | Loss 主体 | 聚合 |
|---|---|---|---|
| SFT | 示范标签 | next-token 交叉熵 | 有效训练 token 平均 |
| DPO | chosen/rejected 偏好 | 负 log-sigmoid 分差 | 偏好对平均 |
| GRPO | 中心化后除组标准差 | `-A * logp` | sequence |
| GRPO_constant | 同上 | `-A * logp` | constant |
| Dr.GRPO | 奖励减组均值 | `-A * logp` | constant |
| RFT | 二元奖励 | `-R * logp` | constant |
| MaxRL | 中心化后除组均值 | `-A * logp` | constant |
| Naive | GRPO 优势 | `-A * logp` | sequence，旧数据 |
| No-clip | GRPO 优势 | `-A * ratio` | sequence |
| GRPO clip | GRPO 优势 | `-min(Aw, A clip(w))` | sequence |
| GSPO | GRPO 优势 | 序列比率的裁剪 loss | 回答平均 |

`importance_reweighting_method` 中 `none / noclip / grpo / gspo` 分别对应不修正、token ratio、token 裁剪与序列裁剪。`none` 是新采样还是 naive，取决于数据是否来自当前策略。CISPO 是讲义中的选做扩展，自定义估计器也没有唯一标准 loss，不能混入这张表当作课程既定必做分支。

## 从公式到实际训练链路

### 代码地图与两条调用链

项目采用手写 PyTorch 训练循环。Transformers 提供模型、tokenizer 与生成接口；PEFT 和 bitsandbytes 为补充路线提供 QLoRA 支持。核心 loss、梯度累积与参数更新由仓库代码组织。

```text
python -m cs336_alignment.cli
  ├─ rl  → rl.train_rl
  │        → 生成 → grader → 完整组优势 → 缓存 old log-prob
  │        → core.grpo_train_step → 当前模型评分 → loss → 更新
  ├─ sft → supplement.train_supplement → PackedSFTDataset → CE
  └─ dpo → supplement.train_supplement → reference cache
           → preference_tokens → score_preference → cached_dpo_loss
```

| 文件 | 关注的接口与职责 |
|---|---|
| [core.py](https://github.com/tanyouqing/CS336-Assignment5/blob/main/cs336_alignment/core.py) | tokenize、log-prob、熵、奖励、优势、policy loss、microbatch 聚合与一次更新 |
| [rl.py](https://github.com/tanyouqing/CS336-Assignment5/blob/main/cs336_alignment/rl.py) | rollout、完整组统计、旧策略评分与训练批次编排 |
| [data.py](https://github.com/tanyouqing/CS336-Assignment5/blob/main/cs336_alignment/data.py) | JSONL、数据划分、packed SFT 与任务解析 |
| [dpo.py](https://github.com/tanyouqing/CS336-Assignment5/blob/main/cs336_alignment/dpo.py) | 偏好对编码、response 评分与 DPO loss |
| [supplement.py](https://github.com/tanyouqing/CS336-Assignment5/blob/main/cs336_alignment/supplement.py) | SFT/DPO 循环、reference 预计算与验证 |
| `runtime.py`、`evaluation.py` | 模型加载、生成、保存恢复、评估与指标 |
| `experiments.py`、`tests/adapters.py` | 实验编排与课程测试接口转接 |

### Token shift 与 response mask 必须一起移动

完整序列由 prompt 和 response 拼接，再构造相邻预测对：

```text
完整 token： P0 P1 R0 R1 EOS
原始角色：    0  0  1  1   1
input_ids：  P0 P1 R0 R1
labels：     P1 R0 R1 EOS
mask：        0  1  1   1
```

`input_ids = ids[:,:-1]`，`labels = ids[:,1:]`，`response_mask = roles[:,1:]`。Mask 描述的是 **label 的角色**：输入位置 `P1` 的目标是首个回答 token `R0`，这一列必须计入回答 loss。Prompt 不作为 RL 的被预测目标，并不意味着 prompt 的上下文路径不参与参数梯度。

模型输入 `[N,T]`，输出 logits `[N,T,V]`；沿词表维计算 `log_softmax`，用 `labels.unsqueeze(-1)` 执行 `gather`，得到实际回答 token 的 log-prob `[N,T]`。优势 `[N]` 扩展成 `[N,1]` 后广播，不必复制成完整矩阵。

熵为 $H(p)=-\sum_v p_v\log p_v$，用于观察分布是否过早变尖；除非明确加进目标，记录熵不意味着已经做 entropy bonus。仓库用 FP32 计算分块 log-softmax，并通过 activation checkpointing 重算来减少保留的中间激活；模型输出 logits 本身仍占显存。

### Rollout batch、optimizer batch 与 microbatch

Rollout batch 是本轮生成的全部回答，group size 是每题回答数；optimizer batch 决定一次参数更新使用哪些回答；microbatch 是显存允许的一次 forward/backward 规模。生成并行数又是另一个参数。

先在**完整题目组**上计算并固定优势，再切训练 batch 与 microbatch，不能切完之后把不同题目拼成新组。Old log-prob 也要在更新前缓存；更新后不能用新模型重新覆盖分母。

如果完整 optimizer batch 有 $N$ 条回答，第 $k$ 个 microbatch 有 $n_k$ 条：

- Sequence 聚合先得到该 microbatch 的均值，再乘 $n_k/N$，累积后等于完整回答平均。
- Constant 聚合让各 microbatch 的有效 loss 和都除同一个完整 batch 的 $Z$；累积后不再额外除 microbatch 数。
- 剪去零优势回答时，仍保留原来的 $N$ 或 $Z$，不能把剩余回答重新平均成另一目标。

例如三条回答拆成 `2 + 1` 时，两个 microbatch 均值的权重应为 `2/3` 与 `1/3`。直接平均两个均值会让最后一条回答权重翻倍。

一次 optimizer 更新只清零一次，完成各 microbatch 的 `backward()` 后，统一做梯度范数裁剪与 `step()`。整批优势为零且没有其他目标时，仓库跳过更新；否则 AdamW 等优化器即使收到零梯度，也可能因状态或 weight decay 改变参数。

### 生成与评分的一致性

训练中的 `logp` 必须重新前向计算；生成阶段通常不保留 autograd 图。`model.train()` 和 `model.eval()` 控制 dropout 等层的行为，`no_grad()` 才控制是否建立梯度图，不能把二者混为一谈。

数学上的精确重要性采样要求分母对应实际行为分布。Temperature、top-p、EOS、停止字符串、最大长度与重新 tokenization 都可能影响这个对应关系。主 RL 代码采用 temperature 1，读取旧模型对训练 token 的分数；仍要核对生成 token 与训练 token、停止边界和 padding 的一致性，而不能只因 tensor 形状相同就认为分布已精确对齐。

双卡 vLLM 路线需要把最新训练权重同步给推理端；单卡 HF 路线在生成与训练之间顺序切换。减少 microbatch 能降低峰值显存，不会自动减少一次 rollout 的生成总成本。

### SFT 与 DPO 的单卡工程

补充路线使用冻结的 4-bit base 加可训练 LoRA adapter，配合梯度 checkpointing 和小 microbatch 控制显存；它改变可训练参数集合，与课程完整全参实验的证据范围不同。加载 adapter 时仍需要对应 base model。

DPO 可以提前计算冻结 reference 对所有偏好样本的分数，训练时只保留当前模型的可微前向。缓存必须与 base、SFT adapter、tokenizer、样本 token 和截断规则绑定；换数据或 checkpoint 后不能静默复用旧缓存。`--adapter` 表示 SFT 初始化，`--resume` 表示已更新的 DPO 训练状态。

正式偏好训练路径只求和 response 的 log-prob；课程单样本 adapter 的完整文本评分写法应结合共享 prompt 抵消的条件理解，不能不检查 token 边界就推广。Reference 与 policy 必须使用相同 prompt、response mask 和序列边界。

## 评估、实验对照与验证边界

### 先建立 prompting baseline

主线对照 `question_only`、`r1_zero` 和 three-shot prompt。它们不仅改变文字模板，也影响格式、探索、回答长度与成功概率。评估时固定数据划分和生成设置，分别记录格式正确与答案正确；grader 通过不等于推理过程每一步都正确。

补充路线从 base、SFT 到 DPO 比较多种能力：MMLU 的知识与选择题表现、GSM8K 的数学正确率、AlpacaEval 的偏好评价、SimpleSafetyTests 的安全行为。自动裁判分数依赖裁判模型和 prompt；不能仅凭验证 loss 代替生成评估。

### 用对照拆解算法变化

| 对照 | 想隔离的因素 | 应同时记录 |
|---|---|---|
| GRPO vs GRPO_constant | 随机回答长度归一化 | 正确率、长度分布、token 数 |
| GRPO_constant vs Dr.GRPO | 奖励标准差缩放 | 组奖励分布、混合组比例 |
| Dr.GRPO vs LOO | 含自身 baseline 的有限组缩放 | 组大小、梯度尺度与方差 |
| Dr.GRPO vs RFT | 是否中心化、是否更新错误回答 | 活跃回答数、有效训练 token |
| constant 变体 vs MaxRL | 难度重加权 | 各难度题目的成功率 |
| naive / noclip / clip / GSPO | 旧数据修正与裁剪方式 | 策略偏移、clip fraction、训练耗时 |

本地主讲义的 off-policy 对照让同一 rollout batch 分成 32 个 optimizer batch，分别使用 token clip 0.2 和 GSPO clip $3\times10^{-4}$ 作为起点。这些是课程实验设置，不是任意模型都适用的最优参数；比较时要同时说明 rollout 数、optimizer 更新数、生成 token 与墙钟预算。

训练日志应区分奖励、验证正确率、长度、熵、梯度范数和裁剪比例。DPO 的隐式奖励准确率比较的是 **policy 与 reference 的 chosen/rejected 分差** 是否为正，不是仅比较当前模型的两个裸 log-prob。RL 代理 loss 的值随每轮采样、优势和比率改变，不能把它当作普通监督 loss，要求全程单调下降。

### 如何验证实现

测试入口和运行说明见仓库 [RUNBOOK](https://github.com/tanyouqing/CS336-Assignment5/blob/main/docs/RUNBOOK.md)。合理顺序是：先验证数学原语与 mask，再检验完整 batch 和 microbatch 梯度一致、零优势过滤前后一致、reference 不变，最后在 Linux/CUDA 上检查真实模型的 rollout、更新、保存与恢复。

仓库区分精简单卡学习配置与课程原版实验配置。小规模 smoke test 可以发现流程错误，不能替代完整 GSM8K 曲线、多随机种子对照或补充作业的正式评测。本次网站更新核对了讲义、笔记和代码，并验证网页格式；没有重新运行模型训练，因此不把实现覆盖、静态检查和真实 GPU 实验混写成同一种“已通过”。

## 材料与延伸阅读

本文整理依据为本地保存的 `rl_to_llm_posttraining_loss_atlas.html`、`THEORY_NOTES_CN.md`、`code_walkthrough.html` 和 `RUNBOOK.md`。核心内容已经整合进正文，项目源码与已发布材料见 [GitHub 仓库](https://github.com/tanyouqing/CS336-Assignment5)。

| 本文内容 | 本地主讲义对应位置 |
|---|---|
| LLM 的 RL 对象、策略梯度 | §4.1.1–4.1.4 |
| Baseline、组均值与优势 | §4.1.5–4.1.6，第 10–11 页的方差练习 |
| 长度归一化与 on-policy loss | §4.1.7–4.2，第 18–20 页 |
| Dr.GRPO、RFT、MaxRL | §5，第 26–31 页 |
| 旧数据、重要性采样、裁剪、GSPO | §6，第 32–37 页 |
| SFT 与 DPO | Supplement §4、§6 |

原讲义：[Reasoning RL](https://github.com/tanyouqing/CS336-Assignment5/blob/main/cs336_spring2026_assignment5_alignment.pdf) 与 [SFT/DPO supplement](https://github.com/tanyouqing/CS336-Assignment5/blob/main/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)。本地主线是 Spring 2026、Version 26.0.0；具体算法和归一化以该版本为准。

算法背景可查 [GAE](https://arxiv.org/abs/1506.02438)、[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)、[Dr.GRPO](https://arxiv.org/abs/2503.20783)、[MaxRL](https://arxiv.org/abs/2602.02710)、[GSPO](https://arxiv.org/abs/2507.18071)、[InstructGPT](https://arxiv.org/abs/2203.02155) 与 [DPO](https://arxiv.org/abs/2305.18290)。这里引用原理，不将论文成绩转写为本项目的实测结果。

站内的 [强化学习数学笔记（上）](/articles/强化学习的数学原理笔记(上)/)、[强化学习数学笔记（下）](/articles/强化学习的数学原理笔记(下)/) 和 [大模型后训练与 Agentic RL](/articles/大模型后训练与Agentic-RL笔记/) 可作为背景阅读；本文专注 Assignment 5 的单轮生成、损失与实现。

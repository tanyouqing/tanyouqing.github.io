---
title: "从基础强化学习到大模型后训练：PPO、DPO、GRPO 与 OPD"
date: "2026-10-03"
description: "衔接强化学习基础，理解 GAE、PPO、RLHF、DPO、GRPO，以及学生采样、教师反馈的 On-Policy Distillation。"
tags: ["RL", "LLM"]
---

# 1. 从已有基础出发：这篇笔记要连接什么

本文承接[强化学习的数学原理笔记（上）](/articles/强化学习的数学原理笔记(上))与[强化学习的数学原理笔记（下）](/articles/强化学习的数学原理笔记(下))。前两篇覆盖 Bellman 方程、MC、TD、Sarsa、Q-learning、函数近似、策略梯度与 Actor-Critic；这里继续回答：怎样用偏好、可验证奖励和教师反馈，优化大模型的生成分布？

本文结合原始论文和官方实现说明训练目标与数据流。原后训练内容整理于 2026-10-03，OPD 内容补充核对于 2026-10-17；训练框架的配置与默认参数应以实际使用的版本为准。

知识依赖关系是：

```text
Policy Gradient / Actor-Critic / Advantage
                  ↓
         GAE → 稳定策略更新 → PPO
                  ↓
        LLM policy 与后训练目标
          ├─ 偏好反馈：RLHF / DPO
          ├─ 可验证反馈：RLVR / GRPO
          └─ 教师概率反馈：OPD
```

RLHF、RLVR 描述反馈来源与训练范式；PPO、GRPO 描述策略优化方法；DPO 描述直接偏好优化方法；OPD 描述在学生自身采样的前缀上向教师蒸馏。这些概念分别对应训练流程中的不同环节。

# 2. 后训练在优化什么：SFT 与 RL 的分工

预训练主要通过大量文本的 next-token prediction 学习语言和知识。后训练进一步塑造模型如何遵循指令、推理、使用工具，以及满足具体任务要求。后训练可以包含监督微调、偏好优化、RL、蒸馏和它们的多阶段组合。

## 2.1 SFT：学习示范行为

给定 prompt $x$ 和示范回答 $y^*$，监督微调的核心目标为：

$$
\mathcal L_{\mathrm{SFT}}(\theta)
=-\mathbb E_{(x,y^*)\sim\mathcal D}\sum_{t=1}^{T}
\log\pi_\theta(y_t^*\mid x,y_{<t}^*).
$$

teacher forcing 表示训练时用示范前缀预测下一个示范 token。用于工具任务时，示范可以包含工具选择、参数、观察后的决策和最终回答；具体哪些角色参与监督取决于数据与训练设定。

## 2.2 RL：通过执行结果调整生成分布

RL 对模型实际采样出的回答或轨迹打分，再提高高回报行为的概率：

$$
J(\theta)=\mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot\mid x)}[R(x,y)].
$$

SFT 提供“怎样做”的示范，RL 提供“这次做得怎样”的反馈。二者经常互补：先用 SFT 建立基本输出格式与工具能力，再用 RL 优化成功率和决策。筛选成功轨迹再做 SFT 也是可用的基线，但其更新机制与直接策略梯度不同。

RL 的反馈不会自动补齐模型完全缺失的能力。若模型从未采样到任何成功行为，奖励再准确也可能缺乏有效学习信号。因此初始策略、任务难度、探索、数据课程与奖励应一起考虑。

# 3. GAE：从 TD error 到适合 PPO 的优势估计

已有 Actor-Critic 更新为 $\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t$。新的问题是如何稳定估计 $\hat A_t$。统一沿用前文的时间约定：动作 $a_t$ 后收到奖励 $r_{t+1}$。

$$
\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
$$

固定策略且 $V_\phi=V^\pi$ 时，$\mathbb E[\delta_t\mid s_t,a_t]=A^\pi(s_t,a_t)$；这是对当前动作优势的采样估计。若只条件于 $s_t$ 并再对 $a_t\sim\pi$ 平均，期望为零。这与前文“真实值下 TD error 的状态条件期望为零”一致。

对于 $n$ 步估计：

$$
\hat A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l r_{t+l+1}
+\gamma^n V_\phi(s_{t+n})-V_\phi(s_t)
=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}.
$$

GAE 将不同长度的估计指数加权，有限轨迹的常用形式为：

$$
\hat A_t^{\mathrm{GAE}}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l},
\qquad
\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}.
$$

$\lambda=0$ 使用一步 TD；$\lambda=1$ 在完整终止轨迹、终止值为零时得到 $G_t-V(s_t)$。较小的 $\lambda$ 更依赖 critic，通常减少方差，但 critic 不准确时可能增加偏差。不能脱离 value 误差、终止方式与折扣目标笼统判断无偏性。[GAE 原始论文，Schulman 等，ICLR 2016](https://arxiv.org/abs/1506.02438)。

## 3.1 手算一次反向传播奖励

设三步奖励为 $(0,0,1)$，当前 $V(s_0),V(s_1),V(s_2)$ 分别为 $(0.5,0.4,0.2)$，真正终止的 $V(s_3)=0$，$\gamma=1,\lambda=0.8$。

| 时刻 | TD error | 从后向前计算 GAE | value target：$\hat A_t+V(s_t)$ |
| --- | --- | --- | --- |
| $t=2$ | $1-0.2=0.8$ | $0.8$ | $1.0$ |
| $t=1$ | $0.2-0.4=-0.2$ | $-0.2+0.8\times0.8=0.44$ | $0.84$ |
| $t=0$ | $0.4-0.5=-0.1$ | $-0.1+0.8\times0.44=0.252$ | $0.752$ |

虽然前两步即时奖励为零，后面的成功仍能回传到前面。训练 actor 时，已计算的 advantage 应停止梯度；训练 critic 时通常回归停止梯度的 value target，避免模型通过改变 target 来降低误差。

## 3.2 真正终止与采样截断

真正终止时未来价值为零；若只是采样器在固定步数切断了一条仍会继续的轨迹，通常应在边界保留 bootstrap。若任务本身规定“达到预算即失败”，预算耗尽则是该任务定义下的终止。GAE 还要阻止递推跨越 reset 后的另一个 episode。因此 bootstrap mask 与轨迹续接 mask 需要分别按语义处理，不能仅看到 `done` 就机械设零。

# 4. PPO：为什么要冻结 old policy 并限制更新

从 $\pi_{\mathrm{old}}$ 收集一批轨迹后，模型会在多个 minibatch/epoch 中更新。此时数据的采样分布固定，而待优化的 $\pi_\theta$ 已发生变化。PPO 使用采样动作的新旧概率比：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}
=\exp\big(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\big).
$$

这个局部 surrogate 修正给定状态下的动作概率。随着策略远离采样策略，旧状态分布与旧优势估计的适用性也会变差。

TRPO 通过新旧策略的 KL 约束控制更新；PPO-Clip 使用更容易优化的目标：

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb E_t\left[
\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

这是要最大化的目标；采用梯度下降实现时，actor loss 取其负数。PPO 用旧策略近期生成的数据做有限次复用，通常仍归入 on-policy 方法；有概率比不代表能无条件使用任意陈旧数据。[TRPO，Schulman 等，2015 预印本](https://arxiv.org/abs/1502.05477)；[PPO，Schulman 等，2017 预印本](https://arxiv.org/abs/1707.06347)。

## 4.1 min 和 clip 怎样共同工作

设 $\epsilon=0.2$，只观察单个样本：

| advantage | ratio | 未裁剪项 | 裁剪项 | 取 min 后 |
| --- | --- | --- | --- | --- |
| $+2$ | $1.5$ | $3.0$ | $2.4$ | $2.4$，继续增大好动作概率不再提高此项 |
| $+2$ | $0.5$ | $1.0$ | $1.6$ | $1.0$，降低好动作概率仍会被惩罚 |
| $-2$ | $0.5$ | $-1.0$ | $-1.6$ | $-1.6$，继续降低坏动作概率不再提高此项 |
| $-2$ | $1.5$ | $-3.0$ | $-2.4$ | $-3.0$，提高坏动作概率仍会被惩罚 |

裁剪移除过度改善这一项的激励，并不硬性保证所有概率比都落在区间内。共享参数会同时改变许多动作概率，因此还需要看 actual/approximate KL、clip fraction、entropy 和任务表现。

## 4.2 一轮 PPO 的数据流

```text
冻结采样策略 π_old
  → 收集状态、动作、奖励、old logprob、old value
  → 计算 GAE 和 value targets
  → 在多个 minibatch 中重新计算 current logprob
  → 更新 actor 与 critic，监控 KL / clipping
  → 同步采样策略，收集下一批新轨迹
```

old logprob 必须来自生成该动作时的行为策略，并在这一批复用期间保持不变。把每次 optimizer step 后的新 logprob 当作 old logprob，会破坏概率比的意义。

# 5. LLM 如何变成一个 RL policy

给定 prompt $x$ 和回答 $y=(y_1,\ldots,y_T)$：

$$
\pi_\theta(y\mid x)=\prod_{t=1}^T\pi_\theta(y_t\mid x,y_{<t}),
\qquad
\log\pi_\theta(y\mid x)=\sum_{t=1}^T\log\pi_\theta(y_t\mid x,y_{<t}).
$$

单轮生成有两种观察粒度。序列级可以视为 contextual bandit：输入 prompt，选择完整回答，得到一次评分。token 级则是序列决策：state 为 prompt 加前缀，action 为下一 token，transition 将 token 拼入前缀。不同粒度对应不同概率比和目标，不能混用。

| 基础 RL 对象 | 单轮 LLM 中的含义 |
| --- | --- |
| state / observation | 实际输入模型的 prompt 与生成前缀 |
| action | 下一 token；也可在另一粒度定义为整段输出 |
| policy | vocabulary 上的 softmax 分布 |
| trajectory | prompt 条件下生成的 token 序列 |
| reward | 偏好模型评分、规则验证或任务结果 |
| critic | 当前前缀继续生成的期望剩余回报 |

若只有最终 reward，序列 REINFORCE 的形式为：

$$
\nabla_\theta J
=\mathbb E\left[(R(x,y)-b(x))
\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\right].
$$

离散采样出的 token 与外部 verifier 往往不可直接求导，策略梯度通过 logprob 给参数提供更新方向。给整段 token 同一个 outcome 权重能训练模型，但不会直接说明哪一步推理导致成功。

# 6. RLHF：奖励模型与 KL 正则化

## 6.1 经典流程与反馈来源

经典 InstructGPT 流程包括示范 SFT、回答排序数据、reward model 和 PPO。这里“human feedback”描述评分来源；反馈也可能来自模型评审或自动规则，不应把所有评分都叫人类反馈。[InstructGPT，Ouyang 等，2022 预印本](https://arxiv.org/abs/2203.02155)。

给定同一 prompt 的偏好对 $(x,y_w,y_l)$，Bradley-Terry 模型为：

$$
P(y_w\succ y_l\mid x)
=\sigma\left(r_\psi(x,y_w)-r_\psi(x,y_l)\right),
$$

$$
\mathcal L_{\mathrm{RM}}(\psi)
=-\mathbb E\log\sigma\left(r_\psi(x,y_w)-r_\psi(x,y_l)\right).
$$

被监督的是同 prompt 下的分差，而非一个绝对的“人类价值分数”。奖励模型是有限数据学出的代理目标，策略过度优化它时可能学到评分漏洞。

## 6.2 reference policy 在约束什么

常见后训练目标为：

$$
J_{\mathrm{KL}}(\theta)=\mathbb E_{x}\left[
\mathbb E_{y\sim\pi_\theta}[r_\psi(x,y)]
-\beta D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\big)
\right].
$$

$\pi_{\mathrm{ref}}$ 通常是冻结的初始 SFT 模型。KL 限制向奖励模型不可靠分布的偏移，但它并不保证模型不会 reward hack 或遗忘。

| 模型/统计量 | 用途 | 典型生命周期 |
| --- | --- | --- |
| current policy $\pi_\theta$ | 当前待更新的 actor | optimizer step 后变化 |
| old / behavior policy | 生成本批数据，作为 PPO ratio 分母 | 每轮 rollout 更新；该批 old logprob 固定 |
| reference policy $\pi_{\mathrm{ref}}$ | 后训练 KL 的锚点 | 通常长期冻结 |
| critic $V_\phi$ | 预测回报并估计 advantage | 与 actor 配合学习 |
| reward model / verifier | 为输出或执行结果评分 | 单独训练或规则实现 |

PPO 的更新幅度控制是 current 与 old 的关系，后训练 KL 是 current 与 reference 的关系。初始时几者可以参数相同，随后承担不同职责。

## 6.3 从 sequence reward 到 token reward

在一次 $\pi_{\mathrm{old}}$ rollout 中，可以先计算并冻结采样 log-ratio：

$$
\ell_t=\log\pi_{\mathrm{old}}(y_t\mid s_t)
-\log\pi_{\mathrm{ref}}(y_t\mid s_t),
\qquad
\tilde r_{t+1}=-\beta\ell_t+\mathbf 1[t=T]r_\psi(x,y).
$$

然后沿 token 轨迹用 $\tilde r$、old value 计算 GAE，再更新 actor/critic。这里是常见 reward-shaping 实现示意；也可将 KL 直接放进 loss，不应同时重复计入两次。

采样的单个 log-ratio 可为负；在对应策略分布下取期望才得到非负的 KL。新策略更新后，固定 rollout 上的旧 KL 统计也不自动等于新策略的精确 KL。要明确记录采样分布与估计器。

# 7. DPO：把偏好目标直接写成 policy loss

考虑固定 $x$、固定 reward 和 $\beta>0$ 的 KL 正则化目标，在 reference 对候选输出有支持且归一化常数有限等条件下，其最优分布满足：

$$
\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)}.
$$

反解 reward：

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x).
$$

代入同 prompt 的 Bradley-Terry 分差，$\log Z(x)$ 抵消。用可训练 policy 参数化这个关系，就得到：

$$
\begin{aligned}
z_\theta={}&\beta\Big[
\log\pi_\theta(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_w\mid x)\\
&-\log\pi_\theta(y_l\mid x)+\log\pi_{\mathrm{ref}}(y_l\mid x)\Big],\\
\mathcal L_{\mathrm{DPO}}={}&-\mathbb E\log\sigma(z_\theta).
\end{aligned}
$$

DPO 通常直接在固定偏好对上训练，无须独立 reward model、critic 或每一步在线 rollout。推导基于特定奖励/偏好建模假设；有限数据、有限容量和分布偏移下，并不保证获得任意真实奖励的全局最优策略。[DPO，Rafailov 等，2023 预印本](https://arxiv.org/abs/2305.18290)。

## 7.1 怎样读懂 DPO 更新

$z_\theta$ 衡量：相对 reference，模型是否更偏向 chosen 而不是 rejected。初始 policy 与 reference 相同则 $z=0$，对应 loss 为 $\log2$；当 $z$ 增大，loss 下降。

基础 DPO 使用回答 token 的条件 logprob 之和来定义序列 logprob，逐 token 的 PPO ratio、平均 token logprob 则对应不同的目标。长度、偏好标注、prompt 配对和 response mask 都会影响训练。

DPO 的 $\beta$ 来自正则化目标，但在有限数据训练时也直接缩放分类 logit 与梯度；不能简单把它解释成“调大就一定更接近 reference”。固定数据 DPO 可以用于 agent 轨迹偏好，但一旦轨迹涉及不同环境观察，必须明确比较对象和条件上下文。

# 8. RLVR 与 GRPO：反馈来源和优化器分别改变什么

## 8.1 RLVR：让任务结果提供反馈

RLVR（Reinforcement Learning with Verifiable Rewards）使用可检查的结果，如数学答案等价性、隐藏单测、SQL 执行结果或文件状态约束。它可以配合 PPO、GRPO 等优化器。

DeepSeek-R1-Zero 的技术报告描述了不先做 SFT 的 RL 实验，使用准确性和格式等规则反馈；DeepSeek-R1 则采用含 cold-start、RL 和数据整理的多阶段流程。这说明某些设定中 RL 可以塑造推理行为，并不能推出所有任务都应跳过 SFT。[DeepSeek-R1，DeepSeek-AI，2025 技术报告](https://arxiv.org/abs/2501.12948)。

可验证不代表验证器覆盖了整个真实目标。通过公开单测、输出合法标签或声称“完成”都可能比真实成功容易，必须核验答案或环境中的最终产物。

## 8.2 GRPO 的组内相对优势

对同一 prompt 用 rollout policy 采样 $G$ 个回答，得到 $R_1,\ldots,R_G$。一种典型 outcome 形式为：

$$
\bar R=\frac1G\sum_{i=1}^G R_i,\qquad
\hat A_i=\frac{R_i-\bar R}{\operatorname{std}(R_1,\ldots,R_G)+\varepsilon_{\mathrm{num}}}.
$$

没有 learned critic；组内 reward 统计量提供相对学习权重。将 $\hat A_i$ 广播到回答 token，再结合 token 概率比和裁剪。例如一种常见目标写法是：

$$
J_{\mathrm{GRPO}}=\mathbb E\left[\frac1G\sum_{i=1}^G\frac1{T_i}\sum_{t=1}^{T_i}
\left\{\min\big(\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)
-\beta\widehat k_{i,t}\right\}\right].
$$

其中 $\widehat k$ 是实现指定的 reference KL 估计项；数值稳定用的 $\varepsilon_{\mathrm{num}}$ 与 clipping 用的 $\epsilon$ 不是同一个参数。GRPO 的组内统计不能直接等同于精确的 $Q^\pi-V^\pi$，且它仍需 rollout、奖励与策略更新系统。[DeepSeekMath，Shao 等，2024 预印本](https://arxiv.org/abs/2402.03300)。

## 8.3 一个组的信号从哪里来

若四个回答奖励为 $(1,0,0,1)$，均值是 $0.5$，使用总体标准差 $0.5$ 且忽略数值稳定项，则 advantage 为 $(1,-1,-1,1)$。这是本文演示约定；实现采用样本标准差时数值会不同。

若奖励为 $(0,0,0,0)$ 或 $(1,1,1,1)$，中心化后的 reward advantage 都为零。该组没有相对优劣信号，但单独 KL 项或其它辅助损失仍可能更新参数。“全失败”要改善探索或任务课程；“全成功”可能需要更难任务或有意义的质量/代价区分。

当组均值包含自身 reward 时，它也不是严格独立于当前动作的 baseline。组中心化、标准差归一化是实际算法的估计选择，会改变梯度的尺度和任务权重，不应宣称它是无条件无偏的优势估计器。

## 8.4 归一化也是优化目标的一部分

按回答长度平均，会让每个回答获得相近总权重；按有效 token 总数平均，会让较长回答贡献更多 token 项。按固定长度常数归一化又有不同权重。标准差缩放还可能改变不同难度 prompt 的贡献。

因此“用了 GRPO”不足以复现实验，还需记录 group size、是否除标准差、KL 系数、token/sequence normalization、截断处理和采样配置。TRL 提供多种 loss 形式；这些选项应按版本核对。[TRL GRPO 官方文档](https://huggingface.co/docs/trl/grpo_trainer)。

进一步阅读时可看 [DAPO，Yu 等，2025 预印本](https://arxiv.org/abs/2503.14476)与 [Understanding R1-Zero-Like Training，Liu 等，2025 预印本](https://arxiv.org/abs/2503.20783)，关注它们讨论的采样、长度和归一化问题，再判断哪些与当前任务有关。

# 9. OPD：在学生自己生成的前缀上向教师学习

## 9.1 为什么不只训练教师写好的答案

OPD（On-Policy Distillation，同策略蒸馏）由学生生成回答，教师在这些回答的每个前缀上提供下一 token 的概率反馈。学生答错的回答同样可以进入训练。

用教师答案做 SFT 时，学生看到的是教师会走到的前缀；部署时却需要接着自己的输出生成。学生一旦走偏，后面的上下文可能从未在示范中出现。OPD 把监督放到学生实际访问的前缀上，缓解这种训练与生成分布不匹配。[GKD，Agarwal 等，2023 预印本、ICLR 2024](https://arxiv.org/abs/2306.13649)

“On-policy”描述前缀由谁生成，不指定唯一损失。GKD 可以选择 forward KL、reverse KL 或 JSD，也可混合不同来源的数据。下面主要介绍 reverse-KL OPD，并区分全词表损失与采样策略梯度两种计算方式。[GKD §3.1](https://arxiv.org/html/2306.13649v3)

## 9.2 教师和学生怎样比较下一 token

记学生为 $p_\theta$，冻结教师为 $q$，学生生成的前缀为 $s_t=(x,y_{<t})$。先把这个前缀视为固定输入，比较同一词表 $\mathcal V$ 上的分布：

$$
\begin{aligned}
D_{\mathrm{FKL}}(q\Vert p_\theta\mid s_t)
&=\sum_{v\in\mathcal V}q(v\mid s_t)
\log\frac{q(v\mid s_t)}{p_\theta(v\mid s_t)},\\
D_{\mathrm{RKL}}(p_\theta\Vert q\mid s_t)
&=\sum_{v\in\mathcal V}p_\theta(v\mid s_t)
\log\frac{p_\theta(v\mid s_t)}{q(v\mid s_t)}.
\end{aligned}
$$

FKL 用教师概率加权，惩罚学生漏掉教师认为可能的选项；RKL 用学生概率加权，更直接惩罚学生把概率分给教师不认可的选项。容量受限时，两者会产生不同折中，常称为 mode-covering 与 mode-seeking。适合采用哪种散度，取决于任务与学生容量。[MiniLLM，Gu 等，ICLR 2024](https://arxiv.org/abs/2306.08543)

能够取得完整 logits 时，可以在采样得到的固定前缀上直接计算全词表 KL，再对学生 logits 反传。学生采样决定训练访问哪些前缀，教师概率决定更新方向；直接把学生 token 当作正确标签做交叉熵，会丢掉教师的监督信息。

## 9.3 只取得已采样 token 的 logprob 时怎样训练

如果教师能返回学生已生成 token 的条件 logprob，就可以使用采样估计。用本批冻结的学生行为策略 $p_{\mathrm{old}}$ 生成 token，保存：

$$
\ell_{t}^{\mathrm{old}}=\log p_{\mathrm{old}}(y_t\mid s_t),
\qquad
\ell_{t}^{T}=\log q(y_t\mid s_t).
$$

为当前 token 定义固定的教师反馈：

$$
A_t=\operatorname{stopgrad}
\left(\ell_t^{T}-\ell_t^{\mathrm{old}}\right).
$$

教师相对学生更认可这个 token 时，$A_t>0$；反之为负。这个学习权重直接由采样 token 的 log-ratio 构造。在相应学生分布下，$-A_t$ 的期望对应非负的 reverse KL，单个采样值则可以为正或为负。

例如可以用 token importance-sampling surrogate：

$$
\rho_t(\theta)=\exp\left(
\log p_\theta(y_t\mid s_t)-\ell_t^{\mathrm{old}}\right),
\qquad
\mathcal L_{\mathrm{OPD}}=-\frac{\sum_t m_t\rho_t(\theta)A_t}
{\sum_t m_t},
$$

其中 $m_t$ 是回答 token 的有效 mask。下降这个 loss，会增加正反馈 token 的概率、降低负反馈 token 的概率。Thinking Machines 的 2025 年 OPD 博客采用这种逐 token 反馈，令折扣为零，只优化当前 token，不累计未来反馈；它是已有 OPD 工作上的一种具体配方。[Thinking Machines：实现与伪代码](https://thinkingmachines.ai/blog/on-policy-distillation/)

这里的 teacher 提供学习目标，old policy 记录采样分布。若另外加入 reference KL，还要明确它的约束对象和系数，避免重复计入同一项。

## 9.4 为什么不能直接对采样 log-ratio 求导

固定前缀 $s$ 时，全词表 RKL 的梯度为：

$$
\nabla_\theta D_{\mathrm{RKL}}
=\mathbb E_{v\sim p_\theta}
\left[\left(\log p_\theta(v\mid s)-\log q(v\mid s)\right)
\nabla_\theta\log p_\theta(v\mid s)\right].
$$

求导中出现的常数 $1$ 项，因为 $\mathbb E[\nabla\log p_\theta]=0$ 而抵消。令 $p_{\mathrm{old}}=p_\theta$，上一节 surrogate 的初始梯度便给出这个条件 KL 梯度的采样估计。

但是，把离散采样结果固定后，直接对 `mean(student_logp - teacher_logp)` 反传，只会得到采样的 score gradient，漏掉分布加权中的 log-ratio，不能实现上述 RKL 梯度。需要完整词表求和，或明确构造带 `stopgrad` 权重的策略梯度 loss。

还要区分固定前缀的条件 KL 与整段序列 KL：后者的梯度也包含前面 token 改变后续前缀分布的影响。GKD 不对采样序列的分布反传；MiniLLM 则分析了当前步与未来回报项。折扣为零的局部 OPD 不能称为完整序列 KL 的精确梯度。[GKD Algorithm 1](https://arxiv.org/html/2306.13649v3)、[MiniLLM §2.2](https://arxiv.org/html/2306.08543v4)

同一批数据更新多步后，固定的 $A_t$、old 前缀与当前学生会逐渐不匹配，因此通常在有限次更新后重新采样。动作概率比仅修正局部动作概率；裁剪、baseline 或混合其他奖励应作为额外设计记录。

## 9.5 一轮 OPD 的数据流

```text
准备 prompts，冻结教师，记录学生采样版本
  → 学生生成回答，保存 token ids、old logprob 与 mask
  → 教师读取相同 prompt 和学生前缀，给已生成 token 打 logprob
  → 构造并冻结逐 token 的教师反馈
  → 学生重新前向，计算 current logprob 和 surrogate loss
  → 更新学生，监控分布距离与任务表现
  → 刷新采样策略，生成下一批回答
```

下面只展示单个更新的损失计算。输入已对齐到每个回答 token，shape 都是 $(B,T)$；`mask` 排除 prompt、padding 和终止后的无效位置。采用有效 token 总数归一化，且假定采样策略与所记录 logprob 的分布一致。

```python
# old_logp 与 teacher_logp 来自本批数据，均不参与反传
advantage = (teacher_logp - old_logp).detach()
ratio = (current_logp - old_logp.detach()).exp()
loss = -(ratio * advantage * mask).sum() / mask.sum()
```

空 mask 的批次需要跳过。教师对整段学生文本做带 causal mask 的 teacher-forcing 前向，可并行获得各位置分数，无需让教师自回归重写每个答案；教师前向本身仍有计算与显存成本。[verl OPD 官方文档](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md)

## 9.6 一个固定前缀上的手算

设下一 token 只有 A、B、C 三种，学生概率为 $(0.6,0.3,0.1)$，教师为 $(0.2,0.5,0.3)$。使用自然对数：

| token | 学生概率 | 教师概率 | 反馈 $\log q-\log p$ |
| --- | --- | --- | --- |
| A | 0.6 | 0.2 | $-1.099$ |
| B | 0.3 | 0.5 | $+0.511$ |
| C | 0.1 | 0.3 | $+1.099$ |

学生抽到 A 时倾向降低其概率，抽到 B 或 C 时倾向提高其概率。共享 softmax 参数会同时改变其他选项，因此不能把每个方向理解成互不影响的独立概率增减。

完整 RKL 约为：

$$
0.6\log 3+0.3\log 0.6+0.1\log(1/3)
\approx 0.3961.
$$

采样反馈的期望是 $-0.3961$，但一次抽样并不等于这个期望。全词表损失在每个前缀上利用所有候选；采样形式更省输出分布的传输与存储，却有采样方差。

## 9.7 适用条件与训练代价

- 监督密度：OPD 在学生自己的前缀上获得逐 token 反馈，比仅用终局成功/失败奖励的 RLVR 更密集。使用过程奖励的 RL 也能提供中间反馈。
- 教师质量：分数表示教师在该前缀下如何续写。前文已经写错时，一个错误结论仍可能有较高条件概率，因此还要用任务结果检查推理质量。
- 初始能力：学生需要采样到有价值的行为。覆盖不足时，可用 SFT 扩展初始能力，再与 OPD 或可验证奖励结合。
- 接口与对齐：直接比较概率需要对应的词表、tokenizer 和 token 边界，以及教师对已生成 token 的条件 logprob。跨 tokenizer 蒸馏需要专门的对齐方法。
- 采样分布：上述推导采用温度为 1 的学生分布。温度、top-k/top-p 改变后，要按实际行为分布检查 logprob、支持集与概率修正。
- 资源成本：教师通常冻结，只做前向；学生仍需 rollout 和反向更新。基础 OPD 可以省去 critic 与组内奖励统计，教师评分则增加了计算和显存开销。

## 9.8 用三个问题区分 PPO、DPO、GRPO 与 OPD

| 方法 | 数据与反馈 | advantage / 学习权重 | 典型训练组件 |
| --- | --- | --- | --- |
| PPO-based RLHF / RLVR | 近期策略 rollout，RM 或 verifier 打分 | critic + GAE | actor、old logprob、critic、奖励；reference 按目标加入 |
| 标准离线 DPO | 固定 chosen/rejected 偏好对 | 相对 reference 的序列 logprob 分差 | actor、reference、偏好数据 |
| GRPO | 同 prompt 多个近期 rollout 与 reward | 组内相对 reward | actor、old logprob、奖励；典型形式无 critic |
| 本节采样式 OPD | 学生 rollout，教师对相同 token 前缀给概率反馈 | 冻结的逐 token 教师/学生 log-ratio | student、teacher、old logprob；不要求偏好对或 learned critic |

先问数据来自哪里，再问什么定义好坏，最后问怎样把反馈转成参数更新。PPO、GRPO 解决策略优化问题，DPO 利用偏好对，OPD 描述学生采样与教师监督的蒸馏方式；它们可以共享基础设施，也可以在训练流程中组合。

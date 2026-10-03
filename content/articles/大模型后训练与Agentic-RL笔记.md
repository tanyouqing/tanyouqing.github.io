---
title: "从基础强化学习到大模型后训练与 Agentic RL"
date: "2026-10-03"
description: "衔接已有强化学习笔记，理解 GAE、PPO、RLHF、DPO、GRPO，以及多步工具智能体的奖励、信用分配与训练系统"
tags: ["强化学习", "大模型", "后训练", "Agentic RL"]
---

# 1. 从已有基础出发：这篇笔记要连接什么

本文承接[强化学习的数学原理笔记（上）](/articles/强化学习的数学原理笔记(上))与[强化学习的数学原理笔记（下）](/articles/强化学习的数学原理笔记(下))。前两篇覆盖 Bellman 方程、MC、TD、Sarsa、Q-learning、函数近似、策略梯度与 Actor-Critic；这里继续回答：怎样用 RL 优化大模型生成的回答，又怎样把训练扩展到会调用工具、修改环境的 agent？

以提供的 HTML 学习路线为组织参考，结合原始论文和官方文档补充推导与实现边界。材料核对日期为 2026-10-03；训练框架的配置与默认参数应以实际使用的版本为准。

知识依赖关系是：

```text
Policy Gradient / Actor-Critic / Advantage
                  ↓
         GAE → 稳定策略更新 → PPO
                  ↓
        LLM policy 与后训练目标
          ├─ 偏好反馈：RLHF / DPO
          └─ 可验证反馈：RLVR / GRPO
                  ↓
         多轮工具交互与 Agentic RL
                  ↓
   环境 + 轨迹 + verifier + 信用分配 + 训练系统
```

RLHF、RLVR 描述反馈来源与训练范式；PPO、GRPO 描述策略优化方法；DPO 描述直接偏好优化方法；agentic 描述交互任务的结构。它们属于不同维度，例如“使用 GRPO 的 agentic RLVR”就是合理组合。

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

这是局部 surrogate 中的动作概率修正，不是对整条轨迹分布进行完全校正。随着策略远离采样策略，旧状态分布与旧优势估计的适用性也会变差。

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

序列 logprob 是回答 token 的条件 logprob 之和。基础 DPO 目标不是逐 token 的 PPO ratio，也不是天然的平均 token logprob。长度、偏好标注、prompt 配对和 response mask 都会影响训练。

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

# 9. 用三个问题区分 PPO、DPO 与 GRPO

| 方法 | 数据与反馈 | advantage / 学习权重 | 典型训练组件 |
| --- | --- | --- | --- |
| PPO-based RLHF / RLVR | 近期策略 rollout，RM 或 verifier 打分 | critic + GAE | actor、old logprob、critic、奖励；reference 可选或按目标加入 |
| 标准离线 DPO | 固定 chosen/rejected 偏好对 | 相对 reference 的序列 logprob 分差 | actor、reference、偏好数据 |
| GRPO | 同 prompt 多个近期 rollout 与 reward | 组内相对 reward | actor、old logprob、奖励；典型形式无 critic |

先问数据来自哪里，再问什么定义好坏，最后问怎样把反馈转成参数更新。不能因为 DPO 没有在线交互就认为它与 RL 目标毫无关系，也不能因为 GRPO 省略 critic 就认为它不需要信用分配。

# 10. Agentic RL：模型行动后，外部世界会改变

单轮文本生成主要沿前缀展开。agent 可以查询数据库、读取文件、执行代码、搜索网页、提交编辑，再根据观察继续决策：

```text
任务 → 模型输出动作 → harness 解析与执行 → 环境变化
                 ↑                         ↓
                 └──── 新上下文/观察 ────────┘
                            ↓
                    成功、失败或预算终止
```

## 10.1 区分真实 state 与模型 observation

设潜在 state $z_k$ 包含文件/数据库状态、任务目标、工具内部状态和 harness 控制状态。环境转移为 $z_{k+1}\sim P(\cdot\mid z_k,a_k)$。模型收到的上下文 $c_k$ 通常只是部分信息，并可能经过截断、检索、摘要或记忆更新。

策略实际是 $\pi_\theta(a_k\mid c_k)$。完整历史可用于 POMDP 的历史条件策略，但有限窗口或摘要未必是充分统计量，不能直接声称“prompt 就是完整 Markov state”。

动作也有两个时间尺度：生成 token 是微观动作，完整 tool call 或一次模型响应是宏观动作。宏观动作可能消耗不同数量 token 或真实时间。若要折扣调用时间/持续时间，可用半马尔可夫视角；若只优化固定预算下成功率，也可以选择不按 token 折扣。必须先定义奖励和时间单位。

## 10.2 训练哪个 policy

“训练 agent”可能是更新工具调用 LLM、更新 router、更新 planner，或选择性训练多个角色。第一轮实验应明确哪些输出来自待训练 policy，哪些来自固定程序、其它模型或工具。改变提示词、检索和记忆也能提高 agent，但若没有更新 policy 参数，应与模型 RL 训练区分报告。

# 11. 奖励与信用分配：最终成功怎样传到具体行动

## 11.1 Outcome、process 与成本

Outcome reward 在结尾检查任务成功；process reward 对中间进展评分。前者易于定义但稀疏，后者更密集但可能把错误过程或评审偏好写进目标。

可以设：

$$
R(\tau)=w_s\,\mathbf1[\text{success}]
-w_c\,\operatorname{cost}(\tau)-w_v\,\operatorname{violations}(\tau).
$$

这是设计选择而非通用最佳公式。权重可能让模型宁可快速失败，也不尝试昂贵但必要的操作。对于必须满足的约束，可以考虑硬性失败规则或约束优化，并分别报告成功、成本和违规率，避免总分掩盖问题。

若额外加入启发式奖励，不自动保证原最优策略不变。理论上的 potential-based shaping 采用 $r'_{k+1}=r_{k+1}+\gamma\Phi(z_{k+1})-\Phi(z_k)$，其保持策略性质还依赖终止处理等条件；把“调用工具次数”或“写了规划”直接加分则是另一个目标。[Reward shaping 原始论文，Ng、Harada 与 Russell，ICML 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)。

## 11.2 将完整任务回报分给所有调用

简单起点是所有待训练调用共享最终 $R(\tau)-b$。即使中间工具和 harness 不可微，若它们的转移机制不依赖训练参数，轨迹概率中的参数梯度仍来自模型动作项：

$$
\nabla_\theta\log P_\theta(\tau)
=\sum_{k\in\mathcal C_\theta}\nabla_\theta\log\pi_\theta(a_k\mid c_k).
$$

这解释了为何可以用最终结果训练多次调用，但成功轨迹里冗余或错误的动作也会一起得到正权重。更细的方案可用调用级 critic、局部可验证奖励、重置到中间状态的分支 rollout，或反事实比较。每种方案都会增加假设、计算成本或评估噪声。

Agent Lightning 原始工作将执行与训练解耦，并分解为模型调用级 transition；其报告的实现将最终回报分配给各调用。该接口并不意味着长时程信用分配已被解决。[Agent Lightning，Luo 等，2025 预印本](https://arxiv.org/abs/2508.03680)。

## 11.3 回合、调用、token 三个粒度

```text
一个 rollout：完整任务 τ
  ├─ call 1：prompt → assistant tokens → tool action
  ├─ 环境观察
  ├─ call 2：新 prompt → assistant tokens → 下一动作
  └─ call 3：新 prompt → 最终回答

回报按 rollout 验证
信用可按 rollout / call / token 分配
最终 policy loss 落在被训练模型生成的 action tokens 上
```

归一化要单独决定：等权 rollout、等权 call 和等权 token 对应不同训练权重。一个 agent 因重复失败产生十次调用，不能仅因为拆成十个训练 sample 就被无意赋予另一条两次调用轨迹的五倍权重。

# 12. Harnessed Agentic RL：保持实际调用上下文

harness 管理工具、上下文、记忆、子任务、错误恢复和终止条件。采用 harnessed 训练时，部署的 harness 继续拥有执行循环，trainer 通过模型服务边界收集并优化调用。

HTML 提及的 [Agent Lightning v1.0，He 等，2026-08 预印本](https://arxiv.org/abs/2608.17528)已核对。论文强调 token 重编码、sample 合并、advantage 和 loss normalization 等问题；以下用教学例子解释它们对训练的影响，并不把该框架的所有实现选择视为唯一解。

## 12.1 字符串相同，不保证 token 前缀相同

一次调用产生 tokens $a_1$，下一次 prompt 由聊天模板重新序列化。即使显示文本相同，role marker、模板、边界分词或摘要也可能改变 token IDs。若前后 token-prefix 不连续，就不能把调用直接拼成一条保持原采样条件的序列。

训练概率比必须在原动作的实际条件上下文计算：

$$
\rho_{k,t}=\exp\left(
\log\pi_\theta(a_{k,t}\mid c_k,a_{k,<t})
-\log\pi_{\mathrm{beh}}(a_{k,t}\mid c_k,a_{k,<t})\right).
$$

不能拿重建后另一个 $c'_k$ 的 current logprob 与旧 $c_k$ 下的 behavior logprob 相减。应保存精确 prompt/action tokens 和采样设置；只有兼容时才合并，否则保留为独立调用样本并维护 rollout 关联。

## 12.2 GRPO 的 group 应保留任务身份

设同一初始任务生成两条 rollout，奖励分别是 $1$ 与 $0$；成功轨迹拆成十个 samples，失败轨迹拆成两个。按 samples 计算均值会得到 $10/12$；按 rollout 计算则是 $1/2$。前者让上下文拆分方式改变了 baseline。

一种可解释的起点是按相同初始任务与可比环境状态组织 rollout group，先计算 rollout-level advantage，再分配给对应调用/token；loss weighting 另行指定。随机环境中，即使初始条件相同，结果仍有转移噪声，应保留 seed 并用多次运行评估。

# 13. 从轨迹到训练张量：mask、logprob 与数据契约

## 13.1 哪些 token 参与 loss

| token 类型 | 模型能否作为上下文读取 | 待训练 actor 的 policy loss |
| --- | --- | --- |
| system / user prompt | 能 | 通常为 0 |
| 待训练模型生成的 reasoning / answer | 能 | 按训练设定为 1 |
| 待训练模型生成的工具名称/参数 | 能 | 通常为 1 |
| tool observation / 环境反馈 | 能 | 为 0 |
| 固定外部模型输出 | 能 | 对当前 actor 通常为 0 |
| padding | 应被 attention mask 屏蔽 | 为 0 |

loss mask 与 attention mask 是不同对象。环境文本不参与 actor loss，仍可能是后续决策必需的上下文。工具反馈由环境产生，不是当前 actor 采样的动作。多轮训练的实际 mask 和模板一致性可参考 [verl multi-turn 官方文档](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)。

## 13.2 关键形状与移位

令 $B$ 为调用样本数，$L$ 为 padded sequence 长度，$K$ 为词表大小：

| 张量 | 典型形状 | 含义 |
| --- | --- | --- |
| `input_ids`、attention mask | $(B,L)$ | 实际序列与可见位置 |
| logits | $(B,L,K)$ | 每个位置预测下一个 token |
| sampled-token logprob | $(B,L-1)$ | `logits[:, :-1]` gather `input_ids[:, 1:]` |
| old/ref/current logprob | $(B,L-1)$ | 相同 token 与条件上下文的三种概率 |
| action loss mask | $(B,L-1)$ | 按被预测 token 对齐的参与更新位置 |
| advantage / value target | 与 action positions 对齐 | 广播或逐位置估计的学习信号 |

常见错误是 logits 未移位、mask 仍对齐输入位置，导致训练了错误 token；另一个错误是在相同 prompt 复制出的上下文上重复计算前几次 assistant 输出的 loss。

## 13.3 rollout 至少应该保存什么

```text
task_id / rollout_id / group_id / call_id / parent_call_id
environment_version / harness_version / tool_schema_version
policy_version / tokenizer_version / chat_template_version
exact_prompt_token_ids / sampled_action_token_ids
behavior_logprobs / temperature / sampling_configuration
action_loss_mask / tool_calls / tool_observations
reward_components / success / termination_reason / costs
seed / retry_id / timestamps
```

采样 temperature、top-p、top-k 会改变行为分布。必须明确 old logprob 表示原模型分布还是实际采样分布，以及 trainer 如何处理两者差异；学习阶段可以先用简单、可核对的采样设置。LoRA 只改变更新参数的方式，不替代 PPO/GRPO 的目标和数据语义。

对无法获取权重或训练接口的黑盒模型，可以收集评估轨迹或优化外层策略，但不能假设能按本篇公式直接更新其参数。

# 14. 训练系统：rollout 与 optimizer 如何接起来

```text
任务采样与环境 reset
  → 冻结行为策略版本
  → 并行 rollout：LLM ↔ harness ↔ 工具环境
  → verifier 检查结果与约束
  → 构造 token/mask/logprob 对齐的 samples
  → 计算 advantages 与训练权重
  → minibatch 更新 policy（PPO 时还更新 critic）
  → 同步推理模型权重
  → 在独立任务上评估并保存 checkpoint
```

同步模式最容易确认哪个策略生成哪批数据。异步模式让推理、工具执行和训练重叠，提高吞吐，但 rollout 可能来自较旧模型。应记录 policy version、限制 staleness，并定义接受、丢弃或校正旧数据的规则。clipping 本身不足以保证任意陈旧轨迹有效。

reset 应恢复真实初始状态。训练任务若共享被上一条轨迹污染的数据库或文件系统，会改变任务分布。并行执行需隔离沙箱；网络重试还应处理去重与幂等，避免同一环境动作重复执行、同一轨迹重复计数。

## 14.1 看哪些指标才能判断真的变好

| 观察维度 | 指标与问题 |
| --- | --- |
| 任务能力 | 独立任务 success rate / pass@1，是否真的完成目标？ |
| 探索 | group reward 方差、全失败/全成功比例，是否有对比信号？ |
| 更新稳定性 | KL、ratio、clip fraction、entropy、gradient norm |
| 成本 | 输出 token、工具调用、时间、超时率 |
| 奖励可靠性 | reward 分项、隐藏检查、hacking 案例 |
| 泛化 | 新任务、扰动环境、改变模板后的表现 |
| 系统一致性 | stale rollout 比例、模板不一致、去重/拼接失败 |

PPO 中再关注 critic loss 与 explained variance；GRPO 中关注组内统计和长度分布。reward 上升而独立成功率不升，可能是评分漏洞、过拟合或评估条件不同。

pass@1、pass@k 和 best-of-N 使用不同采样预算。更高测试时预算带来的改善应与模型权重学习分开分析，比较必须匹配推理预算、工具权限与 harness。

# 15. 为后续训练准备的最小实践路线

以下是按依赖关系安排的学习练习，不是未经测量的算力承诺。

| 阶段 | 最小练习 | 完成后应该解释清楚 |
| --- | --- | --- |
| GAE / PPO | 小离散环境手算 GAE，再跑 mini PPO | advantage、old logprob、clip 正负分支、terminal mask |
| LLM 张量 | 小模型给固定回答计算并核对 logprob | token shift、回答 mask、序列与 token logprob |
| 偏好优化 | 小批 chosen/rejected 数据做 DPO | reference、分差、独立偏好评估 |
| RLVR / GRPO | 自动验证的数学或代码短任务 | group advantage、零方差组、长度权重 |
| Agentic RL | 小型文件/SQL 沙箱里的多步工具任务 | reset、观测、动作、调用级记录、独立 verifier |

## 15.1 一个贯穿例子：在沙箱文件中查找并更新数据

任务为读取虚构库存数据、计算指定条件下的总量、将结果写入指定文件。工具限定为列文件、读文件、执行受限计算、写结果；每个 rollout 从相同只读种子数据和新输出目录开始。

verifier 从隔离的任务定义重新计算正确答案，检查输出路径、数值和输入未被非法修改。固定最大调用/token 预算，最终输出存在但答案错误仍为失败。隐藏 verifier 不由 agent 修改。

先测量初始 SFT policy 成功率。如果工具格式几乎全部错误，应先用少量示范建立能力；若已有部分成功，再为同一任务生成多个 rollout，使用 outcome reward 做 GRPO 起点。保留失败轨迹、调用日志与停止原因。

首先比较原 SFT、成功轨迹筛选后 SFT、同预算 RL。再按实际失败分析决定是否引入调用级 critic、process reward、课程或更细信用分配。固定训练/验证/测试任务划分，既测相似新任务，也测文件名、布局、工具错误等扰动。

## 15.2 首次完整训练前的检查

1. 用同一个样本在推理端和训练端比较实际 token IDs 与条件 logprob。
2. 手工检查一个多轮样本，确认工具观察可读但不参与 actor loss。
3. 初始 current 与 behavior 相同时，ratio 应接近 $1$；误差超出预期时查上下文、采样和数值后端。
4. 手算一组奖励与 advantage，检查全相同奖励、截断和终止边界。
5. 独立运行 verifier，测试伪造结果、修改输入、格式错误与超时。
6. 明确按 rollout、call 还是 token 加权，并确认多调用轨迹权重符合设计。

这些检查比一开始同时尝试大量算法变体更容易定位失败来源。

# 16. 论文与官方资料：按知识依赖阅读

以下论文均链接到原始来源。表中年份使用首次预印本年份，GAE 另标正式 ICLR 发表信息；其它预印本不在本文中推断其最终录用状态。本文的手算例子、数据契约与实践路线为教学综合，不是任何一篇论文的完整复现。

| 材料 | 作者 / 年份与状态 | 解决的问题与阅读重点 |
| --- | --- | --- |
| [GAE](https://arxiv.org/abs/1506.02438) | Schulman 等，2015 预印本；ICLR 2016 | 指数加权 TD residual，理解偏差/方差与边界条件 |
| [TRPO](https://arxiv.org/abs/1502.05477) | Schulman 等，2015 预印本 | 用策略距离约束解释稳定更新的动机 |
| [PPO](https://arxiv.org/abs/1707.06347) | Schulman 等，2017 预印本 | clipped surrogate 与一批数据的多轮更新 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | Ouyang 等，2022 预印本 | SFT、偏好数据、reward model 与 PPO 组成完整流程 |
| [DPO](https://arxiv.org/abs/2305.18290) | Rafailov 等，2023 预印本 | KL 正则目标到偏好分类 loss 的解析连接 |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | Shao 等，2024 预印本 | GRPO 如何以组内统计替代 learned critic |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | DeepSeek-AI，2025 技术报告 | R1-Zero 与 R1 的训练流程、规则奖励与边界 |
| [DAPO](https://arxiv.org/abs/2503.14476) | Yu 等，2025 预印本 | 长序列 RL 的采样与训练设计 |
| [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) | Liu 等，2025 预印本 | base model、reward 与归一化对训练的影响 |
| [Agent Lightning](https://arxiv.org/abs/2508.03680) | Luo 等，2025 预印本 | 将 agent execution 与 RL trainer 解耦，关注 credit 的实现限制 |
| [Agent Lightning v1.0](https://arxiv.org/abs/2608.17528) | He 等，2026 预印本 | harness 下 token 连续性、样本组织与训练权重 |
| [TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer) | 官方动态文档 | 将公式映射到实际 loss、归一化与监控配置 |
| [verl 多轮文档](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html) | 官方动态文档 | multi-turn rollout、assistant mask 与模板一致性 |

阅读时可反复对照五个对象：**谁产生数据、什么定义 reward、谁给出 advantage、哪些 action 参与更新、训练与部署的上下文是否一致**。这五个对象共同决定一次后训练实验究竟在学什么。

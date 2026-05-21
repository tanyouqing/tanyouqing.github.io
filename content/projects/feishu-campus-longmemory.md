---
title: "飞书 LongMemory"
date: "2026-05-21"
description: "面向飞书与 OpenClaw 的个人工作记忆中间层，支持真实事件采集、LLM 记忆抽取、用户画像建模、上下文注入与主动提醒闭环。"
tags: ["Agent", "飞书", "记忆系统"]
tech: ["FastAPI", "SQLite", "OpenClaw", "Feishu", "LLM"]
github: "https://github.com/tanyouqing/feishu-campus-longmemory"
image: "🧠"
---
## 项目介绍

飞书 LongMemory 是一个面向飞书与 OpenClaw Agent Runtime 的个人工作记忆中间层，目标是让 AI 办公助手不再只停留在单轮问答，而是能够从真实办公交互中沉淀长期偏好、用户画像与主动提醒能力。

项目围绕“真实事件 -> 证据存储 -> 记忆/画像抽取 -> 上下文注入 -> 个性化执行 -> 主动服务”的闭环展开。用户在飞书或 OpenClaw 中产生的消息会先进入 Evidence Store，再由规则与 LLM 抽取器生成候选记忆。系统会对候选内容进行证据片段校验、敏感信息过滤和结构化约束，最终写入个人记忆或用户画像，并在 OpenClaw 构建 Prompt 前注入相关上下文。

当前版本已经完成 P0 闭环与用户建模能力，支持周报偏好记忆、用户画像注入、工作提醒等核心场景。

## 核心模块

- **Evidence Store** — 将飞书与 OpenClaw 的真实事件标准化为 `WorkEvent`，完成幂等写入、隐私脱敏与证据追踪。
- **Personal Memory** — 支持工作偏好、提醒偏好等长期记忆的写入、更新、忘记、检索和审计。
- **LLM 抽取层** — 使用 OpenAI-compatible 接口抽取记忆候选，LLM 只生成候选 JSON，最终由后端校验后落库。
- **用户画像建模** — 从真实交互中抽取工作身份、当前阶段、沟通风格、工具使用偏好等画像 patch，并渲染为可注入 Prompt 的 Profile Context。
- **OpenClaw Context 插件** — 在 `before_prompt_build` 阶段调用 `/context/build`，将 User Profile Context 与 Memory Context Pack 注入模型回复前上下文。
- **主动提醒闭环** — 将提醒偏好持久化为 `reminder_jobs`，到期后通过飞书 Bot 真实发送提醒，并将成功或失败结果回写 Evidence。

## 技术亮点

- **Evidence-first 架构** — 所有长期记忆和用户画像都必须关联真实事件证据，避免黑盒式“凭空记忆”。
- **LLM as Candidate Generator** — LLM 只负责语义候选抽取，后端负责类别、置信度、证据片段和敏感信息校验，兼顾智能性与稳定性。
- **规则 fallback** — 当 LLM 不可用、超时或输出不合规时，系统自动回退到规则抽取链路，保证核心能力可持续运行。
- **用户画像 + 记忆双层上下文** — `user_profiles` 维护长期用户建模，`personal_memories` 维护离散工作偏好，二者通过 `/context/build` 组合成短小可控的模型上下文。
- **真实平台闭环** — 项目接入飞书事件、OpenClaw Hook、OpenClaw 插件和飞书 Bot 主动发送，不依赖 mock 事件作为正式能力证明。
- **可解释与可管理** — 支持记忆更新、软删除、提醒取消、审计日志和证据链查询，适合真实办公场景中的长期维护。

## 工程实现

后端使用 Python 3.11 + FastAPI 构建，SQLite 作为 P0 阶段的本地存储，Alembic 管理数据库迁移。核心表包括 `work_events`、`personal_memories`、`memory_evidence_links`、`reminder_jobs`、`memory_audit_logs`、`user_profiles` 和 `user_profile_evidence_links`。

项目提供飞书事件回调、OpenClaw 事件写入、记忆检索、用户画像查询、上下文构建和主动提醒触发等接口，并通过 pytest 覆盖事件入库、记忆抽取、LLM fallback、用户画像、Context Pack、主动提醒和安全边界等关键场景。

## 项目价值

这个项目尝试回答一个实际问题：AI 办公助手如何在尊重隐私和可解释性的前提下，长期理解用户？

相比把聊天记录直接塞进 Prompt，飞书 LongMemory 更强调证据、结构化和可控注入。它让 Agent 能够记住用户的工作偏好，理解用户当前的工作状态，并在合适的时候主动提醒，从而把一次性问答推进到可持续演进的个人办公智能体。

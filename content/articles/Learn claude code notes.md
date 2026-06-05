---
title: "Claude Code 学习笔记"
date: "2026-06-05"
description: "从 agent loop、工具调用、上下文、权限、记忆、子代理与自动化等角度拆解 Claude Code 的基本心智模型。"
tags: ["Claude Code", "Agent", "LLM"]
category: "笔记"
---

# Claude Code 基本心智模型学习笔记

这篇笔记的目标是建立一套理解 Claude Code 的基本心智模型：语言模型负责推理，工具负责行动，执行环境承载副作用，上下文系统负责让模型知道自己做过什么、接下来该做什么。

Claude Code 可以理解为一个围绕模型搭建的 **agentic harness**。它把普通的对话模型放进一个可操作代码仓库、可读取文件、可运行命令、可验证结果的循环里，于是“回答问题”变成了“持续观察、行动、校正”。官方文档也把这个过程概括为：收集上下文、采取行动、验证结果，并在任务完成前持续循环。

## 阅读主线

- **Agent Loop**：模型每一轮都在“看见状态 → 选择工具 → 接收结果 → 再决策”的循环中推进。
- **工具调用**：工具不是附属功能，而是模型接触真实世界的手臂；文件读写、命令执行、搜索、测试都被统一成工具。
- **上下文管理**：上下文窗口是工作记忆，文件、任务列表、日志和摘要是外部记忆；长期任务必须把关键状态写到模型之外。
- **权限与安全**：能力和授权要分离；工具能做什么是一回事，什么时候允许做是另一回事。
- **扩展层**：CLAUDE.md、Skills、MCP、Hooks、Subagents、Plugins 等机制都可以看作叠在核心 agent loop 之上的扩展层。
- **验证闭环**：真正可靠的代码代理不是“生成代码”，而是能在执行后检查、根据反馈修正，并把风险显式暴露给人。

## 最小心智模型

| 层级 | 负责什么 | 常见表现 | 学习重点 |
| --- | --- | --- | --- |
| 模型层 | 推理、规划、解释、选择下一步 | 读懂需求、拆任务、判断失败原因 | 不把模型当数据库，而把它当决策器 |
| 工具层 | 把决策转成真实动作 | 读文件、改文件、跑测试、查资料 | 工具输入输出必须结构化、可追踪 |
| 状态层 | 保存当前任务的工作记忆 | messages、todo、session、日志 | 区分短期上下文和长期外部状态 |
| 权限层 | 控制副作用边界 | ask/allow/deny、只读/可写模式 | 默认可审计，高风险操作要显式确认 |
| 扩展层 | 让 agent 适应项目和团队 | CLAUDE.md、skills、hooks、subagents | 把重复流程固化为可复用约定 |

## 与 Claude Code 官方概念对齐

| 官方概念 | 在心智模型里的位置 | 如何理解 |
| --- | --- | --- |
| Terminal / IDE / Desktop / Web | 交互界面 | 界面决定你如何发起任务、查看 diff、打断或继续；底层仍然是同一类 agent loop。 |
| Local / Cloud / Remote Control | 执行环境 | 决定代码、命令和副作用发生在哪里；本地执行最贴近你的真实开发环境，云端更适合把任务外包出去跑。 |
| Session | 状态容器 | 一次任务会话保存消息、工具调用和结果；新会话通常从新的上下文开始，需要通过文件、记忆或项目说明延续知识。 |
| CLAUDE.md | 项目级长期上下文 | 用来写项目约定、运行命令、测试方式、代码风格和团队偏好，是让模型少猜的第一层配置。 |
| Skills / Commands | 可复用工作流 | 把重复提示词、参考资料、脚本和步骤打包，让模型在合适触发条件下加载。 |
| MCP | 外部工具连接层 | 把 GitHub、数据库、浏览器、内部系统等外部能力暴露为结构化工具。 |
| Hooks | 生命周期自动化 | 在工具调用前后、会话事件或提交前后触发脚本、HTTP 请求或校验逻辑。 |
| Subagents | 隔离的专业执行者 | 为测试、审查、搜索、迁移等子任务创建独立上下文，最后只把结论带回主会话。 |

## 建议的学习路径

1. 先理解 **Agent Loop**：Claude Code 为什么能多轮行动，而不是一次性吐出答案。
2. 再理解 **工具协议**：工具调用为什么必须有输入 schema、调用 ID、结果回传和错误处理。
3. 接着看 **上下文压缩与外部记忆**：长任务为什么不能只依赖聊天历史。
4. 然后看 **权限系统**：可执行命令的 agent 为什么必须有 allow/ask/deny 边界。
5. 最后看 **扩展机制**：什么时候该写 CLAUDE.md，什么时候该用 Skill、Hook、MCP 或 Subagent。

下面的代码和结构都是学习用的抽象化模型。它们刻意保留了“骨架感”，方便理解 Claude Code 类工具为什么能从一次聊天，演化成一个可持续推进工程任务的代理系统。

## Agent Loop
```python
def agent_loop(state):
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=state["messages"],
            tools=TOOLS,
            max_tokens=8000,
        )

        state["messages"].append({
            "role": "assistant",
            "content": response.content,
        })

        if response.stop_reason != "tool_use":
            # transition_reason 暂时理解成这一轮结束后，为什么要继续下一轮
            state["transition_reason"] = None
            return
        
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        state["messages"].append({"role": "user", "content": results})
        state["turn_count"] += 1
        state["transition_reason"] = "tool_result"
```

## 工具调用
1. dispatch map 将工具名映射到处理函数。

```python
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"],
                                        kw["new_text"]),
}
```

2. 循环中按名称查找处理函数。循环体本身与 s01 完全一致。

```python
for block in response.content:
    if block.type == "tool_use":
        handler = TOOL_HANDLERS.get(block.name)
        output = handler(**block.input) if handler \
            else f"Unknown tool: {block.name}"
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
```

- `**` 是 Python 中的 **双星号参数**，用于解包字典作为关键字参数

### 消息规范化
- 在每次 API 调用前运行

#### 步骤
1. 剥离内部字段
2. tool_result 配对补齐
3. 合并连续同角色信息

```python
def normalize_messages(messages: list) -> list:
    """将内部消息列表规范化为 API 可接受的格式。"""
    normalized = []

    for msg in messages:
        # Step 1: 剥离内部字段
        clean = {"role": msg["role"]}
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            clean["content"] = [
                {k: v for k, v in block.items()
                 if k not in ("_internal", "_source", "_timestamp")}
                for block in msg["content"]
            ]
        normalized.append(clean)

    # Step 2: tool_result 配对补齐
    # 收集所有已有的 tool_result ID
    existing_results = set()
    for msg in normalized:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    existing_results.add(block.get("tool_use_id"))

    # 找出缺失配对的 tool_use, 插入占位 result
    for msg in normalized:
        if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if (block.get("type") == "tool_use"
                        and block.get("id") not in existing_results):
                    # 在下一条 user 消息中补齐
                    normalized.append({"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "(cancelled)",
                    }]})

    # Step 3: 合并连续同角色消息
    merged = [normalized[0]] if normalized else []
    for msg in normalized[1:]:
        if msg["role"] == merged[-1]["role"]:
            # 合并内容
            prev = merged[-1]
            prev_content = prev["content"] if isinstance(prev["content"], list) \
                else [{"type": "text", "text": prev["content"]}]
            curr_content = msg["content"] if isinstance(msg["content"], list) \
                else [{"type": "text", "text": msg["content"]}]
            prev["content"] = prev_content + curr_content
        else:
            merged.append(msg)

    return merged
```

## 待办写入
- 维护一个当前会话的外显列表
- 最小条目的属性有：内容、状态（待办、过程中、已完成）、运行时描述
- 另外需要定义最小运行状态（计划多少轮没有被更新），需要有状态约束（同时间最多几个过程中任务）

```python
@dataclass
class PlanItem:
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")

        normalized = []
        in_progress_count = 0
        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("activeForm", "")).strip()

            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1

            normalized.append(PlanItem(
                content=content,
                status=status,
                active_form=active_form,
            ))

        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")

        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def render(self) -> str:
        if not self.state.items:
            return "No session plan yet."

        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)

        completed = sum(1 for item in self.state.items if item.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)
```

- 将 update 接入工具列表中，在 agent loop 中维护

## Sub Agent
- 将派生子代理作为一个工具
- 子代理使用自己的消息列表，只拿必要的工具（不给继续派生子代理的工具，防止递归），只把结果带回父代理

```python
class SubagentContext:
    messages: list #上下文
    tools: list #工具列表
    handlers: dict #工具路由
    max_turns: int #防止子代理无限运行
```

- fork 功能：复制一份父代理的消息列表，在其基础上添加消息，只将结果摘要返回

```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM, messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})
    # Only the final text returns to the parent -- child context is discarded
    return "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"
```

## 技能系统
### 关键数据结构
```python
@dataclass
class SkillManifest:
    name: str          # 技能名称
    description: str   # 技能描述（轻量信息）
    path: Path         # SKILL.md 文件路径

@dataclass
class SkillDocument:
    manifest: SkillManifest   # 元数据
    body: str                 # 完整技能正文（frontmatter 之后的内容）
```

### 技能注册中心 SkillRegistry
启动时扫描 `skills/` 目录下的所有 `SKILL.md`，解析 frontmatter，构建 `documents` 字典。

```python
class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.documents: dict[str, SkillDocument] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta, body = self._parse_frontmatter(path.read_text())
            name = meta.get("name", path.parent.name)
            description = meta.get("description", "No description")
            manifest = SkillManifest(name=name, description=description, path=path)
            self.documents[name] = SkillDocument(manifest=manifest, body=body.strip())
```

提供给系统的两个核心方法：

- `describe_available()`：返回技能列表（名称+描述），用于嵌入 system prompt。
- `load_full_text(name)`：根据名称返回完整技能正文（用 `<skill>` 标签包裹），如果技能不存在则返回错误信息。

### 系统集成
1. 启动时创建全局注册表

```python
SKILLS_DIR = WORKDIR / "skills"
SKILL_REGISTRY = SkillRegistry(SKILLS_DIR)
```

2. 将技能列表注入 system prompt

```python
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill when a task needs specialized instructions before you act.
Skills available:
{SKILL_REGISTRY.describe_available()}
"""
```

3. 定义 `load_skill` 工具，并绑定处理器

```python
TOOLS = [
    {
        "name": "load_skill",
        "description": "Load the full body of a named skill into the current context.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    # ... 其他工具
]

TOOL_HANDLERS = {
    "load_skill": lambda **kw: SKILL_REGISTRY.load_full_text(kw["name"]),
    # ... 其他处理器
}
```

## 上下文压缩
核心目标：当对话历史过长或工具输出过大时，主动压缩内容，释放上下文空间，同时保留关键信息（任务目标、已完成动作、关键文件、决定约束、下一步计划）。

### 主要工作
1. 大的工具调用结果写进磁盘，上下文中只留预览
2. 旧的结果替换成简短占位
3. 整体历史太长时，生成一份连续性摘要

### 压缩后需要保持的信息
- 当前任务目标
- 已完成的关键动作
- 已修改或重点查看过的文件
- 关键决定与约束
- 下一步应该做什么

### Layer 1: micro_compact —— 替换旧工具结果为占位符
每轮调用模型前执行，只处理 `tool_result` 内容，保留最近 N 轮的完整结果，更早的替换为简短占位符。

```python
def micro_compact(messages: list, keep_recent: int = 3) -> list:
    # 遍历消息，收集所有 tool_result 的位置
    tool_results = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for part_idx, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "tool_result":
                tool_results.append((idx, part_idx, part))
    
    # 如果工具结果数量 <= 保留轮数，无需压缩
    if len(tool_results) <= keep_recent:
        return messages
    
    # 找出需要压缩的旧结果（除最后 keep_recent 个之外）
    to_compress = tool_results[:-keep_recent]
    for msg_idx, part_idx, result in to_compress:
        # 仅压缩内容较长的工具结果
        content = result.get("content", "")
        if isinstance(content, str) and len(content) > 100:
            # 替换为简短占位符，保留 tool_use_id 以便关联
            result["content"] = f"[Compressed tool result from earlier step]"
    
    return messages
```

### 处理大工具结果：写磁盘 + 预览
当单个工具输出极大（如读取大文件或运行长命令）时，先将完整内容写入磁盘，上下文中只保留文件路径和摘要预览。

```python
def run_large_tool(command: str) -> str:
    # 执行工具，获取原始输出
    output = execute_command(command)

    # 如果输出长度超过阈值，则保存到磁盘
    if len(output) > 10000:
        transcript_dir = Path(".transcripts")
        transcript_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        file_path = transcript_dir / f"tool_output_{timestamp}.txt"
        file_path.write_text(output, encoding="utf-8")

        # 返回预览信息，包含文件路径
        preview = output[:500] + f"\n... (full output saved to {file_path})"
        return preview
    else:
        return output
```

### Layer 2: auto_compact —— 生成连续性摘要
当估计 token 数超过阈值（如 50000）时触发。流程：保存完整对话到磁盘 → 调用 LLM 生成摘要 → 用摘要替换全部消息。

```python
def estimate_tokens(messages: list) -> int:
    # 简单近似：总字符数 / 4
    total_chars = sum(len(json.dumps(msg, default=str)) for msg in messages)
    return total_chars // 4

def auto_compact(messages: list, transcript_dir: Path = Path(".transcripts")) -> list:
    # 1. 保存完整对话到磁盘，防止信息永久丢失
    transcript_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    transcript_path = transcript_dir / f"transcript_{timestamp}.jsonl"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")

    # 2. 构建摘要提示词，要求保留关键信息
    summary_prompt = f"""Please compress the above conversation into a concise summary that preserves:
- Current task goal
- Key actions already completed
- Files that have been modified or examined
- Important decisions and constraints
- What should be done next

Output the summary in plain text.
"""
    # 3. 调用 LLM 生成摘要（此处为示意，实际使用模型 API）
    summary = call_llm_for_summary(messages, summary_prompt)

    # 4. 用摘要替换全部消息
    compressed_messages = [
        {"role": "user", "content": f"[Conversation Compressed at {timestamp}]\n{summary}"},
        {"role": "assistant", "content": "I have compressed the context. Please continue from the summary above."}
    ]
    return compressed_messages
```

### Layer 3: compact 工具 —— 模型主动请求压缩
允许模型在运行中主动调用 `compact` 工具，触发压缩。工具定义与处理器：


```python
TOOLS = [
    {
        "name": "compact",
        "description": "Request manual compression of the conversation context. Use when context is too long or messy.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # ... 其他工具
]

def handle_compact() -> str:
    # 调用 auto_compact 并返回确认信息
    global messages
    messages = auto_compact(messages)
    return "Context compressed successfully."

# 在 TOOL_HANDLERS 中注册
TOOL_HANDLERS = {
    "compact": lambda **kw: handle_compact(),
    # ...
}
```

### 主循环中的集成
```python
def agent_loop(messages: list) -> None:
    while True:
        # 1. 每轮先做微压缩
        messages = micro_compact(messages)

        # 2. 检查 token 阈值，必要时自动压缩
        if estimate_tokens(messages) > TOKEN_THRESHOLD:
            messages = auto_compact(messages)

        # 3. 调用模型
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        # 4. 处理工具调用（包括手动 compact）
        # ... 执行工具，若有 compact 则更新 messages
```

## 权限系统
核心设计理念：每次工具调用在执行前都经过一个权限管道，按顺序逐级裁决。_管道顺序为：deny 规则 → mode 检查 → allow 规则 → ask 用户。_

### 三种权限模式
程序初期提供三种清晰模式，便于理解和实现。

- `default`：默认模式，完整走管道（deny → mode → allow → ask），写操作通常需要用户确认。
- `plan`：计划模式，拒绝所有写操作，只允许读操作。用于让模型先规划再执行。
- `auto`：自动模式，读操作自动允许，写操作继续走后续管道（最终可能 ask 用户）。

```python
MODES = ("default", "plan", "auto")

# 只读工具集合
READ_ONLY_TOOLS = {"read_file", "bash_readonly"}

# 写工具集合
WRITE_TOOLS = {"write_file", "edit_file", "bash"}
```

### Bash 安全验证器
在进入规则管道之前，先对 bash 命令进行静态模式检查。检测高危模式，严重模式直接 deny，其他模式转为 ask。

```python
class BashSecurityValidator:
    VALIDATORS = [
        ("shell_metachar", r"[;&|`$]"),       # shell 元字符
        ("sudo", r"\bsudo\b"),                 # 提权
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"),  # 递归删除
        ("cmd_substitution", r"\$\("),          # 命令替换
        ("ifs_injection", r"\bIFS\s*="),        # IFS 注入
    ]

    def validate(self, command: str) -> list:
        # 返回触发规则的 (名称, 模式) 列表
        failures = []
        for name, pattern in self.VALIDATORS:
            if re.search(pattern, command):
                failures.append((name, pattern))
        return failures
```

### 权限管理器
核心类，管理当前模式、规则列表，提供 `check()` 方法按管道顺序裁决。

```python
class PermissionManager:
    def __init__(self, mode: str = "default", rules: list = None):
        self.mode = mode
        self.rules = rules or list(DEFAULT_RULES)
        self.consecutive_denials = 0          # 连续拒绝计数
        self.max_consecutive_denials = 3      # 触发提醒阈值

    def check(self, tool_name: str, tool_input: dict) -> dict:
        # 返回 {"behavior": "allow"|"deny"|"ask", "reason": str}
```

#### 管道步骤详解
```markdown
开始
  │
  ▼
是 bash 工具且验证失败？
  │ ├─ 严重模式 → deny
  │ └─ 其他模式 → ask
  ▼
匹配 deny 规则？ ──是──→ deny
  │否
  ▼
模式检查
  ├─ plan + 写工具 → deny
  ├─ plan + 读工具 → allow
  ├─ auto + 只读工具 → allow
  └─ 其他继续
  ▼
匹配 allow 规则？ ──是──→ allow（清零拒绝计数）
  │否
  ▼
默认 ask
```

**步骤 0：Bash 安全预检**（在规则之前执行）

```python
if tool_name == "bash":
    command = tool_input.get("command", "")
    failures = bash_validator.validate(command)
    if failures:
        severe = {"sudo", "rm_rf"}
        if any(f[0] in severe for f in failures):
            return {"behavior": "deny", "reason": f"Bash validator: {desc}"}
        else:
            return {"behavior": "ask", "reason": f"Bash validator flagged: {desc}"}
```

**步骤 1：deny 规则**（不可绕过，最先匹配）

```python
for rule in self.rules:
    if rule["behavior"] != "deny":
        continue
    if self._matches(rule, tool_name, tool_input):
        return {"behavior": "deny", "reason": f"Blocked by deny rule: {rule}"}
```

**步骤 2：模式检查**

```python
if self.mode == "plan":
    if tool_name in WRITE_TOOLS:
        return {"behavior": "deny", "reason": "Plan mode: write operations are blocked"}
    return {"behavior": "allow", "reason": "Plan mode: read-only allowed"}

if self.mode == "auto":
    if tool_name in READ_ONLY_TOOLS or tool_name == "read_file":
        return {"behavior": "allow", "reason": "Auto mode: read-only tool auto-approved"}
    # 写操作继续落到 allow 规则或 ask
```

**步骤 3：allow 规则**

```python
for rule in self.rules:
    if rule["behavior"] != "allow":
        continue
    if self._matches(rule, tool_name, tool_input):
        self.consecutive_denials = 0
        return {"behavior": "allow", "reason": f"Matched allow rule: {rule}"}
```

**步骤 4：默认 ask**

```python
return {"behavior": "ask", "reason": f"No rule matched for {tool_name}, asking user"}
```

#### 规则匹配方法
支持按 `tool` 名称、`path` 通配符（glob）、`content` 通配符（用于 bash 命令）进行匹配。

```python
def _matches(self, rule: dict, tool_name: str, tool_input: dict) -> bool:
    if rule.get("tool") and rule["tool"] != "*":
        if rule["tool"] != tool_name:
            return False
    if "path" in rule and rule["path"] != "*":
        path = tool_input.get("path", "")
        if not fnmatch(path, rule["path"]):
            return False
    if "content" in rule:
        command = tool_input.get("command", "")
        if not fnmatch(command, rule["content"]):
            return False
    return True
```

#### 用户交互
当管道裁决为 `ask` 时，调用 `ask_user` 方法向用户请求批准。支持 `y/n/always`，其中 `always` 会动态添加一条 allow 规则。

```python
def ask_user(self, tool_name: str, tool_input: dict) -> bool:
    preview = json.dumps(tool_input, ensure_ascii=False)[:200]
    print(f"\n  [Permission] {tool_name}: {preview}")
    answer = input("  Allow? (y/n/always): ").strip().lower()
    if answer == "always":
        self.rules.append({"tool": tool_name, "path": "*", "behavior": "allow"})
        return True
    if answer in ("y", "yes"):
        return True
    self.consecutive_denials += 1
    if self.consecutive_denials >= self.max_consecutive_denials:
        print("  [连续拒绝多次，考虑切换到 plan 模式]")
    return False
```

#### 默认规则
```python
DEFAULT_RULES = [
    {"tool": "bash", "content": "rm -rf /", "behavior": "deny"},
    {"tool": "bash", "content": "sudo *", "behavior": "deny"},
    {"tool": "read_file", "path": "*", "behavior": "allow"},
]
```

### 工具执行时的权限集成
在 `agent_loop` 中，每检测到一个 `tool_use` 块，就调用 `perms.check()` 并根据行为分支：

```python
decision = perms.check(block.name, block.input or {})
if decision["behavior"] == "deny":
    output = f"Permission denied: {decision['reason']}"
elif decision["behavior"] == "ask":
    if perms.ask_user(block.name, block.input or {}):
        handler = TOOL_HANDLERS.get(block.name)
        output = handler(**(block.input or {}))
    else:
        output = f"Permission denied by user for {block.name}"
else:  # allow
    handler = TOOL_HANDLERS.get(block.name)
    output = handler(**(block.input or {}))
```

## Hooks
核心理念：在主循环的关键位置预留扩展点（SessionStart, PreToolUse, PostToolUse），外部脚本通过标准接口注入行为，无需改动 `agent_loop` 本身。扩展代码与核心逻辑分离，实现“开闭原则”。

### 三种事件类型
- **SessionStart**：会话启动时触发，可执行初始化脚本。
- **PreToolUse**：工具执行前触发，可修改输入、阻止执行或注入消息。
- **PostToolUse**：工具执行后触发，可处理输出、注入额外上下文。

### 退出码契约
Hook 脚本通过退出码与系统通信，标准输出（stdout）和标准错误（stderr）按约定使用：

| 退出码 | 含义 | 行为 |
| --- | --- | --- |
| 0 | 继续 | 正常执行，stdout 可包含 JSON 扩展指令（如 `updatedInput`、`additionalContext`、`permissionDecision`） |
| 1 | 阻断 | 阻止当前操作，stderr 内容作为阻断原因返回给模型 |
| 2 | 注入消息 | 不阻断执行，但将 stderr 内容作为额外消息注入上下文 |


### HookManager 核心实现
```python
class HookManager:
    def __init__(self, config_path: Path = None, sdk_mode: bool = False):
        self.hooks = {"PreToolUse": [], "PostToolUse": [], "SessionStart": []}
        # 从 .hooks.json 加载配置
        config = json.loads(config_path.read_text())
        for event in HOOK_EVENTS:
            self.hooks[event] = config.get("hooks", {}).get(event, [])

    def run_hooks(self, event: str, context: dict = None) -> dict:
        # 返回 {"blocked": bool, "messages": list[str]}
        # 信任检查：仅当工作区已信任或 SDK 模式时执行 hooks
        if not self._check_workspace_trust():
            return {"blocked": False, "messages": []}
        
        for hook_def in self.hooks.get(event, []):
            # matcher 按工具名过滤（仅 PreToolUse/PostToolUse）
            if matcher and context and matcher != "*" and matcher != tool_name:
                continue
            
            # 设置环境变量传递上下文
            env["HOOK_EVENT"] = event
            env["HOOK_TOOL_NAME"] = context.get("tool_name", "")
            env["HOOK_TOOL_INPUT"] = json.dumps(context.get("tool_input", {}))
            
            r = subprocess.run(command, shell=True, env=env, 
                               capture_output=True, text=True, timeout=30)
            
            if r.returncode == 0:
                # 尝试解析 stdout JSON 中的扩展字段
                hook_output = json.loads(r.stdout)
                if "updatedInput" in hook_output and context:
                    context["tool_input"] = hook_output["updatedInput"]
                if "additionalContext" in hook_output:
                    result["messages"].append(hook_output["additionalContext"])
            elif r.returncode == 1:
                result["blocked"] = True
                result["block_reason"] = r.stderr.strip()
            elif r.returncode == 2:
                result["messages"].append(r.stderr.strip())
```

### 在 agent_loop 中的集成点
```python
def agent_loop(messages: list, hooks: HookManager):
    while True:
        # 模型响应包含 tool_use
        for block in response.content:
            if block.type != "tool_use":
                continue
            ctx = {"tool_name": block.name, "tool_input": block.input}

            # PreToolUse hooks
            pre = hooks.run_hooks("PreToolUse", ctx)
            for msg in pre.get("messages", []):
                results.append(tool_result(msg))           # 注入消息
            if pre.get("blocked"):
                results.append(tool_result(f"Blocked: {reason}"))
                continue                                   # 跳过工具执行

            # 执行工具
            output = handler(**ctx["tool_input"])

            # PostToolUse hooks
            ctx["tool_output"] = output
            post = hooks.run_hooks("PostToolUse", ctx)
            for msg in post.get("messages", []):
                output += f"\n[Hook note]: {msg}"

            results.append(tool_result(output))
```

### 配置文件示例 .hooks.json
```json
{
  "hooks": {
    "SessionStart": [
      {"command": "echo 'Session started' > /tmp/session.log"}
    ],
    "PreToolUse": [
      {"matcher": "bash", "command": "python /scripts/check_safe.py"}
    ],
    "PostToolUse": [
      {"matcher": "write_file", "command": "python /scripts/backup.py"}
    ]
  }
}
```

## Memory 系统
### 核心理念
某些信息应跨越会话边界存在，但不是所有信息都值得记住。

**应该记忆**：用户偏好、反复出现的纠正、无法从代码推导的项目事实、外部资源指针  
**不应记忆**：可从代码重新推导的结构、临时任务状态、密钥凭证

### 存储结构
```plain
.memory/
  MEMORY.md          # 索引文件（轻量摘要，最多200行）
  prefer_tabs.md     # 单个记忆文件
  review_style.md
```

每个记忆文件采用 Frontmatter + Markdown 正文：

```markdown
---
name: prefer_tabs
description: User prefers tabs over spaces
type: user
---
I always want indentation with tabs, not spaces.
```

记忆类型：`user`（偏好）、`feedback`（纠正）、`project`（项目约定）、`reference`（资源指针）

### MemoryManager 工作流程
#### 启动加载
```python
def load_all(self):
    self.memories = {}
    if not self.memory_dir.exists():
        return
    for md_file in sorted(self.memory_dir.glob("*.md")):
        if md_file.name == "MEMORY.md":
            continue
        parsed = self._parse_frontmatter(md_file.read_text())
        if parsed:
            name = parsed.get("name", md_file.stem)
            self.memories[name] = {
                "description": parsed.get("description", ""),
                "type": parsed.get("type", "project"),
                "content": parsed.get("content", ""),
                "file": md_file.name,
            }
```

- 扫描 `.memory/` 下所有 `.md`（排除 `MEMORY.md`）
- 解析 frontmatter，构建内存字典 `{name: {description, type, content}}`

#### 注入系统提示
```python
def load_memory_prompt(self) -> str:
    if not self.memories:
        return ""
    sections = ["# Memories (persistent across sessions)"]
    for mem_type in MEMORY_TYPES:
        typed = {k: v for k, v in self.memories.items() if v["type"] == mem_type}
        if not typed:
            continue
        sections.append(f"## [{mem_type}]")
        for name, mem in typed.items():
            sections.append(f"### {name}: {mem['description']}")
            if mem["content"].strip():
                sections.append(mem["content"].strip())
    return "\n".join(sections)
```

每次调用 LLM 前（`build_system_prompt`）都会重新生成记忆块并插入 system prompt，因此同一会话内新保存的记忆在下轮调用中立即可见。

#### 保存记忆（`save_memory` 工具）
```python
def save_memory(self, name: str, description: str, mem_type: str, content: str) -> str:
    if mem_type not in MEMORY_TYPES:
        return f"Error: type must be one of {MEMORY_TYPES}"
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())
    self.memory_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = (
        f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n{content}\n"
    )
    file_path = self.memory_dir / f"{safe_name}.md"
    file_path.write_text(frontmatter)
    # 更新内存字典
    self.memories[name] = {
        "description": description,
        "type": mem_type,
        "content": content,
        "file": f"{safe_name}.md",
    }
    self._rebuild_index()   # 重建 MEMORY.md 索引
    return f"Saved memory '{name}' [{mem_type}] to {file_path.relative_to(WORKDIR)}"
```

1. 类型校验
2. 名称安全化（生成 `safe_name.md`）
3. 写入独立文件（含 frontmatter）
4. 更新内存字典
5. 重建 `MEMORY.md` 索引（仅保留 name + description + type，限制200行）

```python
def _rebuild_index(self):
    lines = ["# Memory Index", ""]
    for name, mem in self.memories.items():
        lines.append(f"- {name}: {mem['description']} [{mem['type']}]")
        if len(lines) >= MAX_INDEX_LINES:   # 200行限制
            lines.append(f"... (truncated at {MAX_INDEX_LINES} lines)")
            break
    MEMORY_INDEX.write_text("\n".join(lines) + "\n")
```

### Dream Consolidator（异步记忆整理）
可选的后台整理器，在会话间隙执行。设计意图：防止记忆库冗余、冲突、膨胀。

#### 七道门控（全部通过才执行）
```python
def should_consolidate(self) -> tuple[bool, str]:
    now = time.time()
    if not self.enabled:          # 门1
        return False, "disabled"
    if not self.memory_dir.exists() or not memory_files:  # 门2
        return False, "no memory files"
    if self.mode == "plan":       # 门3
        return False, "plan mode"
    if now - self.last_consolidation_time < self.COOLDOWN_SECONDS:  # 门4 (24h)
        return False, "cooldown"
    if now - self.last_scan_time < self.SCAN_THROTTLE_SECONDS:      # 门5 (10min)
        return False, "throttle"
    if self.session_count < self.MIN_SESSION_COUNT:                 # 门6 (5次)
        return False, "insufficient sessions"
    if not self._acquire_lock():   # 门7
        return False, "lock held"
    return True, "ok"
```

#### 锁机制
锁文件 `.memory/.dream_lock` 内容为 `<pid>:<timestamp>`。

```python
def _acquire_lock(self) -> bool:
    if self.lock_file.exists():
        pid_str, timestamp_str = self.lock_file.read_text().strip().split(":", 1)
        pid, lock_time = int(pid_str), float(timestamp_str)
        if (time.time() - lock_time) > self.LOCK_STALE_SECONDS:   # 超时1小时
            self.lock_file.unlink()
        else:
            try:
                os.kill(pid, 0)      # 检查进程是否存活
                return False         # 存活 -> 锁有效
            except OSError:
                self.lock_file.unlink()   # 进程已死 -> 清理锁
    self.lock_file.write_text(f"{os.getpid()}:{time.time()}")
    return True
```

#### 四阶段整理流程
```python
PHASES = [
    "Orient: scan MEMORY.md index for structure and categories",
    "Gather: read individual memory files for full content",
    "Consolidate: merge related memories, remove stale entries",
    "Prune: enforce 200-line limit on MEMORY.md index",
]
```

## System Prompt
### 核心理念
系统提示应从**清晰的模块**组装而成，每个模块有单一来源和单一职责。这样更易于推理、测试和演进。  
关键边界：用 `=== DYNAMIC_BOUNDARY ===` 分隔**静态部分**（可缓存）和**动态部分**（每轮变化）。

### 六层构建管道
`SystemPromptBuilder.build()` 按顺序组装以下部分：

#### 1. 核心指令（Core Instructions）
```python
def _build_core(self) -> str:
    return (
        f"You are a coding agent operating in {self.workdir}.\n"
        "Use the provided tools to explore, read, write, and edit files.\n"
        "Always verify before assuming. Prefer reading files over guessing."
    )
```

基础角色定义和行为约束，通常变化很少。

#### 2. 工具列表（Tool Listing）
```python
def _build_tool_listing(self) -> str:
    lines = ["# Available tools"]
    for tool in self.tools:
        props = tool.get("input_schema", {}).get("properties", {})
        params = ", ".join(props.keys())
        lines.append(f"- {tool['name']}({params}): {tool['description']}")
    return "\n".join(lines)
```

将 `TOOLS` 转换为人类可读列表，展示工具名称、参数和描述。

#### 3. 技能元数据（Skill Listing）
```python
def _build_skill_listing(self) -> str:
    # 遍历 skills/ 目录，解析每个 SKILL.md 的 frontmatter
    # 提取 name 和 description，生成 "- name: description"
```

仅注入技能的**轻量元数据**（名称+描述），完整正文通过 `load_skill` 工具按需加载（s05 概念）。

#### 4. 记忆内容（Memory Section）
```python
def _build_memory_section(self) -> str:
    # 读取 .memory/ 下除 MEMORY.md 外的所有 .md 文件
    # 解析 frontmatter 获取 type, name, description 及正文 body
    # 格式化为 "[{type}] {name}: {description}\n{body}"
```

直接注入**完整记忆内容**（包括正文），非按需加载（s09 简化设计）。

#### 5. CLAUDE.md 链（优先级顺序）
```python
def _build_claude_md(self) -> str:
    sources = []
    # 1. 用户全局 ~/.claude/CLAUDE.md
    # 2. 项目根目录 CLAUDE.md
    # 3. 当前子目录 CLAUDE.md（如果与工作目录不同）
```

所有匹配的文件**全部包含**（非覆盖），按顺序合并。这是 Claude Code 的指令继承机制。

#### 6. 动态上下文（Dynamic Context）
```python
def _build_dynamic_context(self) -> str:
    lines = [
        f"Current date: {datetime.date.today().isoformat()}",
        f"Working directory: {self.workdir}",
        f"Model: {MODEL}",
        f"Platform: {os.uname().sysname}",
    ]
    return "# Dynamic context\n" + "\n".join(lines)
```

每轮变化的信息（日期、平台等），放在 `DYNAMIC_BOUNDARY` 之后，便于缓存静态前缀。

### 边界标记与缓存优化
```python
DYNAMIC_BOUNDARY = "=== DYNAMIC_BOUNDARY ==="

def build(self) -> str:
    # ... 添加静态部分（1-5）
    sections.append(DYNAMIC_BOUNDARY)
    sections.append(self._build_dynamic_context())
    return "\n\n".join(sections)
```

- 静态部分（1-5）在会话中**几乎不变**，生产环境可缓存其 token 表示，避免重复计算。
- 动态部分每轮重新生成，单独追加。

### 每轮动态提醒（System Reminder）
```python
def build_system_reminder(extra: str = None) -> dict:
    content = "<system-reminder>\n" + "\n".join(parts) + "\n</system-reminder>"
    return {"role": "user", "content": content}
```

短生命周期的提示（如“请继续完成任务”）不混入主 System Prompt，而是作为**额外的 user 消息**（带 `<system-reminder>` 标签）注入。这符合 Anthropic API 的推荐实践。

### 主循环中的使用
```python
system = prompt_builder.build()   # 每轮重建
response = client.messages.create(model=MODEL, system=system, ...)
```

## 错误恢复
### 一、三种恢复路径概览
| 触发条件 | 恢复策略 | 关键函数 / 机制 |
| --- | --- | --- |
| `stop_reason == "max_tokens"` | 注入续接消息，重新调用模型 | `CONTINUATION_MESSAGE`，重试计数器 |
| API 报错 `overlong_prompt` | 自动压缩对话历史（LLM 摘要）后重试 | `auto_compact()` |
| 临时网络/连接错误 | 指数退避 + 抖动，重试最多 3 次 | `backoff_delay()`，`APIError` 捕获 |


> 恢复优先级（代码中检查顺序）：  
>
> 1. `max_tokens` → 续接  
> 2. `APIError` 中的 `overlong_prompt` → 压缩  
> 3. 连接/超时错误 → 退避重试  
> 4. 所有重试耗尽 → 优雅失败
>

### 二、核心代码解析（带注释）
#### 1. 常量配置
```python
MAX_RECOVERY_ATTEMPTS = 3          # 每种恢复路径最多尝试次数
BACKOFF_BASE_DELAY = 1.0           # 退避初始延迟（秒）
BACKOFF_MAX_DELAY = 30.0           # 退避最大延迟
TOKEN_THRESHOLD = 50000            # 主动压缩阈值（字符数/4 ≈ token数）
CONTINUATION_MESSAGE = (
    "Output limit hit. Continue directly from where you stopped -- "
    "no recap, no repetition. Pick up mid-sentence if needed."
)
```

#### 2. 策略一：max_tokens 续接恢复
**场景**：模型输出被截断（因 `max_tokens=8000` 限制）。

```python
max_output_recovery_count = 0      # 位于 agent_loop 函数内部，每次调用独立

while True:
    # ... API 调用 ...
    
    if response.stop_reason == "max_tokens":
        max_output_recovery_count += 1
        if max_output_recovery_count <= MAX_RECOVERY_ATTEMPTS:
            # 注入续接提示，告诉模型直接继续输出
            messages.append({"role": "user", "content": CONTINUATION_MESSAGE})
            continue          # 重新进入循环，再次调用 API
        else:
            print("max_tokens recovery exhausted")
            return
    # 成功获得完整输出后重置计数器
    max_output_recovery_count = 0
```

**注意点**：  

- 计数只针对 **同一轮对话中连续 **`max_tokens`** 截断** 的情况。  
- 续接消息需要明确要求模型“不回顾、不重复”，直接从断点继续。

#### 3. 策略二：prompt_too_long → 自动压缩对话历史
**触发**：`APIError` 异常且错误信息包含 `"overlong_prompt"` 或类似关键词。

```python
for attempt in range(MAX_RECOVERY_ATTEMPTS + 1):
    try:
        response = client.messages.create(...)
        break
    except APIError as e:
        error_body = str(e).lower()
        if "overlong_prompt" in error_body or ("prompt" in error_body and "long" in error_body):
            print("Prompt too long. Compacting...")
            messages[:] = auto_compact(messages)   # 原地替换历史
            continue    # 用压缩后的新历史重试 API 调用
        # ... 其他错误处理（退避）...
```

`auto_compact()`** 实现细节**：

```python
def auto_compact(messages: list) -> list:
    # 将整个消息历史转为文本（限制长度防止二次过长）
    conversation_text = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this conversation for continuity. Include:\n"
        "1) Task overview\n2) Current state\n3) Key decisions\n4) Next steps\n\n"
        + conversation_text
    )
    # 调用同一个模型生成摘要
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
    )
    summary = response.content[0].text
    # 返回一个占位消息，包含摘要，用于替换原历史
    return [{"role": "user", "content": 
             f"This session continues from a previous conversation that was compacted.\n"
             f"Summary of prior context:\n\n{summary}\n\n"
             "Continue from where we left off without re-asking the user."}]
```

**效果**：长上下文被压缩成一条摘要消息，大幅减少 token 数。

#### 4. 策略三：临时错误 → 指数退避重试
**捕获异常**：`APIError`（非 `overlong_prompt`）、`ConnectionError`、`TimeoutError`、`OSError`。

```python
def backoff_delay(attempt: int) -> float:
    """指数退避 + 随机抖动"""
    delay = min(BACKOFF_BASE_DELAY * (2 ** attempt), BACKOFF_MAX_DELAY)
    jitter = random.uniform(0, 1)
    return delay + jitter

# 在 API 调用循环中
for attempt in range(MAX_RECOVERY_ATTEMPTS + 1):
    try:
        response = client.messages.create(...)
        break
    except (APIError, ConnectionError, TimeoutError, OSError) as e:
        if attempt < MAX_RECOVERY_ATTEMPTS:
            delay = backoff_delay(attempt)
            print(f"Retrying in {delay:.1f}s (attempt {attempt+1}/{MAX_RECOVERY_ATTEMPTS})")
            time.sleep(delay)
            continue
        else:
            print(f"Failed after {MAX_RECOVERY_ATTEMPTS} retries: {e}")
            return
```

**退避示例**：  
attempt=0 → `1.0 + jitter` 秒  
attempt=1 → `2.0 + jitter` 秒  
attempt=2 → `4.0 + jitter` 秒（不超过 `BACKOFF_MAX_DELAY`）

### 三、主动压缩机制（前摄性恢复）
除了被动处理 `overlong_prompt`，代码还在**每轮工具调用后**主动检查 token 估计值：

```python
# 在工具调用处理完毕后
if estimate_tokens(messages) > TOKEN_THRESHOLD:
    print("[Recovery] Token estimate exceeds threshold. Auto-compacting...")
    messages[:] = auto_compact(messages)
```

`estimate_tokens` 粗略实现：`len(json.dumps(messages)) // 4`（每4字符≈1 token）。  
这可以在上下文膨胀到危险阈值前主动压缩，避免后续 API 调用被拒绝。

### 四、整体 agent_loop 流程图（文字版）
```plain
开始
  │
  ├─> for attempt in 0..MAX_RECOVERY_ATTEMPTS:
  │      try API调用
  │      ├─ 成功 → 跳出重试循环
  │      ├─ overlong_prompt → auto_compact() → continue
  │      └─ 其他异常 → backoff_delay() 等待 → continue
  │
  ├─ 若 response 为空 → 退出
  │
  ├─ 追加 assistant 消息到历史
  │
  ├─ 若 stop_reason == "max_tokens":
  │      ├─ 若未超最大续接次数 → 注入 CONTINUATION_MESSAGE → continue
  │      └─ 否则退出
  │
  ├─ 若 stop_reason != "tool_use" → 正常退出
  │
  ├─ 处理工具调用，追加 tool_result 消息
  │
  ├─ 主动检查 token 估计，若超标则 auto_compact()
  │
  └─ 继续循环
```



## 任务系统
### 一、核心设计思想
- **任务持久化**：每个任务保存为 `.tasks/task_{id}.json` 文件，**不依赖对话上下文**，即使对话被压缩或重置，任务状态依然保留。
- **轻量依赖图**：每个任务记录两个列表：
    - `blockedBy`：当前任务被哪些任务阻塞（必须先完成）。
    - `blocks`：当前任务阻塞哪些后续任务。
- **双向一致性**：当给一个任务添加 `blocks` 时，自动更新被阻塞任务的 `blockedBy` 列表。
- **完成自动解锁**：任务状态变为 `completed` 时，会从所有其他任务的 `blockedBy` 中移除自己，实现依赖解锁。

> 注意：这里任务是**持久工作项**，不是运行时执行槽位（线程/进程）。后续章节才会引入调度器。
>

### 二、TaskManager 类解析
```python
class TaskManager:
    """持久化任务记录存储。类似“磁盘上的工作图”，而非“当前运行的 worker”."""

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)          # 创建 .tasks 目录
        self._next_id = self._max_id() + 1      # 下一个可用 ID

    def _max_id(self) -> int:
        """扫描已有任务文件，返回最大 ID"""
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        """创建一个新任务，初始状态为 pending，无依赖"""
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "blocks": [],
            "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)   # 返回 JSON 供模型查看

    def update(self, task_id: int, status: str = None, owner: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        """更新任务状态、负责人或依赖关系。关键点：双向依赖与自动解锁"""
        task = self._load(task_id)

        # --- 更新简单字段 ---
        if owner is not None:
            task["owner"] = owner
        if status:
            if status not in ("pending", "in_progress", "completed", "deleted"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            # ★ 核心：任务完成时，清除自己作为其他任务的阻塞
            if status == "completed":
                self._clear_dependency(task_id)

        # --- 更新依赖（add_blocks 是“本任务阻塞哪些任务”，即本任务指向下游）---
        if add_blocks:
            # 去重合并
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            # ★ 双向维护：遍历每个被阻塞的任务，将本任务加入其 blockedBy 列表
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass   # 忽略不存在的任务 ID

        # 添加 blockedBy（上游依赖）：只影响本任务，不需要反向更新
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))

        self._save(task)
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int):
        """当任务完成时，将其从所有其他任务的 blockedBy 列表中移除"""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        """生成任务列表的简洁文本（带状态标记）"""
        tasks = [json.loads(f.read_text()) for f in sorted(self.dir.glob("task_*.json"))]
        lines = []
        for t in tasks:
            marker = {"pending":"[ ]", "in_progress":"[>]", "completed":"[x]", "deleted":"[-]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{blocked}")
        return "\n".join(lines) if lines else "No tasks."
```

### 三、任务相关工具接口（供模型调用）
| 工具名 | 对应方法 | 作用 |
| --- | --- | --- |
| `task_create` | `TASKS.create(subject, description)` | 创建新任务 |
| `task_update` | `TASKS.update(...)` | 修改状态/负责人/依赖 |
| `task_list` | `TASKS.list_all()` | 查看所有任务简表 |
| `task_get` | `TASKS.get(task_id)` | 查看单个任务详情 |


这些工具被加入 `TOOL_HANDLERS` 和 `TOOLS`，模型可以通过常规工具调用使用它们。

### 四、依赖图与自动解锁示例
假设当前任务文件如下：

**.tasks/task_1.json**

```json
{"id":1, "subject":"设计数据库", "status":"completed", "blockedBy":[], "blocks":[2]}
```

**.tasks/task_2.json**

```json
{"id":2, "subject":"实现API", "status":"pending", "blockedBy":[1], "blocks":[3]}
```

**.tasks/task_3.json**

```json
{"id":3, "subject":"编写前端", "status":"pending", "blockedBy":[2], "blocks":[]}
```

依赖关系：

```plain
task1 (completed) → task2 (pending, blockedBy 1) → task3 (pending, blockedBy 2)
```

当模型调用 `task_update(task_id=2, status="completed")` 时：

1. `task2.status` 变为 `completed`。
2. 调用 `_clear_dependency(2)`：扫描所有任务，从 `task3.blockedBy` 中移除 `2`。
3. `task3.blockedBy` 变为空，依赖解除了。

如果任务2从未完成，任务3会一直被阻塞，模型在调用 `task_list` 时会看到 `(blocked by: [2])`。

### 五、为什么任务系统能抵御上下文压缩？
- 任务数据存储在磁盘上，不在消息历史中。
- 模型每次需要了解任务状态时，**主动调用** `task_list` 或 `task_get`，从磁盘读取最新数据。
- 即使 LLM 的对话历史被 `auto_compact` 压缩成摘要，任务文件不受影响。
- 这种方式将**工作记忆**（任务图）与**对话历史**分离，适合长期项目。

### 六、使用场景与限制
**场景**：

- 多步骤开发任务，需要跟踪依赖（如“必须先完成A才能开始B”）。
- 团队协作（`owner` 字段可以记录谁负责）。
- 长时间运行的 agent，需要定期恢复上下文。

**限制**：

- 依赖关系是静态的（需要手动更新 `addBlocks` / `addBlockedBy`）。
- 没有自动检测任务完成后的下游任务启动（需模型主动查询并更新状态）。
- 任务图是简单的 DAG（有向无环图），循环依赖不会被检测（但 `_clear_dependency` 在完成时会打破潜在循环）。

### 七、最小示例：模型与任务系统的交互
```plain
用户: 实现一个登录功能，先设计接口，再编写后端，最后测试。
模型: [调用 task_create subject="设计登录接口"]
      [调用 task_create subject="编写登录后端"]
      [调用 task_create subject="编写集成测试"]
      [调用 task_update task_id=2 add_blocked_by=[1]]
      [调用 task_update task_id=3 add_blocked_by=[2]]
      [调用 task_list 展示依赖关系]
```

当设计接口完成后：

```plain
模型: [调用 task_update task_id=1 status="completed"]
      (自动解锁 task2)
      [调用 task_list]  # 显示 task2 不再被阻塞
```

## 后台任务
### 一、核心概念
- **后台任务**：将耗时命令放到独立线程中执行，主线程（agent循环）不阻塞，可以继续做其他事（比如继续与模型对话）。
- **通知队列**：后台任务完成后，结果放入队列；在每次调用 LLM 之前，主循环会清空队列，把完成结果作为合成消息注入对话，让模型得知进度。
- **与 s12 任务系统的区别**：
    - s12 的 `Task` 是持久化工作图（长期存储、有依赖关系）。
    - s13 的 `BackgroundManager` 是**运行时执行槽位**（临时、内存中、非持久化），用于并发执行慢命令。

> 本质：**前台不等待，后台默默跑；跑完通知，模型继续。**
>

### 二、整体架构图
```plain
主线程（agent_loop）                    后台线程（每个任务一个）
+---------------------+                 +----------------------+
| while True:         |                 | subprocess.run(...)  |
|   drain_notifications| <---- 通知队列 ---| 完成后 push 通知       |
|   -> 注入消息给模型   |                 | 写入 output_file      |
|   LLM 调用          |                 +----------------------+
|   处理工具调用       |
|   （可 spawn 新任务） |
+---------------------+
```

### 三、关键组件解析
#### 1. `NotificationQueue`（优先级队列 + 消息折叠）
```python
class NotificationQueue:
    """带优先级和同 key 覆盖的通知队列，避免上下文被旧消息淹没"""
    PRIORITIES = {"immediate": 0, "high": 1, "medium": 2, "low": 3}

    def push(self, message: str, priority: str = "medium", key: str = None):
        with self._lock:
            if key:
                # ★ 折叠：如果已有相同 key 的消息，移除旧的（只保留最新的）
                self._queue = [(p, k, m) for p, k, m in self._queue if k != key]
            self._queue.append((self.PRIORITIES[priority], key, message))
            self._queue.sort(key=lambda x: x[0])   # 按优先级排序

    def drain(self) -> list[str]:
        """取出所有消息并清空队列"""
        with self._lock:
            messages = [m for _, _, m in self._queue]
            self._queue.clear()
            return messages
```

**用途**：  

- 优先级：`immediate` 最高，`low` 最低。  
- 折叠：例如同一个 `task_id` 只保留最新状态，避免模型看到一堆过期状态。

#### 2. `BackgroundManager`（后台任务管理器）
```python
class BackgroundManager:
    def __init__(self):
        self.dir = RUNTIME_DIR          # .runtime-tasks/ 存储日志文件
        self.tasks = {}                 # task_id -> 任务元信息（内存）
        self._notification_queue = []   # 完成通知列表（简单列表，非 NotificationQueue）
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        """立即返回 task_id，不等待命令完成"""
        task_id = str(uuid.uuid4())[:8]
        # ... 初始化 tasks[task_id] = {"status": "running", ...}
        thread = threading.Thread(target=self._execute, args=(task_id, command), daemon=True)
        thread.start()
        return f"Background task {task_id} started: {command[:80]} ..."

    def _execute(self, task_id: str, command: str):
        """在后台线程中执行真正的 shell 命令"""
        # 执行 subprocess.run（阻塞，但在后台线程）
        # 完成后更新 tasks[task_id] 状态
        # 将结果写入 .runtime-tasks/{task_id}.log
        # 最后将通知推入 self._notification_queue
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "preview": preview,      # 输出前500字符的紧凑版本
                "output_file": "...",
            })

    def drain_notifications(self) -> list:
        """主循环调用：获取并清空所有完成通知"""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs

    def detect_stalled(self) -> list[str]:
        """返回运行时间超过 STALL_THRESHOLD_S 的任务 ID（用于告警）"""
        # 遍历 tasks，检查 status == "running" 且 elapsed > 阈值
        return stalled
```

**特点**：

- 任务信息**存内存**（`self.tasks`），不持久化跨会话（但日志文件 `.log` 保留输出）。
- 每个后台任务独立线程，`daemon=True` 保证主线程退出时自动终止。
- 超时限制：`subprocess.run(timeout=300)`，防止永久挂起。

#### 3. 工具接口（供模型调用）
| 工具名 | 功能 | 示例输出 |
| --- | --- | --- |
| `background_run` | 启动后台任务，返回 task_id | `Background task abc123 started: sleep 30 (output_file=.runtime-tasks/abc123.log)` |
| `check_background` | 查询单个或所有后台任务状态 | `abc123: [completed] sleep 30 -> (no output)` |


模型可以：

1. 启动多个后台任务（比如同时编译、测试、下载）。
2. 在后续对话中随时 `check_background` 查看进度。
3. 因为通知队列会在每次 LLM 调用前自动注入完成结果，模型甚至不需要主动查询就能知道任务结束了。

#### 4. 主循环中的集成：**drain 注入 + stall 检测（可选）**
```python
def agent_loop(messages: list):
    while True:
        # ★ 关键：在每次调用 LLM 之前，先处理后台任务的通知
        notifs = BG.drain_notifications()
        if notifs and messages:
            # 将所有完成通知拼接成一段文本，作为 user 消息注入对话历史
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['preview']} (output_file={n['output_file']})"
                for n in notifs
            )
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})

        # 正常调用 LLM（现在对话历史中已经包含后台完成信息）
        response = client.messages.create(...)

        # ... 工具调用处理 ...
```

**注入时机**：  

- 每次循环开始时。  
- 第一次进入循环时，`messages` 可能只有用户原始输入，这时如果有后台通知，也会被注入（但通常后台任务还没结束）。  
- 注入的消息格式为 `<background-results>...</background-results>`，方便模型识别是系统自动插入的通知。

### 四、工作流示例
1. **用户**：`帮我编译这个大型项目，大概要跑5分钟，我先去做别的。`
2. **模型**：调用 `background_run command="make -j8"`  
→ 立即返回 `task_id=abc123`
3. **主循环**：没有其他工具调用，模型输出文本后结束本轮。
4. **后台线程**：执行 `make -j8`，完成后将通知推入队列。
5. **用户**：过一会儿，输入 `编译完了吗？`
6. **主循环**：  
    - 调用 `BG.drain_notifications()`，拿到 `abc123` 的完成通知。  
    - 注入一条 `<background-results>...</background-results>` 到消息历史。  
    - 然后调用 LLM，模型看到通知中的 `status=completed` 和输出文件路径，可以回答“编译完成了，输出见 ...”。
7. **模型**：如果需要，还可以调用 `read_file` 读取编译日志。

### 五、设计亮点与注意事项
| 亮点 | 说明 |
| --- | --- |
| **非阻塞** | 后台线程执行 shell，agent 循环可以立即处理用户新输入或调用其他工具。 |
| **自动注入** | 每次 LLM 调用前自动注入完成结果，模型无需主动轮询（但也可以主动 `check_background`）。 |
| **输出持久化** | 每个后台任务的完整输出保存在 `.runtime-tasks/{task_id}.log`，避免过大内容撑爆 token。 |
| **通知折叠（未用在此例）** | 示例中的 `NotificationQueue` 类定义了但未使用；若使用，可防止同一个任务重复通知。 |
| **超时与卡死检测** | `detect_stalled()` 可被另一个监控线程调用，或定期注入提醒模型“某个任务已经跑了45秒还没完”。 |


**注意事项**：

- 后台任务状态**仅存内存**，程序重启后丢失（但日志文件还在，模型可以手动读文件）。
- 并发线程数无限制，若同时启动很多后台任务，可能会消耗系统资源。
- 通知队列中的内容在每次 LLM 调用前都会被清空并注入，因此**同一批完成通知只会被模型看到一次**（除非模型自己再次查询）。

## 定时调度
### 一、核心思想
Agent 可以为自己安排未来的工作：通过 **cron 表达式** 定义触发时间，后台线程每分钟检查一次，到期时自动将预设的 **提示词** 注入对话历史。这样模型就能“定时醒来”执行任务。

**关键特性**：

- 支持标准 5 字段 cron 表达式（分、时、日、月、周几）。
- 两种触发模式：`recurring`（重复，7 天后自动过期）/ `one-shot`（单次，触发后删除）。
- 两种持久化模式：`durable`（存盘，跨会话）/ `session-only`（仅内存，退出丢失）。
- 分布式锁（`CronLock`）防止多个会话同时触发同一个任务。
- 抖动（`jitter`）：对整点/半点任务增加小偏移，避免所有任务在同一分钟拥挤。
- 错过检测（`detect_missed_tasks`）：启动时检查任务在离线期间是否应被触发。

### 二、Cron 表达式匹配实现
```python
def cron_matches(expr: str, dt: datetime) -> bool:
    """
    检查一个 5 字段 cron 表达式是否匹配给定的时间。
    支持：*（任意）、*/N（步长）、N（精确）、N-M（范围）、N,M（列表）
    """
    fields = expr.strip().split()
    if len(fields) != 5:
        return False
    values = [dt.minute, dt.hour, dt.day, dt.month, dt.weekday()]
    # Python 的 weekday：0=周一；cron 标准：0=周日，需要转换
    cron_dow = (dt.weekday() + 1) % 7
    values[4] = cron_dow

    ranges = [(0,59), (0,23), (1,31), (1,12), (0,6)]
    for field, value, (lo, hi) in zip(fields, values, ranges):
        if not _field_matches(field, value, lo, hi):
            return False
    return True

def _field_matches(field: str, value: int, lo: int, hi: int) -> bool:
    """单个字段匹配逻辑，支持 * / - , 等语法"""
    if field == "*":
        return True
    for part in field.split(","):
        # 处理步长：*/N 或 N-M/S
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            step = int(step_str)
        if part == "*":
            if (value - lo) % step == 0:
                return True
        elif "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if start <= value <= end and (value - start) % step == 0:
                return True
        else:
            if int(part) == value:
                return True
    return False
```

> 注意：cron 的星期字段 0=周日，而 Python `datetime.weekday()` 0=周一，需要转换。
>

### 三、CronScheduler 类（调度器核心）
```python
class CronScheduler:
    def __init__(self):
        self.tasks = []          # 任务列表（内存）
        self.queue = Queue()     # 通知队列，线程安全
        self._stop_event = threading.Event()
        self._thread = None
        self._last_check_minute = -1   # 避免同一分钟内重复触发
```

#### 1. 后台检查循环
```python
def start(self):
    self._load_durable()                # 加载持久化任务
    self._thread = threading.Thread(target=self._check_loop, daemon=True)
    self._thread.start()

def _check_loop(self):
    """每秒检查一次，但真正匹配只在分钟变化时进行"""
    while not self._stop_event.is_set():
        now = datetime.now()
        current_minute = now.hour * 60 + now.minute
        if current_minute != self._last_check_minute:
            self._last_check_minute = current_minute
            self._check_tasks(now)        # 检查所有任务
        self._stop_event.wait(timeout=1)  # 每秒唤醒一次
```

#### 2. 任务匹配与触发
```python
def _check_tasks(self, now: datetime):
    expired = []
    fired_oneshots = []
    for task in self.tasks:
        # 自动过期：重复任务超过 7 天
        age_days = (time.time() - task["createdAt"]) / 86400
        if task["recurring"] and age_days > AUTO_EXPIRY_DAYS:
            expired.append(task["id"])
            continue

        # 抖动处理：如果任务有 jitter_offset，检查时间向前偏移几分钟
        check_time = now
        jitter = task.get("jitter_offset", 0)
        if jitter:
            check_time = now - timedelta(minutes=jitter)

        if cron_matches(task["cron"], check_time):
            notification = f"[Scheduled task {task['id']}]: {task['prompt']}"
            self.queue.put(notification)      # 放入通知队列
            if not task["recurring"]:
                fired_oneshots.append(task["id"])

    # 删除过期和已触发的单次任务
    if expired or fired_oneshots:
        remove_ids = set(expired) | set(fired_oneshots)
        self.tasks = [t for t in self.tasks if t["id"] not in remove_ids]
        self._save_durable()   # 更新持久化文件
```

#### 3. 抖动计算
```python
JITTER_MINUTES = [0, 30]        # 避免在这些分钟触发
JITTER_OFFSET_MAX = 4           # 最大偏移分钟数

def _compute_jitter(self, cron_expr: str) -> int:
    """如果 cron 指向 :00 或 :30，返回一个 1~4 之间的偏移量"""
    fields = cron_expr.strip().split()
    if len(fields) < 1:
        return 0
    minute_field = fields[0]
    try:
        minute_val = int(minute_field)
        if minute_val in JITTER_MINUTES:
            # 根据表达式哈希决定偏移量，保证同一任务每次重启偏移一致
            return (hash(cron_expr) % JITTER_OFFSET_MAX) + 1
    except ValueError:
        pass
    return 0
```

#### 4. 持久化
```python
SCHEDULED_TASKS_FILE = WORKDIR / ".claude" / "scheduled_tasks.json"

def _save_durable(self):
    durable = [t for t in self.tasks if t.get("durable")]
    SCHEDULED_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULED_TASKS_FILE.write_text(json.dumps(durable, indent=2) + "\n")

def _load_durable(self):
    if not SCHEDULED_TASKS_FILE.exists():
        return
    data = json.loads(SCHEDULED_TASKS_FILE.read_text())
    self.tasks = [t for t in data if t.get("durable")]
```

#### 5. 错过任务检测（启动时调用）
```python
def detect_missed_tasks(self) -> list[dict]:
    """检查每个 durable 任务在离线期间是否应该被触发过"""
    now = datetime.now()
    missed = []
    for task in self.tasks:
        last_fired = task.get("last_fired")
        if last_fired is None:
            continue
        last_dt = datetime.fromtimestamp(last_fired)
        check = last_dt + timedelta(minutes=1)
        cap = min(now, last_dt + timedelta(hours=24))
        while check <= cap:
            if cron_matches(task["cron"], check):
                missed.append({"id": task["id"], "cron": task["cron"],
                               "prompt": task["prompt"], "missed_at": check.isoformat()})
                break
            check += timedelta(minutes=1)
    return missed
```

### 四、CronLock：跨会话的分布式锁
```python
class CronLock:
    """基于 PID 文件的锁，防止多个 agent 会话同时触发同一个任务"""
    def acquire(self) -> bool:
        if self._lock_path.exists():
            try:
                stored_pid = int(self._lock_path.read_text().strip())
                os.kill(stored_pid, 0)      # 检查进程是否存活
                return False                # 存活 -> 锁被占用
            except (ProcessLookupError, ...):
                pass                         # 进程已死 -> 锁是过期的，可以取走
        self._lock_path.write_text(str(os.getpid()))
        return True

    def release(self):
        # 只有当前进程的锁才删除
        if self._lock_path.exists():
            stored_pid = int(self._lock_path.read_text().strip())
            if stored_pid == os.getpid():
                self._lock_path.unlink()
```

> 注意：此锁在示例代码中**没有实际集成**到 `CronScheduler` 的触发逻辑中，仅作为教学演示。实际使用时应在 `_check_tasks` 中检查锁。
>

### 五、与 Agent 主循环的集成
```python
def agent_loop(messages: list):
    while True:
        # 每次调用 LLM 前，先清空通知队列，将已触发的任务提示词注入对话
        notifications = scheduler.drain_notifications()
        for note in notifications:
            messages.append({"role": "user", "content": note})

        response = client.messages.create(...)
        # ... 工具调用处理 ...
```

注入的消息格式如：`[Scheduled task abc123]: 请检查日志文件并报告异常`  
模型看到后会正常响应，可以调用工具执行该任务。

### 六、工具接口（供模型调用）
| 工具名 | 功能 | 示例 |
| --- | --- | --- |
| `cron_create` | 创建定时任务 | `cron_create cron="*/5 * * * *" prompt="检查服务状态" recurring=true durable=true` |
| `cron_delete` | 删除任务 | `cron_delete id="abc123"` |
| `cron_list` | 列出所有任务 | `cron_list` |


模型可以动态创建、删除、查看定时任务，实现“自我调度”。

### 七、工作流程示例
1. **用户**：每天上午9点提醒我备份数据库。
2. **模型**：调用 `cron_create cron="0 9 * * *" prompt="现在执行数据库备份" recurring=true durable=true`
3. 后台线程每分钟检查，当时间匹配 `0 9 * * *` 时，将提示词放入通知队列。
4. 下一次用户输入任何消息（或模型主动触发）时，主循环先清空队列，注入提示词。
5. 模型看到后执行备份操作（如调用 `bash` 运行备份脚本）。

### 八、设计要点总结
| 特性 | 实现方式 |
| --- | --- |
| 时间匹配 | 手动解析 cron 表达式，不依赖外部库 |
| 并发安全 | `queue.Queue` 用于跨线程通知；`CronLock` 防止多会话冲突 |
| 持久化 | `durable` 任务存入 `.claude/scheduled_tasks.json` |
| 自动过期 | 重复任务超过 7 天自动删除，防止无限积累 |
| 抖动 | 对整点/半点任务增加 1-4 分钟偏移，分散负载 |
| 错过补偿 | `detect_missed_tasks()` 让用户在离线后决定是否补执行 |



## 总结：把 Claude Code 当成可审计的工程协作者

学习 Claude Code 的关键，不是记住某个命令，而是理解它如何把模型、工具、上下文、权限和验证流程组合起来。一个稳定的使用习惯通常包括：先让它读项目约定，再让它提出可审查的小计划；执行时要求它保留状态、运行检查；收尾时要求它说明改了什么、没验证什么、下一步风险在哪里。

如果只把它当“代码生成器”，你会期待它一次写对；如果把它当“可审计的协作者”，你会自然地设计任务边界、验证路径和回滚点。后者才是使用 agentic coding 工具的核心心智。

## 参考资料

- [Claude Code Overview](https://code.claude.com/docs/en/overview)
- [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)
- [Extend Claude Code](https://code.claude.com/docs/en/features-overview)
- [Claude Code settings](https://code.claude.com/docs/en/configuration)
- [Create custom subagents](https://code.claude.com/docs/en/sub-agents)

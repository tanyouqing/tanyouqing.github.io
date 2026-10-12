---
title: "CS336 L01：分词与字节级 BPE"
date: "2026-07-29"
description: "从 Unicode 和 UTF-8 到 BPE 训练、编码、特殊 token，以及词表大小与序列长度的权衡。"
tags: ["CS336", "LLM"]
category: "笔记"
---

## 1. Tokenizer 的接口与正确性

Tokenizer 至少提供两个操作：

```python
ids = tokenizer.encode(text)
text = tokenizer.decode(ids)
```

最重要的不变量（invariant）是 round trip：

$$
\operatorname{decode}(\operatorname{encode}(x)) = x
$$

其中：

- `encode: str → list[int]`
- `decode: list[int] → str`
- 每个 token id 都对应词表中的一段字节；
- 对正常输入，编码后再解码必须恢复原文。

文本信息在进入模型前已经损坏；即使训练 loss 正常下降，模型也只是在拟合错误的数据表示。

## 2. 为什么不直接按字符、字节或词切分？

Tokenizer 的三个主要目标是：

1. 覆盖性（coverage）：任意文本都能表示；
2. 压缩率（compression ratio）：一个 token 尽量承载更多有效字节；
3. 可控词表（bounded vocabulary）：避免 embedding 与输出层无限增大。

| 方法 | 词表 | 序列长度 | 未见文本 | 主要问题 |
|---|---:|---:|---|---|
| 字符级 | 大 | 长 | 可表示 | Unicode 字符多且长尾严重 |
| 字节级 | 固定 256 | 最长 | 可表示 | 压缩率为 1 byte/token |
| 词级 | 极大 | 短 | 常需 `UNK` | 长尾词、新词、代码和拼写变体难覆盖 |
| 字节级 BPE | 可控 | 较短 | 可回退到 byte | 训练与编码逻辑更复杂 |

> BPE token 不一定是“有语义的词”或“词根”。它只是由语料频率和合并规则得到的字节片段。

## 3. Unicode、UTF-8 与 byte fallback

Python 的字符串是 Unicode 文本；byte-level BPE 先用 UTF-8 转成字节：

```python
"a".encode("utf-8")   # b"\x61"，1 byte
"中".encode("utf-8")  # b"\xe4\xb8\xad"，3 bytes
"🌍".encode("utf-8")  # b"\xf0\x9f\x8c\x8d"，4 bytes
```

初始词表包含全部 256 个 byte：

```python
vocab = {i: bytes([i]) for i in range(256)}
```

因此，任何能编码为 UTF-8 的字符串都能表示，最坏情况只是退化为逐 byte 编码，不需要 `UNK`。

关键细节：

- 一个 Unicode 字符可能由多个 byte token 表示；
- 单独解码某个 token 时，它可能不是完整 UTF-8 字符；
- 应先拼接所有 token 对应的 bytes，再整体执行 UTF-8 decode；
- 对任意 token-id 序列，decoder 还需要明确非法 UTF-8 的处理策略。

## 4. BPE 训练：从 byte 学习词表

### 4.1 直觉

从 byte token 开始，重复合并语料中最频繁的相邻 token pair：

```text
高频片段 → 较大的 token → 更短的序列
低频片段 → 保持细粒度 → 仍然可以表示
```

### 4.2 数学形式

设当前 token 序列为 $z=(z_1,\ldots,z_T)$。相邻 token 对 $(a,b)$ 的出现频次定义为：

$$
c(a,b)
=
\sum_{i=1}^{T-1}
\mathbf{1}\!\left[z_i=a \land z_{i+1}=b\right],
$$

其中，$\mathbf{1}[\cdot]$ 是指示函数：条件成立时取 $1$，否则取 $0$。

每一轮 BPE 训练选择出现频率最高的相邻 token 对：

$$
(a^*,b^*)
=
\operatorname*{arg\,max}_{(a,b)} c(a,b).
$$

随后创建一个新 token：

$$
v_{\text{new}}
=
v_{a^*}\mathbin{\Vert}v_{b^*},
$$

其中，$\Vert$ 表示两个 token 对应 byte 序列的拼接。然后，将序列中所有不重叠的 $(a^*,b^*)$ 替换为新 token。

若基础 byte token 数量为 $256$，特殊 token 数量为 $S$，共执行 $M$ 次有效合并，则最终词表大小为：

$$
|V| = 256 + S + M.
$$

> 如果某一轮不存在可合并的 token 对，或者候选合并已经存在于词表中，实际新增的 token 数可能少于 $M$。因此，严格来说有 $|V| \le 256 + S + M$。

### 4.3 训练伪代码

```text
input: corpus, target_vocab_size, special_tokens

1. 隔离 special tokens
2. 对普通文本做 pre-tokenization
3. 将每个 pre-token 转为 UTF-8 bytes
4. 初始化 256 个 byte token，并加入 special tokens
5. while vocab_size < target_vocab_size:
      统计所有 pre-token 内部的相邻 pair
      按频次与规定的 tie-break 选出 pair
      记录 merge
      把该 pair 的全部不重叠出现替换为新 token
6. return vocab, ordered_merges
```

训练返回两份可复用的 tokenizer 参数：

- `vocab: token_id → bytes`
- `merges: 有顺序的 byte-pair 列表`

## 5. 训练 tokenizer 与使用 tokenizer 是两件事

| 阶段 | 输入 | 核心操作 | 输出 |
|---|---|---|---|
| BPE training | 训练语料 | 反复找最高频 pair | `vocab + ordered_merges` |
| Encoding | 任意新文本 | 按已学习的 merge 优先级合并 | token ids |
| Decoding | token ids | 查表、拼 bytes、UTF-8 decode | 文本 |

编码时**不能重新统计新文本中的最高频 pair**，也不能按字符串长度贪心匹配词表。必须遵循训练得到的 merge priority。

一种清晰但较慢的编码思路：

```text
对每个 pre-token:
  从 UTF-8 bytes 开始
  找出当前相邻 pair 中 rank 最小的可用 merge
  合并它的全部不重叠出现
  直到没有可用 merge
  映射为 token ids
```

> 逐条遍历全部 merge 容易实现，但会浪费大量查找。较高效的编码方式只维护当前序列中可发生的 merge，并按 rank 更新候选。

## 6. 预分词（pre-tokenization）

如果直接在整份文本上做 BPE，merge 可能跨越不希望跨越的边界，训练时的统计和编码时的行为也会难以控制。

预分词先用正则表达式把文本切为较稳定的 chunk，再只在每个 chunk 内运行 BPE：

```text
"Hello, how are you?"
→ ["Hello", ",", " how", " are", " you", "?"]
→ 对每个 chunk 独立执行 byte-level BPE
```

它解释了常见现象：

- `"hello"` 与 `" hello"` 可能得到不同 token；
- 空格经常和后面的词一起编码；
- 标点、数字、缩写和换行有专门边界；
- BPE merge 不跨越 pre-token 边界。

实现要求：训练与编码必须使用同一套 pre-tokenization 规则，否则学到的 merge 与推理时的切分不一致。

## 7. 特殊 token（special tokens）

特殊 token 用来表达普通文本之外的结构，例如：

```text
<|endoftext|>
```

在预训练中，它可以分隔互不相关的文档；在后训练中，同类机制还可表达角色、消息边界或工具调用结构。

特殊 token 应满足：

- 作为一个不可拆分的原子 token；
- 直接映射到唯一 id；
- 不参与普通 BPE merge；
- 训练时不能和左右文本合并；
- 编码时优先识别，随后只对普通片段运行 pre-tokenization 与 BPE；
- 若多个特殊 token 重叠，需要规定确定的匹配策略，例如优先匹配较长的标记。

```text
"doc A<|endoftext|>doc B"
→ 普通片段 + [EOT_ID] + 普通片段
```

> 只把 `<|endoftext|>` 加入 `vocab` 并不等于正确支持它。如果 encode 仍走普通 BPE，它依旧可能被拆开。

## 8. 词表大小与压缩率的权衡

用每个 token 平均承载的字节数衡量压缩率：

$$
\text{compression ratio}
= \frac{\text{UTF-8 bytes 数}}{\text{token 数}}
$$

该值越大，通常表示同一段文本需要的 token 越少。

但词表并非越大越好：

| 增大词表的收益 | 增大词表的代价 |
|---|---|
| 序列通常更短 | embedding 参数增加 |
| context 可容纳更多原文 | 输出 projection / softmax 更贵 |
| attention 的 $T^2$ 成本下降 | 低频 token 的表示更稀疏 |
| 生成步数可能减少 | tokenizer 训练与部署资产更大 |

评测 tokenizer 时至少记录：

- vocabulary size；
- train / held-out compression ratio；
- 不同语言、代码、数字、emoji 的 token 数；
- `decode(encode(x)) == x`；
- encode / train wall time；
- 大文件流式编码时的峰值内存。

## 9. 张量形状与数据流

| 对象 | Shape / 类型 | 含义 |
|---|---|---|
| 原始文本 | `str` | Unicode 字符串 |
| UTF-8 表示 | `bytes`，长度 $B_{\text{utf8}}$ | 0–255 的 byte 序列 |
| token ids | `list[int]`，长度 $T$ | Transformer 的离散输入 |
| batch input | $(B,T)$ | 一批 token 序列 |
| embedding 输出 | $(B,T,d_{\text{model}})$ | 每个 token 的向量表示 |
| logits | $(B,T,V)$ | 每个位置对整个词表的预测 |

## 10. 资源核算

- 词嵌入参数量：$Vd_{\text{model}}$
- 输出层参数量：通常为 $Vd_{\text{model}}$；若 weight tying，则可与 embedding 共享
- logits 元素数：$BTV$
- 标准 self-attention：时间/中间量对 $T$ 含二次项
- FFN 与多数逐 token 层：成本随 $T$ 近似线性增长

因此 tokenizer 同时控制 $V$ 与 $T$：压缩率更高会减小 $T$，但更大的词表会增大与 $V$ 相关的参数和计算。

## 11. BPE 合并示例

下面忽略 pre-tokenization 与特殊 token，仅演示 BPE 核心机制。

语料：

```text
low low lower
```

初始状态可简化写为：

```text
l o w | l o w | l o w e r
```

若前三轮最高频 pair 依次为：

```text
(l, o)  → lo
(lo, w) → low
(low, e) → lowe
```

则：

```text
l o w | l o w | l o w e r
→ lo w | lo w | lo w e r
→ low | low | low e r
→ low | low | lowe r
```

学习到：

```text
vocab 额外包含: b"lo", b"low", b"lowe"
merges 有序记录:
  (b"l", b"o")
  (b"lo", b"w")
  (b"low", b"e")
```

新文本 `"lowest"` 会按照上述 merge 顺序编码；即使 `"est"` 在新文本中出现，也不能临时发明新的 merge。

## 12. 实现结构与边界条件

建议把实现拆成四层：

```text
文本边界层
  special-token split + pre-tokenization
        ↓
BPE 核心层
  pair count + merge + rank
        ↓
词表层
  bytes ↔ token id
        ↓
I/O 层
  encode / encode_iterable / decode
```

关键数据结构：

```python
vocab: dict[int, bytes]
token_to_id: dict[bytes, int]
merges: list[tuple[bytes, bytes]]
merge_rank: dict[tuple[bytes, bytes], int]
special_to_id: dict[str, int]
```

边界条件：

- 空字符串；
- 单字符与多字节 Unicode；
- 连续、首尾和重叠的特殊 token；
- 空格、换行、数字、标点与缩写；
- pair 频次并列时的确定性 tie-break；
- merge 时不能错误处理重叠 pair；
- train 与 encode 必须使用相同 pre-tokenizer；
- `encode_iterable` 不能一次读完整个文件。

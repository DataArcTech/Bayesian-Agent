# Bayesian-Agent：让 Agent 的 Skill 进化从经验主义走向贝叶斯推理

> 仓库地址：[https://github.com/DataArcTech/Bayesian-Agent](https://github.com/DataArcTech/Bayesian-Agent)  
> 文档地址：[https://dataarctech.github.io/Bayesian-Agent/](https://dataarctech.github.io/Bayesian-Agent/)  
> 当前版本：v0.4，arXiv coming soon

过去一年，Agent 框架越来越多。每个框架都在讲工具调用、memory、planner、browser、workflow、multi-agent，但如果把这些概念拨开，本质问题其实很朴素：

**一个 Agent 做错了一次任务之后，它到底有没有真的变聪明？**

很多系统的答案并不令人满意。失败轨迹可能被塞进长上下文，成功经验可能被写成一段 prompt，SOP 可能越积越厚，但这些“经验”通常没有明确的证据权重，也很少回答三个问题：

1. 这条 Skill 到底在什么场景下有效？
2. 它失败时通常死在哪个 failure mode？
3. 继续使用它，是省 token，还是在制造噪声？

这就是我们开源 **Bayesian-Agent: A Bayesian Self-Evolving Agent Framework with Cross-Harness Adaptation** 的原因。

Bayesian-Agent 不是又一个试图替代所有人的 Agent runtime。它更像一个可以挂在不同 Agent harness 旁边的 **Bayesian Skill/SOP evolution layer**：从 Agent 的成功和失败轨迹中提取证据，维护每条 Skill 的 posterior belief，再把更可靠、更省 token、更贴近当前任务的 Skill context 注入下一次运行。

## 一句话介绍

Bayesian-Agent 把 Agent 的每条 Skill / SOP 当成一个关于成功率的假设：

```text
P(success | theta, C, skill)
```

其中：

- `theta` 是底层大模型参数
- `C` 是推理环境，包括 prompt、context、tools、memory、harness feedback
- `skill` 是可复用的任务流程或 SOP

每次 Agent 执行任务后，Bayesian-Agent 会读取经过验证的 trajectory evidence，更新 Skill 的 posterior belief，并在下一次运行时生成 posterior-weighted Skill context。

换句话说，它不是“把失败经历都塞进记忆里”，而是问：

**哪些经验被验证过？哪些经验在当前任务里值得相信？哪些经验应该 patch、split、compress 或 retire？**

## 仓库架构图

下面这张图是当前仓库结构和数据流。核心点是：Bayesian-Agent 的 `core` 不绑定任何具体 Agent runtime；GenericAgent 只是当前实验使用的 backend harness，未来也会上传我们自己的 Agent harness。

<div align="center">
  <img src="../assets/bayesian_agent_repository_architecture.svg" width="900" alt="Bayesian-Agent repository architecture"/>
  <br/>
  <em>Bayesian-Agent repository architecture: a reusable Bayesian Skill/SOP evolution layer across harnesses.</em>
</div>

仓库中最重要的几层是：

- `bayesian_agent/core/`：框架无关的 Bayesian evolution engine
- `bayesian_agent/adapters/`：外部 Agent harness 的适配边界
- `schemas/`：trajectory evidence 和 Skill belief 的通用 JSON schema
- `artifacts/`：GenericAgent + deepseek-v4-flash 的实验结果
- `docs/`：方法、实验、API、adapter 和部署文档

这个边界设计很重要。它意味着 Bayesian-Agent 不需要复制 GenericAgent，也不需要把自己变成一个封闭的大型 Agent 框架。只要某个 Agent framework 能产出统一的 trajectory schema，它就可以接入 Bayesian-Agent。

## 为什么是 Bayesian？

我们可以把大模型系统的优化方式粗略分成两条 MECE 路线：

1. 改变模型参数分布：预训练、SFT、RL、微调等
2. 改变推理条件分布：prompt、context、RAG、tools、memory、harness 等

如果基础模型是在采样：

```text
P(X | theta)
```

那么 Agent 系统其实是在采样：

```text
P(X | theta, C)
```

这里的 `C` 就是整个推理环境。所谓 prompt engineering、context engineering、RAG、tool use、memory、harness engineering，本质上都在改变 `C`，从而改变模型下一步吐字和行动的条件概率分布。

Bayesian-Agent 聚焦的是 Skill / SOP 这一层 `C` 的进化。

传统做法像是：

```text
做过一次 -> 写进经验 -> 下次看到类似任务就用
```

Bayesian-Agent 的做法是：

```text
执行轨迹 -> Verifier -> Evidence -> Posterior -> Skill Context -> 下一次执行
```

这让 Skill 不再是“看起来有用的一段提示词”，而是一个被持续验证、持续修正、持续压缩的操作假设。

## 三个核心优势

### 1. 从零跑：Full Self-Evolving Mode

Bayesian-Agent 可以从零开始运行任务。没有历史 traces，没有人工预置的大量 SOP，也可以在任务执行过程中收集 evidence，并逐步进化 Skill beliefs。

这对应的是研究问题：

**一个 Agent 能不能在没有先验经验的情况下，通过执行和验证自我沉淀 Skill？**

在 SOP-Bench 上，我们用 GenericAgent + `deepseek-v4-flash` 做了原型验证：

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA | deepseek-v4-flash | 80% | 1.34M | 57k | 1.39M | 11.47 |
| SOP-Bench | GA+Bayesian | deepseek-v4-flash | 100% | 1.07M | 52k | 1.12M | 17.86 |

结果很直观：full mode 不只是把准确率从 80% 拉到 100%，token 使用也从 1.39M 降到了 1.12M。

这说明 Skill 进化不是“越写越长”的记忆堆叠。用 posterior belief 过滤和组织经验，反而可能让上下文更短、更准。

### 2. 增量跑：Incremental Repair Mode

从工程落地角度看，更有价值的是增量模式。

很多团队已经有自己的 Agent 框架、自己的业务 harness、自己的评测流水线。要求他们换一套 Agent runtime 并不现实。Bayesian-Agent 的设计不是替换这些系统，而是作为 repair layer 插进去。

流程是：

```text
Base Agent -> Failure Traces -> Bayesian Skill Evolution -> Rerun Failures -> Higher Accuracy
```

也就是：

1. 原来的 Agent 先跑
2. Bayesian-Agent 读取成功和失败轨迹
3. 对失败模式进行 posterior update 和 repair planning
4. 只重跑失败任务

在我们的增量实验里：

| Benchmark | Agent | Model | Final Accuracy | Incremental Input | Incremental Output | Incremental Total | Incremental Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 216k | 10k | 226k | 17.73 |
| Lifelong AgentBench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 71k | 7k | 78k | 25.57 |

这点非常关键：如果一个 Agent 已经能做到 80% 或 90%，Bayesian-Agent 不需要完整重跑所有任务，只需要针对失败部分做增量修复。

这让它从“研究算法”变成了一个很实用的工程模块。

### 3. 跨 Harness：Cross-Harness Adaptation

Bayesian-Agent 最重要的定位，是 **cross-harness adaptation**。

我们现在的实验基于 GenericAgent，但仓库没有 copy 一份 GenericAgent，也没有把自己写成 GenericAgent fork。原因很简单：

**真正有价值的不是某个 harness 本身，而是 Skill/SOP 进化的方法可以跨 harness 复用。**

Bayesian-Agent 把边界定义在两件事上：

- 外部 Agent 需要产出 `TrajectoryEvidence`
- 外部 Agent 能接收 posterior-weighted Skill context

最小 adapter contract 是：

```python
class AgentAdapter(Protocol):
    def run(self, task: Mapping[str, Any], skill_context: str) -> Mapping[str, Any]:
        ...
```

这意味着 GenericAgent、我们后续上传的自研 Agent harness，以及其他 agent frameworks，都可以作为 Bayesian-Agent 的 backend。

我们希望 Bayesian-Agent 成为一个 **Skill evolution substrate**，而不是又一个孤立的 Agent 应用。

## 仓库里现在有什么？

v0.4 已经包含：

- Bayesian Skill registry
- Beta posterior belief update
- failure-mode-aware rewrite policy
- posterior-weighted context builder
- incremental repair utilities
- result summarization CLI
- GenericAgent optional adapter boundary
- trajectory 和 skill belief schema
- SOP-Bench / Lifelong AgentBench 实验 artifacts
- MkDocs 文档站和 GitHub Pages 部署

安装方式：

```bash
git clone https://github.com/DataArcTech/Bayesian-Agent.git
cd Bayesian-Agent
python -m pip install -e .
```

CLI 示例：

```bash
bayesian-agent evolve \
  --results artifacts/ga_deepseek_baseline/sop_results.json \
  --registry temp/bayesian_skill_beliefs.json \
  --context-out temp/skill_context.md
```

Python API 示例：

```python
from bayesian_agent import BayesianSkillRegistry, SkillContextBuilder, TrajectoryEvidence

registry = BayesianSkillRegistry("temp/beliefs.json")
registry.record(
    TrajectoryEvidence(
        task_id="sop_12",
        skill_id="benchmark/sop_bench",
        context="sop_bench",
        outcome="failure",
        failure_mode="xml_wrapped_answer",
        input_tokens=70123,
        output_tokens=4242,
    )
)

print(SkillContextBuilder(registry).render(task_context="sop_bench"))
```

## 和普通 Agent 框架的区别

我觉得可以用一句话概括：

**普通 Agent 框架关注“怎么执行任务”，Bayesian-Agent 关注“执行经验如何被验证、加权、修复、迁移”。**

所以它不是在和 GenericAgent、OpenHands、Claude Code、各类 browser agent 抢同一个位置。

更准确的位置是：

```text
Agent Harness below
Bayesian Skill Evolution above
```

底层 harness 负责执行任务，Bayesian-Agent 负责从执行轨迹中学习，形成可迁移的 Skill posterior。

这也是为什么我们强调 cross-harness adaptation。因为未来真正有价值的 Agent 资产，可能不是某一个 runtime，而是跨 runtime 可复用的 verified operational knowledge。

## Roadmap

接下来我们会继续做几件事：

- 上传我们自己的 Bayesian-Agent harness，让用户不依赖 GenericAgent 也能完整跑 full loop
- 完善更多 agent frameworks 的 adapter examples
- 增加更丰富的 rewrite policy，比如 context-specific split、cost-aware compression、failure taxonomy
- 加入可执行 benchmark runner，方便外部复现实验
- 推进论文版本，arXiv coming soon

## 结语

Agent 的下一阶段，不只是更强的工具调用，也不只是更长的上下文。

更关键的问题是：

**Agent 的经验如何成为可验证、可更新、可迁移的工程资产？**

Bayesian-Agent 给出的答案是：把 Skill/SOP 当成假设，把执行轨迹当成证据，用 Bayesian posterior 管理它们的可靠性和成本。

如果说 prompt engineering 解决的是“怎么问”，context engineering 解决的是“给模型看什么”，harness engineering 解决的是“怎么让模型做事”，那么 Bayesian-Agent 想解决的是：

**做完之后，系统如何真正进化。**

仓库地址：  
[https://github.com/DataArcTech/Bayesian-Agent](https://github.com/DataArcTech/Bayesian-Agent)

文档地址：  
[https://dataarctech.github.io/Bayesian-Agent/](https://dataarctech.github.io/Bayesian-Agent/)

# Bayesian-Agent: A Bayesian Self-Evolving Agent Framework with Cross-Harness Adaptation

<div align="center">
  <img src="assets/banner.png" width="920" alt="Bayesian-Agent banner"/>
</div>

<p align="center">
  🌐 <a href="README.md">English</a> | 🇨🇳 <a href="README_ZH.md">中文</a> |
  📚 <a href="https://dataarctech.github.io/Bayesian-Agent/">Docs</a> |
  🐙 <a href="https://github.com/DataArcTech/Bayesian-Agent">GitHub</a> |
  📄 arXiv Coming Soon
</p>

Bayesian-Agent 是一个面向跨 Agent framework / execution harness 的 Bayesian self-evolving layer，用于把 Agent 的失败轨迹转化为可复用、可验证、带 posterior 权重的 Skills 和 SOPs。

它不是又一个封闭的 monolithic agent framework，而是突出三件事：

- **从零自进化**：没有历史 traces 也能从完整 benchmark 或生产任务中在线沉淀 Skills。
- **增量修复**：接到已有 Agent 后面，读取失败轨迹，只重跑需要修复的任务。
- **跨 harness 适配**：当前适配 GenericAgent，后续可通过统一 trajectory schema 和 adapter boundary 接入其他 agent frameworks。

> v0.4 是第一个独立开源版本。它包含 Bayesian Skill Evolution 核心包、Schema、CLI 工具、实验 artifacts，以及可运行的 GenericAgent adapter boundary。GenericAgent 本身不会被复制、vendoring 或 fork 到本仓库中。

## 📅 News

- **2026-05-09:** 发布 Bayesian-Agent v0.4 独立 package，包含跨 harness Bayesian Skill Evolution 核心 primitives、schemas、CLI utilities 和实验 artifacts。
- **2026-05-09:** 增加可选 GenericAgent adapter boundary，不复制、不 vendoring GenericAgent。
- **2026-05-09:** 发布中英文项目文档和 Bayesian-Agent 方法框架图。

## 🌟 项目简介

LLM Agent 工程正在经历三层演化：

1. **Prompt Engineering**：把任务指令写得更清楚。
2. **Context Engineering**：决定推理时模型能看到什么证据。
3. **Harness Engineering**：把模型放进一个可执行、可观测、可恢复的系统里。

Prompt 可以改善单轮回答。Context 可以改善单次决策。Harness Engineering 解决的是更现实的问题：Agent 要跨工具、文件、测试、日志、记忆和失败恢复连续工作。

在这个语境下，**Skill** 和 **SOP** 不再只是长 prompt，而是 Agent 的核心工程资产。一条好的 Skill 是压缩后的操作知识：

- 先检查什么
- 用哪些工具
- 如何验证进度
- 哪些失败模式要规避
- 什么时候停止、重试或重写流程

Bayesian-Agent 想回答的问题是：既然 Skill 本质上是“如何完成任务”的假设，为什么它的进化要靠经验堆叠，而不是靠证据更新？我们的答案是一个框架无关的进化层：既能从零 bootstrap Skills，也能增量修复已有 Agent，还能在不同 harness 之间迁移，只要它们能产出 verified trajectories。

<div align="center">
  <img src="assets/bayesian_agent_overview.png" width="900" alt="Bayesian-Agent overview"/>
  <br/>
  <em>Bayesian-Agent 将任意兼容 harness 的 verified trajectories 转化为 posterior 加权的 Skills 和 SOPs。</em>
</div>

## 🧠 核心思想

从 MECE 的角度看，大语言模型系统优化只有两条路：

1. 改变模型参数分布，例如预训练、微调、强化学习。
2. 改变推理条件，例如 prompt、context、RAG、工具、记忆和 harness。

Bayesian-Agent 聚焦第二条路。

如果基础模型采样自：

```text
P(X | theta)
```

那么 Agent 系统采样自：

```text
P(X | theta, C)
```

其中 `C` 是推理环境。Skills、SOPs、工具、记忆、检索证据、执行轨迹和 verifier 反馈都属于 `C`。

Bayesian-Agent 把每条 Skill 或 SOP 看作一个关于成功率的假设：

```text
P(success | theta, C, skill)
```

每次得到经过验证的执行轨迹后，框架都会更新该 Skill 的 posterior belief。下一次运行时，Agent 拿到的是 posterior 加权后的 Skill context，而不是未经筛选的长记忆。

### v0.x 里的 “Bayesian” 准确指什么

当前 Bayesian-Agent v0.x 使用的是 **Beta-Bernoulli Bayesian update**：为每条 Skill/SOP 维护一个关于成功率的 posterior belief。它还没有做完整的 Bayesian model selection，也没有显式计算多个 Skill hypothesis 之间的 posterior probability。

对一条 Skill hypothesis `h_k`，定义：

```text
p_k = P(y = 1 | h_k, context)
y_i ~ Bernoulli(p_k)
p_k ~ Beta(alpha_0, beta_0)
```

观察到 `s_k` 次验证成功、`f_k` 次验证失败后：

```text
p_k | D_k ~ Beta(alpha_0 + s_k, beta_0 + f_k)
E[p_k | D_k] = (alpha_0 + s_k) / (alpha_0 + beta_0 + s_k + f_k)
```

当前实现使用 `alpha_0 = beta_0 = 1`，然后用 posterior mean 来排序 Skills、渲染 posterior 加权 context，并触发 `patch`、`split`、`compress`、`retire`、`explore` 等 rewrite actions。

## 📋 核心特性

- **证据加权的 Skill 进化**：从 verified success/failure trajectory 更新 Skill belief。
- **Bayesian Skill Registry**：维护 Beta posterior、失败模式、token 成本、延迟、轮次和 context 分布。
- **面向失败模式的修复**：识别反复出现的错误，生成聚焦的 repair plan。
- **Token-aware context 构建**：只注入真正有 posterior 证据价值的 Skill。
- **从零全量自进化**：完整运行任务，在线收集 evidence，并在无历史 traces 的情况下进化 Skills。
- **已有 Agent 的增量修复层**：读取 baseline agent 的失败轨迹，只重跑失败任务。
- **跨 harness 适配**：当前集成 GenericAgent，后续通过 adapters 接入其他 agent frameworks，而不是复制它们的代码。
- **标准库优先**：v0.4 核心运行时不依赖 Python 标准库之外的包。

## 🧬 自我进化机制

<div align="center">
  <img src="assets/bayesian_agent_framework_v2.svg" width="900" alt="Bayesian-Agent framework"/>
  <br/>
  <em>Bayesian Skill Evolution 方法框架。</em>
</div>

```text
[Agent Trajectory]
      |
      v
[Verifier / Benchmark Grader]
      |
      v
[TrajectoryEvidence: success, failure mode, tokens, turns, latency]
      |
      v
[Bayesian Skill Registry: posterior + cost + contexts]
      |
      v
[Rewrite Policy: compress, patch, split, retire, explore]
      |
      v
[Posterior-Weighted Skill Context]
      |
      v
[Next Agent Run]
```

对每条 Skill 或 benchmark SOP，Bayesian-Agent 会维护：

- 成功率的 Beta posterior
- 经过验证的成功和失败证据
- 失败模式计数
- input、output、total token 统计
- 延迟和轮次统计
- context 分布
- rewrite policy 建议

默认 rewrite policy 保持小而清晰：

| Posterior 信号 | Policy 动作 |
|---|---|
| 多次验证成功 | compress 或 reinforce |
| 失败模式聚集 | patch |
| 不同 context 下表现分化 | split 或 specialize |
| 失败占主导 | retire 或 rewrite |
| 证据稀疏 | explore |

## 🚀 安装

```bash
git clone https://github.com/DataArcTech/Bayesian-Agent.git
cd Bayesian-Agent
python -m pip install -e .
```

当前版本要求 Python 3.9+，运行时不依赖 Python 标准库之外的包。

## ⚡ 快速开始

从已有 Agent 结果中更新 Bayesian Skill registry：

```bash
bayesian-agent evolve \
  --results artifacts/ga_deepseek_baseline/sop_results.json \
  --registry temp/bayesian_skill_beliefs.json \
  --context-out temp/skill_context.md
```

找到需要增量修复的失败任务：

```bash
bayesian-agent repair-plan \
  --baseline artifacts/ga_deepseek_baseline/sop_results.json \
  --out temp/failed_tasks.json
```

汇总一次运行：

```bash
bayesian-agent summarize \
  --results artifacts/bayesian_incremental/results.json \
  --out temp/summary.json
```

跑一次真实 GenericAgent-backed SOP/Lifelong 实验。用 `--model` 在 `deepseek-v4-flash` 和 `deepseek-v4-pro` 之间切换：

```bash
cd Bayesian-Agent
export GENERICAGENT_ROOT="/path/to/GenericAgent"
export DEEPSEEK_API_KEY="sk-..."
export MODEL="deepseek-v4-flash"
"$GENERICAGENT_ROOT/.venv/bin/python" \
  experiments/run_sop_lifelong.py \
  --genericagent-root "$GENERICAGENT_ROOT" \
  --model "$MODEL" \
  --mode all \
  --bench core \
  --out-root "temp/sop_lifelong_${MODEL//-/_}"
```

想先 smoke test 可以加 `--limit 1`，确认脚本和 token 统计正常后再跑全量。

如果要接一个已有 GA baseline 做增量修复，把结果文件通过 `--baseline-results` 传进来即可。脚本只会重跑失败任务：

```bash
"$GENERICAGENT_ROOT/.venv/bin/python" \
  experiments/run_sop_lifelong.py \
  --genericagent-root "$GENERICAGENT_ROOT" \
  --model "$MODEL" \
  --mode bayesian-incremental \
  --bench core \
  --baseline-results artifacts/ga_deepseek_baseline/sop_results.json \
  --baseline-results artifacts/ga_deepseek_baseline/lifelong_results.json \
  --out-root "temp/sop_lifelong_${MODEL//-/_}_incremental_from_ga"
```

## 🐍 Python API

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
        total_tokens=74365,
    )
)

skill_context = SkillContextBuilder(registry).render(task_context="sop_bench")
print(skill_context)
```

## 🔁 三种运行形态

### 🌱 全量 Self-Evolving Mode

Bayesian-Agent 从零开始运行 benchmark tasks，收集 verified evidence，并在运行过程中持续进化 Skills。

这个模式验证的是：在没有先验 benchmark traces 的情况下，Bayesian Skill Evolution 是否能提升 Agent。

### 🛠️ 增量 Repair Mode

Bayesian-Agent 也可以挂载到一个已有 Agent 后面。基础 Agent 先跑一遍，Bayesian-Agent 读取其成功和失败轨迹，更新 Skill posterior，然后只重跑失败任务。

```text
Base Agent -> Failure Traces -> Bayesian Skill Evolution -> Rerun Failures -> Higher Accuracy
```

这是更推荐的生产路径，因为它不需要重新训练模型，也不需要替换原有 harness。

### 🔌 Cross-Harness Adaptation Mode

Bayesian-Agent 不绑定某一个 agent runtime。任何 agent framework 只要能产出统一 trajectory schema，并通过 adapter 接收 posterior 加权的 Skill context，都可以成为 Bayesian-Agent 的后端。

```text
Any Agent Harness -> Trajectory Schema -> Bayesian Skill Registry -> Adapter -> Next Harness Run
```

这让 Bayesian-Agent 更像一个可移植的 Skill/SOP 进化层，而不是又一个封闭 agent framework。

## 📊 实验结果

v0.4 原型基于 GenericAgent 与 `deepseek-v4-flash`，在 SOP-Bench 和 Lifelong AgentBench 上完成验证。

### 🧱 Baseline: GenericAgent + deepseek-v4-flash

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA | deepseek-v4-flash | 80% | 1.34M | 57k | 1.39M | 11.47 |
| Lifelong AgentBench | GA | deepseek-v4-flash | 90% | 649k | 42k | 690k | 26.07 |

### 🌱 全量 Self-Evolving Run

| Benchmark | Agent | Model | Accuracy | Input Tokens | Output Tokens | Total Tokens | Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+Bayesian | deepseek-v4-flash | 100% | 1.07M | 52k | 1.12M | 17.86 |
| Lifelong AgentBench | GA+Bayesian | deepseek-v4-flash | 95% | 666k | 44k | 710k | 26.77 |

全量模式下，Bayesian-Agent 将 SOP-Bench 从 80% 提升到 100%，同时 token 消耗从 1.39M 降到 1.12M。Lifelong AgentBench 从 90% 提升到 95%，token 成本基本相当。

### 🛠️ 增量 Repair Run

增量模式下，Bayesian-Agent 只重跑 GenericAgent 的失败任务：

- SOP-Bench：4 个失败任务，全部修复
- Lifelong AgentBench：2 个失败任务，全部修复

| Benchmark | Agent | Model | Final Accuracy | Incremental Input | Incremental Output | Incremental Total | Incremental Efficiency |
|---|---|---|---:|---:|---:|---:|---:|
| SOP-Bench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 254k | 14k | 268k | 14.93 |
| Lifelong AgentBench | GA+BayesianIncremental | deepseek-v4-flash | 100% | 129k | 10k | 139k | 14.41 |

这说明 Bayesian-Agent 可以作为即插即用的 repair layer：接在一个未达到 100% 准确率的 Agent 后面，用较小的增量推理成本把失败任务补齐。这也是它区别于普通 benchmark agent 的关键：它可以站在 harness 旁边，学习它的失败，并在不替换它的情况下提升它。

实验 artifacts 位于 [`artifacts/`](artifacts/)，方法说明位于 [`docs/method.md`](docs/method.md)。

如果要用另一个模型复现实验形态，只需要改 `--model` 和 `--out-root`：

```bash
export MODEL="deepseek-v4-pro"
"$GENERICAGENT_ROOT/.venv/bin/python" \
  experiments/run_sop_lifelong.py \
  --genericagent-root "$GENERICAGENT_ROOT" \
  --model "$MODEL" \
  --mode all \
  --bench core \
  --out-root "temp/sop_lifelong_${MODEL//-/_}"
```

默认会依次跑三段：GA baseline、Bayesian 全量自进化、Bayesian 基于所选模型新 baseline 的增量修复。结果会写到 `<out-root>/summary.md`。

## 🔌 GenericAgent 与跨 Harness 适配

第一个原型是在 GenericAgent 内部验证的，但 Bayesian-Agent 不是 GenericAgent fork，也不只是 GenericAgent 的附属模块。

开源结构是：

- `bayesian_agent/core/`：框架无关的 Bayesian Skill Evolution 逻辑
- `bayesian_agent/adapters/base.py`：外部 Agent 的最小 adapter contract
- `bayesian_agent/adapters/generic_agent.py`：可选 GenericAgent 集成边界
- `schemas/`：可移植的 trajectory 与 Skill belief schema
- `artifacts/`：可复现实验结果文件

GenericAgent 是当前实验后端。其他 Agent harness 只要能产出统一 trajectory schema，并实现 adapter boundary，也可以接入 Bayesian-Agent。

长期方向是让 Bayesian-Agent 成为多个 agent runtime 共享的 Bayesian Skill/SOP evolution layer：包括 GenericAgent、我们后续会上传的自研 Agent harness，以及其他外部框架。

MinimalAgent adapter 在 v0.4 中按计划暂不提供。

## 🗂️ 仓库结构

```text
bayesian_agent/
  core/                 # Evidence, beliefs, registry, policy, context, repair
  adapters/             # Adapter contract and optional GenericAgent boundary
schemas/                # JSON schemas for trajectories and Skill beliefs
artifacts/              # Baseline, full-mode, and incremental-mode result artifacts
docs/                   # Method and experiment notes
examples/               # Integration notes
tests/                  # Standard-library unittest suite
```

## 🧭 Roadmap

- [x] 将 GenericAgent 原型重构为独立 package core。
- [x] 定义通用 agent run trace schema。
- [x] 实现 Bayesian Skill registry。
- [x] 实现 full self-evolving primitives。
- [x] 实现 incremental repair utilities。
- [x] 增加 GenericAgent optional adapter boundary，不 vendoring GenericAgent。
- [x] 发布实验结果 artifacts。
- [x] 增加英文和中文 README。
- [ ] 增加可在外部 checkout 中执行的 benchmark runners。
- [ ] 增加更丰富的 rewrite policies 和 adapter examples。
- [ ] GenericAgent 边界稳定后再扩展更多 agent harness adapters。
- [ ] 上传我们自己的 Agent harness；当前实验阶段使用 GenericAgent 作为 backend harness。
- [ ] 从 per-Skill Beta-Bernoulli update 升级到更完整的 Bayesian reasoning，包括 Skill hypothesis inference、Bayesian Networks 等 context-aware Bayesian structure、不确定性感知的 Skill selection、Bayesian decision policies 和 online adaptation。

## 🚦 当前状态

Bayesian-Agent v0.4 是早期独立版本。当前 package 可用于 trace ingestion、Bayesian Skill belief update、context rendering、repair planning 和 result summarization，并已经验证从零全量进化、增量修复、跨 harness 适配三条路径。完整 benchmark execution 当前仍使用 GenericAgent 等外部 Agent harness；后续会上传我们自己的 Agent harness。

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DataArcTech/Bayesian-Agent&type=Date)](https://www.star-history.com/#DataArcTech/Bayesian-Agent&Date)

## 📝 Citation

如果你在研究或项目中使用 Bayesian-Agent，欢迎按如下方式引用：

```bibtex
@software{bayesian_agent_2026,
  title = {Bayesian-Agent: A Bayesian Self-Evolving Agent Framework with Cross-Harness Adaptation},
  author = {{Xiaojun Wu}},
  year = {2026},
  url = {https://github.com/DataArcTech/Bayesian-Agent}
}
```

## 📄 License

MIT License. 详见 [`LICENSE`](LICENSE)。

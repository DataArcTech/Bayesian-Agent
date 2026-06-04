# 从零运行中的复杂证据聚合：一次 Skill Rewrite 的完整 Bayesian Evidence Trace

上一篇文章解释了 Bayesian-Agent 的基本公式：

```text
P(y | h_k, x) ∝ P(y | h_k) * Π_j P(x_j | y, h_k)
```

这篇文章只做一件事：把一个从零运行、中途失败、触发 Skill rewrite、再被修复的例子展开到实现细节级别。

重点不是再讲一遍“失败后要改写 Skill”，而是展示当前代码到底把哪些因素放进 evidence model，如何连乘，以及哪一步真正触发 rewrite。

## 一、当前实现到底使用哪些 evidence features

Bayesian-Agent v0.5 的默认 backend 是 `categorical_bayes`。对每条 `TrajectoryEvidence`，当前实现会抽取这些离散特征：

```text
features = {
  context,
  failure_mode,
  token_bucket,
  turn_bucket,
  latency_bucket,
  metadata.*
}
```

其中 bucket 的规则来自 `features_from_event`：

| 特征 | 输入字段 | 当前 bucket |
|---|---|---|
| `token_bucket` | `total_tokens` | `0`, `1_1k`, `1k_10k`, `10k_100k`, `100k_1m`, `1m_plus` |
| `turn_bucket` | `turns` | `0`, `1_2`, `3_5`, `6_10`, `11_20`, `20_plus` |
| `latency_bucket` | `elapsed_seconds` | `0s`, `0s_10s`, `10s_60s`, `1m_5m`, `5m_30m`, `30m_plus` |

`metadata.*` 也不是任意内容都会进入模型。当前实现只接受：

```text
str / int / float / bool
并且 len(str(value)) <= 80
```

进入模型以后，metadata key 会被加上前缀。例如：

```text
metadata = {"output_contract": "csv_expected_output"}
```

会变成：

```text
metadata.output_contract = csv_expected_output
```

复杂 dict、list、过长字符串不会进入这个 categorical likelihood model。

## 二、例子设定：SOP-Bench 从零跑到中途失败

这个例子使用 SOP-Bench，因为它的失败模式比较清楚：Agent 已经算出了流程，但没有把目标 CSV 行的 `expected_output` 写成非空值。

我们从空 registry 开始，只有一条 benchmark Skill：

```text
h = benchmark/sop_bench
success = 0
failure = 0
P_h(success) = 0.500
```

前 3 个任务成功，接着 2 个任务出现同一种失败：

```text
failure_mode = left_expected_output_blank
```

下面这 5 条 evidence 都是当前实现能直接表示的 `TrajectoryEvidence`。成功任务没有 failure mode，进入 feature extraction 后会被表示成：

```text
failure_mode = __none__
```

| Task | Outcome | failure_mode feature | total_tokens | token_bucket | turns | turn_bucket | elapsed_seconds | latency_bucket | metadata.output_contract |
|---|---|---|---:|---|---:|---|---:|---|---|
| `sop_01` | success | `__none__` | 42000 | `10k_100k` | 4 | `3_5` | 45 | `10s_60s` | `csv_expected_output` |
| `sop_02` | success | `__none__` | 51000 | `10k_100k` | 5 | `3_5` | 52 | `10s_60s` | `csv_expected_output` |
| `sop_03` | success | `__none__` | 65000 | `10k_100k` | 7 | `6_10` | 180 | `1m_5m` | `csv_expected_output` |
| `sop_13` | failure | `left_expected_output_blank` | 124000 | `100k_1m` | 12 | `11_20` | 420 | `5m_30m` | `csv_expected_output` |
| `sop_14` | failure | `left_expected_output_blank` | 136000 | `100k_1m` | 14 | `11_20` | 610 | `5m_30m` | `csv_expected_output` |

这里的 metadata 并不是额外编造的维度。SOP-Bench runner 在结果里会写入：

```text
output_contract = csv_expected_output
```

这个字段是短字符串，所以会通过当前 `features_from_event` 的 metadata 过滤规则。

## 三、先验：只看 success/failure 的 class probability

5 条 evidence 之后：

```text
success = 3
failure = 2
N = 5
alpha = 1
Y = {success, failure}
```

当前实现使用 Laplace smoothing：

```text
P_h(y) = (N_y + alpha) / (N + alpha * |Y|)
```

所以：

```text
P_h(success)
= (3 + 1) / (5 + 2)
= 4/7
≈ 0.571

P_h(failure)
= (2 + 1) / (5 + 2)
= 3/7
≈ 0.429
```

如果只看总体成功率，这条 Skill 还没有崩。它不是 0，也不是 100%，而是处在一个“有成功经验，但出现了重复失败”的状态。

## 四、似然：把所有 evidence features 一起连乘

现在我们分析这簇失败的特征：

```text
x_risk = {
  context = sop_bench,
  failure_mode = left_expected_output_blank,
  token_bucket = 100k_1m,
  turn_bucket = 11_20,
  latency_bucket = 5m_30m,
  metadata.output_contract = csv_expected_output
}
```

`failure_mode` 是 verifier 在失败后贴上的诊断标签。它不是 Agent 运行前能预知的输入，而是用来解释失败簇、生成 patch、并在 registry 中保留教训的 evidence feature。

当前实现的 feature likelihood 是：

```text
P_h(x_j = v | y)
= (N_{j,v,y} + alpha) / (N_{j,y} + alpha * |V_j|)
```

其中：

- `N_{j,v,y}`：在标签 `y` 下，第 `j` 个特征取值为 `v` 的次数。
- `N_{j,y}`：在标签 `y` 下，第 `j` 个特征被观察到的总次数。
- `|V_j|`：第 `j` 个特征目前见过的取值个数。
- `alpha = 1`：Laplace smoothing。

对 `x_risk`，每个 likelihood 项如下：

| Feature value | <code>P(value &#124; success, h)</code> | <code>P(value &#124; failure, h)</code> | 解释 |
|---|---:|---:|---|
| `context = sop_bench` | `1.000` | `1.000` | 5 条 evidence 都来自 SOP-Bench，所以这个特征不区分成功/失败 |
| `failure_mode = left_expected_output_blank` | `(0+1)/(3+2)=0.200` | `(2+1)/(2+2)=0.750` | 该失败模式只在失败样本中出现 |
| `token_bucket = 100k_1m` | `(0+1)/(3+2)=0.200` | `(2+1)/(2+2)=0.750` | 高 token bucket 只在这两次失败中出现 |
| `turn_bucket = 11_20` | `(0+1)/(3+3)=0.167` | `(2+1)/(2+3)=0.600` | 长 turn run 更像这簇失败 |
| `latency_bucket = 5m_30m` | `(0+1)/(3+3)=0.167` | `(2+1)/(2+3)=0.600` | 高延迟也和这簇失败同现 |
| `metadata.output_contract = csv_expected_output` | `(3+1)/(3+1)=1.000` | `(2+1)/(2+1)=1.000` | 所有样本都是同一输出契约，因此它是中性条件 |

注意最后一行：metadata 被连乘进去了，但在这个例子里它不改变成功/失败比例，因为成功和失败样本都满足同一个 `output_contract`。这不是 bug，而是证据本身没有区分度。

## 五、后验：风险特征下 failure posterior 被显著拉高

当前代码为了数值稳定，在 `predict_proba` 里使用 log probability 累加：

```text
log_score(y)
= log P_h(y) + sum_j log P_h(x_j | y)
```

数学上等价于：

```text
score(y)
= P_h(y) * Π_j P_h(x_j | y)
```

把上一节所有项连乘：

```text
score(success)
= 4/7
  * 1.000
  * 0.200
  * 0.200
  * 0.167
  * 0.167
  * 1.000
≈ 0.000635

score(failure)
= 3/7
  * 1.000
  * 0.750
  * 0.750
  * 0.600
  * 0.600
  * 1.000
≈ 0.086786
```

归一化：

```text
P_h(failure | x_risk)
= score(failure) / (score(success) + score(failure))
= 0.086786 / (0.000635 + 0.086786)
≈ 0.993

P_h(success | x_risk)
≈ 0.007
```

这一步表达的不是“系统已经知道下一题会失败”，而是：

```text
在已经观察到的 evidence 中，
left_expected_output_blank + high token bucket + long turns + high latency
这一组特征非常像失败簇，而不像成功簇。
```

也就是说，失败不是一个孤立点，而是形成了可解释的 evidence cluster。

## 六、真正触发 Skill rewrite 的条件是什么

这里要非常精确。当前 v0.5 的 `RewritePolicy` 并不是直接读取上面的 `P_h(failure | x_risk)` 来决定 rewrite。

当前代码中的触发顺序可以概括为：

```text
if observations == 0:
    explore
elif beta >= 4 and posterior_success < 0.45:
    retire
elif max(failure_mode_count) >= 2:
    patch
elif context_count >= 3 and observations >= 4:
    split
elif observations >= 3 and posterior_success >= 0.72:
    compress
else:
    explore
```

这 5 条 evidence 之后：

```text
failure_modes = {
  left_expected_output_blank: 2
}
```

所以当前实现触发的是：

```text
rewrite = patch
reason = failures cluster around a recurring mode
confidence = 0.75
```

对应的 posterior audit artifact 会保存类似：

```text
### Bayesian Posterior Audit
Posterior summaries are for ranking, rewrite decisions, and debugging; model-facing prompts should use executable Skill/SOP text.
- benchmark/sop_bench: algorithm=categorical_bayes, posterior_success=0.571, context_success=0.571, alpha=4.0, beta=3.0, observations=5, mean_tokens=83600.0, rewrite=patch, failures=left_expected_output_blank=2
Current task files and runtime feedback remain authoritative.
```

这类内容保存在 `posterior_context_after.md` 或 `belief_after.json` 里，用于审计和解释，不进入 SOP-Bench/Lifelong 的真实模型 prompt。这里的 `posterior_success=0.571` 是不带特定 features 的总体 belief；上面的 `P_h(failure | x_risk)=0.993` 是带完整风险特征的解释性 posterior。两者都来自同一份 evidence，但服务于不同位置：

- `posterior_success`：用于 Skill 排序和审计展示。
- feature-conditioned posterior：用于解释某个 failure cluster 为什么危险。
- `failure_modes` count：当前 v0.5 里实际触发 `patch` 的规则。

## 七、patch 不是泛泛提醒，而是 failure-mode-specific guardrail

SOP-Bench runner 有一个 benchmark-specific patch catalog。对：

```text
failure_mode = left_expected_output_blank
```

当前实现会把 posterior 决策转成模型可执行的 patch section：

```text
### Bayesian Failure-Mode Patches: sop_bench
- failure_mode=left_expected_output_blank observed=2
  - After writing, re-read `test_set_with_outputs.csv` and confirm the target row's `expected_output` is non-empty.
  - If the target cell is empty, write the computed raw category string before finishing.
```

这就是 Skill rewrite 在当前 v0.5 里的具体形态：它不是直接生成一个全新的 child Skill，也不是更新模型参数，而是在下一轮 prompt/context 中加入针对失败模式的可执行约束。

这个 patch 会和稳定的 benchmark guardrails 一起进入下一轮任务 context，例如：

```text
- Read `sop.txt`, `tools.py`, and the target CSV row before acting.
- Compute only the target row and write only its `expected_output` cell.
- Write the raw category string only, for example `manual_review`; never write XML tags, Markdown, quotes, or explanations into the cell.
- Verify the target row's `expected_output` is non-empty before finishing.
```

## 八、修复后的 evidence 如何改变后验

接下来系统用带 patch 的 context 重跑两个失败任务。假设它们都通过 verifier：

| Task | Outcome | failure_mode feature | total_tokens | token_bucket | turns | turn_bucket | elapsed_seconds | latency_bucket | metadata.output_contract |
|---|---|---|---:|---|---:|---|---:|---|---|
| `sop_13_retry` | success | `__none__` | 68000 | `10k_100k` | 7 | `6_10` | 230 | `1m_5m` | `csv_expected_output` |
| `sop_14_retry` | success | `__none__` | 59000 | `10k_100k` | 6 | `6_10` | 190 | `1m_5m` | `csv_expected_output` |

现在总 evidence 变成：

```text
success = 5
failure = 2
N = 7
```

新的 class prior：

```text
P_h(success)
= (5 + 1) / (7 + 2)
= 6/9
≈ 0.667

P_h(failure)
= (2 + 1) / (7 + 2)
= 3/9
≈ 0.333
```

如果看一条健康的、修复后的轨迹特征：

```text
x_healthy = {
  context = sop_bench,
  failure_mode = __none__,
  token_bucket = 10k_100k,
  turn_bucket = 6_10,
  latency_bucket = 1m_5m,
  metadata.output_contract = csv_expected_output
}
```

各项 likelihood 是：

| Feature value | <code>P(value &#124; success, h)</code> | <code>P(value &#124; failure, h)</code> |
|---|---:|---:|
| `context = sop_bench` | `1.000` | `1.000` |
| `failure_mode = __none__` | `(5+1)/(5+2)=0.857` | `(0+1)/(2+2)=0.250` |
| `token_bucket = 10k_100k` | `(5+1)/(5+2)=0.857` | `(0+1)/(2+2)=0.250` |
| `turn_bucket = 6_10` | `(3+1)/(5+3)=0.500` | `(0+1)/(2+3)=0.200` |
| `latency_bucket = 1m_5m` | `(3+1)/(5+3)=0.500` | `(0+1)/(2+3)=0.200` |
| `metadata.output_contract = csv_expected_output` | `1.000` | `1.000` |

连乘：

```text
score(success)
= 0.667
  * 1.000
  * 0.857
  * 0.857
  * 0.500
  * 0.500
  * 1.000
≈ 0.122449

score(failure)
= 0.333
  * 1.000
  * 0.250
  * 0.250
  * 0.200
  * 0.200
  * 1.000
≈ 0.000833
```

归一化：

```text
P_h(success | x_healthy)
= 0.122449 / (0.122449 + 0.000833)
≈ 0.993
```

这就是“失败 -> patch -> 修复成功 -> 后验上升”的完整链条。

## 九、失败记忆不会被洗掉

一个容易误解的点是：修复成功以后，Bayesian-Agent 不会删除之前的失败 evidence。

如果未来又出现和旧失败簇相同的风险特征：

```text
x_risk = {
  context = sop_bench,
  failure_mode = left_expected_output_blank,
  token_bucket = 100k_1m,
  turn_bucket = 11_20,
  latency_bucket = 5m_30m,
  metadata.output_contract = csv_expected_output
}
```

在 5 次成功、2 次失败之后，模型仍然会给出很高的 failure posterior：

```text
P_h(failure | x_risk) ≈ 0.997
```

这很合理。Bayesian-Agent 学到的不是“我现在永远不会犯错”，而是两件更可用的知识：

```text
1. 带 patch 的正常运行轨迹越来越可靠。
2. left_expected_output_blank 这类失败簇仍然需要被 guardrail 约束。
```

所以当前 v0.5 里，即使后续 repair 成功，`failure_modes` 计数仍然会留在 registry 中。第一次出现的 failure mode 只作为 candidate evidence 保存在 audit artifact 中；同一 failure mode 至少出现两次后，context 里才会保留相关 active patch。这比“一错就改 skill”更稳，也能降低单个异常样本导致过拟合的风险。

## 十、这个例子说明了什么

这个复杂例子展示了当前 Bayesian-Agent 的真实工作方式：

```text
从零运行:
  registry 为空，先 explore

成功 evidence:
  success prior 上升，Skill 开始有可靠性证据

失败 evidence:
  failure_mode、token_bucket、turn_bucket、latency_bucket、metadata 一起进入 evidence model

特征连乘:
  风险特征下的 posterior_failure 显著上升

rewrite 触发:
  当前 RewritePolicy 看到同一 failure mode 出现 2 次，触发 patch

context 改写:
  benchmark-specific patch rules 被注入下一轮 prompt；单次失败只进入 audit，不进入 active prompt patch

repair 成功:
  成功 evidence 回写 registry，健康轨迹的 posterior_success 上升

失败记忆:
  旧 failure cluster 仍保留，防止系统忘记已经踩过的坑
```

这里的 Bayesian 并不是一句包装词。它至少体现在三个层次：

1. **Prior**：每条 Skill/SOP 都有平滑后的成功/失败基率。
2. **Likelihood**：context、failure mode、token bucket、turn bucket、latency bucket、metadata 都以 categorical likelihood 的形式统计。
3. **Posterior**：新 evidence 改变下一轮 Skill 排序、failure patch、context 渲染和 repair 行为。

同时也要准确地说，v0.5 不是完整的 Bayesian model selection。它还没有把多个 child Skill hypothesis 放进一个统一的后验竞争框架：

```text
P(h_k | D) ∝ P(D | h_k) P(h_k)
```

当前版本做的是更小但很实用的一步：让每条 Skill/SOP 从 verified trajectories 中形成可解释、可更新、可迁移的 evidence-weighted belief。

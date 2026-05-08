# Python API

The Python API is small and built around evidence, beliefs, registries, policies, and context rendering.

## TrajectoryEvidence

```python
from bayesian_agent import TrajectoryEvidence

event = TrajectoryEvidence(
    task_id="sop_12",
    skill_id="benchmark/sop_bench",
    context="sop_bench",
    outcome="failure",
    failure_mode="xml_wrapped_answer",
    input_tokens=70123,
    output_tokens=4242,
)
```

`total_tokens` is automatically computed when omitted.

## BayesianSkillRegistry

```python
from bayesian_agent import BayesianSkillRegistry

registry = BayesianSkillRegistry("temp/beliefs.json")
belief = registry.record(event)

print(belief.success_probability)
```

Use an in-memory registry for tests:

```python
registry = BayesianSkillRegistry.in_memory()
```

## SkillContextBuilder

```python
from bayesian_agent import SkillContextBuilder

context = SkillContextBuilder(registry).render(task_context="sop_bench", limit=5)
print(context)
```

The renderer orders beliefs by context match, posterior success probability, observation count, and token cost.

## RewritePolicy

```python
from bayesian_agent.core.policy import RewritePolicy

decision = RewritePolicy().decide(belief)
print(decision.action, decision.reason, decision.confidence)
```

Default actions are:

- `explore`
- `compress`
- `patch`
- `split`
- `retire`

## End-to-End Example

```python
from bayesian_agent import BayesianSkillRegistry, SkillContextBuilder, TrajectoryEvidence

events = [
    TrajectoryEvidence(
        task_id="task_1",
        skill_id="skill/search_then_verify",
        context="qa",
        outcome="success",
        input_tokens=1200,
        output_tokens=300,
    ),
    TrajectoryEvidence(
        task_id="task_2",
        skill_id="skill/search_then_verify",
        context="qa",
        outcome="failure",
        failure_mode="missing_verification",
        input_tokens=1400,
        output_tokens=350,
    ),
]

registry = BayesianSkillRegistry.in_memory()
registry.record_many(events)

print(SkillContextBuilder(registry).render(task_context="qa"))
```

## Public Import Surface

The package root exports:

```python
from bayesian_agent import BayesianSkillRegistry, SkillContextBuilder, TrajectoryEvidence
```

Lower-level types are available from `bayesian_agent.core`.

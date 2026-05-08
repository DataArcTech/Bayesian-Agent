# GenericAgent Integration

Bayesian-Agent does not copy or vendor GenericAgent.

The current GenericAgent prototype should be treated as an external host agent. A local integration can be built by implementing:

```python
from bayesian_agent.adapters.base import AgentAdapter


class LocalGenericAgentAdapter:
    def run(self, task, skill_context):
        # 1. Load your local GenericAgent checkout.
        # 2. Inject skill_context into the task prompt.
        # 3. Run GenericAgent.
        # 4. Return a trajectory-like dict with task_id, success, tokens, and failure_mode.
        ...
```

The public adapter in `bayesian_agent.adapters.generic_agent.GenericAgentAdapter` is intentionally a thin boundary placeholder. It imports nothing from GenericAgent at module import time.

Recommended environment variable for local experiments:

```bash
export GENERIC_AGENT_HOME=/path/to/GenericAgent
```

Then use Bayesian-Agent CLI tools to evolve a registry and compute repair plans:

```bash
bayesian-agent evolve \
  --results artifacts/ga_deepseek_baseline/sop_results.json \
  --registry temp/bayesian_skill_beliefs.json \
  --context-out temp/skill_context.md

bayesian-agent repair-plan \
  --baseline artifacts/ga_deepseek_baseline/sop_results.json \
  --out temp/failed_tasks.json
```

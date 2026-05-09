# Contributing to Bayesian-Agent

Thanks for helping improve Bayesian-Agent.

Bayesian-Agent is not meant to become a monolithic agent runtime. Its core value is a Bayesian Skill/SOP evolution layer that can work with many execution harnesses. Contributions should preserve that boundary.

## Project Principles

- **Keep the core harness-agnostic**: `bayesian_agent/core/` must not import GenericAgent, browser runtimes, model SDKs, benchmark runners, or framework-specific code.
- **Do not vendor external agents**: integrations should point to local checkouts, packages, APIs, or commands. Do not copy another agent framework into this repository.
- **Evidence first**: adapter outputs should be convertible into verified `TrajectoryEvidence`.
- **Small interfaces over large abstractions**: prefer one clear adapter boundary over a deep plugin hierarchy.
- **Standard-library-first**: avoid runtime dependencies unless there is a strong reason.
- **Tests before confidence**: every adapter contribution should include focused tests that run with `python3 -m unittest discover -v`.

## Repository Map

```text
bayesian_agent/
  core/                 # Framework-agnostic evidence, beliefs, registry, policy, context, repair
  adapters/             # Adapter protocol and optional external harness boundaries
schemas/                # Portable JSON schemas for trajectories and Skill beliefs
artifacts/              # Experiment result artifacts
docs/                   # Documentation site content
tests/                  # unittest test suite
```

## Adding a New Adapter

Adapters are how Bayesian-Agent connects to external agent frameworks.

The rule is simple:

```text
External Harness Run -> trajectory-like mapping -> TrajectoryEvidence -> Bayesian Skill Registry
Bayesian Skill Context -> Adapter -> External Harness Next Run
```

An adapter should execute one task with posterior-weighted Skill context and return a trajectory-like mapping that Bayesian-Agent can normalize.

### 1. Start From the Protocol

All adapters should satisfy `AgentAdapter`:

```python
from typing import Any, Mapping, Protocol, runtime_checkable

@runtime_checkable
class AgentAdapter(Protocol):
    def run(self, task: Mapping[str, Any], skill_context: str) -> Mapping[str, Any]:
        """Run one task with Bayesian Skill context and return a trajectory-like mapping."""
        ...
```

The adapter receives:

- `task`: a benchmark or application task mapping
- `skill_context`: posterior-weighted Skill/SOP context rendered by Bayesian-Agent

The adapter returns a mapping with enough information to build `TrajectoryEvidence`.

### 2. Create an Adapter Module

Create a focused file under `bayesian_agent/adapters/`.

Example:

```text
bayesian_agent/adapters/my_harness.py
```

Skeleton:

```python
"""Optional MyHarness integration boundary.

This module should not vendor MyHarness or import heavy dependencies eagerly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class MyHarnessAdapter:
    """Adapter for a local or installed MyHarness runner."""

    root: str

    def integration_note(self) -> str:
        return (
            "MyHarness integration is optional. Set root to a local MyHarness checkout; "
            "Bayesian-Agent does not copy or vendor MyHarness code."
        )

    def run(self, task: Mapping[str, Any], skill_context: str) -> Mapping[str, Any]:
        """Run one task and return a trajectory-like result.

        Keep this method as a thin boundary around the external harness.
        """
        raise NotImplementedError(
            "Wire this method to your local MyHarness runner."
        )
```

If the external harness has expensive imports, import them inside `run()` or behind a helper so `import bayesian_agent` remains lightweight.

### 3. Return a Trajectory-Like Mapping

The returned mapping should include fields compatible with `TrajectoryEvidence.from_run(...)`.

Minimum useful shape:

```python
{
    "task_id": "sop_12",
    "success": False,
    "input_tokens": 70123,
    "output_tokens": 4242,
    "total_tokens": 74365,
    "turns": 8,
    "elapsed_seconds": 31.5,
    "failure_mode": "xml_wrapped_answer",
    "summary": "The model wrapped the final answer in XML.",
    "metadata": {
        "harness": "my_harness",
        "model": "deepseek-v4-flash"
    }
}
```

Bayesian-Agent will attach `skill_id` and `context` when converting benchmark results into evidence, or your integration can construct `TrajectoryEvidence` directly.

The portable schema is documented in:

```text
schemas/trajectory.schema.json
```

### 4. Preserve the Harness Boundary

Good adapter behavior:

- calls an installed package, local checkout, CLI command, API, or user-provided callable
- returns normalized trajectory data
- keeps harness-specific logs in `metadata`
- fails with a clear error when the external harness is not configured

Avoid:

- copying external framework source code into `bayesian_agent/adapters/`
- importing a large framework at module import time
- hiding benchmark graders inside the adapter
- returning only free-form text without token usage or success signal
- changing `bayesian_agent/core/` to fit one harness

### 5. Add Tests

Add tests under `tests/`.

For a lightweight boundary adapter, test that it:

- satisfies `AgentAdapter`
- does not eagerly import the external framework
- preserves constructor configuration
- returns or documents a clear integration error

Example:

```python
import unittest

from bayesian_agent.adapters.base import AgentAdapter
from bayesian_agent.adapters.my_harness import MyHarnessAdapter


class MyHarnessAdapterTests(unittest.TestCase):
    def test_adapter_is_runtime_checkable(self):
        adapter = MyHarnessAdapter(root="/tmp/my-harness")
        self.assertIsInstance(adapter, AgentAdapter)

    def test_adapter_does_not_vendor_harness(self):
        adapter = MyHarnessAdapter(root="/tmp/not-installed")
        self.assertIn("does not copy or vendor", adapter.integration_note())


if __name__ == "__main__":
    unittest.main()
```

Run:

```bash
python3 -m unittest discover -v
python3 -m compileall bayesian_agent
```

### 6. Document the Adapter

Update docs when the adapter is user-visible:

- `docs/adapters.md`: explain setup and usage
- `docs/quick-start.md`: add a short example if the adapter is ready for users
- `README.md` / `README_ZH.md`: mention only mature integrations

If the adapter is only a boundary placeholder, say so clearly.

### 7. Adapter PR Checklist

Before opening a PR, make sure:

- [ ] The adapter does not vendor or copy external framework code.
- [ ] The adapter does not import heavy external dependencies at module import time.
- [ ] The adapter returns a trajectory-like mapping or documents how it emits `TrajectoryEvidence`.
- [ ] Harness-specific diagnostics go into `metadata`.
- [ ] Tests pass with `python3 -m unittest discover -v`.
- [ ] Compilation passes with `python3 -m compileall bayesian_agent`.
- [ ] Documentation explains setup, expected inputs, and returned evidence.

## General Development Workflow

1. Create a small branch.
2. Keep changes scoped to one concern.
3. Add or update tests.
4. Run verification:

```bash
python3 -m unittest discover -v
python3 -m compileall bayesian_agent
git diff --check
```

5. If docs changed, run:

```bash
uv run --with mkdocs-material mkdocs build --strict
```

6. Open a PR with:

- what changed
- why it matters
- how it was tested
- any remaining limitations

## Commit Style

Use short, descriptive commit messages:

```text
feat: add my-harness adapter boundary
test: cover my-harness adapter protocol
docs: document adapter contribution workflow
```

## Questions

If you are unsure whether an integration belongs in core or adapters, put it in adapters. The core package should remain portable, evidence-first, and harness-agnostic.

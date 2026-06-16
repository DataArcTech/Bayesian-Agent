# Bayesian Harness Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `bayesian_agent` its own focused harness core instead of letting benchmark modules directly glue external backends together.

**Architecture:** Keep external runners behind adapters, add a BA-owned harness package for task envelopes, prompt preparation, run artifacts, and memory lifecycle, and add a three-layer memory package inspired by hippocampus/intermediate-state/cortex. Benchmark code should depend on this narrow harness boundary and keep grading/evolution policy local.

**Tech Stack:** Python standard library, `unittest`, existing `BayesianSkillRegistry` and benchmark helpers.

---

### Task 1: Harness And Memory API Tests

**Files:**
- Create: `tests/test_harness_core.py`
- Create: `tests/test_memory_layers.py`

- [ ] **Step 1: Write failing harness tests**

```python
from pathlib import Path
import tempfile
import unittest

from bayesian_agent.harness import AgentHarness, HarnessTask


class HarnessCoreTests(unittest.TestCase):
    def test_harness_wraps_adapter_run_and_writes_artifacts(self):
        class Adapter:
            def run_task(self, *, prompt, workspace, max_turns):
                self.prompt = prompt
                return {"transcript": "ok", "input_tokens": 2, "output_tokens": 3}

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "task"
            adapter = Adapter()
            harness = AgentHarness(adapter)
            run = harness.run_task(HarnessTask(task_id="t1", prompt="solve", workspace=workspace, max_turns=4))

            self.assertEqual(adapter.prompt, "solve")
            self.assertEqual(run["task_id"], "t1")
            self.assertEqual(run["total_tokens"], 5)
            self.assertTrue((workspace / "harness_run.json").exists())

    def test_harness_injects_three_layer_memory_context(self):
        class Adapter:
            def run_task(self, *, prompt, workspace, max_turns):
                return {"transcript": prompt}

        with tempfile.TemporaryDirectory() as td:
            harness = AgentHarness(Adapter())
            harness.memory.hippocampus.remember("recent clue")
            harness.memory.state.set("phase", "draft")
            harness.memory.cortex.remember("stable rule")
            run = harness.run_task(HarnessTask(task_id="t1", prompt="solve", workspace=Path(td), memory_context=True))

            self.assertIn("Hippocampus", run["transcript"])
            self.assertIn("recent clue", run["transcript"])
            self.assertIn("phase: draft", run["transcript"])
            self.assertIn("stable rule", run["transcript"])
```

- [ ] **Step 2: Write failing memory tests**

```python
from pathlib import Path
import tempfile
import unittest

from bayesian_agent.memory import ThreeLayerMemory


class MemoryLayerTests(unittest.TestCase):
    def test_three_layer_memory_names_are_stable(self):
        memory = ThreeLayerMemory()

        self.assertEqual(memory.layer_names(), ["hippocampus", "state", "cortex"])

    def test_cortex_persists_while_fast_layers_can_reset(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cortex.json"
            memory = ThreeLayerMemory(cortex_path=path)
            memory.hippocampus.remember("fast")
            memory.state.set("working", "middle")
            memory.cortex.remember("durable")
            memory.save()

            memory.reset_fast_layers()
            reloaded = ThreeLayerMemory(cortex_path=path)

            self.assertEqual(memory.hippocampus.items, [])
            self.assertEqual(memory.state.values, {})
            self.assertIn("durable", reloaded.cortex.items)
```

- [ ] **Step 3: Verify red**

Run: `python -m unittest tests.test_harness_core tests.test_memory_layers -v`
Expected: fail because `bayesian_agent.harness` and `bayesian_agent.memory` do not exist yet.

### Task 2: Three-Layer Memory Package

**Files:**
- Create: `bayesian_agent/memory/__init__.py`
- Create: `bayesian_agent/memory/layers.py`

- [ ] **Step 1: Implement minimal memory layers**

```python
@dataclass
class HippocampusMemory:
    items: list[str] = field(default_factory=list)
    max_items: int = 12

    def remember(self, text: str) -> None:
        if text:
            self.items.append(str(text))
            self.items = self.items[-self.max_items:]
```

- [ ] **Step 2: Add intermediate state and cortical JSON persistence**

Use `StateMemory` for current task/session key-value state and `CorticalMemory` for durable JSON-backed items.

- [ ] **Step 3: Verify green for memory tests**

Run: `python -m unittest tests.test_memory_layers -v`
Expected: pass.

### Task 3: Harness Core Package

**Files:**
- Create: `bayesian_agent/harness/__init__.py`
- Create: `bayesian_agent/harness/types.py`
- Create: `bayesian_agent/harness/core.py`

- [ ] **Step 1: Implement `HarnessTask`**

Create a dataclass with `task_id`, `prompt`, `workspace`, `max_turns`, `skill_context`, `metadata`, and `memory_context`.

- [ ] **Step 2: Implement `AgentHarness`**

`AgentHarness.run_task()` should prepare memory context, call `adapter.run_task(prompt=..., workspace=..., max_turns=...)`, normalize token totals, add `task_id`, write `harness_task.json` and `harness_run.json`, and remember a concise run summary in hippocampus/state.

- [ ] **Step 3: Verify green for harness tests**

Run: `python -m unittest tests.test_harness_core -v`
Expected: pass.

### Task 4: Benchmark Wiring

**Files:**
- Modify: `bayesian_agent/benchmarks/sop_lifelong.py`
- Modify: `bayesian_agent/benchmarks/realfin.py`
- Modify: `experiments/run_benchmarks.py`
- Modify: `tests/test_experiment_scripts.py`

- [ ] **Step 1: Add a harness factory**

`experiments/run_benchmarks.py` should build the adapter, then wrap it in `AgentHarness`, while preserving adapter-specific dry-run information.

- [ ] **Step 2: Replace direct adapter task calls**

Benchmark modules should call `harness.run_task(HarnessTask(...))`. Grading remains in benchmark modules, and Bayesian belief updates still use verified graded results.

- [ ] **Step 3: Verify benchmark script tests**

Run: `python -m unittest tests.test_experiment_scripts -v`
Expected: pass.

### Task 5: Full Verification And Smoke Experiment

**Files:**
- Generated outputs under `results/` or `artifacts/` only.

- [ ] **Step 1: Run unit test suite**

Run: `python -m unittest discover -v`
Expected: all repository tests pass.

- [ ] **Step 2: Run deepseek-v4-flash SOP/Lifelong smoke**

Run: `python experiments/run_benchmarks.py --harness genericagent --model deepseek-v4-flash --bench core --mode all --limit 1 --max-turns 8 --out-root results/harness_core_deepseek_v4_flash`
Expected: completes `sop` and `lifelong` baseline/full/incremental smoke runs or reports the concrete missing credential/backend blocker.

- [ ] **Step 3: Inspect summaries**

Read `results/harness_core_deepseek_v4_flash/sop/summary.md` and `results/harness_core_deepseek_v4_flash/lifelong/summary.md`; report accuracy/tokens and any blockers.

### Self-Review

- Spec coverage: harness core, adapter decoupling, three-layer memory, and SOP/Lifelong deepseek smoke are covered.
- Placeholder scan: no `TBD`, no hidden follow-up implementation steps.
- Type consistency: benchmark calls use `AgentHarness` and `HarnessTask`; adapters remain task-execution backends.

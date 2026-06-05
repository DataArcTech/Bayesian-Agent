import tempfile
import unittest
from pathlib import Path

from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.memory import ThreeLayerMemory


class MemoryLayerTests(unittest.TestCase):
    def test_three_layer_memory_names_are_stable(self):
        memory = ThreeLayerMemory()

        self.assertEqual(memory.layer_names(), ["hippocampus", "state", "cortex"])

    def test_memory_context_renders_only_three_layers(self):
        memory = ThreeLayerMemory()
        memory.hippocampus.remember("recent clue")
        memory.state.set("phase", "draft")
        memory.cortex.remember("stable rule")

        context = memory.render_context()

        self.assertIn("### Hippocampus", context)
        self.assertIn("recent clue", context)
        self.assertIn("### Intermediate State", context)
        self.assertIn("phase: draft", context)
        self.assertIn("### Cortex", context)
        self.assertIn("stable rule", context)
        self.assertNotIn("L4", context)

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

    def test_cortex_records_verified_evidence_in_registry(self):
        with tempfile.TemporaryDirectory() as td:
            memory = ThreeLayerMemory(cortex_path=Path(td) / "cortex.json", registry_path=Path(td) / "beliefs.json")

            memory.cortex.record_evidence(
                TrajectoryEvidence(
                    task_id="sop_01",
                    skill_id="benchmark/sop_bench",
                    context="sop_bench",
                    outcome="success",
                    total_tokens=7,
                )
            )

            self.assertEqual(memory.cortex.registry.get("benchmark/sop_bench").observations, 1)


if __name__ == "__main__":
    unittest.main()

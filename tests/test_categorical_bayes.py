import json
import tempfile
import unittest
from pathlib import Path

from bayesian_agent import BayesianSkillRegistry, SkillBelief, TrajectoryEvidence
from bayesian_agent.cli import main
from bayesian_agent.core.algorithms.categorical_bayes import CategoricalBayesState, NaiveBayesState, features_from_event
from bayesian_agent.core.algorithms.naive_bayes import NaiveBayesState as LegacyNaiveBayesState
from bayesian_agent.core.context import SkillContextBuilder


class CategoricalBayesAlgorithmTests(unittest.TestCase):
    def test_categorical_bayes_predicts_success_conditioned_on_context(self):
        state = CategoricalBayesState()
        for idx in range(3):
            state.update(TrajectoryEvidence(task_id=f"s{idx}", skill_id="skill/a", context="sop", outcome="success"))
        for idx in range(3):
            state.update(
                TrajectoryEvidence(
                    task_id=f"f{idx}",
                    skill_id="skill/a",
                    context="lifelong",
                    outcome="failure",
                    failure_mode="sql_error",
                )
            )

        sop_probability = state.predict_success({"context": "sop"})
        lifelong_probability = state.predict_success({"context": "lifelong"})

        self.assertGreater(sop_probability, lifelong_probability)
        self.assertGreater(sop_probability, 0.5)
        self.assertLess(lifelong_probability, 0.5)

    def test_recurring_failure_mode_raises_contextual_failure_posterior(self):
        state = CategoricalBayesState()
        for idx in range(2):
            state.update(
                TrajectoryEvidence(
                    task_id=f"s{idx}",
                    skill_id="benchmark/sop_bench",
                    context="sop_bench",
                    outcome="success",
                    metadata={"output_contract": "csv_expected_output"},
                )
            )
        for idx in range(2):
            state.update(
                TrajectoryEvidence(
                    task_id=f"f{idx}",
                    skill_id="benchmark/sop_bench",
                    context="sop_bench",
                    outcome="failure",
                    failure_mode="left_expected_output_blank",
                    metadata={"output_contract": "csv_expected_output"},
                )
            )

        posterior = state.predict_proba(
            {
                "context": "sop_bench",
                "metadata.output_contract": "csv_expected_output",
                "failure_mode": "left_expected_output_blank",
            }
        )

        self.assertAlmostEqual(posterior["failure"], 0.75)
        self.assertAlmostEqual(posterior["success"], 0.25)

    def test_feature_extraction_buckets_runtime_signals(self):
        event = TrajectoryEvidence(
            task_id="t",
            skill_id="skill/a",
            context="sop",
            outcome="failure",
            failure_mode="wrong_row",
            total_tokens=42_000,
            turns=7,
            elapsed_seconds=15.2,
            metadata={"model": "deepseek-v4-flash", "large": {"skip": True}},
        )

        features = features_from_event(event)

        self.assertEqual(features["context"], "sop")
        self.assertEqual(features["failure_mode"], "wrong_row")
        self.assertEqual(features["token_bucket"], "10k_100k")
        self.assertEqual(features["turn_bucket"], "6_10")
        self.assertEqual(features["latency_bucket"], "10s_60s")
        self.assertEqual(features["metadata.model"], "deepseek-v4-flash")
        self.assertNotIn("metadata.large", features)

    def test_registry_defaults_to_categorical_bayes_for_new_beliefs(self):
        registry = BayesianSkillRegistry.in_memory()
        registry.record(TrajectoryEvidence(task_id="a", skill_id="skill/a", context="ctx", outcome="success"))

        belief = registry.get("skill/a")

        self.assertEqual(belief.algorithm, "categorical_bayes")
        self.assertIn("categorical_bayes", belief.to_dict())
        self.assertIsInstance(NaiveBayesState(), CategoricalBayesState)

    def test_naive_bayes_algorithm_alias_is_accepted(self):
        registry = BayesianSkillRegistry.in_memory(algorithm="naive_bayes")
        registry.record(TrajectoryEvidence(task_id="a", skill_id="skill/a", context="ctx", outcome="success"))

        self.assertEqual(registry.algorithm, "categorical_bayes")
        self.assertEqual(registry.get("skill/a").algorithm, "categorical_bayes")

    def test_legacy_naive_bayes_import_path_is_supported(self):
        self.assertIs(LegacyNaiveBayesState, CategoricalBayesState)

    def test_beta_bernoulli_backend_remains_available(self):
        belief = SkillBelief(skill_id="skill/beta", algorithm="beta_bernoulli")
        belief.update(TrajectoryEvidence(task_id="s", skill_id="skill/beta", context="ctx", outcome="success"))
        belief.update(TrajectoryEvidence(task_id="f", skill_id="skill/beta", context="ctx", outcome="failure"))

        self.assertEqual(belief.algorithm, "beta_bernoulli")
        self.assertEqual(belief.alpha, 2.0)
        self.assertEqual(belief.beta, 2.0)
        self.assertAlmostEqual(belief.predict_success_probability(context="ctx"), 0.5)

    def test_frequentist_backend_uses_empirical_success_rate_without_prior(self):
        registry = BayesianSkillRegistry.in_memory(algorithm="frequentist")
        registry.record(TrajectoryEvidence(task_id="s", skill_id="skill/freq", context="ctx", outcome="success"))

        belief = registry.get("skill/freq")
        self.assertEqual(belief.algorithm, "frequentist")
        self.assertEqual(belief.alpha, 1.0)
        self.assertEqual(belief.beta, 0.0)
        self.assertAlmostEqual(belief.success_probability, 1.0)

        registry.record(TrajectoryEvidence(task_id="f", skill_id="skill/freq", context="ctx", outcome="failure"))
        belief = registry.get("skill/freq")

        self.assertAlmostEqual(belief.success_probability, 0.5)
        self.assertEqual(belief.to_dict()["frequentist"]["successes"], 1.0)

    def test_legacy_registry_without_algorithm_loads_as_beta_bernoulli(self):
        raw = {
            "version": 1,
            "skills": {
                "skill/legacy": {
                    "skill_id": "skill/legacy",
                    "alpha": 3.0,
                    "beta": 1.0,
                    "posterior_success": 0.75,
                    "observations": 2,
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "beliefs.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            registry = BayesianSkillRegistry(path)
            belief = registry.get("skill/legacy")

            self.assertEqual(registry.algorithm, "beta_bernoulli")
            self.assertEqual(belief.algorithm, "beta_bernoulli")
            self.assertAlmostEqual(belief.success_probability, 0.75)

    def test_context_builder_uses_contextual_categorical_bayes_probability(self):
        registry = BayesianSkillRegistry.in_memory(algorithm="categorical_bayes")
        registry.record(TrajectoryEvidence(task_id="a1", skill_id="skill/a", context="sop", outcome="success"))
        registry.record(TrajectoryEvidence(task_id="a2", skill_id="skill/a", context="sop", outcome="success"))
        registry.record(TrajectoryEvidence(task_id="b1", skill_id="skill/b", context="sop", outcome="failure", failure_mode="bad"))
        registry.record(TrajectoryEvidence(task_id="b2", skill_id="skill/b", context="lifelong", outcome="success"))
        registry.record(TrajectoryEvidence(task_id="b3", skill_id="skill/b", context="lifelong", outcome="success"))

        context = SkillContextBuilder(registry).render(task_context="sop", limit=2)

        self.assertLess(context.find("skill/a"), context.find("skill/b"))
        self.assertIn("algorithm=categorical_bayes", context)
        self.assertIn("context_success", context)

    def test_cli_evolve_accepts_algorithm_choice(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td) / "results.json"
            registry_path = Path(td) / "beliefs.json"
            results.write_text(
                json.dumps({"results": {"bench": [{"task_id": "a", "success": True, "total_tokens": 10}]}}),
                encoding="utf-8",
            )

            code = main(["evolve", "--results", str(results), "--registry", str(registry_path), "--algorithm", "beta_bernoulli"])

            self.assertEqual(code, 0)
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(data["algorithm"], "beta_bernoulli")
            self.assertEqual(data["skills"]["benchmark/bench"]["algorithm"], "beta_bernoulli")

    def test_cli_evolve_accepts_frequentist_algorithm(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td) / "results.json"
            registry_path = Path(td) / "beliefs.json"
            results.write_text(
                json.dumps({"results": {"bench": [{"task_id": "a", "success": True, "total_tokens": 10}]}}),
                encoding="utf-8",
            )

            code = main(["evolve", "--results", str(results), "--registry", str(registry_path), "--algorithm", "frequentist"])

            self.assertEqual(code, 0)
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(data["algorithm"], "frequentist")
            self.assertEqual(data["skills"]["benchmark/bench"]["posterior_success"], 1.0)


if __name__ == "__main__":
    unittest.main()

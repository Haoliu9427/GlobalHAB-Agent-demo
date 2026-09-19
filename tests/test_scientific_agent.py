import json
import unittest
from unittest.mock import patch

from globalhab_demo.data import generate_demo_data
from globalhab_demo.scientific_agent import ScientificTools, run_scientific_agent

AID = "downstream__14d__logistic"


class ScientificAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = generate_demo_data(days=720, seed=42)

    def test_tools_budget_freeze_and_common_rows(self):
        t = ScientificTools(self.frame, budget=4)
        with self.assertRaises(ValueError):
            t.execute("exec", {"code": "bad"})
        self.assertEqual(t.used, 0)
        first = t.execute("evaluate_hypothesis", {"action_id": AID})
        with self.assertRaises(ValueError):
            t.execute("finish", {"action_id": AID})
        second = t.execute("evaluate_hypothesis", {"action_id": "local__45d__logistic"})
        self.assertEqual(first["n_scored"], second["n_scored"])
        with self.assertRaises(ValueError):
            t.execute("evaluate_hypothesis", {"action_id": AID})
        for kind in ("reversed", "permutation"):
            t.execute("negative_control", {"action_id": AID, "kind": kind})
        with self.assertRaises(ValueError):
            t.execute("evaluate_hypothesis", {"action_id": "local__7d__logistic"})
        self.assertEqual(t.used, 4)
        t.execute("finish", {"action_id": AID})
        self.assertIsNotNone(t.final_test())
        with self.assertRaises(ValueError):
            t.execute("inspect_data", {})

    def test_holdout_labels_cannot_change_feedback(self):
        a = ScientificTools(self.frame)
        changed = self.frame.copy()
        mask = changed.date.ge(a.test_start) & changed.region.eq(a.holdout)
        changed.loc[mask, "hab_event"] = 1-changed.loc[mask, "hab_event"]
        b = ScientificTools(changed)
        for tool, args in [("evaluate_hypothesis", {"action_id": AID}),
                           ("negative_control", {"action_id": AID, "kind": "permutation"})]:
            ra, rb = a.execute(tool, args), b.execute(tool, args)
            self.assertEqual(ra["pr_auc"], rb["pr_auc"])
            self.assertEqual(ra["brier"], rb["brier"])
        train, val, test = a._split(a._features(AID))
        self.assertLess(train.date.max(), val.date.min())
        self.assertLess(val.date.max(), test.date.min())
        self.assertNotIn(a.holdout, train.region.unique())
        self.assertNotIn(a.holdout, val.region.unique())

    def test_planner_receives_controls_and_executes_only_requested(self):
        calls = [
            ("evaluate_hypothesis", {"action_id": AID}),
            ("negative_control", {"action_id": AID, "kind": "reversed"}),
            ("negative_control", {"action_id": AID, "kind": "permutation"}),
            ("finish", {"action_id": AID}),
        ]
        seen = []
        def planner(messages):
            seen.append(json.loads(json.dumps(messages)))
            tool, args = calls[len(seen)-1]
            return json.dumps({"tool": tool, "arguments": args, "rationale": "检查证据"}), {"total_tokens": 10}
        result = run_scientific_agent(self.frame, planner, budget=4, model_label="TEST scripted adapter")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["experiments_executed"], 3)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["known_total_tokens"], 40)
        self.assertIn("candidate_minus_control_ap", seen[2][-1]["content"])
        self.assertNotIn("one_shot_final_holdout", json.dumps(seen))
        self.assertNotIn("synthetic_ground_truth", json.dumps(seen))

    def test_model_failure_never_falls_back(self):
        def fail(messages):
            raise RuntimeError("secret-value-from-provider")
        r = run_scientific_agent(self.frame, fail)
        self.assertEqual(r["status"], "model_service_error")
        self.assertEqual(r["experiments_executed"], 0)
        self.assertIsNone(r["final"])
        self.assertNotIn("secret-value", json.dumps(r))

    def test_invalid_calls_terminate_without_fits(self):
        r = run_scientific_agent(self.frame, lambda _: ('{"tool":"exec","arguments":{},"rationale":"no"}', {}), budget=4)
        self.assertEqual(r["status"], "planning_limit")
        self.assertEqual(r["experiments_executed"], 0)
        self.assertEqual(len(r["audit"]), 9)
        self.assertFalse(r["token_usage_complete"])


if __name__ == "__main__":
    unittest.main()

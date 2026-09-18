"""Behavior checks for shared inputs, threshold changes and visible zero rows."""
import unittest
import numpy as np
from globalhab_demo.joint_bio import default_thresholds, evaluate_joint, perturb_joint, matrix_figure
from globalhab_demo.bio_response import evaluate_intervention_robustness, INTERVENTIONS
from globalhab_demo.research_figures import robustness_distribution


class JointAssessmentTests(unittest.TestCase):
    def test_common_environment_and_object_specific_thresholds(self):
        thresholds = default_thresholds()
        baseline = evaluate_joint(65, 28, 5, thresholds)
        self.assertEqual(len(baseline), 12)
        self.assertEqual(baseline[baseline['因子'].eq('藻华压力')]['当前输入'].nunique(), 1)
        thresholds.loc[thresholds['评估对象'].eq('贝类养殖'), '藻华上限'] = 80
        changed = evaluate_joint(65, 28, 5, thresholds)
        before = baseline[baseline['对象'].eq('网箱鱼')]['阈值比'].to_numpy()
        after = changed[changed['对象'].eq('网箱鱼')]['阈值比'].to_numpy()
        np.testing.assert_array_equal(before, after)
        shell = changed[changed['对象'].eq('贝类养殖') & changed['因子'].eq('藻华压力')].iloc[0]
        self.assertFalse(shell['超过自定阈值'])
        self.assertAlmostEqual(shell['阈值比'], 65 / 80)

    def test_low_oxygen_direction_and_exact_threshold(self):
        thresholds = default_thresholds()
        exact = evaluate_joint(60, 30, 4, thresholds)
        self.assertFalse(exact['超过自定阈值'].any())
        low = evaluate_joint(60, 30, 3, thresholds)
        self.assertTrue(low[low['因子'].eq('溶解氧')]['超过自定阈值'].all())
        with self.assertRaises(ValueError):
            evaluate_joint(60, 30, 0, thresholds)

    def test_perturbation_coverage_and_bounds(self):
        frames = perturb_joint(100, 30, .5, default_thresholds())
        self.assertEqual(frames['情景编号'].nunique(), 27)
        self.assertEqual(len(frames), 27 * 4 * 3)
        fig = matrix_figure(frames, stability=True)
        self.assertEqual(np.asarray(fig.data[0].z).shape, (4, 3))
        self.assertTrue(((np.asarray(fig.data[0].z) >= 0) & (np.asarray(fig.data[0].z) <= 1)).all())

    def test_zero_frequency_and_tied_interventions_stay_visible(self):
        result = evaluate_intervention_robustness(65, 1.4, 6.4, 20, 100, 42, 72)
        fig = robustness_distribution(result['detail'], result['summary'])
        medians = [t for t in fig.data if t.type == 'scatter' and t.mode == 'markers']
        self.assertEqual(len(medians), len(INTERVENTIONS))
        self.assertEqual(len({float(t.y[0]) for t in medians}), len(INTERVENTIONS))
        self.assertEqual(result['detail'].scenario_id.nunique(), 81)
        labels = set(fig.layout.yaxis.ticktext)
        self.assertIn('维持监测', labels)
        self.assertIn('转移准备', labels)


if __name__ == '__main__':
    unittest.main()

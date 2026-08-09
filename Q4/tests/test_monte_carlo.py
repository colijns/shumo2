import importlib.util
import os
import unittest


Q4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    'q4_monte_carlo', os.path.join(Q4_DIR, 'monte_carlo.py'))
mc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mc)
geo = mc.geo


class TestCosts(unittest.TestCase):

    def test_cost_frontier_respects_budget_and_is_maximal(self):
        budget = 8.0
        rows = mc.cost_frontier(budget, [0, 100, 300, 10000])
        self.assertEqual([row[0] for row in rows], [0, 100, 300])
        for n_a, n_b in rows:
            self.assertLessEqual(mc.total_cost(n_a, n_b), budget + 1e-12)
            self.assertGreater(mc.total_cost(n_a, n_b + 1), budget)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            mc.cost_frontier(-1.0, [0])

    def test_common_trial_is_monotone_for_nested_candidates(self):
        pairs = [(0, 0), (30, 0), (60, 0), (60, 100), (60, 200)]
        for seed in range(5):
            results = mc.common_random_trial(pairs, seed)
            states = [int(result['conductive']) for result in results]
            self.assertEqual(states, sorted(states))

    def test_common_trial_aggregation(self):
        pairs = [(0, 0), (1, 0)]
        trials = [mc.common_random_trial(pairs, seed) for seed in range(3)]
        rows = mc.aggregate_common_trials(pairs, trials)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['m'], 3)
        self.assertEqual(rows[0]['x'], 0)

    def test_particle_costs(self):
        self.assertAlmostEqual(geo.COST_PER_A, 0.0148440252882, places=10)
        self.assertAlmostEqual(geo.COST_PER_B, 0.00167551608191, places=10)

    def test_equal_cost_frontier(self):
        pairs = mc.equal_cost_frontier(612, [0, 306, 612])
        self.assertEqual(pairs[-1], (612, 0))
        budget = mc.total_cost(612, 0)
        for n_a, n_b in pairs:
            self.assertLessEqual(mc.total_cost(n_a, n_b), budget + 1e-12)
            self.assertLess(budget - mc.total_cost(n_a, n_b),
                            geo.COST_PER_B + 1e-12)

    def test_volume_fraction(self):
        phi_a, phi_b = mc.volume_fractions(1, 1)
        self.assertAlmostEqual(phi_a, geo.V_A / geo.V_BOX)
        self.assertAlmostEqual(phi_b, geo.V_B / geo.V_BOX)

    def test_wilson_bounds(self):
        p, lo, hi = mc.wilson_ci(90, 100)
        self.assertEqual(p, 0.9)
        self.assertLess(lo, p)
        self.assertGreater(hi, p)


if __name__ == '__main__':
    unittest.main()

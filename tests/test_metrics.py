import unittest

import numpy as np

from src.experiment import ndcg_at_k, recall_at_k


class MetricTests(unittest.TestCase):
    def test_perfect_ndcg(self):
        self.assertAlmostEqual(ndcg_at_k(np.array([2, 3, 4]), {2, 3}, 3), 1.0)

    def test_ndcg_and_recall_with_no_hits(self):
        ranked = np.array([1, 2, 3])
        self.assertEqual(ndcg_at_k(ranked, {8, 9}, 3), 0.0)
        self.assertEqual(recall_at_k(ranked, {8, 9}, 3), 0.0)

    def test_recall(self):
        self.assertAlmostEqual(recall_at_k(np.array([1, 2, 3]), {2, 3, 4, 5}, 3), 0.5)


if __name__ == "__main__":
    unittest.main()

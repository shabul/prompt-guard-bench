import unittest

from gemma_benchmark import binary_metrics, dataset_digest, nearest_rank


class BenchmarkHelperTests(unittest.TestCase):
    def test_digest_includes_label_and_split(self):
        rows = {"train": [{"text": "hello", "label": 1}]}
        original = dataset_digest(rows)
        self.assertNotEqual(original, dataset_digest({"train": [{"text": "hello", "label": 0}]}))
        self.assertNotEqual(original, dataset_digest({"test": rows["train"]}))

    def test_nearest_rank_at_35_samples(self):
        self.assertEqual(nearest_rank(list(range(1, 36)), 0.95), 34)
        with self.assertRaises(ValueError):
            nearest_rank([], 0.95)

    def test_binary_metrics_uses_score_and_confusion_counts(self):
        rows = [
            {"label": 1, "score": 0.8, "forward_latency_ms": 1},
            {"label": 1, "score": 0.2, "forward_latency_ms": 2},
            {"label": 0, "score": 0.7, "forward_latency_ms": 3},
            {"label": 0, "score": 0.1, "forward_latency_ms": 4},
        ]
        result = binary_metrics(rows, 0.6)
        self.assertEqual(result["confusion_matrix"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["mean_latency_ms"], 2.5)


if __name__ == "__main__":
    unittest.main()

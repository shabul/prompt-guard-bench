import unittest

from research.paired_recipe import paired_bootstrap


class PairedRecipeTests(unittest.TestCase):
    def test_cluster_bootstrap_is_paired_and_deterministic(self):
        rows = [
            {"recipe_code": group, "label": group % 2, "predicted": group % 2}
            for group in range(20)
        ]
        other = [{**row, "predicted": 1 - row["label"]} for row in rows]
        reports = {
            "gemma": {"predictions": {"test": rows}},
            "tfidf": {"predictions": {"test": other}},
        }
        first = paired_bootstrap(reports, "gemma", "tfidf", repetitions=100, seed=7)
        second = paired_bootstrap(reports, "gemma", "tfidf", repetitions=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["groups"], 20)
        self.assertEqual(first["observed"]["macro_f1"], 1.0)
        self.assertEqual(first["observed"]["balanced_accuracy"], 1.0)
        self.assertEqual(first["excluded_one_class_replicates"], 0)


if __name__ == "__main__":
    unittest.main()

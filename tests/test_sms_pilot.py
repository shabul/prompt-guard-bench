import unittest

from pilot_sms import choose_f1_threshold, make_splits


class SmsPilotTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        rows = [
            {"id": f"{label}-{index}", "text": f"message {label}-{index}", "label": label}
            for label in (0, 1) for index in range(600)
        ]
        first = make_splits(rows, seed=42)
        second = make_splits(rows, seed=42)
        self.assertEqual(
            {name: [row["id"] for row in split] for name, split in first.items()},
            {name: [row["id"] for row in split] for name, split in second.items()},
        )
        ids = [row["id"] for split in first.values() for row in split]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({name: len(split) for name, split in first.items()}, {"train": 128, "validation": 200, "test": 400})

    def test_threshold_is_chosen_from_validation_scores(self):
        choice = choose_f1_threshold([0.1, 0.4, 0.7, 0.9], [0, 1, 0, 1])
        self.assertEqual(choice["threshold"], 0.4)
        self.assertEqual(choice["validation_f1"], 0.8)


if __name__ == "__main__":
    unittest.main()

import unittest

from pilot_recipe import make_recipe_splits


class RecipePilotTests(unittest.TestCase):
    def test_recipe_groups_and_rows_are_disjoint(self):
        rows = [
            {"id": f"{recipe}-{label}-{index}", "text": f"review {recipe}-{label}-{index}", "label": label, "recipe_code": recipe}
            for recipe in range(100) for label in (0, 1) for index in range(16)
        ]
        splits, groups = make_recipe_splits(rows, seed=42)
        self.assertEqual({name: len(split) for name, split in splits.items()}, {"train": 128, "validation": 200, "test": 320})
        self.assertEqual({name: len(group) for name, group in groups.items()}, {"train": 60, "validation": 20, "test": 20})
        for name, selected in splits.items():
            self.assertTrue({row["recipe_code"] for row in selected}.issubset(set(groups[name])))
        all_ids = [row["id"] for selected in splits.values() for row in selected]
        self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == "__main__":
    unittest.main()

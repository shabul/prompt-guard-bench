import unittest

from judgment_bench import BinaryTaskSpec, LogisticHeadArtifact, score_embedding


def example_artifact():
    return {
        "version": "binary_logistic_v1",
        "task_id": "example-v1",
        "model_id": "test/encoder",
        "model_revision": "abc123",
        "score_semantics": "uncalibrated_logistic_decision_function",
        "train_data_sha256": "a" * 64,
        "mean": [1.0, 2.0],
        "scale": [2.0, 4.0],
        "coef": [2.0, -1.0],
        "intercept": 0.25,
        "validation_threshold": {"threshold": 0.5, "rule": "validation only"},
    }


class ContractTests(unittest.TestCase):
    def test_task_requires_distinct_defined_labels(self):
        self.assertEqual(BinaryTaskSpec("x", "no", "yes", "demo").positive_label, "yes")
        with self.assertRaises(ValueError):
            BinaryTaskSpec("x", "same", "same", "demo")

    def test_scoring_uses_saved_normalization_and_threshold(self):
        artifact = LogisticHeadArtifact.from_mapping(example_artifact())
        prediction = score_embedding([3.0, 6.0], artifact, model_id="test/encoder", model_revision="abc123")
        self.assertAlmostEqual(prediction.score, 1.25)
        self.assertEqual(prediction.label, 1)
        self.assertEqual(prediction.score_semantics, "uncalibrated_logistic_decision_function")
        self.assertEqual(prediction.input_status, "ok")

    def test_truncation_never_emits_decision(self):
        artifact = LogisticHeadArtifact.from_mapping(example_artifact())
        prediction = score_embedding([3.0, 6.0], artifact, model_id="test/encoder", model_revision="abc123", truncated=True)
        self.assertEqual(prediction.input_status, "truncated")
        self.assertIsNone(prediction.label)

    def test_mismatch_and_invalid_head_fail_closed(self):
        artifact = LogisticHeadArtifact.from_mapping(example_artifact())
        with self.assertRaises(ValueError):
            score_embedding([3.0, 6.0], artifact, model_id="other", model_revision="abc123")
        with self.assertRaises(ValueError):
            score_embedding([3.0, 6.0], artifact, model_id="test/encoder", model_revision="wrong")
        with self.assertRaises(ValueError):
            score_embedding([3.0], artifact, model_id="test/encoder", model_revision="abc123")
        record = example_artifact()
        record["scale"] = [0.0, 4.0]
        with self.assertRaises(ValueError):
            LogisticHeadArtifact.from_mapping(record)

    def test_unpinned_pilot_head_needs_explicit_override(self):
        record = example_artifact()
        del record["model_revision"]
        artifact = LogisticHeadArtifact.from_mapping(record)
        with self.assertRaises(ValueError):
            score_embedding([3.0, 6.0], artifact, model_id="test/encoder")
        prediction = score_embedding([3.0, 6.0], artifact, model_id="test/encoder", allow_unpinned_model=True)
        self.assertEqual(prediction.label, 1)


if __name__ == "__main__":
    unittest.main()

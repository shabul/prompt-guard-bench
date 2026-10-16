"""Count reviewer overlap in the fixed recipe split without exporting IDs or text."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from pilot_recipe import SOURCE_FILENAME, SOURCE_SHA256, SOURCE_URL, load_recipe_rows, make_recipe_splits


def main():
    rows, _ = load_recipe_rows()
    splits, _ = make_recipe_splits(rows)
    selected = {row["id"]: split for split, group in splits.items() for row in group}
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != SOURCE_SHA256:
        raise ValueError("recipe archive hash differs")
    with zipfile.ZipFile(io.BytesIO(archive)) as contents:
        frame = pd.read_csv(io.BytesIO(contents.read(SOURCE_FILENAME)))
    if not {"text", "user_id"}.issubset(frame.columns):
        raise ValueError("source reviewer columns absent")

    reviewer_by_id = {}
    for _, source in frame.iterrows():
        value = source["text"]
        if not isinstance(value, str) or not value.strip():
            continue
        row_id = hashlib.sha256(value.strip().casefold().encode("utf-8")).hexdigest()
        if row_id in selected and row_id not in reviewer_by_id:
            reviewer_by_id[row_id] = None if pd.isna(source["user_id"]) else str(source["user_id"])
    if set(reviewer_by_id) != set(selected):
        raise ValueError("could not join every selected review to a source reviewer")

    reviewers = {
        split: {reviewer_by_id[row["id"]] for row in group if reviewer_by_id[row["id"]] is not None}
        for split, group in splits.items()
    }
    missing = {
        split: sum(reviewer_by_id[row["id"]] is None for row in group)
        for split, group in splits.items()
    }
    development = reviewers["train"] | reviewers["validation"]
    test_rows_with_dev_reviewer = sum(
        reviewer_by_id[row["id"]] in development
        for row in splits["test"]
        if reviewer_by_id[row["id"]] is not None
    )
    result = {
        "source_archive_sha256": SOURCE_SHA256,
        "selected_rows_by_split": {name: len(group) for name, group in splits.items()},
        "reviewer_id_missing_by_split": missing,
        "distinct_reviewers_by_split": {name: len(ids) for name, ids in reviewers.items()},
        "reviewer_intersection_counts": {
            f"{left}_{right}": len(reviewers[left] & reviewers[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        },
        "test_rows_with_reviewer_seen_in_development": test_rows_with_dev_reviewer,
        "note": "counts only; reviewer IDs, names, and review text are not exported",
    }
    Path("recipe_reviewer_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

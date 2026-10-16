from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SOURCE_TO_LABEL = {
    "First Party Collection/Use": "data_collection",
    "Third Party Sharing/Collection": "third_party_sharing",
    "User Choice/Control": "consent_choices",
    "Data Security": "security_protection",
    "Data Retention": "retention_deletion",
    "Do Not Track": "advertising_tracking",
    "User Access, Edit and Deletion": "user_rights_opt_out",
}

LABELS = tuple(SOURCE_TO_LABEL.values())
REQUIRED_COLUMNS = ("website_id", "policy_text", "opp_category", "policy_labels")


def clean_text(text: str) -> str:
    """Normalize markup entities and whitespace without changing clause meaning."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    frame["policy_text"] = frame["policy_text"].map(clean_text)
    frame["target"] = frame["opp_category"].map(SOURCE_TO_LABEL)

    unsupported = sorted(set(frame.loc[frame["target"].isna(), "opp_category"]))
    if unsupported:
        raise ValueError(
            "Unsupported OPP-115 categories are present; review the mapping before "
            f"training: {unsupported}"
        )

    inconsistent = frame.loc[frame["policy_labels"] != frame["target"]]
    if not inconsistent.empty:
        raise ValueError(
            f"{len(inconsistent)} prepared labels disagree with their original "
            "OPP-115 categories."
        )

    invalid = frame.loc[
        frame["website_id"].eq("") | frame["policy_text"].eq("") | frame["opp_category"].eq("")
    ]
    if not invalid.empty:
        raise ValueError(f"Dataset contains {len(invalid)} rows with missing required values.")

    # Repeated annotations of the exact same clause/category do not add independent evidence.
    frame = frame.drop_duplicates(["website_id", "policy_text", "target"]).reset_index(drop=True)
    return frame


def split_by_website(
    frame: pd.DataFrame, seed: int = 42, test_size: float = 0.2, validation_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible train/validation/test sets with disjoint website groups."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Split proportions must be between 0 and 1.")
    labels = set(frame["target"])
    if labels != set(LABELS):
        raise ValueError(f"Expected all {len(LABELS)} supported labels; found {sorted(labels)}.")

    groups = frame["website_id"].to_numpy()
    targets = frame["target"].to_numpy()
    chosen: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None

    # GroupShuffleSplit is used only on website IDs. Search deterministic seeds for
    # partitions that retain every class, especially the rare Do Not Track class.
    for attempt in range(1000):
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
        train_val_idx, test_idx = next(outer.split(frame, targets, groups))
        train_val = frame.iloc[train_val_idx]
        test = frame.iloc[test_idx]
        if set(test["target"]) != labels:
            continue

        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=seed + 10_000 + attempt,
        )
        train_idx, validation_idx = next(
            inner.split(train_val, train_val["target"], train_val["website_id"])
        )
        train = train_val.iloc[train_idx]
        validation = train_val.iloc[validation_idx]
        if set(train["target"]) == labels and set(validation["target"]) == labels:
            chosen = (train.copy(), validation.copy(), test.copy())
            break

    if chosen is None:
        raise ValueError(
            "Could not create website-disjoint splits containing every class after "
            "1000 deterministic attempts. Review rare-class website coverage."
        )

    train, validation, test = chosen
    group_sets = [set(part["website_id"]) for part in chosen]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Website leakage detected between data splits.")
    return train, validation, test

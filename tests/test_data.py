import pandas as pd
import pytest

from contract_lens.data import LABELS, load_dataset, split_by_website


def _frame() -> pd.DataFrame:
    rows = []
    categories = {
        "First Party Collection/Use": "data_collection",
        "Third Party Sharing/Collection": "third_party_sharing",
        "User Choice/Control": "consent_choices",
        "Data Security": "security_protection",
        "Data Retention": "retention_deletion",
        "Do Not Track": "advertising_tracking",
        "User Access, Edit and Deletion": "user_rights_opt_out",
    }
    for website_index in range(35):
        for category, label in categories.items():
            rows.append(
                {
                    "website_id": f"site-{website_index}",
                    "policy_text": f"Sample policy clause for {label} at site {website_index}.",
                    "opp_category": category,
                    "policy_labels": label,
                }
            )
    return pd.DataFrame(rows)


def test_load_dataset_uses_original_category_and_deduplicates_exact_clause(tmp_path):
    frame = _frame()
    frame.loc[0, "policy_text"] = "  Data&nbsp; is collected.  "
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)

    loaded = load_dataset(path)

    assert loaded.iloc[0]["policy_text"] == "Data is collected."
    assert len(loaded) == len(frame) - 1
    assert set(loaded["target"]) == set(LABELS)


def test_load_dataset_rejects_prepared_label_disagreement(tmp_path):
    frame = _frame()
    frame.loc[0, "policy_labels"] = "invented_label"
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="disagree"):
        load_dataset(path)


def test_split_by_website_has_disjoint_groups_and_all_labels():
    frame = _frame()
    frame["target"] = frame["policy_labels"]

    train, validation, test = split_by_website(frame)

    groups = [set(part["website_id"]) for part in (train, validation, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert all(set(part["target"]) == set(LABELS) for part in (train, validation, test))

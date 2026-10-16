import pandas as pd

from contract_lens.data import SOURCE_TO_LABEL
from contract_lens.inference import ClausePredictor
from contract_lens.training import train


def test_train_writes_comparable_models_and_loadable_attention_checkpoint(tmp_path):
    rows = []
    for site_number in range(35):
        for category, label in SOURCE_TO_LABEL.items():
            rows.append(
                {
                    "website_id": f"site-{site_number}",
                    "policy_text": (
                        f"This clause about {label} describes policy details "
                        f"for website number {site_number}."
                    ),
                    "opp_category": category,
                    "policy_labels": label,
                }
            )
    dataset_path = tmp_path / "clauses.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    artifact_dir = tmp_path / "artifacts"

    metrics = train(
        dataset_path,
        artifact_dir,
        seed=7,
        epochs=1,
        batch_size=128,
        max_length=30,
        embedding_dim=12,
        hidden_dim=8,
    )

    assert set(metrics["models"]) == {
        "tfidf_logistic_regression",
        "bilstm",
        "bilstm_attention",
    }
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv")
    assert manifest.groupby("website_id")["split"].nunique().max() == 1
    for model_name in (
        "bilstm_attention",
        "bilstm",
        "tfidf_logistic_regression",
    ):
        predictor = ClausePredictor(artifact_dir, model_name)
        assert len(predictor.predict(["We collect information."])) == 1
    assert (artifact_dir / "confusion_matrix_bilstm_attention.csv").is_file()
    assert (artifact_dir / "errors_bilstm_attention.csv").is_file()

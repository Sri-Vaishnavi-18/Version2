from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from contract_lens.data import LABELS, SOURCE_TO_LABEL, load_dataset, split_by_website
from contract_lens.models import BiLSTMClassifier, Vocabulary

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


class ClauseDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocabulary: Vocabulary, max_length: int):
        self.sequences = [
            torch.tensor(vocabulary.encode(text, max_length), dtype=torch.long)
            for text in frame["policy_text"]
        ]
        self.labels = torch.tensor(
            [LABEL_TO_ID[label] for label in frame["target"]], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def collate_clauses(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True, padding_value=0), lengths, torch.stack(labels)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def predict_neural(
    model: BiLSTMClassifier,
    frame: pd.DataFrame,
    vocabulary: Vocabulary,
    max_length: int,
    device: torch.device,
) -> list[str]:
    model.eval()
    dataset = ClauseDataset(frame, vocabulary, max_length)
    loader = DataLoader(dataset, batch_size=128, collate_fn=collate_clauses)
    predictions: list[int] = []
    with torch.inference_mode():
        for token_ids, lengths, _ in loader:
            logits = model(token_ids.to(device), lengths.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return [ID_TO_LABEL[index] for index in predictions]


def fit_neural(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    vocabulary: Vocabulary,
    artifact_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    max_length: int,
    embedding_dim: int,
    hidden_dim: int,
    use_attention: bool,
    device: torch.device,
) -> tuple[BiLSTMClassifier, list[dict[str, float]]]:
    set_seed(seed)
    train_ds = ClauseDataset(train, vocabulary, max_length)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_clauses,
    )
    model = BiLSTMClassifier(
        len(vocabulary.token_to_id),
        len(LABELS),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        use_attention=use_attention,
    ).to(device)
    counts = np.bincount(
        [LABEL_TO_ID[label] for label in train["target"]], minlength=len(LABELS)
    )
    class_weights = len(train) / (len(LABELS) * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    val_true = validation["target"].tolist()
    model_label = "BiLSTM + Attention" if use_attention else "BiLSTM (no attention)"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for token_ids, lengths, labels in train_loader:
            token_ids, lengths, labels = token_ids.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(token_ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        val_pred = predict_neural(model, validation, vocabulary, max_length, device)
        val_f1 = f1_score(val_true, val_pred, labels=LABELS, average="macro", zero_division=0)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / len(train_ds),
                "validation_macro_f1": float(val_f1),
            }
        )
        print(
            f"{model_label} epoch {epoch}/{epochs}: "
            f"train_loss={history[-1]['train_loss']:.4f}, "
            f"validation_macro_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Neural model training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, history


def _save_neural_checkpoint(
    model: BiLSTMClassifier,
    vocabulary: Vocabulary,
    artifact_dir: Path,
    use_attention: bool,
    max_length: int,
    embedding_dim: int,
    hidden_dim: int,
) -> None:
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "vocab_size": len(vocabulary.token_to_id),
            "num_classes": len(LABELS),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "dropout": 0.3,
            "use_attention": use_attention,
            "max_length": max_length,
            "labels": list(LABELS),
        },
        artifact_dir / ("bilstm_attention.pt" if use_attention else "bilstm.pt"),
    )
    if use_attention:
        with (artifact_dir / "tokenizer.json").open("w", encoding="utf-8") as file:
            json.dump(vocabulary.to_dict(), file, ensure_ascii=False, indent=2)


def train(
    dataset_path: Path,
    artifact_dir: Path,
    seed: int = 42,
    epochs: int = 12,
    batch_size: int = 64,
    patience: int = 3,
    max_length: int = 160,
    embedding_dim: int = 96,
    hidden_dim: int = 64,
    cpu_threads: int = 1,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least one.")
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)
    train_frame, validation_frame, test_frame = split_by_website(data, seed=seed)
    vocabulary = Vocabulary.build(train_frame["policy_text"].tolist())

    split_manifest = pd.concat(
        [
            train_frame[["website_id"]].assign(split="train"),
            validation_frame[["website_id"]].assign(split="validation"),
            test_frame[["website_id"]].assign(split="test"),
        ]
    ).drop_duplicates()
    split_manifest.to_csv(artifact_dir / "split_manifest.csv", index=False)

    baseline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    baseline.fit(train_frame["policy_text"], train_frame["target"])
    joblib.dump(baseline, artifact_dir / "tfidf_logistic_regression.joblib")

    metrics: dict[str, Any] = {
        "dataset": str(dataset_path),
        "rows_after_exact_duplicate_removal": len(data),
        "website_count": int(data["website_id"].nunique()),
        "label_counts": {
            label: int(count)
            for label, count in data["target"].value_counts().reindex(LABELS, fill_value=0).items()
        },
        "split_sizes": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "split_websites": {
            "train": int(train_frame["website_id"].nunique()),
            "validation": int(validation_frame["website_id"].nunique()),
            "test": int(test_frame["website_id"].nunique()),
        },
        "device": str(device),
        "models": {},
    }
    test_truth = test_frame["target"].tolist()
    baseline_test = baseline.predict(test_frame["policy_text"]).tolist()
    metrics["models"]["tfidf_logistic_regression"] = score_predictions(test_truth, baseline_test)

    neural_histories: dict[str, list[dict[str, float]]] = {}
    for use_attention, model_name in ((False, "bilstm"), (True, "bilstm_attention")):
        model, history = fit_neural(
            train_frame,
            validation_frame,
            vocabulary,
            artifact_dir,
            seed,
            epochs,
            batch_size,
            patience,
            max_length,
            embedding_dim,
            hidden_dim,
            use_attention,
            device,
        )
        neural_histories[model_name] = history
        _save_neural_checkpoint(
            model, vocabulary, artifact_dir, use_attention, max_length, embedding_dim, hidden_dim
        )
        predictions = predict_neural(model, test_frame, vocabulary, max_length, device)
        metrics["models"][model_name] = score_predictions(test_truth, predictions)
        metrics["models"][model_name]["test_predictions"] = predictions

    metrics["models"]["tfidf_logistic_regression"]["test_predictions"] = baseline_test
    metrics["training_history"] = neural_histories
    pd.DataFrame(
        [
            {"model": name, **epoch}
            for name, history in neural_histories.items()
            for epoch in history
        ]
    ).to_csv(artifact_dir / "training_history.csv", index=False)

    for model_name, model_metrics in metrics["models"].items():
        predictions = model_metrics["test_predictions"]
        matrix = confusion_matrix(test_truth, predictions, labels=LABELS)
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
            artifact_dir / f"confusion_matrix_{model_name}.csv"
        )
        errors = [
            {
                "website_id": website_id,
                "true_label": true,
                "predicted_label": predicted,
                "policy_text": text,
            }
            for website_id, true, predicted, text in zip(
                test_frame["website_id"], test_truth, predictions, test_frame["policy_text"]
            )
            if true != predicted
        ]
        pd.DataFrame(
            errors,
            columns=["website_id", "true_label", "predicted_label", "policy_text"],
        ).to_csv(artifact_dir / f"errors_{model_name}.csv", index=False)
        del model_metrics["test_predictions"]

    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    with (artifact_dir / "label_schema.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"labels": list(LABELS), "source_category_mapping": dict(SOURCE_TO_LABEL)},
            file,
            ensure_ascii=False,
            indent=2,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate AI Contract Lens.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Working Prototype/opp115_policy_clauses_prepared.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset does not exist: {args.data}")

    result = train(
        args.data,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        max_length=args.max_length,
        cpu_threads=args.cpu_threads,
    )
    print(f"Device: {result['device']}")
    print(f"Website-disjoint split sizes: {result['split_sizes']}")
    for name, scores in result["models"].items():
        print(
            f"{name}: macro-F1={scores['macro_f1']:.4f}, "
            f"weighted-F1={scores['weighted_f1']:.4f}, "
            f"accuracy={scores['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()

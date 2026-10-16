from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import torch

from contract_lens.data import LABELS
from contract_lens.models import BiLSTMClassifier, Vocabulary

LABEL_DETAILS = {
    "data_collection": {
        "name": "Data collection and use",
        "plain_english": "This clause describes information the organization collects or how it uses that information.",
        "review_prompt": "Check what information is collected, why it is needed, and whether optional collection can be declined.",
    },
    "third_party_sharing": {
        "name": "Sharing with third parties",
        "plain_english": "This clause discusses information shared with, received from, or handled by another organization.",
        "review_prompt": "Check who receives the information, for what purpose, and whether onward sharing is described.",
    },
    "consent_choices": {
        "name": "Consent and choices",
        "plain_english": "This clause describes a choice or control available to a user, or how a choice affects the service.",
        "review_prompt": "Check whether the choice is clear, optional, and easy to change or withdraw.",
    },
    "security_protection": {
        "name": "Data security",
        "plain_english": "This clause describes safeguards or security practices for information.",
        "review_prompt": "Check which safeguards are stated and whether the policy explains their scope and limits.",
    },
    "retention_deletion": {
        "name": "Retention and deletion",
        "plain_english": "This clause discusses how long information is kept or when it may be deleted.",
        "review_prompt": "Check how long information is retained and what conditions trigger deletion.",
    },
    "advertising_tracking": {
        "name": "Advertising and tracking choices",
        "plain_english": "This clause concerns tracking or choices related to tracking and advertising.",
        "review_prompt": "Check which tracking is covered and whether the described control applies across services or devices.",
    },
    "user_rights_opt_out": {
        "name": "User access, correction, and deletion rights",
        "plain_english": "This clause describes a user's ability to access, change, or delete information.",
        "review_prompt": "Check how to make a request, what identity checks apply, and whether exceptions are listed.",
    },
}

MAX_TEXT_CHARACTERS = 1_000_000
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_KEY_TERM_PATTERNS = (
    re.compile(r"\bpersonal\s+(?:information|data)\b", re.IGNORECASE),
    re.compile(r"\bsensitive\s+(?:information|data)\b", re.IGNORECASE),
    re.compile(r"\bcookies?\b", re.IGNORECASE),
    re.compile(r"\bthird[- ]part(?:y|ies)\b", re.IGNORECASE),
    re.compile(r"\bpartners?\b|\baffiliates?\b", re.IGNORECASE),
    re.compile(r"\bcollect(?:s|ed|ing|ion)?\b", re.IGNORECASE),
    re.compile(r"\btargeted\s+advertising\b|\badvertising\b", re.IGNORECASE),
    re.compile(r"\bconsent\b|\bopt[- ]out\b", re.IGNORECASE),
    re.compile(r"\bretention\b|\bretain(?:ed|s|ing)?\b", re.IGNORECASE),
    re.compile(r"\bdelet(?:e|ion|ed|ing)\b|\beras(?:e|ure|ed)\b", re.IGNORECASE),
    re.compile(r"\bsecurity\b|\bencrypt(?:ion|ed|s)?\b", re.IGNORECASE),
    re.compile(r"\baccess\b|\bcorrect(?:ion|ed)?\b", re.IGNORECASE),
    re.compile(r"\blocation\b", re.IGNORECASE),
)


def split_into_clauses(text: str) -> list[str]:
    """Split on paragraph and sentence boundaries while retaining source wording."""
    clauses: list[str] = []
    for paragraph in text.splitlines():
        start = 0
        for match in _SENTENCE_BOUNDARY.finditer(paragraph):
            clause = paragraph[start : match.start()].strip()
            if clause:
                clauses.append(clause)
            start = match.end()
        final = paragraph[start:].strip()
        if final:
            clauses.append(final)
    return clauses


def extract_key_terms(text: str) -> list[str]:
    """Return only terms that literally appear in the supplied evidence."""
    terms: list[str] = []
    seen: set[str] = set()
    for pattern in _KEY_TERM_PATTERNS:
        for match in pattern.finditer(text):
            term = match.group(0)
            normalized = term.casefold()
            if normalized not in seen:
                seen.add(normalized)
                terms.append(term)
    return terms


class ClausePredictor:
    def __init__(self, artifact_dir: str | Path, model_name: str = "bilstm_attention"):
        self.artifact_dir = Path(artifact_dir)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: BiLSTMClassifier | None = None
        self.vocabulary: Vocabulary | None = None
        self.pipeline: Any | None = None
        self.max_length = 160
        if self.device.type == "cpu":
            torch.set_num_threads(1)

        if model_name == "tfidf_logistic_regression":
            self.pipeline = joblib.load(self.artifact_dir / "tfidf_logistic_regression.joblib")
        elif model_name in {"bilstm", "bilstm_attention"}:
            self._load_neural()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _load_neural(self) -> None:
        checkpoint_path = self.artifact_dir / f"{self.model_name}.pt"
        tokenizer_path = self.artifact_dir / "tokenizer.json"
        if not checkpoint_path.is_file() or not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Trained {self.model_name} model/tokenizer not found in {self.artifact_dir}. "
                "Run the training command before analyzing documents."
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if tuple(checkpoint["labels"]) != LABELS:
            raise ValueError("The checkpoint label schema does not match this application.")
        with tokenizer_path.open(encoding="utf-8") as file:
            self.vocabulary = Vocabulary(json.load(file))
        if len(self.vocabulary.token_to_id) != checkpoint["vocab_size"]:
            raise ValueError("The saved tokenizer size does not match the model checkpoint.")
        self.max_length = int(checkpoint["max_length"])
        self.model = BiLSTMClassifier(
            checkpoint["vocab_size"],
            checkpoint["num_classes"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
            use_attention=checkpoint["use_attention"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, clauses: list[str]) -> list[dict[str, Any]]:
        if not clauses:
            return []
        if self.pipeline is not None:
            probabilities = self.pipeline.predict_proba(clauses)
            model_labels = self.pipeline.named_steps["classifier"].classes_
            results = []
            for row in probabilities:
                index = int(row.argmax())
                label = str(model_labels[index])
                results.append({"label": label, "score": float(row[index])})
            return results

        if self.model is None or self.vocabulary is None:
            raise RuntimeError("The neural model or tokenizer has not been initialized.")
        results = []
        with torch.inference_mode():
            for start in range(0, len(clauses), 128):
                batch = clauses[start : start + 128]
                encoded = [
                    self.vocabulary.encode(clause, self.max_length) for clause in batch
                ]
                lengths = torch.tensor([len(row) for row in encoded], device=self.device)
                width = int(lengths.max().item())
                token_ids = torch.zeros((len(encoded), width), dtype=torch.long, device=self.device)
                for index, row in enumerate(encoded):
                    token_ids[index, : len(row)] = torch.tensor(row, device=self.device)
                probabilities = torch.softmax(self.model(token_ids, lengths), dim=1)
                scores, indices = probabilities.max(dim=1)
                results.extend(
                    {
                        "label": LABELS[int(label_index)],
                        "score": float(score),
                    }
                    for score, label_index in zip(scores.cpu(), indices.cpu())
                )
        return results


def analyze_text(text: str, predictor: ClausePredictor) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("Policy text is empty.")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError(
            f"Policy text exceeds the {MAX_TEXT_CHARACTERS:,}-character analysis limit."
        )
    clauses = split_into_clauses(text)
    predictions = predictor.predict(clauses)
    findings = [
        {
            **prediction,
            "name": LABEL_DETAILS[prediction["label"]]["name"],
            "plain_english": LABEL_DETAILS[prediction["label"]]["plain_english"],
            "review_prompt": LABEL_DETAILS[prediction["label"]]["review_prompt"],
            "key_terms": extract_key_terms(clause),
            "evidence": clause,
        }
        for clause, prediction in zip(clauses, predictions)
    ]
    category_counts = {
        label: sum(finding["label"] == label for finding in findings) for label in LABELS
    }
    return {
        "model": predictor.model_name,
        "clause_count": len(clauses),
        "findings": findings,
        "category_counts": category_counts,
        "key_terms": list(
            dict.fromkeys(
                term for finding in findings for term in finding["key_terms"]
            )
        ),
        "summary": [
            f"{LABEL_DETAILS[label]['name']}: {category_counts[label]} clause(s) found."
            for label in LABELS
            if category_counts[label]
        ],
    }

from contract_lens.inference import analyze_text, split_into_clauses


class FixedPredictor:
    model_name = "test-only"

    def predict(self, clauses):
        return [
            {"label": "data_collection", "score": 0.75}
            for _ in clauses
        ]


def test_clause_splitting_preserves_original_wording():
    text = "We collect account data.  We also use cookies!\nYou can opt out."

    assert split_into_clauses(text) == [
        "We collect account data.",
        "We also use cookies!",
        "You can opt out.",
    ]


def test_analysis_returns_original_evidence_and_summary():
    text = "We collect account data."

    result = analyze_text(text, FixedPredictor())

    assert result["findings"][0]["evidence"] == text
    assert result["findings"][0]["label"] == "data_collection"
    assert result["findings"][0]["key_terms"] == ["collect"]
    assert result["category_counts"]["data_collection"] == 1
    assert result["summary"] == ["Data collection and use: 1 clause(s) found."]

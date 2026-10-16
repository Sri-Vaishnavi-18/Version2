from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from contract_lens.data import LABELS
from contract_lens.inference import LABEL_DETAILS, ClausePredictor, analyze_text
from contract_lens.sources import MAX_POLICY_BYTES, extract_pdf_text, fetch_policy_url

ARTIFACT_DIR = Path(os.environ.get("CONTRACT_LENS_ARTIFACTS", "artifacts"))
MODEL_FILES = {
    "BiLSTM + Attention": "bilstm_attention.pt",
    "BiLSTM (no attention)": "bilstm.pt",
    "TF-IDF + Logistic Regression": "tfidf_logistic_regression.joblib",
}
MODEL_KEYS = {
    "BiLSTM + Attention": "bilstm_attention",
    "BiLSTM (no attention)": "bilstm",
    "TF-IDF + Logistic Regression": "tfidf_logistic_regression",
}

st.set_page_config(page_title="AI Contract Lens", page_icon="🔎", layout="wide")
st.title("AI Contract Lens")
st.caption(
    "Evidence-linked review of privacy-policy clauses. This research prototype is not legal advice "
    "and does not determine whether a policy complies with any law."
)


def uploaded_text(uploaded_file) -> str:
    content = uploaded_file.getvalue()
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Uploaded files must be 20 MB or smaller.")
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise ValueError("TXT uploads must use UTF-8 encoding.") from error
    elif suffix == ".pdf":
        text = extract_pdf_text(content)
    else:
        raise ValueError("Upload a PDF or plain-text (.txt) file.")
    if not text:
        raise ValueError("No extractable text was found in the uploaded file.")
    return text


@st.cache_resource(show_spinner=False)
def get_predictor(artifact_dir: str, model_name: str) -> ClausePredictor:
    return ClausePredictor(artifact_dir, model_name)


@st.cache_data(show_spinner=False)
def read_metrics(artifact_dir: str) -> dict:
    metrics_path = Path(artifact_dir) / "metrics.json"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open(encoding="utf-8") as file:
        return json.load(file)


available_models = [
    title
    for title, filename in MODEL_FILES.items()
    if (ARTIFACT_DIR / filename).is_file()
]
if available_models:
    st.success(f"Trained model artifacts found in `{ARTIFACT_DIR}`.")
else:
    st.warning(
        f"No trained model artifacts found in `{ARTIFACT_DIR}`. Train the project first "
        "using the commands in README.md; analysis is disabled rather than using a placeholder."
    )

analysis_tab, evaluation_tab, about_tab = st.tabs(
    ["Analyze a policy", "Model evaluation", "Data and limitations"]
)

with analysis_tab:
    source_mode = st.radio(
        "Policy source",
        ["Paste text", "Upload PDF/TXT", "Public policy URL"],
        horizontal=True,
    )
    policy_text = ""
    document_name = "pasted-policy"
    if source_mode == "Paste text":
        policy_text = st.text_area(
            "Paste policy text",
            height=220,
            placeholder="Paste the Terms, Privacy Policy, or cookie notice text here.",
        )
    elif source_mode == "Upload PDF/TXT":
        upload = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
        if upload is not None:
            document_name = upload.name
            try:
                policy_text = uploaded_text(upload)
                st.caption(f"Extracted {len(policy_text):,} characters from {upload.name}.")
            except (ValueError, OSError) as error:
                st.error(str(error))
    else:
        url = st.text_input("Public policy URL", placeholder="https://example.com/privacy")
        if st.button("Fetch policy text"):
            try:
                with st.spinner("Fetching the public policy page..."):
                    policy_text = fetch_policy_url(url)
                st.session_state["fetched_policy_text"] = policy_text
                st.session_state["fetched_policy_url"] = url
            except (ValueError, OSError) as error:
                st.error(str(error))
        fetched_url = st.session_state.get("fetched_policy_url", "")
        policy_text = (
            st.session_state.get("fetched_policy_text", "") if url == fetched_url else ""
        )
        document_name = fetched_url if policy_text else url or "policy-url"
        if policy_text:
            st.caption(f"Fetched {len(policy_text):,} characters from {document_name}.")

    if source_mode == "Public policy URL" and st.button("Clear fetched text"):
        st.session_state.pop("fetched_policy_text", None)
        st.session_state.pop("fetched_policy_url", None)
        st.rerun()

    if available_models:
        selected_model = st.selectbox("Classifier", available_models, index=0)
    else:
        selected_model = None

    analysis_key = hashlib.sha256(
        f"{document_name}\0{selected_model}\0{policy_text}".encode("utf-8")
    ).hexdigest()
    if st.button("Analyze clauses", type="primary", disabled=not bool(available_models)):
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_report", None)
        if not policy_text.strip():
            st.error("Provide policy text before running an analysis.")
        else:
            try:
                with st.spinner("Classifying clauses..."):
                    predictor = get_predictor(
                        str(ARTIFACT_DIR), MODEL_KEYS[selected_model]
                    )
                    result = analyze_text(policy_text, predictor)
                report = {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "document": document_name,
                    "model": selected_model,
                    "disclaimer": (
                        "Predictions are research outputs, not legal advice or a compliance decision. "
                        "Scores are model probabilities and may be miscalibrated."
                    ),
                    "summary": result["summary"],
                    "key_terms": result["key_terms"],
                    "findings": result["findings"],
                }
                st.session_state["analysis_result"] = result
                st.session_state["analysis_report"] = report
                st.session_state["analysis_key"] = analysis_key
            except (ValueError, OSError, RuntimeError) as error:
                st.error(str(error))

    result = (
        st.session_state.get("analysis_result")
        if st.session_state.get("analysis_key") == analysis_key
        else None
    )
    report = (
        st.session_state.get("analysis_report")
        if st.session_state.get("analysis_key") == analysis_key
        else None
    )
    if result and report:
        st.subheader("Plain-English summary")
        st.caption(
            "These are topic counts and review prompts, not ratings of risk or legal compliance. "
            "The classifier is forced to choose among seven topics and has no 'other/none' class."
        )
        summary_columns = st.columns(4)
        for index, label in enumerate(LABELS):
            summary_columns[index % len(summary_columns)].metric(
                LABEL_DETAILS[label]["name"], result["category_counts"][label]
            )
        for summary_item in result["summary"]:
            st.write(f"- {summary_item}")
        st.markdown("**Key terms appearing in the evidence:**")
        st.write(", ".join(result["key_terms"]) if result["key_terms"] else "No configured key terms matched.")

        st.subheader("Clause findings and evidence")
        st.caption("Evidence below is the source clause as supplied; model scores are not certainty.")
        for index, finding in enumerate(result["findings"], start=1):
            with st.expander(
                f"{index}. {finding['name']} — model score {finding['score']:.1%}"
            ):
                st.markdown(f"**Plain English:** {finding['plain_english']}")
                st.markdown(f"**Review prompt:** {finding['review_prompt']}")
                st.markdown(
                    "**Key terms in this clause:** "
                    + (", ".join(finding["key_terms"]) or "No configured key terms matched.")
                )
                st.markdown("**Original evidence:**")
                st.code(finding["evidence"], language=None)

        json_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=[
                "label",
                "name",
                "score",
                "plain_english",
                "review_prompt",
                "key_terms",
                "evidence",
            ],
        )
        writer.writeheader()
        writer.writerows(result["findings"])
        left, right = st.columns(2)
        left.download_button(
            "Download JSON report",
            data=json_bytes,
            file_name="ai_contract_lens_report.json",
            mime="application/json",
        )
        right.download_button(
            "Download CSV findings",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="ai_contract_lens_findings.csv",
            mime="text/csv",
        )

with evaluation_tab:
    metrics = read_metrics(str(ARTIFACT_DIR))
    if not metrics:
        st.info(
            "No measured evaluation report is available yet. Train the models to generate "
            "website-disjoint test metrics, confusion matrices, and error analysis."
        )
    else:
        st.markdown(
            "All displayed values are measured on the held-out website test split. "
            "Macro-F1 weights each class equally; class imbalance can make accuracy misleading."
        )
        st.caption(
            f"Device: {metrics['device']} · "
            f"Train/validation/test rows: {metrics['split_sizes']} · "
            f"Websites per split: {metrics['split_websites']}"
        )
        for title, model_name in MODEL_KEYS.items():
            model_metrics = metrics.get("models", {}).get(model_name)
            if not model_metrics:
                continue
            st.subheader(title)
            columns = st.columns(4)
            columns[0].metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
            columns[1].metric("Macro precision", f"{model_metrics['macro_precision']:.3f}")
            columns[2].metric("Macro recall", f"{model_metrics['macro_recall']:.3f}")
            columns[3].metric("Macro-F1", f"{model_metrics['macro_f1']:.3f}")
            class_report = model_metrics["per_class"]
            class_rows = [
                {
                    "Class": LABEL_DETAILS[label]["name"],
                    "Precision": class_report[label]["precision"],
                    "Recall": class_report[label]["recall"],
                    "F1": class_report[label]["f1-score"],
                    "Support": int(class_report[label]["support"]),
                }
                for label in LABELS
            ]
            st.dataframe(class_rows, hide_index=True, use_container_width=True)

            confusion_path = ARTIFACT_DIR / f"confusion_matrix_{model_name}.csv"
            if confusion_path.is_file():
                import pandas as pd

                matrix = pd.read_csv(confusion_path, index_col=0)
                matrix.index = [LABEL_DETAILS[label]["name"] for label in matrix.index]
                matrix.columns = [LABEL_DETAILS[label]["name"] for label in matrix.columns]
                st.markdown("**Confusion matrix** (rows: true class; columns: predicted class)")
                st.dataframe(matrix, use_container_width=True)
            error_path = ARTIFACT_DIR / f"errors_{model_name}.csv"
            if error_path.is_file():
                import pandas as pd

                errors = pd.read_csv(error_path)
                st.markdown(f"**Held-out errors:** {len(errors)} misclassified clauses")
                if not errors.empty:
                    st.dataframe(errors.head(20), hide_index=True, use_container_width=True)

with about_tab:
    st.subheader("Training data")
    st.write(
        "Supervised labels come only from the supplied OPP-115 human annotations. The "
        "prepared policy-clause CSV contains 17,567 rows before exact duplicate removal, "
        "spanning 115 website IDs; its seven target labels were checked against the original "
        "OPP-115 category names."
    )
    st.write(
        "The separate cookie-banner CSV is not used for supervised training: every row is "
        "marked `heuristic_from_observed_text_review_required`, and 122 rows have no label. "
        "Those labels are not treated as human ground truth."
    )
    st.subheader("Limitations")
    st.markdown(
        "- OPP-115 policies were collected around 2015 and do not represent current policies or all jurisdictions.\n"
        "- The target categories are mapped directly from OPP-115 categories; a predicted category is not a legal risk rating.\n"
        "- The model must choose among the seven supported categories; there is no 'other/none' class, so out-of-scope text can be assigned the closest available topic.\n"
        "- Clause-level classification can miss context, negation, exceptions, and layout or PDF extraction errors.\n"
        "- Confidence scores are not calibrated and findings need human verification against the evidence shown.\n"
        "- This tool does not determine compliance, provide legal advice, or infer whether a company actually follows its policy.\n"
        "- Individual analyses are limited to 1,000,000 extracted characters."
    )
    st.caption(f"Maximum policy URL response size: {MAX_POLICY_BYTES // (1024 * 1024)} MB.")

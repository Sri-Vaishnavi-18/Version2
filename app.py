from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from contract_lens.data import LABELS
from contract_lens.inference import (
    LABEL_DETAILS,
    ClausePredictor,
    analyze_text,
    split_into_clauses,
)
from contract_lens.sources import extract_pdf_text, fetch_policy_url

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("CONTRACT_LENS_ARTIFACTS", ROOT / "artifacts"))
MODEL_NAMES = {
    "BiLSTM + Attention": "bilstm_attention",
    "BiLSTM without Attention": "bilstm",
    "TF-IDF + Logistic Regression": "tfidf_logistic_regression",
}
MODEL_FILES = {
    "bilstm_attention": "bilstm_attention.pt",
    "bilstm": "bilstm.pt",
    "tfidf_logistic_regression": "tfidf_logistic_regression.joblib",
}
HERO_IMAGE = ROOT / "img3.jpg"
FEATURE_IMAGE = ROOT / "img1.jpg"
SUPPORT_IMAGE = ROOT / "img2.avif"
PAGE_OPTIONS = ("Analyze Policy", "Model Comparison")

st.set_page_config(
    page_title="AI Contract Lens",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def image_data_uri(path: Path) -> str:
    """Read a repository image for use as a responsive CSS background."""
    if not path.is_file():
        return ""
    suffix = path.suffix.lower()
    mime_type = "image/avif" if suffix == ".avif" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


hero_image = image_data_uri(HERO_IMAGE)
hero_background = (
    f'url("{hero_image}")'
    if hero_image
    else "linear-gradient(120deg, #102b2a, #176c61)"
)

st.markdown(
    f"""
    <style>
    :root{{--ink:#102b2a;--muted:#647570;--paper:#f7f8f5;--line:#e3eae5;
      --teal:#0c766d;--deep:#07574f;--lime:#c5ef7b}}
    html{{scroll-behavior:smooth}}
    .stApp{{background:var(--paper);color:var(--ink)}}
    .stApp p,.stApp label{{color:var(--ink)}}
    [data-testid="stHeader"]{{background:transparent}}
    [data-testid="stToolbar"],#MainMenu,footer{{visibility:hidden}}
    .block-container{{max-width:1200px;padding:1.2rem 2rem 3rem}}
    .brand{{font-size:1.08rem;font-weight:850;letter-spacing:-.04em;color:var(--ink);
      margin:.3rem 0 1rem}}
    .brand span{{display:inline-grid;place-items:center;width:34px;height:34px;margin-right:.55rem;
      border-radius:11px;background:linear-gradient(135deg,var(--teal),#36a48f);color:white}}
    .hero{{min-height:470px;display:flex;align-items:center;position:relative;overflow:hidden;
      padding:clamp(1.7rem,5vw,4.3rem);border-radius:28px;color:white;
      background-image:linear-gradient(90deg,rgba(7,24,25,.88) 0%,rgba(7,31,31,.69) 50%,
        rgba(7,31,31,.18) 100%),{hero_background};
      background-size:cover;background-position:center 43%;
      box-shadow:0 25px 65px rgba(10,45,42,.2);animation:enter .7s ease both}}
    .hero-copy{{max-width:690px}}
    .eyebrow,.kicker{{text-transform:uppercase;letter-spacing:.14em;font-size:.74rem;font-weight:850}}
    .eyebrow{{color:#d0f096}}
    .hero h1{{color:#fff!important;font-size:clamp(2.8rem,7vw,6rem);line-height:.97;
      letter-spacing:-.075em;font-weight:850;margin:1rem 0}}
    .hero p{{max-width:39rem;color:#e0ece6;font-size:1.05rem;line-height:1.7}}
    .kicker{{color:var(--teal);margin:0 0 .55rem}}
    .section-title{{font-size:clamp(1.8rem,4vw,3rem);line-height:1.05;letter-spacing:-.055em;
      font-weight:850;margin:.2rem 0 .7rem;color:var(--ink)}}
    .muted{{color:var(--muted);line-height:1.7}}
    .panel,.model-card,.feature-card{{height:100%;border:1px solid var(--line);border-radius:20px;
      background:rgba(255,255,255,.92);padding:1.2rem;
      box-shadow:0 12px 35px rgba(20,50,43,.05);transition:transform .2s,box-shadow .2s}}
    .panel:hover,.model-card:hover,.feature-card:hover{{transform:translateY(-3px);
      box-shadow:0 18px 42px rgba(20,50,43,.1)}}
    .feature-image img{{height:285px!important;width:100%!important;object-fit:cover!important;
      border-radius:17px}}
    .support-image img{{height:125px!important;width:100%!important;object-fit:cover!important;
      border-radius:15px}}
    .stImage img{{border-radius:16px}}
    div.stButton>button,div.stDownloadButton>button{{border-radius:12px;
      border:1px solid #cfe1d7;color:#174b42;font-weight:750;background:white;
      transition:all .18s ease}}
    div.stButton>button:hover,div.stDownloadButton>button:hover{{background:#eaf5ed;
      border-color:#80b9a0;color:#093c35;transform:translateY(-1px)}}
    div.stButton>button[kind="primary"]{{background:linear-gradient(120deg,#0d756a,#14907a);
      color:white;border:0;box-shadow:0 8px 20px rgba(12,118,109,.2);
      background-size:180% 100%;animation:button-glow 6s ease infinite}}
    div.stButton>button[kind="primary"]:hover{{background:#075b52;color:white;
      box-shadow:0 12px 26px rgba(12,118,109,.28);transform:translateY(-2px)}}
    [data-testid="stWidgetLabel"] p{{color:var(--ink)!important;font-weight:750}}
    [data-testid="stTextArea"] textarea{{background:#252630!important;color:#f7f8fb!important;
      border:1px solid #252630!important;border-radius:12px!important;min-height:190px;
      padding:1rem!important;line-height:1.65;transition:border-color .2s,box-shadow .2s}}
    [data-testid="stTextArea"] textarea::placeholder{{color:#b9bdc9!important;opacity:1}}
    [data-testid="stTextArea"] textarea:focus{{border-color:#36a48f!important;
      box-shadow:0 0 0 3px rgba(12,118,109,.18)!important}}
    div[data-testid="stMetric"]{{background:white;border:1px solid var(--line);
      padding:.9rem;border-radius:15px}}
    div[data-testid="stMetricValue"]{{color:var(--deep)}}
    div[data-testid="stExpander"]{{background:white;border:1px solid var(--line);border-radius:15px}}
    .evidence{{white-space:normal;overflow-wrap:anywhere;line-height:1.8;font-size:1rem;
      color:#183d35;padding:1rem;border-left:4px solid #78b899;border-radius:0 12px 12px 0;
      background:#f2f8f3}}
    .stTabs [data-baseweb="tab-list"]{{gap:.4rem;border-bottom:1px solid var(--line)}}
    .stTabs [data-baseweb="tab"]{{height:3.2rem;padding:0 1.1rem;font-weight:750}}
    .stTabs [aria-selected="true"]{{color:var(--teal)!important}}
    div[data-testid="stRadioGroup"] [role="radiogroup"]{{gap:.5rem}}
    div[data-testid="stRadioGroup"] label[data-testid="stRadioOption"]{{
      padding:.55rem .9rem;border:1px solid var(--line);border-radius:999px;background:white}}
    div[data-testid="stRadioGroup"] label[data-testid="stRadioOption"] p{{
      color:var(--ink)!important;font-weight:700}}
    div[data-testid="stRadioGroup"] label[data-testid="stRadioOption"][data-selected="true"]{{
      background:#e7f3e9;border-color:#95c4a1}}
    @keyframes enter{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}}}}
    @keyframes button-glow{{0%,100%{{background-position:0% 50%}}50%{{background-position:100% 50%}}}}
    @media(max-width:720px){{.block-container{{padding:.7rem 1rem 2rem}}
      .hero{{min-height:390px;border-radius:20px;padding:1.5rem;background-position:58% center}}
      .feature-image img{{height:190px!important}}}}
    @media(prefers-reduced-motion:reduce){{*,*::before,*::after{{scroll-behavior:auto!important;
      animation-duration:.01ms!important;transition-duration:.01ms!important}}}}
    </style>
    """,
    unsafe_allow_html=True,
)


def uploaded_text(file) -> str:
    content = file.getvalue()
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Uploaded files must be 20 MB or smaller.")
    suffix = Path(file.name).suffix.lower()
    if suffix == ".pdf":
        text = extract_pdf_text(content)
    elif suffix == ".txt":
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise ValueError("TXT uploads must use UTF-8 encoding.") from error
    else:
        raise ValueError("Upload a PDF or plain-text (.txt) file.")
    if not text:
        raise ValueError("No extractable text was found in the uploaded file.")
    return text


@st.cache_resource(show_spinner=False)
def get_predictor(artifact_dir: str, model_key: str) -> ClausePredictor:
    return ClausePredictor(artifact_dir, model_key)


@st.cache_data(show_spinner=False)
def load_metrics(artifact_dir: str) -> dict:
    path = Path(artifact_dir) / "metrics.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


available_for_compare = {
    key: (ARTIFACTS / filename).is_file() for key, filename in MODEL_FILES.items()
}
attention_ready = available_for_compare["bilstm_attention"] and (
    ARTIFACTS / "tokenizer.json"
).is_file()

st.markdown(
    '<div class="brand"><span aria-hidden="true">⌕</span>AI Contract Lens</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Privacy policy intelligence</div>
        <h1>See what’s in<br>the fine print.</h1>
        <p>Turn dense policy language into clear, evidence-linked topics.
        Review complete sentences, keep the context, and stay in control.</p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if st.button("Analyze a Contract  ↓", type="primary", key="hero_analyze"):
    st.session_state["active_page"] = PAGE_OPTIONS[0]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = PAGE_OPTIONS[0]
st.radio(
    "Project pages",
    PAGE_OPTIONS,
    horizontal=True,
    key="active_page",
    label_visibility="collapsed",
)

if st.session_state["active_page"] == "Analyze Policy":
    st.markdown('<div id="analyzer"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="kicker">Policy analyzer</p>'
        '<h2 class="section-title">Start with a document.<br>Finish with evidence.</h2>'
        '<p class="muted">The primary classifier is the trained BiLSTM + Attention model. '
        'Comparison models are available only on the Model Comparison page.</p>',
        unsafe_allow_html=True,
    )

    image_col, steps_col = st.columns([1, 1.15], gap="large", vertical_alignment="center")
    with image_col:
        if FEATURE_IMAGE.is_file():
            with st.container(key="feature_image"):
                st.image(
                    str(FEATURE_IMAGE),
                    caption="Keep the source wording close while you review.",
                    use_container_width=True,
                    alt="A person reviewing a document at a desk.",
                )
    with steps_col:
        st.markdown(
            '<div class="panel"><div class="kicker">A simple review flow</div>'
            '<p><strong>1.</strong> Paste text, upload PDF/TXT, or fetch a public URL.</p>'
            '<p><strong>2.</strong> Classify policy sentences with BiLSTM + Attention.</p>'
            '<p><strong>3.</strong> Check complete evidence and export your findings.</p></div>',
            unsafe_allow_html=True,
        )
        if SUPPORT_IMAGE.is_file():
            with st.container(key="support_image"):
                st.image(
                    str(SUPPORT_IMAGE),
                    caption="Designed for clear, human-led review.",
                    use_container_width=True,
                    alt="Supporting image for policy review.",
                )

    if not attention_ready:
        st.warning(
            f"BiLSTM + Attention checkpoint and tokenizer were not both found in `{ARTIFACTS}`. "
            "Add `bilstm_attention.pt` and `tokenizer.json` to enable analysis."
        )

    with st.container():
        source = st.radio(
            "Policy source",
            ("Paste text", "Upload PDF/TXT", "Public policy URL"),
            horizontal=True,
            key="source_choice",
        )
        text = ""
        document = "pasted-policy"
        if source == "Paste text":
            text = st.text_area(
                "Paste policy text",
                height=190,
                placeholder="Paste Terms, a Privacy Policy, or a cookie notice.",
                key="policy_text",
            )
        elif source == "Upload PDF/TXT":
            file = st.file_uploader("Choose a PDF or TXT file", type=("pdf", "txt"))
            if file:
                document = file.name
                try:
                    text = uploaded_text(file)
                    st.caption(f"Extracted {len(text):,} characters.")
                except (ValueError, OSError) as error:
                    st.error(str(error))
        else:
            url = st.text_input(
                "Public policy URL",
                placeholder="https://example.com/privacy",
                key="policy_url",
            )
            fetch_col, clear_col = st.columns(2)
            if fetch_col.button("Fetch policy", key="fetch_policy"):
                try:
                    with st.spinner("Fetching public policy text..."):
                        st.session_state["fetched_text"] = fetch_policy_url(url)
                        st.session_state["fetched_url"] = url
                    st.rerun()
                except (ValueError, OSError) as error:
                    st.error(str(error))
            if clear_col.button("Clear fetched text", key="clear_policy"):
                st.session_state.pop("fetched_text", None)
                st.session_state.pop("fetched_url", None)
                st.rerun()
            if url == st.session_state.get("fetched_url"):
                text = st.session_state.get("fetched_text", "")
                document = url
                if text:
                    st.success(f"Fetched {len(text):,} characters.")

        if text:
            st.caption(
                f"{len(text):,} characters · {len(split_into_clauses(text)):,} sentences ready to analyze"
            )

        fingerprint = hashlib.sha256(
            f"{document}\0bilstm_attention\0{text}".encode("utf-8")
        ).hexdigest()
        if st.button(
            "Analyze with BiLSTM + Attention",
            type="primary",
            use_container_width=True,
            disabled=not attention_ready,
            key="run_analysis",
        ):
            st.session_state.pop("analysis", None)
            if not text.strip():
                st.error("Provide policy text before running an analysis.")
            else:
                try:
                    with st.spinner("Classifying complete policy sentences..."):
                        predictor = get_predictor(str(ARTIFACTS), "bilstm_attention")
                        result = analyze_text(text, predictor)
                    report = {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "document": document,
                        "model": "bilstm_attention",
                        "summary": result["summary"],
                        "key_terms": result["key_terms"],
                        "findings": result["findings"],
                        "disclaimer": "Research prototype; not legal advice or a compliance decision.",
                    }
                    st.session_state["analysis"] = {
                        "key": fingerprint,
                        "result": result,
                        "report": report,
                    }
                except (ValueError, OSError, RuntimeError) as error:
                    st.error(str(error))

    saved = st.session_state.get("analysis", {})
    if saved.get("key") == fingerprint:
        result = saved["result"]
        report = saved["report"]
        st.markdown("### Analysis overview")
        metric_cols = st.columns(4)
        for index, label in enumerate(LABELS):
            metric_cols[index % 4].metric(
                LABEL_DETAILS[label]["name"], result["category_counts"][label]
            )
        chart = pd.DataFrame(
            {
                "Topic": [LABEL_DETAILS[label]["name"] for label in LABELS],
                "Sentences": [result["category_counts"][label] for label in LABELS],
            }
        ).set_index("Topic")
        st.bar_chart(chart, horizontal=True, color="#0c766d")
        st.caption(
            "Model scores are not calibrated certainty. The classifier must choose one of "
            "the seven supported topics."
        )

        st.markdown("### Sentence findings")
        for index, finding in enumerate(result["findings"], start=1):
            with st.expander(
                f"{index:02d} · {finding['name']} · model score {finding['score']:.1%}"
            ):
                st.write(f"**Plain English:** {finding['plain_english']}")
                st.write(f"**Review prompt:** {finding['review_prompt']}")
                if finding["key_terms"]:
                    st.write("**Key terms:** " + ", ".join(finding["key_terms"]))
                st.markdown("**Complete source sentence**")
                st.markdown(
                    f'<div class="evidence">{html.escape(finding["evidence"])}</div>',
                    unsafe_allow_html=True,
                )

        csv_buffer = io.StringIO()
        fields = ("label", "name", "score", "plain_english", "review_prompt", "key_terms", "evidence")
        writer = csv.DictWriter(csv_buffer, fieldnames=fields)
        writer.writeheader()
        for finding in result["findings"]:
            writer.writerow({**finding, "key_terms": ", ".join(finding["key_terms"])})
        json_col, csv_col = st.columns(2)
        json_col.download_button(
            "Download JSON report",
            json.dumps(report, ensure_ascii=False, indent=2),
            "ai_contract_lens_report.json",
            "application/json",
            use_container_width=True,
        )
        csv_col.download_button(
            "Download CSV findings",
            csv_buffer.getvalue().encode("utf-8"),
            "ai_contract_lens_findings.csv",
            "text/csv",
            use_container_width=True,
        )

elif st.session_state["active_page"] == "Model Comparison":
    st.markdown('<div id="comparison"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="kicker">Model comparison</p>'
        '<h2 class="section-title">How the classifiers compare.</h2>'
        '<p class="muted">AI Contract Lens classifies policy clauses into seven OPP-115 topics. '
        'The primary app uses BiLSTM + Attention; the other classifiers are benchmarks. '
        'Cookie-banner heuristic labels are excluded from supervised training.</p>',
        unsafe_allow_html=True,
    )
    metrics = load_metrics(str(ARTIFACTS))
    if not metrics:
        st.info("No measured test metrics found. Include the completed training `metrics.json` artifact.")
    else:
        st.caption(
            f"Website-disjoint test evaluation · Device: {metrics.get('device', 'not recorded')} · "
            f"Rows: {metrics.get('split_sizes', {})} · Websites: {metrics.get('split_websites', {})}"
        )
        comparison_rows = []
        for title, key in MODEL_NAMES.items():
            scores = metrics.get("models", {}).get(key)
            if scores:
                comparison_rows.append(
                    {
                        "Model": title,
                        "Accuracy": scores["accuracy"],
                        "Macro-F1": scores["macro_f1"],
                        "Weighted-F1": scores["weighted_f1"],
                    }
                )
        if comparison_rows:
            comparison = pd.DataFrame(comparison_rows).set_index("Model")
            st.bar_chart(comparison, color=["#0c766d", "#7565e8", "#c5a647"])
            st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
        else:
            st.info("The metrics file contains no model scores.")

    st.markdown("#### Individual model results")
    model_tabs = st.tabs(list(MODEL_NAMES))
    for tab, (title, key) in zip(model_tabs, MODEL_NAMES.items()):
        with tab:
            scores = metrics.get("models", {}).get(key) if metrics else None
            available = available_for_compare[key]
            if not available:
                st.info(f"The trained `{MODEL_FILES[key]}` model artifact is not present.")
            if not scores:
                st.info("No measured test metrics are recorded for this model.")
                continue

            st.markdown(f"#### {title}")
            kpi = st.columns(4)
            for column, metric, label in zip(
                kpi,
                ("accuracy", "macro_precision", "macro_recall", "macro_f1"),
                ("Accuracy", "Macro precision", "Macro recall", "Macro-F1"),
            ):
                column.metric(label, f"{scores[metric]:.3f}")
            report = scores["per_class"]
            class_rows = [
                {
                    "Topic": LABEL_DETAILS[label]["name"],
                    "Precision": report[label]["precision"],
                    "Recall": report[label]["recall"],
                    "F1": report[label]["f1-score"],
                    "Support": int(report[label]["support"]),
                }
                for label in LABELS
            ]
            st.dataframe(
                pd.DataFrame(class_rows).style.format(
                    {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
                ),
                hide_index=True,
                use_container_width=True,
            )
            confusion_path = ARTIFACTS / f"confusion_matrix_{key}.csv"
            if confusion_path.is_file():
                matrix = pd.read_csv(confusion_path, index_col=0)
                matrix.index = [LABEL_DETAILS[label]["name"] for label in matrix.index]
                matrix.columns = [LABEL_DETAILS[label]["name"] for label in matrix.columns]
                st.caption("Confusion matrix · rows: actual topic; columns: predicted topic")
                st.dataframe(matrix, use_container_width=True)
            error_path = ARTIFACTS / f"errors_{key}.csv"
            if error_path.is_file():
                with st.expander("Inspect held-out classification errors"):
                    errors = pd.read_csv(error_path)
                    st.caption(f"{len(errors)} test clauses were misclassified.")
                    if not errors.empty:
                        st.dataframe(errors.head(20), hide_index=True, use_container_width=True)

st.markdown(
    '<div style="margin-top:2.4rem;padding-top:1rem;border-top:1px solid #e3eae5;'
    'color:#647570;font-size:.85rem">AI Contract Lens · Research prototype · Not legal advice</div>',
    unsafe_allow_html=True,
)

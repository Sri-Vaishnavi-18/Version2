# Pactio — Project, Model, and Viva Guide

> **Purpose:** Use this as the detailed project reference and professor-question preparation guide. It describes the code in this repository, distinguishes the final implementation from earlier exploratory work, and does not claim unverified experimental results.

## 1. Project in one minute

Pactio is the product name for this research prototype. It takes policy text and classifies each clause into one of seven privacy-practice topics. A finding is shown with the original clause as evidence, a short plain-English explanation, a review prompt, and a model score. The system compares:

1. TF-IDF + Logistic Regression (classical baseline)
2. A scratch-trained BiLSTM without attention
3. A scratch-trained BiLSTM with attention (the main deep-learning model)

The neural models use randomly initialized trainable embeddings. There is no BERT, transformer, external pretrained language model, or pretrained word-vector table. Training and test partitions are separated by `website_id`, not by random clause rows, to reduce website-level leakage.

The app accepts pasted text, PDF/TXT files, and accessible public policy URLs. It provides topic counts, literal key-term matches, evidence-linked findings, and JSON/CSV exports. It is not legal advice or a compliance/risk determination.

## 2. Problem definition and scope

### Task

Given a policy clause, predict its OPP-115 privacy-practice category. This is **single-label, seven-class classification**: each training row has one category, and the model returns one of the seven known categories.

### What the task is not

- It does not determine whether a clause is good, bad, risky, lawful, or compliant.
- It does not prove that an organization follows its written policy.
- It does not provide a calibrated probability of correctness.
- It does not generate a free-form legal summary. Plain-English descriptions and review prompts are application text associated with predicted categories.
- It does not include a reject/unknown class. Consequently, text outside the seven training categories is still assigned one of them.

## 3. Dataset inspection and label provenance

The workspace contains the OPP-115 corpus and a prepared policy-clause CSV at `Working Prototype/opp115_policy_clauses_prepared.csv`. The prepared CSV was inspected before implementation:

- 17,567 rows before exact duplicate removal in the loader.
- 115 distinct `website_id` values.
- Required columns include `website_id`, `policy_text`, `opp_category`, and `policy_labels`.
- Seven observed OPP-115 categories map to seven supplied target strings.
- Every prepared `policy_labels` value agreed with the mapping from its original `opp_category` in the inspected file. The loader checks that agreement again and stops if it finds a mismatch.
- Rows without website IDs, text, or categories cause an explicit error; unsupported categories also cause an error rather than being silently discarded.
- Exact duplicate `(website_id, cleaned policy_text, target)` rows are removed.

### Category mapping

| OPP-115 source category | Training target | User-facing topic |
|---|---|---|
| `First Party Collection/Use` | `data_collection` | Data collection and use |
| `Third Party Sharing/Collection` | `third_party_sharing` | Sharing with third parties |
| `User Choice/Control` | `consent_choices` | Consent and choices |
| `Data Security` | `security_protection` | Data security |
| `Data Retention` | `retention_deletion` | Retention and deletion |
| `Do Not Track` | `advertising_tracking` | Advertising and tracking choices |
| `User Access, Edit and Deletion` | `user_rights_opt_out` | User access, correction, and deletion rights |

These are renamed existing OPP-115 category annotations, **not newly invented labels**. In a viva, describe them as OPP-115 category labels. Do not imply the model was trained on independent lawyer judgments of risk.

### Cleaning and duplicate handling

`contract_lens.data.clean_text` decodes HTML entities, strips text matching HTML tags, collapses runs of whitespace, and trims the ends. The loader uses the original category to derive the target and verifies the prepared target. It only removes rows whose website, cleaned text, and target are all identical. The same text on different websites is retained, but the website grouping keeps those rows in one partition if they share the same `website_id`.

### Why the cookie-banner CSV is excluded

`Working Prototype/cookie_banners_prepared.csv` has 561 rows. Its `label_source` is `heuristic_from_observed_text_review_required` for all rows, and 122 rows lack a banner label. Those labels are heuristic and require human review, so the final supervised training and reported final-model comparison exclude this CSV. Treating those labels as verified ground truth would overstate their reliability.

### Data license and age

The supplied OPP-115 readme says its annotations are for research, teaching, and scholarship with non-commercial terms. Check the corpus documentation and obtain any needed permissions before redistribution or commercial use. The data describe historical policies collected around 2015, not current policies. Cite the OPP-115 corpus paper in academic submissions.

## 4. Leakage-safe split and imbalance

The split implementation is in `contract_lens.data.split_by_website`.

1. `GroupShuffleSplit` first separates a held-out test group of websites (20% of website groups).
2. A second grouped split reserves 20% of the remaining website groups for validation. This is approximately 64% train, 16% validation, and 20% test by website, subject to the group-level sizes.
3. The code tries deterministic random seeds until each partition contains every observed label. It fails with an explicit error if that is not possible.
4. A final intersection check verifies there are no `website_id` overlaps.
5. The vocabulary is built from the training partition only. Validation/test text does not determine vocabulary membership.

This is preferable to splitting clause rows at random: clauses from the same website are correlated in vocabulary, style, and policy wording. A row-level split could put almost-identical clauses from the same site on both sides and inflate apparent generalization.

The class frequencies are imbalanced. The neural loss uses per-class weights calculated from training data:

\[
w_k = \frac{N}{K n_k}
\]

where \(N\) is the number of training clauses, \(K\) the number of classes, and \(n_k\) the number of training clauses in class \(k\). The TF-IDF logistic-regression baseline also uses scikit-learn's `class_weight="balanced"`.

This compensates for frequency imbalance during training; it does not add examples, guarantee fair recall, or make a small class statistically well represented. OPP-115's `Do Not Track` class is especially small, so its per-class test result deserves careful discussion.

## 5. Model code explained

The reusable model code is in `src/contract_lens/models.py`; training and evaluation are in `src/contract_lens/training.py`.

### 5.1 Tokenization and vocabulary

The tokenizer uses a regular expression to split lowercase text into alphabetic tokens (including simple apostrophes), numbers, and punctuation. It does not use a pretrained tokenizer.

- `<PAD>` has token ID 0 and pads variable-length batches.
- `<UNK>` has token ID 1 for words not in the training vocabulary.
- The vocabulary is capped at 20,000 entries by default, and a token must occur at least twice in training data.
- A clause is truncated to 160 tokens by default. Empty text encodes to `<UNK>`, although the data loader rejects empty training clauses.

### 5.2 Learned embedding layer

The embedding table is a trainable matrix \(E\in\mathbb{R}^{V\times d}\), with vocabulary size \(V\) and embedding dimension \(d=96\) by default. Token \(x_t\) is mapped to vector:

\[
e_t = E[x_t]
\]

The padding row is configured with `padding_idx=0`. Embeddings are initialized and optimized as part of this project; they are not downloaded pretrained embeddings.

### 5.3 LSTM recurrence

The LSTM is a recurrent neural network designed to retain or discard information through gates. For input \(x_t\) and prior hidden state \(h_{t-1}\), a standard LSTM computes:

\[
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i)\\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f)\\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g)\\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o)\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t\\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
\]

Here \(i_t\), \(f_t\), and \(o_t\) are input, forget, and output gates; \(g_t\) is the candidate cell update; \(c_t\) is the cell state; \(\sigma\) is sigmoid; and \(\odot\) is element-wise multiplication. The forget gate controls retained prior memory, while the input gate controls new information.

### 5.4 Bidirectionality

A unidirectional LSTM reads tokens from left to right. A BiLSTM reads the clause in both directions:

\[
\overrightarrow{h_t} = \mathrm{LSTM}_{\rightarrow}(e_1,\ldots,e_t),
\qquad
\overleftarrow{h_t} = \mathrm{LSTM}_{\leftarrow}(e_T,\ldots,e_t)
\]

The representation at token \(t\) is their concatenation:

\[
h_t = [\overrightarrow{h_t};\overleftarrow{h_t}]
\]

The default hidden size is 64 in **each** direction, so each token representation has 128 features. Bidirectionality provides preceding and following context when labeling a clause.

The code uses `pack_padded_sequence` so the LSTM does not process padding as if it were part of a clause. It pads only to the longest sequence in each batch.

### 5.5 Attention pooling (main model)

Attention assigns a learned weight to each valid token representation. In this implementation, a learned linear projection produces a scalar score:

\[
u_t = v^\top h_t
\]

Padding positions are masked. Scores are normalized with softmax over the valid clause tokens:

\[
\alpha_t = \frac{\exp(u_t)}{\sum_{j=1}^{T}\exp(u_j)}
\]

The clause vector is the weighted sum:

\[
c = \sum_{t=1}^{T}\alpha_t h_t
\]

The classifier maps this vector to seven logits:

\[
z = W_c\,\mathrm{Dropout}(c) + b_c
\]

Attention is a learned pooling mechanism that lets the classifier weight token representations differently. **Attention weights are not guaranteed faithful explanations.** The interface therefore shows the original clause, not just a highlighted attention score, as evidence.

### 5.6 BiLSTM without attention (ablation)

The comparison model has the same trainable embedding and bidirectional LSTM. It uses the final forward and backward hidden states concatenated as its clause representation:

\[
c_{\text{plain}} = [\overrightarrow{h_T};\overleftarrow{h_1}]
\]

The final classifier is otherwise similar. Comparing it with attention asks whether learned token-level pooling helps relative to final-state pooling. This is an ablation comparison, not a different dataset.

### 5.7 Output probabilities

For class \(k\), softmax turns logits \(z_k\) into a distribution:

\[
p_k = \frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}
\]

The predicted label is \(\hat y=\arg\max_k p_k\). The UI displays the maximum model probability as a “model score”; it may be overconfident or miscalibrated and should not be called a verified confidence or risk score.

### 5.8 Objective and optimization

Training uses weighted multi-class cross-entropy. For true class \(y_i\), predicted distribution \(p_i\), and class weight \(w_{y_i}\), the per-example loss is:

\[
\ell_i = -w_{y_i}\log p_{i,y_i}
\]

PyTorch's weighted cross-entropy reduction averages using the weights of examples in the minibatch. The optimizer is AdamW with learning rate \(10^{-3}\). A 0.3 dropout layer is used, gradient norms are clipped to 1.0, and mini-batches default to 64. Training defaults to 12 epochs with early stopping after 3 epochs without validation macro-F1 improvement. The best validation macro-F1 state is saved for each neural model.

The random seed (default 42) is set for Python, NumPy, and PyTorch. GPU kernels and library versions may still introduce some nondeterminism, so exact bit-for-bit reproducibility across machines is not guaranteed.

## 6. Baseline: TF-IDF + Logistic Regression

The baseline is a scikit-learn `Pipeline`:

1. `TfidfVectorizer` lowercases the clause, extracts unigrams and bigrams, requires a term in at least two documents, caps the vocabulary at 100,000 features, uses sublinear term-frequency scaling, and applies the vectorizer's default L2 normalization.
2. `LogisticRegression` uses balanced class weights, the `lbfgs` solver, a maximum of 1,000 iterations, and the run seed.

For term \(t\) in document \(d\), TF-IDF is broadly:

\[
\mathrm{tfidf}(t,d) = \mathrm{tf}(t,d)\times\mathrm{idf}(t)
\]

With scikit-learn's smoothed inverse document frequency:

\[
\mathrm{idf}(t) = \log\left(\frac{1+N}{1+\mathrm{df}(t)}\right)+1
\]

where \(N\) is the number of training documents and \(\mathrm{df}(t)\) is the number containing term \(t\). Sublinear term frequency scales a nonzero count \(f\) as \(1+\log(f)\). The resulting feature vector is L2-normalized by default.

Logistic Regression learns a linear decision boundary in this sparse feature space. It is a useful, fast, interpretable-in-feature-space baseline, but it does not have the recurrent contextual representation of the neural models.

## 7. Models compared and models explored

### Final implemented, trainable, and comparable models

| Model | Input features | Learner | Purpose |
|---|---|---|---|
| TF-IDF + Logistic Regression | Sparse one-/two-token TF-IDF features | Balanced multiclass linear classifier | Classical baseline |
| BiLSTM without attention | Learned embeddings and bidirectional sequence encoding | Final forward/backward states + classifier | Neural ablation |
| BiLSTM + Attention | Same learned embeddings and BiLSTM | Masked attention pooling + classifier | Main deep-learning model |

All three use the same seven OPP-115 topic labels and the same saved website-group split when trained by `contract_lens.training.train`. Compare their **test** metrics from the same successful run, not metrics from different splits or training history.

### Earlier exploratory dual-input idea

The workspace also contained an earlier Colab prototype/notebook describing a dual-input architecture with separate policy-text and cookie-banner branches, attention pooling, policy and banner multi-label prediction heads, and an optional policy/banner relationship head. It also described extracting observable banner controls and comparing them with policy topics.

This earlier design is **exploratory and not the final deployed model**:

- The supplied banner labels are heuristic and review-required, not verified human labels.
- The relationship head was only to be trained if reviewed relationship annotations were provided; none should be assumed from the policy/banner text alone.
- The earlier notebook documented an optional synthetic smoke-test path. Results from a synthetic run would not be research evidence and must not be mixed with OPP-115 results.
- It is not one of the three classifiers shown in the final app's benchmark.

If asked about it, explain that the project narrowed the final supervised experiment to the trustworthy OPP-115 clause labels rather than presenting heuristic banner labels or unreviewed relationships as ground truth.

### How to describe experimental outcomes

The user previously reported **about 85% accuracy** from a Colab run. This assistant did not inspect that Colab run's `metrics.json`; therefore, this guide cannot independently confirm which model, split, or partition produced 85%. Before presenting it, verify:

1. It is the `accuracy` field for a model in `artifacts/metrics.json`.
2. It is from the held-out **test** split, not the training or validation metric.
3. You also report macro-F1, weighted-F1, per-class precision/recall/F1, the website-level split, and the confusion matrix.
4. The same test partition was used for all three model comparisons.

Do not claim that attention is better than the baseline unless your actual measured test results show it. No fabricated or placeholder metrics belong in the report or presentation.

## 8. Evaluation: definitions and formulae

For one class in a one-vs-rest view:

- True positive (TP): actual class and predicted class match.
- False positive (FP): predicted as this class, but actual class is different.
- False negative (FN): actual class is this class, but the prediction is different.
- True negative (TN): neither actual nor predicted class is this class.

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}
\]

\[
\mathrm{F1}=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
\]

F1 is defined as 0 by the implementation for divisions with no positive predictions/support (`zero_division=0`).

For \(K=7\) classes:

- **Accuracy:** total correct predictions divided by all test rows. Can look high when common categories dominate.
- **Macro precision/recall/F1:** calculate each metric per class, then give every class equal weight:

\[
\mathrm{MacroF1}=\frac{1}{K}\sum_{k=1}^{K}\mathrm{F1}_k
\]

- **Weighted-F1:** average per-class F1 weighted by the number of true examples of each class:

\[
\mathrm{WeightedF1}=\sum_{k=1}^{K}\frac{n_k}{N}\mathrm{F1}_k
\]

- **Confusion matrix:** a \(7\times7\) table. Rows are true labels, columns predicted labels. Diagonal cells are correct predictions; off-diagonal cells show which categories the model confuses.
- **Error analysis:** `errors_<model>.csv` holds each held-out misclassification with the website ID, true category, predicted category, and original clause. Review representative errors to understand ambiguity and label boundaries.

In an imbalanced dataset, do not report accuracy alone. Discuss the rare classes and the support (number of held-out examples) underlying each score.

## 9. Training artifacts

The configured output directory (often `artifacts/`) is produced by training:

| File | Contents |
|---|---|
| `bilstm_attention.pt` | Attention model state dictionary and model configuration |
| `bilstm.pt` | No-attention BiLSTM state dictionary and configuration |
| `tokenizer.json` | Training-only vocabulary, including padding and unknown tokens |
| `tfidf_logistic_regression.joblib` | Fitted TF-IDF + Logistic Regression pipeline |
| `metrics.json` | Measured test metrics and per-class classification reports |
| `confusion_matrix_<model>.csv` | Test confusion matrix for each model |
| `errors_<model>.csv` | Held-out prediction errors and original clause text |
| `training_history.csv` | Neural epoch train loss and validation macro-F1 |
| `split_manifest.csv` | Website IDs assigned to train, validation, or test |
| `label_schema.json` | Label names and original category mapping |

The Pactio React frontend is served by FastAPI and uses `bilstm_attention.pt` plus `tokenizer.json` for its analysis endpoint. The optional Streamlit research interface also supports model comparison and reads metrics and diagnostic CSVs when present. Keep test/error data private if clause text is sensitive. Never commit the dataset or environment folder to GitHub.

The Pactio analyzer uses only BiLSTM + Attention. TF-IDF + Logistic Regression and BiLSTM without Attention remain isolated in the optional Streamlit Model Comparison page.

## 10. Application flow

1. User pastes text, uploads a PDF/TXT, or enters a public HTTP/HTTPS URL.
2. PDF text is extracted using `pypdf`; text files use UTF-8; webpages are converted to visible text with BeautifulSoup. Image-only PDFs and script-rendered pages may not yield usable text.
3. Text is divided into paragraphs/sentences for the classifier. The original clause slice is retained as evidence.
4. FastAPI validates and converts the source to text, then `ClausePredictor` loads the saved BiLSTM + Attention checkpoint/tokenizer; it does not retrain during app use.
5. The model returns one topic and one maximum class probability for every clause. FastAPI groups clauses by topic for the React results screen.
6. The interface attaches a fixed plain-English explanation/review prompt and finds literal key-term matches from a short configured regex list. Key terms are not generated by the model and do not change its predicted class.
7. Pactio displays topic counts and original clause evidence and exports JSON/TXT. The optional Streamlit interface exports JSON/CSV and reads the training-produced test metrics, confusion matrices, and error files.

URL retrieval has safeguards for non-HTTP schemes, embedded credentials, non-standard ports, localhost/private IP resolutions, redirect counts, request timeouts, and a 5 MB response limit. User analyses are limited to one million characters. Treat uploaded text and exported error-analysis files as potentially sensitive.

## 11. Libraries and their roles

| Library | Role |
|---|---|
| Python | Application and model implementation |
| PyTorch (`torch`) | Embeddings, BiLSTM, attention, loss, optimizer, checkpointing, inference |
| Pandas | Load and validate clause data; write split, metrics, confusion matrix, and errors |
| scikit-learn | Grouped splitting, TF-IDF, Logistic Regression, and evaluation metrics |
| NumPy | Class-frequency weights, random seeding, numerical utilities |
| React | Pactio's responsive public-facing frontend and interactive results |
| Vite | Frontend development server and production asset build |
| FastAPI + Uvicorn | Text/URL/file analysis API and production serving for the built frontend |
| Streamlit | Optional model-comparison and research interface |
| pypdf | Extract text from PDF documents |
| requests | Fetch public policy URLs with bounded timeouts and streaming |
| BeautifulSoup | Convert HTML pages into visible text |
| joblib | Save/load the fitted scikit-learn pipeline |
| pytest | Unit and end-to-end smoke tests |

## 12. Tests and what they establish

Tests are under `tests/`:

- `test_data.py`: text normalization, duplicate removal, label mismatch rejection, disjoint website groups, class coverage.
- `test_models.py`: reserved vocabulary IDs, unknown-token behavior, finite output shapes for both neural architectures.
- `test_inference.py`: sentence segmentation and preservation of original evidence.
- `test_api.py`: input validation, grouped evidence response, file upload, and trained-model readiness reporting.
- `test_sources.py`: rejects local/invalid URL cases.
- `test_training_integration.py`: runs a tiny **test fixture** through all three pipelines, writes artifacts, verifies the split manifest, and reloads every model.

The integration fixture is artificial test data used only to check code wiring; it is not the corpus, not a source of project metrics, and must not be described as a model experiment. The latest local run completed with 22 passing tests; rerun tests after later code changes before claiming that version is validated.

## 13. How to run and deploy

### Google Colab training

Upload the project files and prepared CSV into Colab's runtime (or mount Drive). From the folder containing `pyproject.toml`:

```python
!pip install -e .
```

Train:

```python
!python -m contract_lens.training \
  --data "Working Prototype/opp115_policy_clauses_prepared.csv" \
  --output "/content/artifacts" \
  --seed 42 --epochs 12 --batch-size 64
```

PyTorch uses CUDA if it is available. Confirm with `torch.cuda.is_available()` and print the GPU name before training. Download `/content/artifacts` after the command completes; runtime files disappear when a temporary Colab runtime is deleted.

### Local Pactio application

Install Python 3.10+, Node.js 20.19+, and the project dependencies. Build the React frontend and start the API:

```powershell
python -m pip install -e ".[test]"
Set-Location pactio-frontend\pactio-frontend
npm install
npm run build
Set-Location ..\..
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

The built React app and API are served together at `http://127.0.0.1:8000`. The API needs `artifacts/bilstm_attention.pt` and `artifacts/tokenizer.json`; `/api/health` reports whether the model loads successfully. For Render deployment, use the repository's Docker-based Blueprint so Node/npm are provided inside the build image; provision the two model files as Render secret files rather than committing them.

The optional comparison interface can still be run with `python -m streamlit run app.py`. This is a separate legacy/research UI; it is not the entry point for the supplied Pactio React design. GitHub hosting the source is distinct from access control for a running website. The supplied frontend has no sign-in screen, so any public deployment should be treated as accessible to visitors allowed by the hosting platform.

## 14. Limitations and responsible interpretation

1. **Historical distribution:** policies are from around 2015; language and legal frameworks change.
2. **Domain coverage:** a finite set of 115 sites cannot represent every service, language, jurisdiction, or policy style.
3. **Class imbalance:** a rare category can have unstable test metrics even with class weighting.
4. **Annotation semantics:** category labels are not judgments of harm, severity, legality, or compliance.
5. **Forced choice:** no unknown/out-of-scope category or rejection threshold is implemented.
6. **Context loss:** a sentence split can separate definitions, exceptions, negation, or references from the surrounding context.
7. **Extraction:** image-only PDFs, complex layouts, and dynamically rendered pages can lose information.
8. **Score calibration:** the displayed softmax or Logistic Regression probability has not been calibrated; it is a model score, not certainty.
9. **Evidence limitations:** the original clause is shown as provenance, but showing evidence does not establish that a model reasoned correctly.
10. **No legal judgment:** humans should review the full policy and consult qualified counsel for legal determinations.

## 15. Likely professor questions and suggested answers

### Q1. What problem are you solving?

We classify policy clauses into seven privacy-practice categories based on OPP-115 annotations and show the original clause with each prediction. The prototype organizes policy text for human review; it does not make legal-compliance decisions.

### Q2. Why did you use a BiLSTM?

The task is sequence classification. A BiLSTM creates context-aware token representations using both left and right context. It gives a from-scratch deep-learning comparison without using a pretrained transformer.

### Q3. What does attention add?

It computes a normalized score over BiLSTM token states and forms a weighted clause representation. This lets the classifier weight token states differently rather than only using the last hidden states. The weights are not guaranteed to be faithful explanations, so the app uses the original clause as evidence.

### Q4. Why use a TF-IDF baseline?

It is a simple, strong, reproducible classical baseline. It tests whether sparse word/bigram features plus a linear classifier are sufficient before attributing performance to the neural model.

### Q5. What is the difference between the two BiLSTMs?

Both share the learned embedding and bidirectional LSTM. The no-attention model concatenates the final forward and backward states. The attention model computes a masked softmax over every token state and weighted-sums them before classification.

### Q6. Did you use BERT or pretrained embeddings?

No. The embedding matrix is initialized by PyTorch and learned from the training partition. The only tokenizer is a regular-expression tokenizer with a vocabulary built from training text.

### Q7. How did you prevent data leakage?

We split by `website_id` using `GroupShuffleSplit`, then verify that the website sets for train, validation, and test are pairwise disjoint. Splitting individual clauses randomly could place clauses from the same policy/site in both train and test.

### Q8. Why class weighting?

The category counts are unequal. The neural loss weights class \(k\) inversely according to its training frequency, and the Logistic Regression baseline uses scikit-learn's balanced class weights. We still report per-class results because weighting cannot create more examples or remove uncertainty.

### Q9. What is macro-F1 and why report it?

F1 is the harmonic mean of precision and recall. Macro-F1 averages F1 equally across categories, so a common class does not dominate it. It complements accuracy for this imbalanced seven-class dataset.

### Q10. What does your 85% accuracy mean?

Only if verified in the run's `metrics.json`, it is the proportion of held-out test clauses correctly classified for that model and split. We also need macro-F1, per-class support/precision/recall/F1, the confusion matrix, and confirmation that the metric is test—not train/validation. The number alone does not establish generalization or legal accuracy.

### Q11. Why not train on the cookie-banner data?

The supplied rows are labeled as heuristic and review-required, and some lack labels. Using those labels as ground truth would introduce unverified supervision. The final benchmark therefore uses the human-annotated OPP-115 categories only.

### Q12. What was the dual-input model?

An earlier exploratory notebook described policy-text and cookie-banner branches with attention, multi-label outputs, and an optional relationship head. It is not the final deployed classifier. Banner labels and policy/banner relation labels need human review before they can support a valid supervised experiment.

### Q13. Does a high score mean a policy is safe or compliant?

No. The target is an OPP-115 topic category, not a risk label or legal decision. The output helps find relevant passages for a person to review.

### Q14. What happens to a clause outside the training topics?

The current classifier is forced to select one of seven known classes because no unknown class or confidence-rejection rule is implemented. This is a known limitation; out-of-scope clauses can receive a misleading topic.

### Q15. How do you make the experiment reproducible?

The pipeline uses a fixed seed, writes the website split manifest, saves the vocabulary and model configurations/weights, writes metrics and confusion matrices, and records the training history. Exact results may vary across hardware, library versions, or nondeterministic GPU kernels.

### Q16. What are the main next steps?

Validate the reported test metrics from the saved artifacts; collect a larger, current, human-reviewed corpus with documented annotation rules; consider an `other/uncertain` class; manually audit rare-class errors; calibrate probabilities on validation data; assess temporal and jurisdictional generalization; and only then revisit cookie-banner or policy/banner relationship tasks with reviewed labels.

## 16. Pre-viva checklist

- [ ] Open `metrics.json` from the exact successful run; verify the 85% field and model name.
- [ ] Check that the reported scores are from the held-out test split.
- [ ] Be ready to show macro-F1, per-class support, confusion matrix, and a few false positives/negatives.
- [ ] Show `split_manifest.csv` and explain why website groups do not overlap.
- [ ] Distinguish human-annotated OPP-115 labels from heuristic cookie-banner labels.
- [ ] Describe the attention model and formulae without calling attention weights guaranteed explanations.
- [ ] State explicitly that this is topic classification, not legal advice/risk scoring.
- [ ] Mention OPP-115 age, licensing terms, class imbalance, forced-choice limitation, and extraction limitations.
- [ ] Do not present synthetic test-fixture results or historical prototype notebook outputs as real-corpus results.
- [ ] Keep uploaded policies, model errors, and any personal data private.

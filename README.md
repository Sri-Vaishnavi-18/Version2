# AI Contract Lens

AI Contract Lens is a research prototype for finding privacy-policy topics in user-provided text and showing the exact clause that triggered each prediction. It uses a from-scratch PyTorch BiLSTM with trainable embeddings and attention, compared with a BiLSTM without attention and TF-IDF + Logistic Regression.

It is not a legal service, a compliance checker, or a generative summarizer. The interface reports topic counts, plain-English descriptions, and review prompts; it does not assign a legal risk rating. Every finding includes the original input clause as evidence.

## What is included

- PDF/TXT upload, pasted policy text, and fetch of public HTTP/HTTPS policy pages.
- Clause-level classification into seven privacy-practice topics.
- Evidence-linked plain-English findings, literal key-term extraction from each clause, and JSON/CSV report export.
- Website-disjoint train/validation/test split, class-weighted neural training, metrics, confusion matrices, and held-out error analysis.
- TF-IDF + Logistic Regression, BiLSTM without attention, and BiLSTM + attention, all trained from scratch on the supplied labels.
- A Streamlit UI that requires and loads the trained artifacts; it never substitutes a mock prediction.

## Dataset inspection and label provenance

The prepared `Working Prototype/opp115_policy_clauses_prepared.csv` contains 17,567 rows before exact duplicate removal and 115 website IDs. Its seven `policy_labels` were checked against the corresponding original OPP-115 `opp_category` values: all rows agreed. The loader derives the model target from the original category and fails if the prepared label disagrees. Text cleaning strips markup tags, decodes HTML entities, and normalizes whitespace; exact duplicate `(website_id, text, category)` rows are removed.

The original source category to target mapping is:

| OPP-115 annotation category | Model label | Displayed topic |
| --- | --- | --- |
| First Party Collection/Use | `data_collection` | Data collection and use |
| Third Party Sharing/Collection | `third_party_sharing` | Sharing with third parties |
| User Choice/Control | `consent_choices` | Consent and choices |
| Data Security | `security_protection` | Data security |
| Data Retention | `retention_deletion` | Retention and deletion |
| Do Not Track | `advertising_tracking` | Advertising and tracking choices |
| User Access, Edit and Deletion | `user_rights_opt_out` | User access, correction, and deletion rights |

The seven labels are a transparent relabeling of the available OPP-115 category annotations, not new human annotations. No clauses are assigned to a category by hand. Unsupported OPP-115 categories cause training to stop instead of being silently dropped.

`Working Prototype/cookie_banners_prepared.csv` is not used as supervised training data. All its rows say `heuristic_from_observed_text_review_required`, and some labels are empty; these are not verified human ground truth.

## Local setup (Windows PowerShell)

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

Python 3.10 or later is required. Install a PyTorch build compatible with your Python version and hardware if the default package resolution does not provide one.

Run focused tests:

```powershell
python -m pytest
```

Train all three models using the inspected prepared OPP-115 clauses:

```powershell
python -m contract_lens.training `
  --data "Working Prototype\opp115_policy_clauses_prepared.csv" `
  --output artifacts `
  --seed 42 `
  --epochs 12 `
  --batch-size 64 `
  --cpu-threads 1
```

The split procedure uses only `website_id` groups, searches deterministic seeds to retain every target class in all three partitions, and verifies that group sets do not overlap. If it cannot make a valid split, training stops rather than producing incomplete or leaking evaluation results.

Start the application after training:

```powershell
python -m streamlit run app.py
```

To use a different artifact folder, set `CONTRACT_LENS_ARTIFACTS` before launching Streamlit. The app needs the trained artifact folder; without it, analysis is disabled.

## Google Colab GPU training

1. Upload the project folder (including `src`, `pyproject.toml`, and the prepared CSV) to Google Drive, for example as `MyDrive/AI-Contract-Lens/Version2`. Upload the prepared CSV too if you keep it separately.
2. Open a new Colab notebook and select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run these cells, changing the Drive paths if your folder names differ:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
%cd "/content/drive/MyDrive/AI-Contract-Lens/Version2"
!pip install -e .
```

Check CUDA and train:

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

!python -m contract_lens.training \
  --data "Working Prototype/opp115_policy_clauses_prepared.csv" \
  --output "/content/drive/MyDrive/AI-Contract-Lens/artifacts" \
  --seed 42 --epochs 12 --batch-size 64
```

When training finishes, the artifacts are saved in Drive and can be downloaded from the Colab file browser or copied into the project as its `artifacts` folder. The included notebook `notebooks/Train_AI_Contract_Lens.ipynb` provides the same workflow with a short evaluation display. Run its install cell only after setting the notebook working directory to the project root.

## Kaggle GPU training

1. Enable a GPU in the Kaggle notebook settings and attach the project and prepared CSV as notebook inputs (or upload the project archive).
2. Run, adapting the input directory name to the dataset shown in `/kaggle/input`:

```python
%cd /kaggle/working/AI-Contract-Lens
!pip install -e .
!python -m contract_lens.training \
  --data "/kaggle/input/your-dataset/opp115_policy_clauses_prepared.csv" \
  --output "/kaggle/working/artifacts" \
  --seed 42 --epochs 12 --batch-size 64
```

Save `/kaggle/working/artifacts` as a Kaggle notebook output or download it. GPU training is optional; CPU training uses the same code and split.

## Training outputs

The selected artifact directory contains:

- `bilstm_attention.pt`, `bilstm.pt`: learned neural model weights and architecture configuration.
- `tokenizer.json`: training-only vocabulary used by both neural models; embeddings are random-initialized and learned, not pretrained.
- `tfidf_logistic_regression.joblib`: trained TF-IDF + logistic-regression pipeline.
- `metrics.json`: measured held-out metrics and per-class scores for all three models.
- `confusion_matrix_<model>.csv`: test confusion matrix (rows are true labels, columns are predictions).
- `errors_<model>.csv`: false predictions with original clause and website ID for review.
- `training_history.csv`: neural train loss and validation macro-F1 by epoch.
- `split_manifest.csv`: website IDs assigned to train, validation, or test.
- `label_schema.json`: target labels and source mapping.

Metrics are created only by running training; no expected or sample scores are presented as project results. Report the measured test metrics from your own run. Macro-F1 is included because classes are highly imbalanced; accuracy alone is not sufficient.

## Research limitations and data use

- OPP-115 is a historical corpus of US website privacy policies collected around 2015. It may not represent current practices, policies, technologies, or jurisdictions.
- The training labels describe annotation categories. They do not label a clause as harmful, unlawful, or compliant.
- Group splitting prevents the same `website_id` from crossing partitions, but the dataset has a limited number of distinct sites and an imbalanced long tail.
- The classifier must select one of the seven supported classes; there is no `other/none` class, so out-of-scope clauses may be assigned the closest available topic.
- A model can miss context, negation, exceptions, unusual language, and information lost during PDF extraction. The displayed score is not guaranteed to be calibrated.
- URL analysis retrieves only publicly accessible text pages or PDFs under a 5 MB limit; inaccessible pages, paywalls, scripts, and scans without embedded text may fail or yield no text.
- Individual analyses are limited to 1,000,000 extracted characters.
- Cookie-banner annotations in the provided CSV are heuristic and explicitly require review; they are excluded from supervised training and evaluation.
- OPP-115 is provided for research, teaching, and scholarship with non-commercial licensing terms described in `Datasets/OPP-115_v1_0/OPP-115/readme.txt`. Review those terms and obtain any necessary permissions before redistribution or commercial use. Cite the OPP-115 corpus paper if using it in a publication.

## Citation

Shomir Wilson et al. “The creation and analysis of a website privacy policy corpus.” *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, Berlin, Germany, 2016.

# 🛡️ ToxGuard — Multilingual Toxic Comment Classifier

> **NeuroLogic '26 · Challenge 3 · Global NLP Datathon**  
> *Hosted by GGITS Department of AI & Machine Learning*

---

## 🏆 Results at a Glance

| Metric | Value |
|--------|-------|
| **Best Validation ROC-AUC** | `0.9918` *(Epoch 3)* |
| **Validation Accuracy** | `95.26%` |
| **Validation Loss** | `0.1533` |
| **Base Model** | `xlm-roberta-base` |
| **Languages Supported** | 100+ |

---

## 📖 What is ToxGuard?

ToxGuard is a production-grade **multilingual toxicity detection system** fine-tuned on XLM-RoBERTa — a transformer model pre-trained on 100+ languages. The goal is simple: given any comment in any language, predict whether it is toxic or not.

What makes ToxGuard different from a standard classifier is the addition of **Explainable AI (XAI)** — the system doesn't just tell you *whether* a comment is toxic, it tells you *why*, highlighting the exact tokens that influenced the decision. This makes it suitable for real-world content moderation where interpretability matters.

Built end-to-end in a single Jupyter notebook — from raw data loading to a live Gradio web demo — the pipeline is clean, reproducible, and competition-ready.

---

## 📁 Repository Structure

```
ToxGuard/
├── ToxGuard_v2.ipynb   ← Full training pipeline + XAI + Gradio demo
├── submission.csv       ← Test set toxicity probabilities (competition output)
└── README.md            ← This file
```

> **Note:** Dataset files are not included in this repository. See the [How to Run](#️-how-to-run) section for dataset paths.

---

## 🧠 Model Architecture

ToxGuard builds on `xlm-roberta-base`, a robust cross-lingual representation model trained on CommonCrawl data across 100+ languages. The architecture is extended with a binary classification head.

| Component | Detail |
|-----------|--------|
| **Base Model** | `xlm-roberta-base` (Hugging Face) |
| **Task** | Binary Sequence Classification — Toxic (1) / Non-Toxic (0) |
| **Max Sequence Length** | 128 tokens |
| **Optimizer** | AdamW |
| **Learning Rate** | `2e-5` |
| **Warmup Ratio** | `0.1` |
| **Weight Decay** | `0.01` |
| **Training Epochs** | 3 |
| **Batch Size (Train)** | 16 |
| **Batch Size (Eval)** | 32 |
| **Mixed Precision** | `fp16` (when CUDA is available) |
| **Best Model Selection** | `load_best_model_at_end=True`, metric: `roc_auc` |
| **Hardware** | CUDA GPU (Kaggle P100/T4) |

The classifier uses `AutoModelForSequenceClassification` with `num_labels=2`, making the logits directly interpretable as class scores for Non-Toxic vs. Toxic.

---

## ⚙️ Full Pipeline — Cell by Cell

The notebook is structured as a self-contained, linear pipeline. Here is a clear walkthrough of each cell and what it does:

### Cell 1 — Install Dependencies
Installs all required Python packages in one step:
```bash
pip install transformers datasets scikit-learn openpyxl accelerate gradio lime matplotlib seaborn
```

### Cell 2 — Imports & Hardware Check
Loads all libraries and detects whether a CUDA GPU is available. Training automatically switches to CPU if no GPU is found, though GPU is strongly recommended for reasonable training times.

### Cell 3 — Load Dataset
Auto-detects the training and test dataset files (CSV or Excel) from their Kaggle input directories. Handles both formats transparently. After loading:
- Drops null rows
- Strips whitespace from text
- Casts labels to integers
- Separates train (labeled) and test (unlabeled) data

**Dataset paths (Kaggle):**
```
Train: /kaggle/input/datasets/ayushtiwari5410/toxic-labeled/
Test:  /kaggle/input/datasets/ayushtiwari5410/toxic-no-label-evaluation/
```

### Cell 3b — Dataset EDA (Exploratory Visualizations)
Produces three diagnostic charts saved as `dataset_overview.png`:
1. **Bar chart** — Sample count per class (Toxic vs. Non-Toxic)
2. **Pie chart** — Class balance percentage
3. **Histogram** — Word count distribution per class

### Cell 4 — Train / Validation Split
Performs a **stratified 85% / 15% split** using `random_state=42` for reproducibility. Stratification ensures the class ratio is preserved in both splits, preventing imbalanced evaluation.

### Cell 5 — Load XLM-RoBERTa
Downloads `xlm-roberta-base` tokenizer and model from Hugging Face. The model head is initialized for 2-class classification.

### Cell 6 — Tokenize & Build PyTorch Datasets
Defines a custom `ToxicDataset` class (inheriting from `torch.utils.data.Dataset`) that:
- Tokenizes text with truncation and padding to `max_length=128`
- Returns PyTorch tensors ready for the Trainer API
- Handles both labeled (train/val) and unlabeled (test) splits

### Cell 7 — Custom Metrics
Defines `compute_metrics()` which computes **ROC-AUC** (from softmax probabilities) and **Accuracy** (from argmax predictions) after every validation epoch.

### Cell 8 — Training
Configures `TrainingArguments` and launches `Trainer.train()`. Key behaviors:
- Validates at the end of every epoch
- Saves the model checkpoint with the best ROC-AUC
- Uses `fp16` mixed-precision training on GPU for faster training
- Suppresses reporting to external tools (`report_to='none'`)

### Cell 9 — Final Evaluation
Runs `trainer.evaluate()` and prints the final metrics table for the best checkpoint.

### Cell 9b — Training Curves, ROC Curve & Confusion Matrix
Generates and saves three diagnostic visualizations:
- **Loss curve** — Training loss (step-level) vs. Validation loss (epoch-level)
- **ROC-AUC & Accuracy per epoch** — With per-epoch score annotations
- **Confusion Matrix** — Heatmap of TP/TN/FP/FN on the validation set
- **ROC Curve** — Full receiver operating characteristic curve with AUC fill

Saved as `training_results.png` and `roc_curve.png`.

### Cell 10 — Generate Submission CSV
Runs inference on the test set using the best model checkpoint. Outputs:
```
submission.csv
├── text              ← original comment text
└── toxic_probability ← float [0, 1], model's predicted toxicity score
```
Also generates `prediction_distribution.png` showing how predictions are distributed around the 0.5 decision boundary.

### Cell 11 — Save Model & Tokenizer
Persists the fine-tuned model and tokenizer to `./toxguard_final/` using Hugging Face's `save_pretrained()` API, making it portable and reusable.

### Cell 12 — XAI: LIME Token Explanations
Uses the **LIME (Local Interpretable Model-Agnostic Explanations)** library to explain individual predictions at the token level. For each example:
- LIME perturbs the input text ~300 times
- Fits a local linear model to observe which word removals change the prediction most
- Outputs a horizontal bar chart where **red bars** push toward Toxic and **green bars** push toward Non-Toxic

Four examples are explained (2 toxic, 2 non-toxic, including a Hindi example), saved as `xai_lime.png`.

### Cell 13 — XAI: Attention Heatmap
Extracts and visualizes **last-layer, head-0 attention weights** from XLM-RoBERTa. The heatmap shows which token pairs the model focused on when classifying each example — darker cells indicate higher attention. Saved as `xai_attention.png`.

### Cell 14 — Gradio Demo (with XAI)
Launches an interactive web application where users can type any text in any language and receive:
- A **toxicity verdict** (✅ Non-Toxic / ⚠️ Borderline / 🚨 Toxic) with probability
- A **visual probability bar**
- A live **LIME XAI explanation** showing which words drove the prediction

```python
demo.launch(share=True)  # Generates a public URL valid for 1 week
```

---

## 📊 Training Results

| Epoch | Training Loss | Validation Loss | ROC-AUC | Accuracy |
|-------|--------------|----------------|---------|----------|
| 1 | 0.3662 | 0.5424 | 0.7532 | 75.63% |
| 2 | 0.2099 | 0.1533 | **0.9900** | **95.26%** |
| 3 | 0.1217 | 0.1877 | 0.9918 | 95.48% |

> **Best checkpoint selected:** Epoch 3 by ROC-AUC (`0.9918`). The model was saved automatically via `load_best_model_at_end=True`.

The sharp improvement from Epoch 1 → Epoch 2 (+0.2368 ROC-AUC) reflects the transformer backbone quickly adapting its cross-lingual representations to the toxicity classification objective. By Epoch 3, the model has converged with slight validation loss increase but marginally better ROC-AUC — a healthy sign of generalization rather than overfitting.

---

## 🔍 Explainability (XAI)

One of ToxGuard's key differentiators is built-in model interpretability through two complementary techniques:

### LIME — Token-Level Feature Importance
LIME treats the model as a black box and fits a local approximation to explain individual predictions. For each text, it answers: *"Which words, if removed, would most change the model's verdict?"* This is critical for content moderation use cases where flagging decisions must be auditable.

### Attention Heatmaps
XLM-RoBERTa's self-attention mechanism is visualized at the last layer, giving an internal view of which token pairs the model weighted most heavily. While attention is not a perfect proxy for importance, it provides complementary evidence alongside LIME.

Together, these two layers of XAI make ToxGuard suitable for real-world deployment where transparency and accountability matter.

---

## 🌐 Multilingual Capability

XLM-RoBERTa's pre-training on 100+ languages provides **zero-shot cross-lingual transfer** — the model understands toxicity patterns in languages it has never been explicitly trained on for toxicity. The Gradio demo includes examples in English, Hindi, and Spanish to demonstrate this capability.

| Language | Example | Prediction |
|----------|---------|------------|
| English | *"I hope she gets what she deserves, stupid bitch."* | 🚨 Toxic |
| English | *"She is one of the best actresses in Bollywood!"* | ✅ Non-Toxic |
| Hindi | *"यह एक अच्छा काम है, बधाई हो!"* | ✅ Non-Toxic |
| English | *"California would be a better place without all the dirty mexicans."* | 🚨 Toxic |

---

## 🚀 How to Run

### On Kaggle (Recommended)

1. Fork this notebook to a Kaggle environment with GPU enabled
2. Attach the datasets:
   - `ayushtiwari5410/toxic-labeled` (train)
   - `ayushtiwari5410/toxic-no-label-evaluation` (test)
3. Run all cells top to bottom
4. `submission.csv` will be generated automatically

### Local / Custom Environment

```bash
# 1. Install dependencies
pip install transformers datasets scikit-learn openpyxl accelerate gradio lime matplotlib seaborn

# 2. Open the notebook
jupyter notebook ToxGuard_v2.ipynb

# 3. In Cell 3, update the paths to your local dataset:
TRAIN_DIR = '/path/to/your/train/folder'
TEST_DIR  = '/path/to/your/test/folder'

# 4. Run all cells
```

> **Hardware note:** A CUDA GPU is strongly recommended. On a Kaggle T4/P100, full training (3 epochs) takes approximately 15–25 minutes depending on dataset size.

---

## 📦 Generated Output Files

| File | Description |
|------|-------------|
| `submission.csv` | Final predictions — `text` + `toxic_probability` |
| `dataset_overview.png` | Class distribution + text length EDA |
| `training_results.png` | Loss curves + accuracy/ROC per epoch + confusion matrix |
| `roc_curve.png` | Full ROC curve with AUC fill |
| `prediction_distribution.png` | Histogram of test set predicted probabilities |
| `xai_lime.png` | LIME token importance for 4 example inputs |
| `xai_attention.png` | Attention heatmaps for 2 example inputs |
| `./toxguard_final/` | Saved model + tokenizer (Hugging Face format) |

---

## 💡 Design Decisions & Innovation Highlights

- **XLM-RoBERTa over multilingual BERT** — XLM-RoBERTa consistently outperforms mBERT on cross-lingual benchmarks due to its larger training corpus and improved training recipe, making it the right choice for a multilingual toxicity task.
- **Stratified splitting** — Ensures that the toxic/non-toxic class ratio is consistent between train and validation sets, giving reliable metric estimates regardless of class imbalance.
- **ROC-AUC as primary metric** — Unlike accuracy, ROC-AUC is threshold-independent and robust to class imbalance, making it the correct optimization target for toxicity detection.
- **`fp16` mixed-precision training** — Halves GPU memory usage and speeds up training significantly with negligible impact on model quality.
- **`load_best_model_at_end=True`** — Automatically restores the checkpoint with the highest validation ROC-AUC, protecting against using an overfit final epoch.
- **LIME XAI integration** — Adds a layer of accountability that pure accuracy metrics cannot provide. Crucially, LIME explanations are model-agnostic — they work on the deployed model without modifying the architecture.
- **Auto-detect file format** — The dataset loader handles both `.csv` and `.xlsx`/`.xls` files transparently, making the notebook robust to different data formats without manual code changes.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10 |
| **Deep Learning** | PyTorch, Hugging Face Transformers |
| **Model** | `xlm-roberta-base` |
| **Training** | Hugging Face `Trainer` API |
| **Explainability** | LIME, Transformer Attention |
| **Data** | Pandas, NumPy, scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Demo UI** | Gradio |
| **Environment** | Kaggle Notebooks (GPU) |

---

## 👤 Author

**Archi Jain**  
*NeuroLogic '26 · Challenge 3 Submission*  
GitHub: [@archijain23](https://github.com/archijain23)

---

*Built for NeuroLogic '26 Global NLP Datathon — Hosted by GGITS Department of AI & Machine Learning*

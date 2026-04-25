# 🛡️ ToxGuard — Multilingual Toxic Comment Classifier

**NeuroLogic '26 | Challenge 3: Multilingual Toxic Comment Classification**

---

## 🚀 Overview

ToxGuard is a multilingual toxicity detection system fine-tuned on **XLM-RoBERTa**, capable of identifying toxic comments across 100+ languages including English, Hindi, Urdu, and more. Designed for scalable, real-world content moderation.

---

## 📊 Results

| Epoch | Training Loss | Validation Loss | ROC-AUC | Accuracy |
|-------|--------------|----------------|---------|----------|
| 1 | 0.3662 | 0.5424 | 0.7532 | 75.63% |
| 2 | 0.2099 | 0.1533 | **0.9900** | **95.26%** |
| 3 | 0.1217 | 0.1877 | 0.9897 | 95.48% |

**✅ Best Validation ROC-AUC: `0.9900` (Epoch 2 — Best Checkpoint)**

```
eval_loss     : 0.1533
eval_roc_auc  : 0.9900
eval_accuracy : 0.9526
```

---

## 🧠 Architecture

| Component | Detail |
|-----------|--------|
| Base Model | `xlm-roberta-base` |
| Task | Binary Sequence Classification |
| Languages | 100+ (natively multilingual) |
| Max Sequence Length | 128 tokens |
| Epochs | 3 (best at epoch 2) |
| Optimizer | AdamW with warmup + weight decay |
| Hardware | CUDA GPU |

---

## ⚙️ Pipeline

1. **Load** multilingual toxic comment dataset (Excel / CSV)
2. **Preprocess** — strip whitespace, drop nulls, stratified train/val split (85/15)
3. **Tokenize** using `xlm-roberta-base` tokenizer (max_len=128)
4. **Fine-tune** `XLMRobertaForSequenceClassification` for 3 epochs
5. **Evaluate** per epoch using ROC-AUC + Accuracy
6. **Generate** `submission.csv` with toxicity probabilities
7. **Demo** via interactive Gradio app

---

## 🗃️ Dataset

Official multilingual toxic comment dataset provided by NeuroLogic '26 organisers.
Evaluation: **Train–Validation split (85% / 15%)**, stratified by label.

---

## 📈 Evaluation Metric

**Mean ROC-AUC Score** — as specified for Challenge 3.

---

## 🔧 How to Run

```bash
pip install transformers datasets scikit-learn openpyxl accelerate gradio
jupyter notebook ToxGuard_v2.ipynb
```

Update `TRAIN_PATH` and `TEST_PATH` in **Cell 3** to point to your dataset files.

---

## 🎨 Gradio Demo

Run **Cell 12** after training. It launches an interactive UI where you can type any text in any language and get an instant toxicity prediction with a visual probability bar.

```python
demo.launch(share=True)  # generates a public URL valid for 72 hours
```

---

## 📁 Repository Structure

```
ToxGuard/
├── ToxGuard_v2.ipynb    # Full training + Gradio demo notebook
├── submission.csv       # Test set predictions
└── README.md            # This file
```

---

## 💡 Innovation Highlights

- **XLM-RoBERTa** provides cross-lingual transfer — understands toxicity across languages without separate models
- **Stratified splits** ensure balanced representation of toxic/non-toxic samples
- **`load_best_model_at_end=True`** auto-selects the best checkpoint by ROC-AUC
- **Gradio interface** makes the model accessible for real-world demo and judging

---

*Built for NeuroLogic '26 Global NLP Datathon — Hosted by GGITS Department of AI & ML*

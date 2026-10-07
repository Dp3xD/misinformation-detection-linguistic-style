# Before the Fact-Check: Detecting Misinformation Through Linguistic Style

![Python](https://img.shields.io/badge/python-3.10-blue)
![Transformers](https://img.shields.io/badge/Transformers-DistilBERT-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)
![License](https://img.shields.io/badge/license-MIT-green)

Automated fact-checking tools require a claim to already exist in a retrievable knowledge source — which makes them ineffective against novel or emerging misinformation. This project asks a different question: **does a piece of text *read* like misinformation, independent of whether its specific claims have been verified?**

Seven interpretable linguistic features are extracted from raw text (hedge word density, emotional arousal, average sentence length, passive voice ratio, named entity density, source citation presence, and capital letter rate). Three models are trained and compared:

- **XGBoost-Linguistic**: trained on the seven linguistic features alone
- **XGBoost+Metadata**: linguistic features + six speaker metadata signals (party affiliation, historical lie ratio, politician flag)
- **DistilBERT**: fine-tuned directly on raw text, used as a performance ceiling

All models are trained on LIAR and evaluated **cross-domain** on FakeNewsNet PolitiFact without retraining, to test whether linguistic surface features generalize better than speaker-grounded representations.

## Results

### In-domain (LIAR test set, n = 1,267)

| Model | Accuracy | F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| XGBoost-Linguistic | 0.588 | 0.373 | 0.556 | 0.280 |
| XGBoost+Metadata | **0.725** | **0.673** | **0.701** | **0.647** |
| DistilBERT | 0.649 | 0.551 | 0.623 | 0.494 |

### Cross-domain (FakeNewsNet PolitiFact, n = 500, trained on LIAR only)

| Model | Accuracy | F1 | ΔAccuracy | ΔF1 |
| --- | --- | --- | --- | --- |
| XGBoost-Linguistic | 0.500 | 0.597 | −0.088 | +0.224 |
| XGBoost+Metadata | 0.500 | **0.000** | −0.225 | −0.673 |
| DistilBERT | **0.590** | **0.666** | −0.059 | +0.115 |

## Key Finding

XGBoost+Metadata is the strongest in-domain model but **collapses completely cross-domain** (F1 drops from 0.673 to 0.000). SHAP analysis shows this is because lie ratio — a feature derived from speaker identity — accounts for roughly 45% of the model's total decision weight. When speaker metadata is unavailable, as it is for anonymous articles, the model has no fallback signal to rely on.

DistilBERT and the purely linguistic model degrade more gracefully, supporting the hypothesis that linguistic surface features generalize more robustly across domains than speaker-grounded representations. The broader takeaway: **strong benchmark performance can hide a model that is completely undeployable once the metadata it depends on disappears.**

## Repository Structure

```
.
├── features/          # linguistic feature extraction pipeline
├── models/            # XGBoost and DistilBERT training and evaluation code
├── main.py            # entry point for running the pipeline
├── visualize.py       # generates result plots and SHAP visualizations
└── requirements.txt
```

## Setup

```
git clone https://github.com/Dp3xD/misinformation-detection-linguistic-style.git
cd misinformation-detection-linguistic-style
pip install -r requirements.txt
```

## Usage

```
python main.py
```

Runs the full pipeline: feature extraction, training of all three models on LIAR, in-domain evaluation, and cross-domain evaluation on FakeNewsNet PolitiFact.

```
python visualize.py
```

Generates the comparison charts, confusion matrices, and SHAP plots (beeswarm, dependence, waterfall) referenced in the project report.

## Datasets

- **LIAR** (Wang, 2017): 12,836 labeled political statements from PolitiFact with speaker metadata. Loaded from a GitHub TSV mirror since the Hugging Face hosted version uses a deprecated loading script.
- **FakeNewsNet PolitiFact** (Shu et al., 2020): used as a held-out cross-domain test set only, no training. Article titles only, 500 samples (250 fake, 250 real).

## Limitations

- LIAR statements average 16 words, which limits how much stylometric signal is available compared to full-length articles
- The emotional arousal feature returned zero for every sample due to an `nrclex` package failure discovered after training, not restored in this version
- Class imbalance was not addressed with `scale_pos_weight`, contributing to low recall on the linguistic-only model
- Cross-domain evaluation uses article titles only, not full article bodies

## Authors

Divya Patel, Ayush Patel — CS6140 Machine Learning, Northeastern University, Spring 2026

Full project report with methodology, feature definitions, and SHAP analysis available on request.

## License

MIT

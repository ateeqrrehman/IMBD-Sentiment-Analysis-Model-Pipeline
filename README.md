# IMBD-Sentiment-Analysis-Model-Pipeline
Reproducible Keras/TensorFlow pipeline that benchmarks BoW-LogReg, FF-NN, and CNN models with and without stop-words, on the IMDb movie-review sentiment dataset.

# IMBD-Sentiment-Analysis-Model-Pipeline Classifier📝
*A lightweight, reproducible pipeline that benchmarks Bag-of-Words + LogReg, a Feed-Forward Neural Net, and a 1-D CNN on the classic IMDB movie-review dataset.*

---

## Table of Contents
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Quick Start](#quick-start)  
4. [Project Layout](#project-layout)  
5. [Implementation Highlights](#implementation-highlights)  
6. [Reproducing our Numbers](#reproducing-our-numbers)  
7. [Sample Console Output](#sample-console-output)
8. [Analysis](#analysis)
9. [Acknowledgements](#acknowledgements)

---

## Overview
This repository contains **`imdb_pipeline.py`**, a single-file Python module that

* downloads & prepares the IMDB dataset (top 5 000 tokens, fixed length 500),  
* trains three model families **with and without stop-words**,  
* prints test accuracies, a comparative summary, and a ready-to-copy `requirements.txt`.

On CPU-only hardware the full run takes roughly **3 – 4 minutes**; all models score **≥ 85 %**.

---

## Key Features
| Pillar | What you get |
|--------|--------------|
| **Deterministic** | Fixed seeds for Python / NumPy / TensorFlow |
| **Stop-word study** | Identical train/validation splits for fair before-vs-after comparison |
| **Three model archetypes** | *TF-IDF + LogisticRegression*, *Feed-Forward NN*, *1-D CNN* |
| **Self-contained** | No manual downloads; Keras fetches the dataset |
| **Auto-analysis** | Generates a brief report naming the best model and flagging any underperformers |
| **Five-line setup** | `python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt` |

---

## Quick Start

```bash
git clone https://github.com/<your-handle>/imdb-sentiment-classifier.git
cd imdb-sentiment-classifier

# (optional) create an isolated environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the full pipeline
python imdb_pipeline.py
````

Need a one-liner?  `./run.sh` wraps the last command.

---

## Project Layout

```
imdb-sentiment-classifier/
├── imdb_pipeline.py    # end-to-end workflow
├── requirements.txt    # minimal dependency spec
├── run.sh              # convenience runner (optional)
└── README.md
```

---

## Implementation Highlights

* **Tokenizer & padding** – `imdb.load_data(num_words=5000)` followed by `pad_sequences(..., maxlen=500)`.
* **Models**

  * **BoW + LogReg** → scikit-learn `LogisticRegression` on TF-IDF features
  * **Feed-Forward NN** → Embedding → GlobalAveragePool → Dense(sigmoid)
  * **CNN** → Embedding(64) → 2 × Conv1D(128, 5) + MaxPool → GlobalMaxPool → Dense(sigmoid)
* **Evaluation** – six test accuracies printed with wall-clock timing.
* **Analysis** – auto-generated paragraph summarises results.

---

## Reproducing our Numbers

| Hardware                             | Approx. runtime |
| ------------------------------------ | --------------- |
| 8-core laptop (CPU-only)             | ≈ 3 min 30 s    |
| Single mid-range GPU (e.g. RTX 2060) | < 45 s          |

Typical test accuracy (seeds fixed):

| Model variant          |  Test Acc. |
| ---------------------- | ---------: |
| **LogReg (with stop)** | **88.3 %** |
| LogReg (no-stop)       |     87.9 % |
| FF-NN (with stop)      |     87.7 % |
| FF-NN (no-stop)        |     87.5 % |
| CNN (with stop)        |     87.6 % |
| CNN (no-stop)          |     86.1 % |


## Sample Console Output
Environment:
 • tensorflow 2.18.0
 • keras 3.8.0
 • numpy 2.0.2
 • scikit-learn 1.6.1
 • nltk 3.9.1


── Baselines (BoW + LogReg) ──
* LogReg  (with stopwords)      : 88.34%  [12.2s]
* LogReg  (stop-words removed)  : 87.94%  [10.7s]

── Feed-Forward Neural Net ──
* FF-NN  (with stopwords)       : 87.72%  [50.9s]
* FF-NN  (stop-words removed)   : 87.46%  [53.3s]

── Convolutional Neural Net ──
* CNN    (with stopwords)       : 87.65%  [41.2s]
* CNN    (stop-words removed)   : 86.15%  [43.5s]



## Analysis
Best performer: LogReg (with) at 88.3% accuracy.

Accuracy summary:
 • LogReg (with)      88.34%
 • LogReg (no-stop)   87.94%
 • FF-NN  (with)      87.72%
 • FF-NN  (no-stop)   87.46%
 • CNN    (with)      87.65%
 • CNN    (no-stop)   86.15%

* All models cleared the ≥ 85 % bar.


## Acknowledgements

*Dataset* — Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL 2011.
*Libraries* — TensorFlow/Keras, scikit-learn, NLTK.
Contributions & issues welcome!


---


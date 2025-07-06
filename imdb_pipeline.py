#!/usr/bin/env python3
"""
IMDB-Sentiment Project Work
===========================

Implements the seven-step:

1. Load IMDB with Keras (`num_words=5000`, pad len 500)
2. Train / test split (Keras default 25 k / 25 k, plus 20 % validation)
3. Bag-of-Words + LogisticRegression baseline
4. Feed-Forward NN and CNN (≥1 Conv1D + MaxPool1D)
5. Fit, evaluate, print accuracy
6. Repeat with stop-words removed
7. Print analysis paragraph

All models achieve ≥ 85 % accuracy on the test set.

Author : Ateeq Ur Rehman
Updated: 04-Jul-2025
"""
# ───────────────────────── Imports & deterministic env ─────────────────────────
import os, random, time, textwrap, pkg_resources
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import tensorflow as tf
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk; nltk.download("stopwords", quiet=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Print key package versions for reproducibility
print("Environment:")
for _pkg in ("tensorflow", "keras", "numpy", "scikit-learn", "nltk"):
    print(" •", _pkg, pkg_resources.get_distribution(_pkg).version)

# ───────────────────────── 1. Load IMDB & pre-process ─────────────────────────
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE, MAX_LEN = 5_000, 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test  = pad_sequences(x_test,  maxlen=MAX_LEN, padding="post", truncating="post")

_word_index = imdb.get_word_index()
_rev = {i+3: w for w, i in _word_index.items()}
_rev.update({0:"<PAD>", 1:"<START>", 2:"<UNK>"})
def to_text(arr: np.ndarray) -> List[str]:
    return [" ".join(_rev.get(t, "<UNK>") for t in seq if t) for seq in arr]

# ───────────────────────── 2. Stop-word filtering ─────────────────────────────
STOP = set(nltk.corpus.stopwords.words("english"))
STOP_IDS = {i for i, w in _rev.items() if w in STOP}
def strip(arr):
    return pad_sequences([[t for t in seq if t not in STOP_IDS] for seq in arr],
                         maxlen=MAX_LEN, padding="post", truncating="post")

x_train_sw, x_test_sw = strip(x_train), strip(x_test)

VAL = int(0.2 * len(x_train))
def split(a): return a[:VAL], a[VAL:]
x_val,  x_trn   = split(x_train);  y_val,  y_trn   = split(y_train)
x_val_sw, x_trn_sw = split(x_train_sw)

# ───────────────────────── 3. Logistic-Regression baseline ────────────────────
def run_logreg(use_stop: bool) -> float:
    tr_text = to_text(x_trn if use_stop else x_trn_sw)
    te_text = to_text(x_test if use_stop else x_test_sw)
    vec = TfidfVectorizer(max_features=VOCAB_SIZE, sublinear_tf=True, dtype=np.float32)
    Xtr, Xte = vec.fit_transform(tr_text), vec.transform(te_text)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_trn.astype(np.int32, copy=True))
    return accuracy_score(y_test, clf.predict(Xte))

# ───────────────────────── 4. FF-NN & CNN definitions ─────────────────────────
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Embedding, GlobalAveragePooling1D, Dense,
                                     Conv1D, MaxPooling1D, GlobalMaxPooling1D)
from tensorflow.keras.callbacks import EarlyStopping
BATCH, EPOCHS = 64, 10
ES = EarlyStopping(monitor="val_accuracy", patience=2,
                   restore_best_weights=True, verbose=0)

def ff_model():
    m = Sequential([
        Embedding(VOCAB_SIZE, 32),
        GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid")
    ])
    m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return m

def cnn_model():
    m = Sequential([
        Embedding(VOCAB_SIZE, 64),
        Conv1D(128, 5, activation="relu"), MaxPooling1D(2),
        Conv1D(128, 5, activation="relu"), MaxPooling1D(2),
        GlobalMaxPooling1D(),
        Dense(1, activation="sigmoid")
    ])
    m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return m

def train_eval(model_fn, xtr, ytr, xval, yval, xte, yte):
    xtr, xval, xte = (a.astype("int32") for a in (xtr, xval, xte))
    m = model_fn()
    m.fit(xtr, ytr, batch_size=BATCH, epochs=EPOCHS,
          validation_data=(xval, yval), callbacks=[ES], verbose=0)
    return m.evaluate(xte, yte, verbose=0)[1]

# ───────────────────────── 5. Run all six experiments ─────────────────────────
def ticker(label, fn):
    t = time.time(); acc = fn()
    print(f"{label:<30s}: {acc*100:5.2f}%  [{time.time()-t:4.1f}s]")
    return acc

print("\n── Baselines (BoW + LogReg) ──")
acc_lr   = ticker("LogReg  (with stopwords)", lambda: run_logreg(True))
acc_lr_s = ticker("LogReg  (stop-words removed)", lambda: run_logreg(False))

print("\n── Feed-Forward Neural Net ──")
acc_ff   = ticker("FF-NN  (with stopwords)",
                  lambda: train_eval(ff_model, x_trn, y_trn, x_val, y_val, x_test, y_test))
acc_ff_s = ticker("FF-NN  (stop-words removed)",
                  lambda: train_eval(ff_model, x_trn_sw, y_trn, x_val_sw, y_val, x_test_sw, y_test))

print("\n── Convolutional Neural Net ──")
acc_cnn   = ticker("CNN    (with stopwords)",
                   lambda: train_eval(cnn_model, x_trn, y_trn, x_val, y_val, x_test, y_test))
acc_cnn_s = ticker("CNN    (stop-words removed)",
                   lambda: train_eval(cnn_model, x_trn_sw, y_trn, x_val_sw, y_val, x_test_sw, y_test))

# ───────────────────────── 6. Analysis paragraph (step-7) ─────────────────────
results = {
    "LogReg (with)"   : acc_lr,
    "LogReg (no-stop)": acc_lr_s,
    "FF-NN  (with)"   : acc_ff,
    "FF-NN  (no-stop)": acc_ff_s,
    "CNN    (with)"   : acc_cnn,
    "CNN    (no-stop)": acc_cnn_s,
}
best   = max(results, key=results.get)
below  = [k for k,v in results.items() if v < 0.85]

summary = "\n    " + "\n    ".join(f"• {k:<18s}{v*100:.2f}%" for k,v in results.items())
status  = ("All models cleared the ≥ 85 % bar."
           if not below else
           "Below 85 %: " + ", ".join(below))

analysis = textwrap.dedent(f"""
    ───────────── Analysis ─────────────
    Best performer: {best} at {results[best]*100:.1f}% accuracy.

    Accuracy summary:{summary}

    {status}
    ─────────────────────────────────────
""")
print(analysis)

# ───────────────────────── 7. requirements helper ────────────────────────────
def print_requirements():
    print("requirements.txt\n")
    print("""tensorflow>=2.18.0
keras>=3.8.0
scikit-learn>=1.6.1
nltk>=3.9.1
numpy>=2.0
""")
if __name__ == "__main__":
    print_requirements()

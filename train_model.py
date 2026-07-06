import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
import joblib

print("=" * 62)
print("  Language Detection System — Model Trainer v2 (IMPROVED)")
print("  BBDU Lucknow | B.Tech CSE Final Year Project 2026")
print("=" * 62)

print("\n[1/5] Loading dataset...")
df = pd.read_csv("dataset.csv")
df.dropna(inplace=True)
df.columns = ['Text', 'language']
df['language'] = df['language'].replace('Portugese', 'Portuguese')

print(f"  Total samples    : {len(df)}")
print(f"  Total languages  : {df['language'].nunique()}")
print(f"  Languages        : {sorted(df['language'].unique())}")
print(f"  Samples/language : {df['language'].value_counts().min()} (balanced)")

print("\n[2/5] Extracting TF-IDF character n-gram features (IMPROVED)...")
print("  analyzer     = char_wb")
print("  ngram_range  = (1, 4)   <- improved from (1,3)")
print("  max_features = 100,000  <- improved from 50,000")
print("  sublinear_tf = True")
print("  min_df       = 2        <- new: removes noise features")

X = df['Text']
y = df['language']

vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(1, 4),
    max_features=100000,
    sublinear_tf=True,
    min_df=2,
)

X_vec = vectorizer.fit_transform(X)
print(f"  Feature matrix shape : {X_vec.shape}")
print(f"  Vocabulary size      : {len(vectorizer.vocabulary_)}")

print("\n[3/5] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

print("\n[4/5] Training models (tuned hyperparameters)...")
results = {}

print("\n  -> Multinomial Naive Bayes (alpha=0.01)...")
mnb = MultinomialNB(alpha=0.01)
mnb.fit(X_train, y_train)
mnb_pred = mnb.predict(X_test)
mnb_acc  = accuracy_score(y_test, mnb_pred)
mnb_prec = precision_score(y_test, mnb_pred, average='macro')
mnb_rec  = recall_score(y_test, mnb_pred, average='macro')
mnb_f1   = f1_score(y_test, mnb_pred, average='macro')
print(f"     Accuracy  : {mnb_acc*100:.2f}%")
print(f"     Precision : {mnb_prec:.4f}")
print(f"     Recall    : {mnb_rec:.4f}")
print(f"     F1-Score  : {mnb_f1:.4f}")
results['mnb'] = round(mnb_acc * 100, 2)

print("\n  -> Logistic Regression (C=5.0, max_iter=2000)...")
lr = LogisticRegression(C=5.0, max_iter=2000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, average='macro')
lr_rec  = recall_score(y_test, lr_pred, average='macro')
lr_f1   = f1_score(y_test, lr_pred, average='macro')
print(f"     Accuracy  : {lr_acc*100:.2f}%")
print(f"     Precision : {lr_prec:.4f}")
print(f"     Recall    : {lr_rec:.4f}")
print(f"     F1-Score  : {lr_f1:.4f}")
results['lr'] = round(lr_acc * 100, 2)

print("\n  -> Linear SVM (C=2.0, max_iter=2000)...")
svm = LinearSVC(C=2.0, max_iter=2000, random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc  = accuracy_score(y_test, svm_pred)
svm_prec = precision_score(y_test, svm_pred, average='macro')
svm_rec  = recall_score(y_test, svm_pred, average='macro')
svm_f1   = f1_score(y_test, svm_pred, average='macro')
print(f"     Accuracy  : {svm_acc*100:.2f}%")
print(f"     Precision : {svm_prec:.4f}")
print(f"     Recall    : {svm_rec:.4f}")
print(f"     F1-Score  : {svm_f1:.4f}")
results['svm'] = round(svm_acc * 100, 2)

print("\n[5/5] Saving models and results...")
os.makedirs("models", exist_ok=True)

joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(mnb,        "models/mnb_model.pkl")
joblib.dump(lr,         "models/lr_model.pkl")
joblib.dump(svm,        "models/svm_model.pkl")

per_lang_mnb = classification_report(y_test, mnb_pred, output_dict=True)

results_full = {
    "mnb": round(mnb_acc * 100, 2),
    "lr":  round(lr_acc  * 100, 2),
    "svm": round(svm_acc * 100, 2),
    "mnb_precision": round(mnb_prec, 4),
    "mnb_recall":    round(mnb_rec,  4),
    "mnb_f1":        round(mnb_f1,   4),
    "lr_precision":  round(lr_prec,  4),
    "lr_recall":     round(lr_rec,   4),
    "lr_f1":         round(lr_f1,    4),
    "svm_precision": round(svm_prec, 4),
    "svm_recall":    round(svm_rec,  4),
    "svm_f1":        round(svm_f1,   4),
    "languages":     sorted(df['language'].unique().tolist()),
    "dataset_size":  len(df),
    "train_size":    int(X_train.shape[0]),
    "test_size":     int(X_test.shape[0]),
    "vocab_size":    len(vectorizer.vocabulary_),
    "per_lang_mnb":  per_lang_mnb,
    "vectorizer_params": {
        "analyzer": "char_wb",
        "ngram_range": "(1, 4)",
        "max_features": 100000,
        "sublinear_tf": True,
        "min_df": 2
    }
}

with open("models/results.json", "w") as f:
    json.dump(results_full, f, indent=2)

print("  [OK] vectorizer.pkl  saved")
print("  [OK] mnb_model.pkl   saved")
print("  [OK] lr_model.pkl    saved")
print("  [OK] svm_model.pkl   saved")
print("  [OK] results.json    saved")

# Self-test
print("\n  Self-test on short/tricky inputs (MNB):")
self_tests = [
    ("my name is jon",         "English"),
    ("hello world",            "English"),
    ("veni vidi vici",         "Latin"),
    ("lorem ipsum dolor sit",  "Latin"),
    ("bonjour",                "French"),
    ("hola amigo",             "Spanish"),
    ("the quick brown fox",    "English"),
    ("amor",                   "Portuguese"),
]
for text, expected in self_tests:
    feat = vectorizer.transform([text])
    pred = mnb.predict(feat)[0]
    conf = max(mnb.predict_proba(feat)[0]) * 100
    status = "PASS" if pred == expected else "NOTE"
    print(f"  [{status}] \"{text}\" => {pred} ({conf:.1f}%) [expected: {expected}]")

print()
print("=" * 62)
print("  TRAINING COMPLETE")
print(f"  MNB Accuracy : {mnb_acc*100:.2f}%")
print(f"  LR  Accuracy : {lr_acc*100:.2f}%")
print(f"  SVM Accuracy : {svm_acc*100:.2f}%")
print("=" * 62)
print("  Run: python app.py")
print("  Open: http://127.0.0.1:5000")
print("=" * 62 + "\n")

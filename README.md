# LexiDetect - Language Detection System Using Machine Learning

## What does it do ??
- **Automatically detects the natural language of any text input across **22 languages** using:
- **TF-IDF vectorization** at character n-gram level (analyzer=char_wb, ngram=(1,3))
- **Multinomial Naive Bayes** (primary model) — 98.02% accuracy
- **Logistic Regression** — 98.46% accuracy
- **Linear SVM** — 98.77% accuracy
- 

---

## Project Structure
```
language_detection/
├── dataset.csv          ← 22,000 samples across 22 languages
├── train_model.py       ← Run this FIRST to train and save models
├── app.py               ← Flask web application
├── requirements.txt     ← Python dependencies
├── models/              ← Auto-created after training
│   ├── vectorizer.pkl
│   ├── mnb_model.pkl
│   ├── lr_model.pkl
│   ├── svm_model.pkl
│   └── results.json
└── templates/
    └── index.html       ← Web interface
```

---

## How to Run (Step by Step)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the Models
```bash
python train_model.py
```
This will:
- Load and clean the dataset (22,000 samples, 22 languages)
- Extract TF-IDF character n-gram features (50,000 features)
- Train all 3 models and evaluate on test set
- Save all models to the `models/` folder

### Step 3 — Run the Web App
```bash
python app.py
```

### Step 4 — Open in Browser
```
http://127.0.0.1:5000
```

---

## Supported Languages
Arabic · Chinese · Dutch · English · Estonian · French · Hindi · Indonesian ·  
Japanese · Korean · Latin · Persian · Portuguese · Pushto · Romanian · Russian ·  
Spanish · Swedish · Tamil · Thai · Turkish · Urdu

---

## API Endpoints

### POST /api/detect
Detect language of input text.

**Request:**
```json
{
  "text": "Hello, how are you?",
  "model": "mnb"
}
```
model options: `"mnb"` (Naive Bayes), `"lr"` (Logistic Regression), `"svm"` (Linear SVM)

**Response:**
```json
{
  "language": "English",
  "code": "en",
  "flag": "🇬🇧",
  "script": "Latin",
  "family": "Germanic",
  "confidence": 98.45,
  "model_used": "Multinomial Naive Bayes",
  "top3": [
    {"language": "English", "probability": 98.45},
    {"language": "French",  "probability": 1.20},
    {"language": "Spanish", "probability": 0.35}
  ],
  "warning": null,
  "char_count": 19,
  "word_count": 4
}
```

### GET /api/stats
Get system information and model accuracy.

---

## Key Technical Details
| Component | Detail |
|---|---|
| Vectorizer | TfidfVectorizer |
| Analyzer | char_wb (character + word boundary) |
| N-gram range | (1, 3) — unigrams, bigrams, trigrams |
| Max features | 50,000 |
| Sublinear TF | True (log normalization) |
| Train/Test split | 67% / 33%, random_state=42 |
| MNB alpha | 0.1 (Laplace smoothing) |
| LR C | 1.0, max_iter=1000 |
| SVM C | 1.0, max_iter=1000 |

## Future Enhancement
Interested One can perform the enhancement needed in it project. Feel free to fork the project.

## Author

Farhan Abid 
All right reserverd @2026

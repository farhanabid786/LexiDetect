"""
Language Detection System - Flask Web Application 
"""

from flask import Flask, request, jsonify, render_template
import joblib
import json
import os
import unicodedata

app = Flask(__name__)

# ── Load models at startup 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

vectorizer = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))

models = {
    'mnb': joblib.load(os.path.join(MODELS_DIR, 'mnb_model.pkl')),
    'lr':  joblib.load(os.path.join(MODELS_DIR, 'lr_model.pkl')),
    'svm': joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl')),
}

with open(os.path.join(MODELS_DIR, 'results.json')) as f:
    training_results = json.load(f)

# ── Language metadata 
LANG_META = {
    "English":    {"code": "en", "flag": "🇬🇧", "script": "Latin",          "family": "Germanic"},
    "Hindi":      {"code": "hi", "flag": "🇮🇳", "script": "Devanagari",     "family": "Indo-Aryan"},
    "French":     {"code": "fr", "flag": "🇫🇷", "script": "Latin",          "family": "Romance"},
    "Spanish":    {"code": "es", "flag": "🇪🇸", "script": "Latin",          "family": "Romance"},
    "Arabic":     {"code": "ar", "flag": "🇸🇦", "script": "Arabic",         "family": "Semitic"},
    "Chinese":    {"code": "zh", "flag": "🇨🇳", "script": "Han",            "family": "Sino-Tibetan"},
    "Japanese":   {"code": "ja", "flag": "🇯🇵", "script": "Hiragana/Kanji", "family": "Japonic"},
    "Korean":     {"code": "ko", "flag": "🇰🇷", "script": "Hangul",         "family": "Koreanic"},
    "Russian":    {"code": "ru", "flag": "🇷🇺", "script": "Cyrillic",       "family": "Slavic"},
    "Portuguese": {"code": "pt", "flag": "🇧🇷", "script": "Latin",          "family": "Romance"},
    "Thai":       {"code": "th", "flag": "🇹🇭", "script": "Thai",           "family": "Tai-Kadai"},
    "Tamil":      {"code": "ta", "flag": "🇮🇳", "script": "Tamil",          "family": "Dravidian"},
    "Urdu":       {"code": "ur", "flag": "🇵🇰", "script": "Arabic-Urdu",   "family": "Indo-Aryan"},
    "Persian":    {"code": "fa", "flag": "🇮🇷", "script": "Arabic-Persian", "family": "Iranian"},
    "Turkish":    {"code": "tr", "flag": "🇹🇷", "script": "Latin",          "family": "Turkic"},
    "Indonesian": {"code": "id", "flag": "🇮🇩", "script": "Latin",          "family": "Austronesian"},
    "Swedish":    {"code": "sv", "flag": "🇸🇪", "script": "Latin",          "family": "Germanic"},
    "Romanian":   {"code": "ro", "flag": "🇷🇴", "script": "Latin",          "family": "Romance"},
    "Dutch":      {"code": "nl", "flag": "🇳🇱", "script": "Latin",          "family": "Germanic"},
    "Latin":      {"code": "la", "flag": "🏛️",  "script": "Latin",          "family": "Italic"},
    "Estonian":   {"code": "et", "flag": "🇪🇪", "script": "Latin",          "family": "Uralic"},
    "Pushto":     {"code": "ps", "flag": "🇦🇫", "script": "Arabic-Pashto", "family": "Iranian"},
}

MODEL_LABELS = {
    'mnb': 'Multinomial Naive Bayes',
    'lr':  'Logistic Regression',
    'svm': 'Linear SVM',
}


# ── Script-based pre-detection 
# For scripts that are UNIQUE to one language in our set, we can detect
# the language with near-100% certainty BEFORE running the ML model.
# This solves the problem of "Korean/Chinese/Japanese/Hindi" being misclassified.

def get_dominant_script(text):
    script_counts = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        # Devanagari (Hindi, Sanskrit, etc.) — U+0900 to U+097F
        if 0x0900 <= cp <= 0x097F:
            script_counts['Devanagari'] = script_counts.get('Devanagari', 0) + 1
        # Arabic (Arabic, Urdu, Persian, Pushto) — U+0600 to U+06FF + extended
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
            script_counts['Arabic'] = script_counts.get('Arabic', 0) + 1
        # Cyrillic (Russian) — U+0400 to U+04FF
        elif 0x0400 <= cp <= 0x04FF:
            script_counts['Cyrillic'] = script_counts.get('Cyrillic', 0) + 1
        # CJK Unified (Chinese) — U+4E00 to U+9FFF
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            script_counts['CJK'] = script_counts.get('CJK', 0) + 1
        # Hiragana / Katakana (Japanese) — U+3040 to U+30FF
        elif 0x3040 <= cp <= 0x30FF:
            script_counts['Hiragana'] = script_counts.get('Hiragana', 0) + 1
        # Hangul (Korean) — U+AC00 to U+D7AF and Jamo
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            script_counts['Hangul'] = script_counts.get('Hangul', 0) + 1
        # Thai — U+0E00 to U+0E7F
        elif 0x0E00 <= cp <= 0x0E7F:
            script_counts['Thai'] = script_counts.get('Thai', 0) + 1
        # Tamil — U+0B80 to U+0BFF
        elif 0x0B80 <= cp <= 0x0BFF:
            script_counts['Tamil'] = script_counts.get('Tamil', 0) + 1
        else:
            script_counts['Latin'] = script_counts.get('Latin', 0) + 1

    if not script_counts:
        return None, 0.0

    total = sum(script_counts.values())
    dominant = max(script_counts, key=script_counts.get)
    ratio = script_counts[dominant] / total
    return dominant, ratio


def script_predetect(text):
    script, ratio = get_dominant_script(text)

    if ratio < 0.6:
        # Mixed scripts — let ML handle it
        return None, None

    if script == 'Devanagari':
        return 'Hindi', round(ratio * 100, 1)
    elif script == 'Cyrillic':
        return 'Russian', round(ratio * 100, 1)
    elif script == 'CJK':
        return 'Chinese', round(ratio * 100, 1)
    elif script == 'Hiragana':
        return 'Japanese', round(ratio * 100, 1)
    elif script == 'Hangul':
        return 'Korean', round(ratio * 100, 1)
    elif script == 'Thai':
        return 'Thai', round(ratio * 100, 1)
    elif script == 'Tamil':
        return 'Tamil', round(ratio * 100, 1)
    elif script == 'Arabic':
        # Arabic script is shared by Arabic, Urdu, Persian, Pushto — let ML decide
        return None, None

    return None, None


# ── Routes 
@app.route('/')
def index():
    return render_template('index.html',
        languages=sorted(LANG_META.keys()),
        training_results=training_results,
        lang_meta=LANG_META
    )


@app.route('/api/detect', methods=['POST'])
def detect():
    data      = request.get_json(force=True)
    text      = data.get('text', '').strip()
    model_key = data.get('model', 'mnb')

    # ── Input validation ──
    if not text:
        return jsonify({'error': 'Input cannot be empty.'}), 400
    if model_key not in models:
        return jsonify({'error': f'Unknown model: {model_key}'}), 400
    if text.replace(' ', '').replace('\n', '').isdigit():
        return jsonify({'error': 'Input contains no recognizable linguistic features.'}), 400

    # ── Preprocessing: Unicode NFC normalization ──
    text_norm = unicodedata.normalize('NFC', text)

    # ── LAYER 1: Script-based pre-detection (fast, near-100% accurate) ──
    script_lang, script_conf = script_predetect(text_norm)

    warning = None
    if len(text_norm) < 20:
        warning = "Very short input — confidence may be lower. Try entering a full sentence for best results."

    if script_lang is not None:
        # Script detection was conclusive — skip ML model
        meta = LANG_META.get(script_lang, {
            "code": "??", "flag": "🌐", "script": "Unknown", "family": "Unknown"
        })
        return jsonify({
            'language':   script_lang,
            'code':       meta['code'],
            'flag':       meta['flag'],
            'script':     meta['script'],
            'family':     meta['family'],
            'confidence': script_conf,
            'model_used': f'{MODEL_LABELS[model_key]} + Script Detection',
            'top3': [{'language': script_lang, 'probability': script_conf}],
            'warning':    warning,
            'char_count': len(text_norm),
            'word_count': len(text_norm.split()),
            'detection_method': 'script',
        })

    # ── LAYER 2: ML model inference ──
    features = vectorizer.transform([text_norm])
    clf       = models[model_key]
    predicted = clf.predict(features)[0]

    confidence = None
    top3       = []

    if hasattr(clf, 'predict_proba'):
        proba    = clf.predict_proba(features)[0]
        classes  = clf.classes_
        all_prob = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        confidence = round(float(max(proba)) * 100, 2)
        top3 = sorted(all_prob.items(), key=lambda x: x[1], reverse=True)[:3]
    else:
        top3 = [(predicted, 1.0)]

    meta = LANG_META.get(predicted, {
        "code": "??", "flag": "🌐", "script": "Unknown", "family": "Unknown"
    })

    return jsonify({
        'language':   predicted,
        'code':       meta['code'],
        'flag':       meta['flag'],
        'script':     meta['script'],
        'family':     meta['family'],
        'confidence': confidence,
        'model_used': MODEL_LABELS[model_key],
        'top3': [{'language': l, 'probability': round(p * 100, 2)} for l, p in top3],
        'warning':    warning,
        'char_count': len(text_norm),
        'word_count': len(text_norm.split()),
        'detection_method': 'ml',
    })


@app.route('/api/stats')
def stats():
    return jsonify({
        'supported_languages': len(LANG_META),
        'languages':  list(LANG_META.keys()),
        'models':     list(MODEL_LABELS.values()),
        'vectorizer': {
            'type':         'TF-IDF',
            'analyzer':     'char_wb',
            'ngram_range':  '(1, 4)',
            'max_features': 100000,
            'sublinear_tf': True,
            'min_df': 2,
        },
        'accuracy': {
            'mnb': training_results['mnb'],
            'lr':  training_results['lr'],
            'svm': training_results['svm'],
        },
        'dataset_size': training_results.get('dataset_size', 22000),
        'train_size':   training_results.get('train_size'),
        'test_size':    training_results.get('test_size'),
    })


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print(" LexiDetect - Language Detection System v2")
    print("=" * 55)
    print(f"  Languages Supported : {len(LANG_META)}")
    print(f"  Vectorizer          : {training_results.get('vectorizer', 'TF-IDF char n-gram (1,4)')}")
    print(f"  MNB Accuracy        : {training_results['mnb']}%")
    print(f"  LR  Accuracy        : {training_results['lr']}%")
    print(f"  SVM Accuracy        : {training_results['svm']}%")
    print("=" * 55)
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 55 + "\n")
    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000))
)

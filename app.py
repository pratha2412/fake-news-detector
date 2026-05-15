from flask import Flask, request, jsonify, render_template
import os
import re
import json
import numpy as np
import joblib
from train_model import train
from preprocess import preprocess_text

app = Flask(__name__)

# Constants
SENSATIONAL_WORDS = [
    "BREAKING", "SHOCKING", "EXPOSED", "SECRET", "LEAKED",
    "BOMBSHELL", "COVERUP", "CONSPIRACY", "HOAX", "FRAUD",
    "CENSORED", "BANNED", "DELETED", "WAKE UP", "SHEEPLE",
    "THEY DON'T WANT YOU TO KNOW", "SHARE BEFORE REMOVED",
    "MAINSTREAM MEDIA", "DEEP STATE", "NEW WORLD ORDER"
]

HEDGING_PHRASES = [
    "according to", "researchers say", "study shows", "data suggests",
    "experts believe", "published in", "peer reviewed", "cited by",
    "findings indicate", "evidence suggests", "scientists report",
    "journal of", "university of", "conducted by", "sample size"
]

EMOTION_WORDS = [
    "outrage", "furious", "disgusting", "betrayed", "corrupt",
    "evil", "lying", "criminal", "destroy", "danger", "threat",
    "panic", "crisis", "disaster", "catastrophe", "collapse"
]

model_A = None
model_B = None
model_stats = None

def load_models():
    global model_A, model_B, model_stats
    if not os.path.exists('model/classifier.pkl'):
        print("Models not found. Training models inline...")
        train()
    
    model_A = joblib.load('model/classifier.pkl')
    category_path = 'model/category_classifier.pkl'
    model_B = joblib.load(category_path) if os.path.exists(category_path) else None
    with open('model/model_stats.json', 'r') as f:
        model_stats = json.load(f)

# Load models on startup
with app.app_context():
    load_models()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({
        "status": "ready" if model_A is not None else "training",
        "stats": model_stats
    })

@app.route('/stats')
def stats():
    if model_stats:
        return jsonify(model_stats)
    return jsonify({"error": "Model stats not found"}), 404

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    text = data['text']
    if len(text) < 20:
        return jsonify({"error": "Text too short. Minimum 20 characters required."}), 400
        
    if len(text) > 10000:
        text = text[:10000]
        
    cleaned_text = preprocess_text(text)
    
    # 1. Pipeline A: Fake/Real Prediction (calibrated probabilities)
    prob_fake = float(model_A.predict_proba([cleaned_text])[0][1])
    # Note: verdict logic moved to step 5 to incorporate heuristics

        
    # 2. Pipeline B: Categories (optional when trained on 20 Newsgroups)
    if model_B is not None:
        probs_B = model_B.predict_proba([cleaned_text])[0]
        top_3_indices = probs_B.argsort()[-3:][::-1]
        categories = model_stats['categories']
        newsgroup_category = categories[top_3_indices[0]]
        related_categories = [categories[idx] for idx in top_3_indices[1:]]
    else:
        newsgroup_category = "n/a"
        related_categories = []
    
    # 3. Extract top contributing TF-IDF features
    tfidf_A = model_A.named_steps['tfidf']
    calibrated = model_A.named_steps['clf']
    base_clf = calibrated.calibrated_classifiers_[0].estimator
    
    text_vector = tfidf_A.transform([cleaned_text])
    feature_indices = text_vector.nonzero()[1]
    
    contributions = []
    feature_names = tfidf_A.get_feature_names_out()
    coefs = base_clf.coef_[0]
    
    for idx in feature_indices:
        val = text_vector[0, idx]
        coef = coefs[idx]
        contributions.append((feature_names[idx], val * coef))
        
    # Sort contributions
    contributions.sort(key=lambda x: x[1]) # ascending
    
    # Positive contributions drive it towards FAKE (1), negative towards REAL (0)
    top_fake_words = [word for word, score in reversed(contributions[-5:]) if score > 0]
    top_real_words = [word for word, score in contributions[:5] if score < 0]
    
    # 4. Compute sub-scores
    word_count = len(cleaned_text.split())
    if word_count == 0: word_count = 1
    
    # Sensationalism score
    sensational_count = sum(1 for word in SENSATIONAL_WORDS if word.lower() in cleaned_text.lower())
    caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
    exclamations = text.count('!')
    
    sens_raw = (sensational_count * 2) + (caps_words * 0.5) + exclamations
    sensationalism_score = min(10.0, (sens_raw / word_count) * 50)
    
    # Emotional score
    emo_count = sum(1 for word in EMOTION_WORDS if word.lower() in cleaned_text.lower())
    emotional_score = min(10.0, (emo_count / word_count) * 100)
    
    # Source score (hedging/citations)
    source_count = sum(1 for phrase in HEDGING_PHRASES if phrase.lower() in cleaned_text.lower())
    source_score = min(10.0, (source_count / word_count) * 200) # Higher is usually better for real news, map to 0-10
    
    # Complexity score
    words = cleaned_text.split()
    avg_word_len = sum(len(w) for w in words) / word_count if words else 0
    sentences = max(1, len(re.split(r'[.!?]+', cleaned_text)))
    avg_sentence_len = word_count / sentences
    complexity_score = min(10.0, (avg_word_len * 0.5) + (avg_sentence_len * 0.2))

    # 5. Hybrid Confidence Calculation
    # ML base confidence
    base_fake_conf = prob_fake * 100
    
    # Adjust based on heuristics
    # + up to 30 points for high sensationalism
    # + up to 20 points for high emotion
    # + 25 points if NO sources are cited (common in fake/satire)
    # - up to 20 points if well-sourced
    
    heuristic_adjustment = (sensationalism_score * 3.0) + (emotional_score * 2.0)
    
    if source_score == 0:
        heuristic_adjustment += 25.0
    else:
        heuristic_adjustment -= (source_score * 2.0)
        
    final_fake_conf = base_fake_conf + heuristic_adjustment
    confidence_fake = max(0.0, min(100.0, final_fake_conf))
    confidence_real = 100.0 - confidence_fake
    
    if confidence_fake >= 75:
        verdict = "FAKE"
    elif confidence_fake >= 55:
        verdict = "LIKELY FAKE"
    elif confidence_fake >= 40:
        verdict = "MIXED"
    elif confidence_fake >= 25:
        verdict = "LIKELY REAL"
    else:
        verdict = "REAL"
    
    output = {
        "verdict": verdict,
        "confidence_fake": round(confidence_fake, 1),
        "confidence_real": round(confidence_real, 1),
        "newsgroup_category": newsgroup_category,
        "related_categories": related_categories,
        "top_fake_words": top_fake_words,
        "top_real_words": top_real_words,
        "sensationalism_score": round(sensationalism_score, 1),
        "emotional_score": round(emotional_score, 1),
        "source_score": round(source_score, 1),
        "complexity_score": round(complexity_score, 1),
        "word_count": word_count,
        "model_info": {
            "algorithm": "LinearSVC + TF-IDF",
            "training_samples": model_stats['training_samples'],
            "test_accuracy": model_stats['test_accuracy']
        }
    }
    
    return jsonify(output)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

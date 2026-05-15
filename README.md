# Fake News Detector

A machine learning-powered web application that detects fake news articles using **TF-IDF vectorization** and a **Linear Support Vector Classifier (LinearSVC)**. Built with Flask and scikit-learn, it provides explainable predictions with heuristic-based scoring.

---

## Project Structure

```
fake-news/
├── fake-news-detector/
│   ├── app.py               # Flask web server & API endpoints
│   ├── train_model.py        # Model training pipeline (Pipeline A & B)
│   ├── preprocess.py         # Shared text cleaning utilities
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile            # Container build instructions
│   ├── Procfile              # Heroku deployment config
│   ├── static/
│   │   └── style.css         # Dark-themed UI styles
│   ├── templates/
│   │   └── index.html        # Single-page frontend (HTML + JS)
│   └── model/
│       ├── classifier.pkl         # Trained Pipeline A model
│       ├── category_classifier.pkl # Trained Pipeline B model
│       └── model_stats.json       # Training metrics & metadata
└── README.md
```

---

## Approach & Architecture

The project uses a **hybrid approach** combining:

1. **Machine Learning Pipeline (ML)** — A TF-IDF + LinearSVC model trained on the 20 Newsgroups dataset. Six politically/religiously charged newsgroups are labeled as "fake" to simulate sensational/unreliable content, while the remaining 14 groups serve as "real".

2. **Heuristic Scoring** — Five rule-based scores (sensationalism, emotional tone, source citations, complexity, TF-IDF word contributions) that make the prediction explainable and adjust the ML confidence.

3. **Dual Pipeline System**:
   - **Pipeline A**: Binary classifier (Fake vs Real) with calibrated probabilities
   - **Pipeline B**: Multi-class classifier that identifies which topic category the article belongs to

---

## Training Methods & Algorithms Used

### 1. Text Vectorization — TF-IDF (Term Frequency-Inverse Document Frequency)
- **What it does**: Converts raw text into numerical feature vectors by assigning importance weights to words. Words that appear frequently in a document but rarely across the corpus get higher scores (they are more "distinctive").
- **Configuration**: max 75,000 features, unigram + bigram (ngram_range=(1,2)), minimum document frequency of 2, max 85% document frequency, sublinear TF scaling (`1 + log(tf)`), stop words removed.

### 2. Pipeline A — Linear Support Vector Classifier (LinearSVC)
- **Algorithm**: LinearSVC finds the optimal hyperplane (decision boundary) that best separates "fake" articles from "real" articles in the high-dimensional TF-IDF feature space. It uses a linear kernel (no kernel trick needed since text data is already high-dimensional).
- **Parameters**: `C=0.5` (regularization strength — lower = stronger regularization to prevent overfitting), `class_weight='balanced'` (automatically adjusts weights inversely proportional to class frequencies to handle imbalanced data), `dual='auto'` (auto-selects solver), `max_iter=5000`.
- **Calibration**: Wrapped in `CalibratedClassifierCV` with `cv=3` (3-fold cross-validation) and `method='sigmoid'` (Platt scaling) to convert raw SVM decision scores into proper probability estimates (0 to 1).

### 3. Pipeline B — Multinomial Naive Bayes (MultinomialNB)
- **Algorithm**: A probabilistic classifier based on Bayes' theorem, well-suited for discrete data like word counts. It assumes feature independence given the class label ("naive" assumption) and models word occurrence as a multinomial distribution.
- **Parameters**: `alpha=0.01` (Laplace smoothing parameter — prevents zero probabilities for unseen words; small value keeps the model closer to raw data).

### 4. Cross-Validation — Stratified K-Fold
- **Method**: 5-fold Stratified K-Fold cross-validation ensures each fold maintains the same class distribution as the original dataset. Provides a more reliable estimate of model generalization compared to a single train/test split.
- **Scoring metric**: F1-score (harmonic mean of precision and recall).

### 5. Feature Importance Extraction (SVM Coefficients)
- The trained LinearSVC has a `coef_` array: each feature (word/bigram) has a learned weight coefficient. Positive coefficients push predictions toward "fake", negative toward "real". The magnitude of `coefficient × term frequency` determines per-prediction word contributions for explainability.

### 6. Hybrid Confidence (ML + Heuristics)
- The raw ML probability from Pipeline A is adjusted by rule-based heuristic scores (sensationalism, emotional tone, source citations). This hybrid approach compensates for the fact that the model was trained on 20 Newsgroups (which may not perfectly represent real-world fake news) by incorporating linguistic patterns known to correlate with misinformation.

### 7. 20 Newsgroups Labeling Strategy
- 6 out of 20 newsgroups are labeled as "fake" (class 1):
  - 3 political: `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`
  - 3 religious: `talk.religion.misc`, `alt.atheism`, `soc.religion.christian`
- Rationale: These groups contain opinionated, emotionally charged, and often unverifiable content — characteristics shared with fake news — while the remaining 14 groups (science, technology, sports, etc.) tend to be more factual.

---

## File-by-File Explanation

### `preprocess.py`
Shared text cleaning used by both training and inference. It strips HTML tags, removes email/newsgroup headers (From, Subject, etc.), removes URLs, and normalizes whitespace. Ensures consistent input between training and prediction.

### `train_model.py`
Handles all model training:

- **`_build_tfidf_vectorizer()`** — Creates a TF-IDF vectorizer with config: max 75,000 features, unigram+bigram n-grams, min document frequency of 2, max 85% document frequency, sublinear TF scaling.

- **`_load_custom_dataset()`** — Optionally loads external CSV datasets (news.csv, fake_real_news.csv, dataset.csv) if present. Falls back to 20 Newsgroups if no CSV found.

- **`_load_20newsgroups()`** — Loads 20 Newsgroups, removes headers/footers/quotes, and maps 6 groups (talk.politics.guns, talk.politics.mideast, talk.politics.misc, talk.religion.misc, alt.atheism, soc.religion.christian) as the "fake" class (label=1).

- **`train()`** — Main entry point:
  1. Loads data (custom CSV or 20 Newsgroups)
  2. Trains **Pipeline A**: `TF-IDF → CalibratedClassifierCV(LinearSVC)` with 3-fold cross-validation for probability calibration
  3. Trains **Pipeline B**: `TF-IDF → MultinomialNB` for multi-class category prediction
  4. Evaluates with accuracy, F1 score, and 5-fold cross-validation
  5. Extracts top-20 most influential words for both fake and real classes from the SVM coefficients
  6. Saves both models and a stats JSON to the `model/` directory

### `app.py`
Flask application serving three endpoints:

- **`GET /`** — Renders the index.html frontend
- **`GET /status`** — Returns model readiness and accuracy stats
- **`POST /analyze`** — Main prediction endpoint that:
  1. Validates input (20–10,000 characters)
  2. Preprocesses the text
  3. Gets Pipeline A probability of "fake"
  4. Gets Pipeline B top-3 category predictions
  5. Extracts top-5 words driving the prediction via TF-IDF feature contributions (coefficient × term frequency)
  6. Computes 4 heuristic scores:
     - **Sensationalism** — Counts of all-caps words, exclamation marks, and a predefined list of sensational keywords (BREAKING, SHOCKING, EXPOSED, etc.)
     - **Emotional Tone** — Counts of emotionally charged words (outrage, furious, corrupt, etc.)
     - **Source Score** — Counts of hedging/citation phrases (peer reviewed, study shows, etc.)
     - **Complexity** — Average word length and sentence length
  7. **Hybrid Confidence**: Combines ML probability with heuristic adjustments (sensationalism +30pts max, emotion +20pts max, no sources +25pts, good sources -20pts)
  8. Maps the final confidence (0-100) to a 5-level verdict: **FAKE**, **LIKELY FAKE**, **MIXED**, **LIKELY REAL**, **REAL**

### `templates/index.html`
Single-page dark-themed frontend with:
- Example buttons (Science, Politics, Medical, Religion, Tech) that fill the textarea instantly
- Status bar showing model readiness
- Analyze button that POSTs to `/analyze`
- Results panel with animated confidence bars, score cards, word chips, category tags, and model info
- Pure vanilla JavaScript — no framework dependencies

### `static/style.css`
Dark-themed CSS with CSS custom properties (variables). Uses DM Sans (body) and Space Mono (headings/monospace elements). Includes animations for the results panel fade-in and confidence bar transitions.

### `requirements.txt`
```
flask, scikit-learn, nltk, numpy, joblib, pandas, gunicorn
```

### `Dockerfile`
Multi-stage Python 3.11 container that installs dependencies, copies the app, and runs `train_model.py` at build time so the container starts with a pre-trained model. Serves via Gunicorn on port 5000.

### `Procfile`
Heroku deployment entry point — runs Gunicorn with 2 workers.

---

## Step-by-Step Workflow

### How the Project Works (End-to-End)

1. **Data Loading**
   - Checks for custom CSV files (`data/news.csv`, etc.)
   - If none found, falls back to the built-in 20 Newsgroups dataset
   - Six controversial newsgroups are labeled as "fake" (class 1), rest as "real" (class 0)

2. **Preprocessing**
   - HTML tags removed
   - Newsgroup headers stripped
   - URLs removed
   - Whitespace normalized

3. **TF-IDF Vectorization**
   - Converts raw text into numerical feature vectors
   - Weighs words by their importance across the corpus
   - Limits to 75,000 most informative unigrams and bigrams

4. **Pipeline A Training (Fake/Real)**
   - LinearSVC with balanced class weights
   - Wrapped in CalibratedClassifierCV with sigmoid calibration for probability outputs
   - Evaluated with accuracy, F1 score, and 5-fold stratified cross-validation

5. **Pipeline B Training (Category)**
   - Multinomial Naive Bayes for 20-class topic classification
   - Only trained when using the 20 Newsgroups dataset
   - ~69.6% category accuracy

6. **Model Persistence**
   - Both pipelines saved as `.pkl` files
   - Metrics (accuracy, F1, CV scores, top words) saved as JSON

7. **Web Server Startup**
   - Flask loads saved models on startup
   - Serves the frontend at `GET /`

8. **User Inputs Text**
   - User pastes (or clicks an example) article text
   - Frontend validates minimum 20 characters
   - Sends POST request to `/analyze`

9. **Backend Analysis**
   - Text is preprocessed identically to training
   - Pipeline A predicts fake probability
   - Pipeline B predicts topic category (top 3)
   - TF-IDF feature contributions extracted for explainability
   - Four heuristic scores computed
   - ML + heuristics combined into a hybrid confidence score
   - Verdict assigned based on confidence thresholds

10. **Results Display**
    - Verdict badge with color coding (red=fake, green=real, yellow=mixed)
    - Animated confidence bars for fake/real percentages
    - Four score cards with mini bars
    - Top words driving the prediction (fake-trigger words vs real-indicator words)
    - Identified categories with primary and related tags
    - Model info footer showing algorithm and training stats

---

## How to Run

```bash
# Install dependencies
cd fake-news-detector
pip install -r requirements.txt

# Train the model (if not already saved)
python train_model.py

# Run the Flask server
python app.py

# Or with Gunicorn (production)
gunicorn --bind 0.0.0.0:5000 --workers 2 app:app

# Or with Docker
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

Open `http://localhost:5000` in your browser.

---

## Model Performance

| Metric              | Score  |
|----------------------|--------|
| Test Accuracy        | 92.96% |
| Test F1 (Fake Class) | 86.39% |
| CV F1 Mean           | 88.34% |
| Fake Precision       | 89.66% |
| Fake Recall          | 83.35% |
| Real Precision       | 94.06% |
| Real Recall          | 96.48% |
| Category Accuracy    | 69.56% |

Trained on **11,314 samples** from the 20 Newsgroups dataset.

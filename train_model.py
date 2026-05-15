import os
import json
import numpy as np
import joblib
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score

from preprocess import preprocess_text

FAKE_GROUPS = {
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc',
    'alt.atheism',
    'soc.religion.christian',
}

CUSTOM_DATA_PATHS = [
    'data/news.csv',
    'data/fake_real_news.csv',
    'data/dataset.csv',
]


def _build_tfidf_vectorizer():
    return TfidfVectorizer(
        stop_words='english',
        max_features=75000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
    )


def _load_custom_dataset():
    """Load a real fake-news CSV if present (columns: text or title+text, label)."""
    for path in CUSTOM_DATA_PATHS:
        if not os.path.exists(path):
            continue
        print(f"Loading custom dataset from {path}...")
        df = pd.read_csv(path)
        label_col = next(
            (c for c in df.columns if c.lower() in ('label', 'class', 'target')),
            None,
        )
        if label_col is None:
            continue

        if 'text' in df.columns:
            texts = df['text'].astype(str)
        elif 'title' in df.columns and 'text' in df.columns:
            texts = (df['title'].astype(str) + ' ' + df['text'].astype(str))
        elif 'title' in df.columns:
            texts = df['title'].astype(str)
        else:
            continue

        labels = df[label_col].astype(str).str.upper().str.strip()
        y = np.array([1 if v in ('FAKE', '1', 'TRUE', 'FAKE NEWS') else 0 for v in labels])
        X = [preprocess_text(t) for t in texts]
        mask = [len(x) >= 20 for x in X]
        X = [x for x, keep in zip(X, mask) if keep]
        y = y[mask]
        if len(X) < 100:
            continue
        print(f"  Loaded {len(X)} samples from custom dataset.")
        return X, y
    return None, None


def _load_20newsgroups():
    print("Loading 20 Newsgroups dataset...")
    data_train = fetch_20newsgroups(
        subset='train', remove=('headers', 'footers', 'quotes')
    )
    data_test = fetch_20newsgroups(
        subset='test', remove=('headers', 'footers', 'quotes')
    )

    target_names = data_train.target_names
    fake_indices = [i for i, name in enumerate(target_names) if name in FAKE_GROUPS]

    X_train = [preprocess_text(t) for t in data_train.data]
    X_test = [preprocess_text(t) for t in data_test.data]
    y_train_binary = np.array([1 if y in fake_indices else 0 for y in data_train.target])
    y_test_binary = np.array([1 if y in fake_indices else 0 for y in data_test.target])
    y_train_multi = data_train.target
    y_test_multi = data_test.target

    return (
        X_train, X_test, y_train_binary, y_test_binary,
        y_train_multi, y_test_multi, target_names,
    )


def train():
    custom_X, custom_y = _load_custom_dataset()
    use_custom = custom_X is not None

    if use_custom:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train_binary, y_test_binary = train_test_split(
            custom_X, custom_y, test_size=0.2, random_state=42, stratify=custom_y
        )
        y_train_multi = y_train_binary
        y_test_multi = y_test_binary
        target_names = ['real', 'fake']
        X_train_multi = X_train
    else:
        (
            X_train, X_test, y_train_binary, y_test_binary,
            y_train_multi, y_test_multi, target_names,
        ) = _load_20newsgroups()
        X_train_multi = X_train

    print("Training Pipeline A (Binary: Fake/Real)...")
    pipeline_A = Pipeline([
        ('tfidf', _build_tfidf_vectorizer()),
        ('clf', CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight='balanced', dual='auto', max_iter=5000),
            cv=3,
            method='sigmoid',
        )),
    ])
    pipeline_A.fit(X_train, y_train_binary)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        Pipeline([
            ('tfidf', _build_tfidf_vectorizer()),
            ('clf', LinearSVC(C=0.5, class_weight='balanced', dual='auto', max_iter=5000)),
        ]),
        X_train,
        y_train_binary,
        cv=cv,
        scoring='f1',
    )

    print("Training Pipeline B (Multi-class categories)...")
    pipeline_B = Pipeline([
        ('tfidf', _build_tfidf_vectorizer()),
        ('clf', MultinomialNB(alpha=0.01)),
    ])
    if not use_custom:
        pipeline_B.fit(X_train_multi, y_train_multi)

    print("Evaluating Pipeline A...")
    pred_A = pipeline_A.predict(X_test)
    acc_A = accuracy_score(y_test_binary, pred_A)
    f1_A = f1_score(y_test_binary, pred_A)
    rep_A = classification_report(y_test_binary, pred_A, output_dict=True)

    print(f"Model trained on {len(X_train)} samples")
    print(f"Test accuracy: {acc_A * 100:.2f}%")
    print(f"Test F1 (fake class): {f1_A * 100:.2f}%")
    print(f"CV F1 mean: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 200:.2f}%)")
    print(
        f"Fake precision: {rep_A['1']['precision'] * 100:.2f}%, "
        f"recall: {rep_A['1']['recall'] * 100:.2f}%"
    )
    print(
        f"Real precision: {rep_A['0']['precision'] * 100:.2f}%, "
        f"recall: {rep_A['0']['recall'] * 100:.2f}%"
    )

    acc_B = None
    if not use_custom:
        print("Evaluating Pipeline B...")
        pred_B = pipeline_B.predict(X_test)
        acc_B = accuracy_score(y_test_multi, pred_B)
        print(f"Pipeline B Test accuracy: {acc_B * 100:.2f}%")

    tfidf_A = pipeline_A.named_steps['tfidf']
    calibrated = pipeline_A.named_steps['clf']
    base_clf = calibrated.calibrated_classifiers_[0].estimator
    feature_names = np.array(tfidf_A.get_feature_names_out())
    coef = base_clf.coef_[0]

    top_fake_words = feature_names[coef.argsort()[-20:][::-1]].tolist()
    top_real_words = feature_names[coef.argsort()[:20]].tolist()

    os.makedirs('model', exist_ok=True)
    joblib.dump(pipeline_A, 'model/classifier.pkl')
    if not use_custom:
        joblib.dump(pipeline_B, 'model/category_classifier.pkl')

    stats = {
        "training_samples": len(X_train),
        "dataset": "custom" if use_custom else "20newsgroups",
        "test_accuracy": round(acc_A, 4),
        "test_f1": round(f1_A, 4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "fake_precision": round(rep_A['1']['precision'], 4),
        "fake_recall": round(rep_A['1']['recall'], 4),
        "real_precision": round(rep_A['0']['precision'], 4),
        "real_recall": round(rep_A['0']['recall'], 4),
        "top_fake_words": top_fake_words,
        "top_real_words": top_real_words,
        "categories": target_names,
    }
    if acc_B is not None:
        stats["category_accuracy"] = round(acc_B, 4)

    with open('model/model_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)

    print("Training complete. Models saved to 'model/' directory.")


if __name__ == '__main__':
    train()

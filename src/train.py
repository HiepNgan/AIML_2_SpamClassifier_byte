"""Step 3 - Train and compare spam classifiers.

Run from the project root:
    python src/train.py

What it does:
  1. Split data 80/20 (stratified, so both sets keep ~12% spam).
  2. Score 3 candidate models with 5-fold cross-validation on the TRAIN set only.
  3. Pick the best by F1 (spam class), refit it on the full train set.
  4. Save the model + the untouched test set for evaluate.py.
"""
import json
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data_utils import OUTPUT_DIR, ROOT, clean_text, load_dataset

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
TEST_SPLIT_PATH = ROOT / "data" / "test_split.csv"
SEED = 42


def build_features():
    """Two views of the same message, glued side by side.

    - word n-grams catch phrases like "call now" or "free entry"
    - char n-grams catch tricks like "fr33", "c@ll", "W1N"
    """
    words = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                            min_df=2, sublinear_tf=True)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            min_df=3, sublinear_tf=True, lowercase=True)
    return FeatureUnion([("words", words), ("chars", chars)])


def candidate_models():
    # Every model must output probabilities, because the task asks for
    # confidence scores. LinearSVC doesn't, so it gets wrapped in a calibrator.
    return {
        "Naive Bayes": ComplementNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(
            C=10, class_weight="balanced", max_iter=2000),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced"), cv=3),
    }


def compare_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []

    for name, clf in candidate_models().items():
        pipe = Pipeline([("features", build_features()), ("clf", clf)])
        start = time.perf_counter()
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=["accuracy", "precision", "recall", "f1"])
        rows.append({
            "model": name,
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
            "seconds": time.perf_counter() - start,
        })
        print(f"  {name:<20} F1 = {rows[-1]['f1']:.4f}")

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=0.2,
        stratify=df["target"], random_state=SEED)

    print(f"Train: {len(X_train)} tin | Test: {len(X_test)} tin (giữ kín đến bước đánh giá)")
    print("\nSo sánh model bằng 5-fold cross-validation:")
    results = compare_models(X_train, y_train)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n" + results.round(4).to_string(index=False))

    best_name = results.loc[0, "model"]
    print(f"\n=> Model tốt nhất: {best_name}. Train lại trên toàn bộ tập train...")

    best = Pipeline([("features", build_features()),
                     ("clf", candidate_models()[best_name])])
    best.fit(X_train, y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    pd.DataFrame({"text": X_test, "target": y_test}).to_csv(TEST_SPLIT_PATH, index=False)

    with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as f:
        json.dump({"model": best_name, "cv_f1": round(results.loc[0, "f1"], 4),
                   "train_size": len(X_train), "test_size": len(X_test),
                   "seed": SEED}, f, indent=2)

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"Đã lưu model: {MODEL_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

"""Step 4 - Evaluate the saved model on the held-out test set.

Run from the project root (after train.py):
    python src/evaluate.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from data_utils import OUTPUT_DIR
from train import MODEL_DIR, MODEL_PATH, TEST_SPLIT_PATH

LABELS = ["ham", "spam"]


def load_artifacts():
    if not MODEL_PATH.exists() or not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError("Chưa có model hoặc tập test. Chạy python src/train.py trước.")
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_SPLIT_PATH, keep_default_na=False)
    return model, test["text"], test["target"]


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: raw counts. Right: share of each true class, easier to read
    # when ham outnumbers spam 7 to 1.
    for ax, norm, title in [(axes[0], None, "Counts"),
                            (axes[1], "true", "Normalized by true class")]:
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=LABELS, normalize=norm,
            values_format=".2%" if norm else "d", cmap="Blues",
            colorbar=False, ax=ax)
        ax.set_title(title)

    fig.suptitle(f"Confusion matrix - {model_name} (test set)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_confusion_matrix.png", dpi=120)
    plt.close(fig)


def plot_precision_recall(y_true, y_prob):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, name="spam", ax=ax)
    ax.set_title("Precision-Recall curve (test set)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_precision_recall.png", dpi=120)
    plt.close(fig)


def save_mistakes(texts, y_true, y_pred, y_prob):
    """Wrong predictions are where you learn the most about a model."""
    df = pd.DataFrame({"text": texts, "actual": y_true, "predicted": y_pred,
                       "spam_prob": y_prob.round(3)})
    wrong = df[df["actual"] != df["predicted"]].copy()
    wrong["error_type"] = wrong["actual"].map({0: "false positive (ham -> spam)",
                                               1: "false negative (spam -> ham)"})
    wrong[["actual", "predicted"]] = wrong[["actual", "predicted"]].replace({0: "ham", 1: "spam"})
    wrong.to_csv(OUTPUT_DIR / "misclassified.csv", index=False)
    return wrong


def main():
    model, X_test, y_test = load_artifacts()
    model_name = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))["model"]

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({"true_negative": int(tn), "false_positive": int(fp),
                    "false_negative": int(fn), "true_positive": int(tp)})

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_precision_recall(y_test, y_prob)
    wrong = save_mistakes(X_test, y_test, y_pred, y_prob)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "test_size": len(y_test),
                   **{k: round(v, 4) if isinstance(v, float) else v
                      for k, v in metrics.items()}}, f, indent=2)

    print(f"Model: {model_name} | Test set: {len(y_test)} tin\n")
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\nTin thường bị chặn nhầm (FP): {fp}")
    print(f"Spam lọt lưới (FN):          {fn}")
    print(f"\nĐã lưu: confusion matrix, PR curve, metrics.json, "
          f"misclassified.csv ({len(wrong)} dòng) vào {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Step 5 - Use the trained model on new messages.

Run from the project root. Three ways to use it:

    python src/predict.py "WINNER! Claim your free prize now"   # one or more messages
    python src/predict.py --demo                                # 10 test-set examples -> CSV
    python src/predict.py                                       # interactive mode
"""
import argparse
import sys

import joblib
import pandas as pd

from data_utils import OUTPUT_DIR
from train import MODEL_PATH, TEST_SPLIT_PATH

DEFAULT_THRESHOLD = 0.5


class SpamDetector:
    def __init__(self, model_path=MODEL_PATH, threshold=DEFAULT_THRESHOLD):
        if not model_path.exists():
            sys.exit("Chưa có model. Chạy python src/train.py trước.")
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def predict(self, messages):
        """Return one dict per message: label, spam probability, confidence."""
        probs = self.model.predict_proba(list(messages))[:, 1]
        results = []
        for text, p in zip(messages, probs):
            is_spam = p >= self.threshold
            results.append({
                "text": text,
                "predicted": "spam" if is_spam else "ham",
                "spam_prob": round(float(p), 4),
                # how sure the model is about the label it picked
                "confidence": round(float(p if is_spam else 1 - p), 4),
            })
        return results


def print_results(results, show_actual=False):
    for r in results:
        tag = "🚫 SPAM" if r["predicted"] == "spam" else "✅ HAM "
        line = f"{tag} {r['confidence']:>7.1%}"
        if show_actual:
            mark = "✓" if r["actual"] == r["predicted"] else "✗"
            line += f"  (thật: {r['actual']:<4} {mark})"
        preview = r["text"] if len(r["text"]) <= 70 else r["text"][:67] + "..."
        print(f"{line}  | {preview}")


def run_demo(detector, n_per_class=5, seed=7):
    """Pick spam and ham samples from the test set so ground truth is known."""
    test = pd.read_csv(TEST_SPLIT_PATH, keep_default_na=False)
    sample = (test.groupby("target", group_keys=False)
                  .sample(n=n_per_class, random_state=seed)
                  .sample(frac=1, random_state=seed))   # shuffle ham/spam together

    results = detector.predict(sample["text"].tolist())
    for r, target in zip(results, sample["target"]):
        r["actual"] = "spam" if target == 1 else "ham"

    print_results(results, show_actual=True)
    correct = sum(r["actual"] == r["predicted"] for r in results)
    print(f"\nĐúng {correct}/{len(results)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "example_predictions.csv"
    cols = ["text", "actual", "predicted", "spam_prob", "confidence"]
    pd.DataFrame(results)[cols].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {out}")


def run_interactive(detector):
    print("Nhập tin nhắn để kiểm tra (Enter trống để thoát).")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        print_results(detector.predict([text]))


def main():
    parser = argparse.ArgumentParser(description="SMS spam classifier")
    parser.add_argument("messages", nargs="*", help="message(s) to classify")
    parser.add_argument("--demo", action="store_true",
                        help="predict 10 test-set samples and save them to CSV")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="spam probability cut-off (default 0.5)")
    args = parser.parse_args()

    detector = SpamDetector(threshold=args.threshold)

    if args.demo:
        run_demo(detector)
    elif args.messages:
        print_results(detector.predict(args.messages))
    else:
        run_interactive(detector)


if __name__ == "__main__":
    main()

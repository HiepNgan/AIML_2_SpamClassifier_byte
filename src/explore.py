"""Step 2 - Explore the dataset before training anything.

Run from the project root:
    python src/explore.py
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # save images only, no popup windows
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import OUTPUT_DIR, add_length_features, clean_text, load_dataset

STOPWORDS = {
    "the", "a", "an", "to", "you", "i", "and", "is", "in", "u", "it", "of",
    "for", "me", "my", "your", "on", "have", "that", "are", "be", "at", "so",
    "but", "can", "do", "will", "this", "not", "with", "we", "or", "get",
    "if", "ur", "just", "no", "now", "all", "from", "was", "am", "up",
}

sns.set_theme(style="whitegrid")
COLORS = {"ham": "#4C72B0", "spam": "#DD8452"}


def print_overview(df):
    counts = df["label"].value_counts()
    total = len(df)
    print(f"Tổng số tin nhắn sau khi làm sạch: {total}")
    for label, n in counts.items():
        print(f"  {label:<5}: {n:>5}  ({n / total:.1%})")

    print("\nĐộ dài trung bình (ký tự):")
    print(df.groupby("label")["char_len"].mean().round(1).to_string())


def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = ["ham", "spam"]
    sns.countplot(data=df, x="label", order=order, hue="label",
                  palette=COLORS, legend=False, ax=ax)
    for bar in ax.patches:
        ax.annotate(f"{int(bar.get_height())}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Class distribution")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_class_distribution.png", dpi=120)
    plt.close(fig)


def plot_length(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="char_len", hue="label", palette=COLORS,
                 bins=60, stat="density", common_norm=False, ax=ax)
    ax.set_xlim(0, 250)
    ax.set_title("Message length: spam is usually longer")
    ax.set_xlabel("Characters")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_message_length.png", dpi=120)
    plt.close(fig)


def top_words(texts, n=15):
    counter = Counter()
    for text in texts:
        counter.update(w for w in clean_text(text).split()
                       if w not in STOPWORDS and len(w) > 1)
    return counter.most_common(n)


def plot_top_words(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, label in zip(axes, ["ham", "spam"]):
        words = top_words(df.loc[df["label"] == label, "text"])
        names, freqs = zip(*words)
        ax.barh(names[::-1], freqs[::-1], color=COLORS[label])
        ax.set_title(f"Top words in {label}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_top_words.png", dpi=120)
    plt.close(fig)

    print("\nTừ hay gặp trong SPAM:")
    print("  " + ", ".join(w for w, _ in top_words(df.loc[df.label == "spam", "text"], 10)))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = add_length_features(load_dataset())

    print_overview(df)
    plot_class_balance(df)
    plot_length(df)
    plot_top_words(df)

    print(f"\nĐã lưu 3 biểu đồ vào: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

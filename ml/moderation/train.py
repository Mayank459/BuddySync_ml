"""Toxicity pipeline: data -> preprocess -> split -> train -> test -> save.

Char n-gram tf-idf + logistic regression: handles Hinglish spellings, runs on CPU in microseconds.
data/messages.csv is a synthetic seed. Replace with real labelled data, e.g. Hinglish abuse
datasets such as LCS2-IIITD, plus your own moderator decisions (plan §8).
Run from ml/:  python -m moderation.train
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from common.utils import load, save_artifact


def main():
    df = load("messages")                                          # 1. data
    df["text"] = df.text.str.lower().str.strip()                   # 2. preprocess
    Xtr, Xte, ytr, yte = train_test_split(df.text, df.label, test_size=0.25,
                                          stratify=df.label, random_state=0)  # 3. split
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                          LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xtr, ytr)  # 4. train
    print(classification_report(yte, model.predict(Xte), target_names=["ok", "abusive"], digits=3))  # 5. test
    print("saved", save_artifact({"model": model, "model_version": "tox-charlr-v0"}, "moderation"))


if __name__ == "__main__":
    main()

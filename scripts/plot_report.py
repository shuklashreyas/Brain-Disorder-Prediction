
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_cnn_probs(path):
    df = pd.read_csv(path)
    return df


def compute_cnn_metrics(df):
    p_cols = [c for c in df.columns if c.startswith("p_")]
    df["cnn_pred_idx"] = df[p_cols].values.argmax(axis=1)
    csv_order = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]
    mapping = {name: idx for idx, name in enumerate(csv_order)}
    df["true_idx"] = df["true_label"].map(mapping)

    y_true = df["true_idx"].values
    y_pred = df["cnn_pred_idx"].values

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    return cm, metrics, csv_order


def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(7, 6))
    if _HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CNN Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fit_binary_approx(df):
    df = df.copy()
    df["ad_binary"] = df["true_label"].isin(["Mild Impairment", "Moderate Impairment"]).astype(int)

    X = df[["p_mild_impairment", "p_moderate_impairment", "p_no_impairment", "p_very_mild_impairment"]].values
    y = df["ad_binary"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    clf = LogisticRegression(solver="liblinear")
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return fpr, tpr, roc_auc, metrics


def plot_roc(fpr, tpr, roc_auc, out_path):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Binary ROC (approx. Bayesian logistic on CNN probs)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_summary_table(cnn_metrics, bin_metrics_approx, bayes_json_path=None):
    rows = []

    rows.append({
        "model": "CNN (per-class probs)",
        "accuracy": cnn_metrics["accuracy"],
        "precision": cnn_metrics["precision"],
        "recall": cnn_metrics["recall"],
        "f1": cnn_metrics["f1"],
    })

    rows.append({
        "model": "Binary (approx. Bayes - LR)",
        "accuracy": bin_metrics_approx["accuracy"],
        "precision": bin_metrics_approx["precision"],
        "recall": bin_metrics_approx["recall"],
        "f1": bin_metrics_approx["f1"],
    })

    if bayes_json_path and os.path.exists(bayes_json_path):
        try:
            bj = json.load(open(bayes_json_path, "r"))
            tm = bj.get("test_metrics", {})
            if tm:
                rows.append({
                    "model": "Binary Bayesian (from bayesian_results.json)",
                    "accuracy": tm.get("accuracy", [None])[0],
                    "precision": tm.get("precision", [None])[0],
                    "recall": tm.get("recall", [None])[0],
                    "f1": tm.get("f1", [None])[0],
                })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    return df


def main(args):
    csv_path = args.csv or "bayesiandata/cnn_probs_for_bayes.csv"
    df = load_cnn_probs(csv_path)

    cm, cnn_metrics, labels = compute_cnn_metrics(df)
    cm_out = os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png")
    plot_confusion_matrix(cm, labels, cm_out)
    print("Saved CNN confusion matrix to:", cm_out)

    fpr, tpr, roc_auc, bin_metrics_approx = fit_binary_approx(df)
    roc_out = os.path.join(RESULTS_DIR, "binary_roc_approx.png")
    plot_roc(fpr, tpr, roc_auc, roc_out)
    print("Saved binary ROC (approx) to:", roc_out)

    bayes_json = args.bayes_json or "results/bayesian_results.json"
    summary_df = build_summary_table(cnn_metrics, bin_metrics_approx, bayes_json_path=bayes_json)
    csv_out = os.path.join(RESULTS_DIR, "model_test_metrics_summary.csv")
    summary_df.to_csv(csv_out, index=False)
    print("Saved summary CSV to:", csv_out)

    try:
        fig, ax = plt.subplots(figsize=(8, len(summary_df) * 0.8 + 1))
        ax.axis('off')
        tbl = ax.table(cellText=np.round(summary_df.select_dtypes([np.number]).values, 4).tolist(),
                       colLabels=summary_df.columns, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        png_out = os.path.join(RESULTS_DIR, "model_test_metrics_summary.png")
        plt.savefig(png_out, bbox_inches='tight', dpi=200)
        plt.close()
        print("Saved summary table image to:", png_out)
    except Exception as e:
        print("Could not render table image:", e)

    print("All done. If you want the exact Bayesian ROC/AUC from the brms model, run `src/bayesian_model.R` in R and save the posterior predictions.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="Path to cnn probs CSV", default=None)
    p.add_argument("--bayes-json", help="Path to bayesian_results.json", default=None)
    args = p.parse_args()
    main(args)

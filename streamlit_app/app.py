import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


st.set_page_config(page_title="Alzheimer’s Detection", layout="wide")

st.title("Deep Learning & Bayesian Modeling for Early Alzheimer’s Detection")


# Tabs / Pages
tabs = st.tabs(["Landing", "Visualization"])


with tabs[0]:
    st.header("Project Overview")
    st.write(
        "This project compares a CNN classifier and Bayesian models for early detection of Alzheimer's using MRI-derived features and CNN predictions."
    )
    st.write("Highlights:")
    st.markdown(
        "- CNN classification on MRI images\n- Bayesian logistic / ordinal models trained on ROI features and CNN probabilities\n- Interactive visualizations of test performance"
    )
    st.subheader("Sample Images")
    sample_dir = "sample_imgs"
    if os.path.exists(sample_dir) and os.path.isdir(sample_dir):
        allowed = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [f for f in sorted(os.listdir(sample_dir)) if f.lower().endswith(allowed)]
        if len(files) == 0:
            st.info(f"No image files found in `{sample_dir}`.")
        else:
            max_show = st.slider("Images to show", min_value=1, max_value=min(12, len(files)), value=min(8, len(files)))
            cols = st.columns(min(4, max_show))
            for i, fname in enumerate(files[:max_show]):
                try:
                    img_path = os.path.join(sample_dir, fname)
                    img = Image.open(img_path)
                    with cols[i % len(cols)]:
                        st.image(img, caption=fname, use_container_width=True)
                except Exception as e:
                    st.write(f"Could not load {fname}: {e}")
    else:
        st.info(f"(Optional) Place example images in a `{sample_dir}/` folder to display them here.")



with tabs[1]:
    st.header("Interactive Visualization")
    st.write("Choose a visualization type and interact with the results below.")

    csv_path = "bayesiandata/cnn_probs_for_bayes.csv"
    df = None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Could not read {csv_path}: {e}")
    else:
        st.warning(f"CSV not found: {csv_path}")

    option = st.selectbox("Visualization", ["CNN Confusion Matrix", "Binary ROC (approx)", "Summary Table"])

    if option == "CNN Confusion Matrix":
        if df is None:
            st.error("CNN probability CSV is required to compute the confusion matrix.")
        else:
            # compute predicted labels
            p_cols = [c for c in df.columns if c.startswith("p_")]
            df["pred_idx"] = df[p_cols].values.argmax(axis=1)
            csv_order = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]
            mapping = {name: idx for idx, name in enumerate(csv_order)}
            df["true_idx"] = df["true_label"].map(mapping)

            cm = confusion_matrix(df["true_idx"].values, df["pred_idx"].values)

            normalize = st.checkbox("Normalize rows (per-true-class)")
            cm_display = cm.astype(float)
            if normalize:
                row_sums = cm_display.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cm_display = cm_display / row_sums

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm_display, cmap="Blues")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(csv_order)))
            ax.set_xticklabels(csv_order, rotation=45, ha="right")
            ax.set_yticks(range(len(csv_order)))
            ax.set_yticklabels(csv_order)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = cm[i, j]
                    txt = f"{val}" if not normalize else f"{cm_display[i,j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", color="black")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("CNN Test Confusion Matrix")
            st.pyplot(fig)

    elif option == "Binary ROC (approx)":
        roc_img = "results/binary_roc_approx.png"
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Binary ROC (approx)")
        else:
            st.info("No precomputed ROC image found; attempting to compute an approximate ROC from the CSV.")
            if df is None:
                st.error("CSV required to compute ROC.")
            else:
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import roc_curve, auc
                from sklearn.model_selection import train_test_split

                df2 = df.copy()
                df2["ad_binary"] = df2["true_label"].isin(["Mild Impairment", "Moderate Impairment"]).astype(int)
                X = df2[["p_mild_impairment", "p_moderate_impairment", "p_no_impairment", "p_very_mild_impairment"]].values
                y = df2["ad_binary"].values
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
                clf = LogisticRegression(solver="liblinear")
                clf.fit(X_tr, y_tr)
                y_proba = clf.predict_proba(X_te)[:, 1]
                fpr, tpr, _ = roc_curve(y_te, y_proba)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Binary ROC (approx.)")
                ax.legend()
                st.pyplot(fig)

    else: 
        summary_csv = "results/model_test_metrics_summary.csv"
        if os.path.exists(summary_csv):
            sdf = pd.read_csv(summary_csv)
            st.dataframe(sdf)
            st.download_button("Download summary CSV", sdf.to_csv(index=False), file_name="model_test_metrics_summary.csv")
        else:
            st.info("Summary CSV not found. Attempting to build a minimal summary from available data.")
            rows = []
            bj = "results/bayesian_results.json"
            if os.path.exists(bj):
                try:
                    bjd = pd.read_json(bj)
                    tm = bjd.get("test_metrics", {})
                    if tm:
                        rows.append({
                            "model": "Binary Bayesian (from JSON)",
                            "accuracy": tm.get("accuracy", [None])[0],
                            "precision": tm.get("precision", [None])[0],
                            "recall": tm.get("recall", [None])[0],
                            "f1": tm.get("f1", [None])[0],
                        })
                except Exception:
                    pass

            if df is not None:
                p_cols = [c for c in df.columns if c.startswith("p_")]
                df["pred_idx"] = df[p_cols].values.argmax(axis=1)
                csv_order = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]
                mapping = {name: idx for idx, name in enumerate(csv_order)}
                df["true_idx"] = df["true_label"].map(mapping)
                y_true = df["true_idx"].values
                y_pred = df["pred_idx"].values
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                rows.append({
                    "model": "CNN (from probs)",
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                })

            if rows:
                sdf = pd.DataFrame(rows)
                st.dataframe(sdf)
                st.download_button("Download summary CSV", sdf.to_csv(index=False), file_name="model_test_metrics_summary.csv")
            else:
                st.warning("No data available to build summary table.")

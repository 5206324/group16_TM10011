"""Centrale evaluatie-, rapportage- en plotfuncties voor nested CV."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def _get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model heeft geen predict_proba of decision_function.")


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_samples": len(y_true),
    }


def evaluate_split(model, X, y, split_name):
    y_true = np.asarray(y)
    y_pred = model.predict(X)
    y_prob = _get_probabilities(model, X)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    return metrics


def make_fold_record(
    fold_idx,
    classifier_name,
    model,
    best_params,
    inner_cv_score,
    selected_features,
    X_train,
    y_train,
    X_test,
    y_test,
):
    train_metrics = evaluate_split(model, X_train, y_train, "train")
    test_metrics = evaluate_split(model, X_test, y_test, "test")

    record = {
        "fold": fold_idx,
        "model_name": classifier_name,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "selected_features": list(selected_features),
        "n_features": len(selected_features),
        "best_params": dict(best_params),
        "inner_cv_score": inner_cv_score,
        "pipeline": model,
    }

    for key, value in train_metrics.items():
        record[f"train_{key}"] = value
    for key, value in test_metrics.items():
        record[f"test_{key}"] = value

    return record


def records_to_dataframe(fold_records):
    return pd.DataFrame(fold_records)


def summarize_classifiers(results_df):
    summary = (
        results_df.groupby("model_name")
        .agg(
            folds=("fold", "count"),
            mean_test_auc=("test_auc", "mean"),
            std_test_auc=("test_auc", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_test_f1=("test_f1", "mean"),
            std_test_f1=("test_f1", "std"),
            mean_test_sensitivity=("test_sensitivity", "mean"),
            mean_test_specificity=("test_specificity", "mean"),
            mean_test_ppv=("test_ppv", "mean"),
            mean_test_npv=("test_npv", "mean"),
            mean_n_features=("n_features", "mean"),
        )
        .reset_index()
        .sort_values(["mean_test_auc", "mean_test_accuracy"], ascending=False)
    )

    std_cols = [col for col in summary.columns if col.startswith("std_")]
    summary[std_cols] = summary[std_cols].fillna(0.0)
    return summary


def select_best_models(results_df):
    best_by_classifier = (
        results_df.sort_values(["test_auc", "test_accuracy"], ascending=False)
        .groupby("model_name", sort=False)
        .head(1)
        .copy()
    )

    best_overall = (
        results_df.sort_values(["test_auc", "test_accuracy"], ascending=False)
        .iloc[0]
        .copy()
    )

    return best_by_classifier, best_overall


def print_classifier_summary(summary_df):
    print("\n=== CLASSIFIER OVERZICHT (mean over 5 folds) ===")
    print(summary_df.round(4).to_string(index=False))


def print_best_model_report(best_record, summary_df):
    classifier_name = best_record["model_name"]
    classifier_summary = summary_df.loc[
        summary_df["model_name"] == classifier_name
    ].iloc[0]

    print("\n=== BESTE INDIVIDUELE FOLD-MODEL ===")
    print(f"Modelnaam: {best_record['model_name']}")
    print(f"Fold: {best_record['fold']}")
    print(f"Aantal train samples: {best_record['n_train_samples']}")
    print(f"Aantal test samples: {best_record['n_test_samples']}")
    print(f"Gekozen features: {best_record['selected_features']}")
    print(f"Aantal features: {best_record['n_features']}")
    print(f"Hyperparameters: {best_record['best_params']}")
    print(f"Inner CV score (AUC): {best_record['inner_cv_score']:.4f}")
    print("\nOp testdata toegepast:")
    print(f"AUC: {best_record['test_auc']:.4f}")
    print(f"Accuracy: {best_record['test_accuracy']:.4f}")
    print(f"Sensitivity/Recall: {best_record['test_sensitivity']:.4f}")
    print(f"Specificity: {best_record['test_specificity']:.4f}")
    print(f"PPV/Precision: {best_record['test_ppv']:.4f}")
    print(f"NPV: {best_record['test_npv']:.4f}")
    print(f"F1: {best_record['test_f1']:.4f}")
    print("\nOp traindata toegepast:")
    print(f"AUC: {best_record['train_auc']:.4f}")
    print(f"Accuracy: {best_record['train_accuracy']:.4f}")
    print(f"F1: {best_record['train_f1']:.4f}")
    print(
        f"\nSTD van 5 modellen van beste classifier ({classifier_name}) - "
        f"test AUC: {classifier_summary['std_test_auc']:.4f}"
    )
    print(
        f"STD van 5 modellen van beste classifier ({classifier_name}) - "
        f"test Accuracy: {classifier_summary['std_test_accuracy']:.4f}"
    )


def plot_train_test_roc(best_record):
    train_fpr, train_tpr, _ = roc_curve(
        best_record["train_y_true"], best_record["train_y_prob"]
    )
    test_fpr, test_tpr, _ = roc_curve(
        best_record["test_y_true"], best_record["test_y_prob"]
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        train_fpr,
        train_tpr,
        label=(
            f"Train ROC | AUC={best_record['train_auc']:.3f}, "
            f"Acc={best_record['train_accuracy']:.3f}"
        ),
    )
    plt.plot(
        test_fpr,
        test_tpr,
        label=(
            f"Test ROC | AUC={best_record['test_auc']:.3f}, "
            f"Acc={best_record['test_accuracy']:.3f}"
        ),
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Train vs Test ROC - Beste model ({best_record['model_name']})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_classifier_comparison(best_by_classifier, summary_df, classifiers=None):
    if classifiers is None:
        classifiers = ["LogisticRegression", "RandomForest", "XGBoost"]

    plt.figure(figsize=(8, 6))

    for classifier_name in classifiers:
        rows = best_by_classifier.loc[
            best_by_classifier["model_name"] == classifier_name
        ]
        if rows.empty:
            continue

        record = rows.iloc[0]
        summary_row = summary_df.loc[
            summary_df["model_name"] == classifier_name
        ].iloc[0]
        fpr, tpr, _ = roc_curve(
            record["test_y_true"], record["test_y_prob"]
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{classifier_name} | best fold {int(record['fold'])} | "
                f"AUC={record['test_auc']:.3f} | "
                f"Acc={record['test_accuracy']:.3f} | "
                f"stdAUC5={summary_row['std_test_auc']:.3f}"
            ),
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC van beste individuele fold-model per classifier")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

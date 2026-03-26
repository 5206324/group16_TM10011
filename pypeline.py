# %% --- Step 1: Data exploration ---
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path.cwd()))

from Stap_1_Data_inladen import data_lipo
from Stap_1b_visualisatie import plot_baseline_comparison
from Stap_2_kfolds_splitsing import kfold
from Stap_3_inner_loop import inner_loop
from Stap_7_Data_verzameling import (
    make_fold_record,
    plot_classifier_comparison,
    plot_train_test_roc,
    print_best_model_report,
    print_classifier_summary,
    records_to_dataframe,
    select_best_models,
    summarize_classifiers,
)


# Data inladen
data = data_lipo("Lipo_radiomicFeatures.csv")
if "label" in data.columns:
    data = data.set_index(data.columns[0])


# %% --- Step 1b: Baseline visualisatie ---
X_baseline = data.select_dtypes(include=[np.number])
if "label" in X_baseline.columns:
    X_baseline = X_baseline.drop(columns=["label"])
y_baseline = data["label"].map({"lipoma": 0, "liposarcoma": 1})

plot_baseline_comparison(X_baseline, y_baseline)


# %% --- Step 2: Outer loop ---
le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])
print(f"Klassen succesvol omgezet: {dict(zip(le.classes_, le.transform(le.classes_)))}")

folds = kfold(data, target_column="label", n_splits=5)
alle_fold_records = []
alle_fold_features = []

for i, pakketje in enumerate(folds, start=1):
    print(f"\n--- Fold {i} ---")
    X_train_outer, X_test_outer = pakketje["X_train"], pakketje["X_test"]
    y_train_outer, y_test_outer = pakketje["y_train"], pakketje["y_test"]

    beste_model, naam, best_params, inner_cv_score = inner_loop(
        X_train_outer, y_train_outer
    )

    rfecv_stap = beste_model.named_steps["rfecv"]
    filter_stap = beste_model.named_steps["feature_filter"]
    namen_na_filter = filter_stap.columns_to_keep_
    winnende_features = [
        namen_na_filter[j] for j in range(len(namen_na_filter)) if rfecv_stap.support_[j]
    ]

    df_fold = pd.DataFrame(
        {
            "Fold": i,
            "Model": naam,
            "Feature": winnende_features,
        }
    )
    alle_fold_features.append(df_fold)

    fold_record = make_fold_record(
        fold_idx=i,
        classifier_name=naam,
        model=beste_model,
        best_params=best_params,
        inner_cv_score=inner_cv_score,
        selected_features=winnende_features,
        X_train=X_train_outer,
        y_train=y_train_outer,
        X_test=X_test_outer,
        y_test=y_test_outer,
    )
    alle_fold_records.append(fold_record)

    print(f"Fold {i} klaar ({naam})")
    print(f"Optimaal aantal features: {rfecv_stap.n_features_}")
    print(f"Winnende features: {winnende_features}")

    scores = rfecv_stap.cv_results_["mean_test_score"]
    x_range = np.arange(
        rfecv_stap.min_features_to_select,
        rfecv_stap.min_features_to_select + len(scores) * rfecv_stap.step,
        rfecv_stap.step,
    )

    plt.figure(figsize=(8, 4))
    plt.plot(
        x_range,
        scores,
        color="#2E86C1",
        marker="o",
        markersize=4,
        label="Mean AUC",
    )
    plt.axvline(
        x=rfecv_stap.n_features_,
        color="#E74C3C",
        linestyle="--",
        label=f"Optimum: {rfecv_stap.n_features_}",
    )
    plt.title(f"RFECV Curve - Fold {i} ({naam})")
    plt.xlabel("Aantal Features")
    plt.ylabel("AUC Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %% --- Step 3: Feature stabiliteit ---
df_eindrapport = pd.concat(alle_fold_features, ignore_index=True)
print("\n--- OVERZICHT ALLE FOLDS ---")
print(df_eindrapport)

frequentie_tabel = df_eindrapport["Feature"].value_counts().reset_index()
frequentie_tabel.columns = ["Radiomics Feature", "Aantal keer gekozen (max 5)"]
print("\n--- STABILITEITS-ANALYSE ---")
print(frequentie_tabel)


# %% --- Step 4: Centrale modelrapportage ---
results_df = records_to_dataframe(alle_fold_records)
summary_df = summarize_classifiers(results_df)
best_by_classifier, best_overall = select_best_models(results_df)

print_classifier_summary(summary_df)
print_best_model_report(best_overall, summary_df)

plot_train_test_roc(best_overall)
plot_classifier_comparison(best_by_classifier, summary_df)

# %%
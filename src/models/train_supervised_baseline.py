import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ============================================================
# LOAD DATA
# ============================================================

feature_files = sorted(
    glob.glob("features_*_extended/jet_substructure_features_extended_quark_gluon_only.csv"),
    key=os.path.getmtime
)

if len(feature_files) == 0:
    raise FileNotFoundError(
        "No extended feature file found. "
        "Run calculate_extended_features.py first."
    )

feature_file = feature_files[-1]
print("Using extended feature file:", feature_file)

df = pd.read_csv(feature_file)

print("\nOriginal rows:", len(df))
print("Truth label counts:")
print(df["truth_label"].value_counts())


# ============================================================
# CHOOSE FEATURES
# ============================================================

FEATURE_COLUMNS = [
    # Original variables
    "jet_mass",
    "multiplicity",
    "ptD",
    "N95",

    # New variables
    "jet_width",
    "leading_constituent_pt_fraction",
    "charged_multiplicity",
    "neutral_multiplicity",
    "mean_constituent_pt",
    "max_constituent_pt"
]

required_columns = FEATURE_COLUMNS + ["truth_label"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df.dropna(subset=required_columns).copy()
df = df[df["truth_label"].isin(["quark", "gluon"])].copy()

# Convert labels into numbers:
# quark = 0
# gluon = 1
df["label"] = df["truth_label"].map({
    "quark": 0,
    "gluon": 1
})

X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
y = df["label"].to_numpy(dtype=int)

print("\nUsable rows:", len(df))
print("\nFeatures being used:")
for f in FEATURE_COLUMNS:
    print(" -", f)

print("\nLabel counts:")
print(df["truth_label"].value_counts())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X,
    y,
    df,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

print("\nTraining labels:")
print(df_train["truth_label"].value_counts())

print("\nTesting labels:")
print(df_test["truth_label"].value_counts())


# ============================================================
# DEFINE MODELS
# ============================================================

models = {
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    ),

    # This random forest is intentionally less aggressive than before
    # to reduce overfitting.
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced"
    ),

    "simple_neural_network": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )
    )
}


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_dir = "supervised_extended_results"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

summary_rows = []

for model_name, model in models.items():
    print("\n" + "=" * 70)
    print("Training model:", model_name)
    print("=" * 70)

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("Training accuracy:", train_acc)
    print("Testing accuracy:", test_acc)

    cm = confusion_matrix(y_test, test_pred)

    print("\nConfusion matrix on TEST set:")
    print(cm)

    print("\nClassification report on TEST set:")
    print(classification_report(
        y_test,
        test_pred,
        target_names=["quark", "gluon"]
    ))

    if hasattr(model, "predict_proba"):
        test_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, test_prob)
        print("Test ROC AUC:", auc)
    else:
        test_prob = None
        auc = np.nan

    summary_rows.append({
        "model": model_name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_auc": auc
    })

    # Save test predictions
    result_df = df_test.copy()
    result_df["true_label_number"] = y_test
    result_df["predicted_label_number"] = test_pred
    result_df["predicted_label"] = np.where(test_pred == 1, "gluon", "quark")
    result_df["correct"] = result_df["truth_label"] == result_df["predicted_label"]

    if test_prob is not None:
        result_df["predicted_gluon_probability"] = test_prob

    result_path = os.path.join(output_dir, f"{model_name}_test_predictions.csv")
    result_df.to_csv(result_path, index=False)

    print("Saved predictions:", result_path)

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["quark", "gluon"]
    )

    disp.plot()
    plt.title(f"{model_name} confusion matrix")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print("Saved confusion matrix:", cm_path)

    # ROC curve
    if test_prob is not None:
        RocCurveDisplay.from_predictions(
            y_test,
            test_prob,
            name=model_name
        )

        plt.title(f"{model_name} ROC curve")
        plt.tight_layout()

        roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
        plt.savefig(roc_path, dpi=200)
        plt.close()

        print("Saved ROC curve:", roc_path)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(output_dir, "model_summary.csv")
summary.to_csv(summary_path, index=False)

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)
print(summary)

print("\nSaved summary:", summary_path)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_model = models["random_forest"]
rf_importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": rf_importances
}).sort_values("importance", ascending=False)

importance_path = os.path.join(output_dir, "random_forest_feature_importance.csv")
importance_df.to_csv(importance_path, index=False)

print("\nRandom forest feature importance:")
print(importance_df)
print("Saved:", importance_path)

plt.figure(figsize=(9, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Random forest feature importance")
plt.gca().invert_yaxis()
plt.tight_layout()

importance_plot_path = os.path.join(output_dir, "random_forest_feature_importance.png")
plt.savefig(importance_plot_path, dpi=200)
plt.close()

print("Saved:", importance_plot_path)


# ============================================================
# SIMPLE SCATTER PLOTS FOR BEST MODEL
# ============================================================

best_model_name = summary.sort_values("test_auc", ascending=False).iloc[0]["model"]
best_predictions_path = os.path.join(output_dir, f"{best_model_name}_test_predictions.csv")
best_df = pd.read_csv(best_predictions_path)

print("\nBest model by test AUC:", best_model_name)

# Plot multiplicity vs pT_D colored by prediction
plt.figure(figsize=(8, 6))

for label in ["quark", "gluon"]:
    sub = best_df[best_df["predicted_label"] == label]
    plt.scatter(
        sub["multiplicity"],
        sub["ptD"],
        s=10,
        alpha=0.5,
        label=f"predicted {label}"
    )

plt.xlabel("Multiplicity")
plt.ylabel("pT_D")
plt.title(f"Multiplicity vs pT_D, colored by {best_model_name} prediction")
plt.legend()
plt.tight_layout()

path = os.path.join(output_dir, f"{best_model_name}_multiplicity_vs_ptD_prediction.png")
plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)


# Plot jet width vs leading pT fraction colored by truth
plt.figure(figsize=(8, 6))

for label in ["quark", "gluon"]:
    sub = best_df[best_df["truth_label"] == label]
    plt.scatter(
        sub["jet_width"],
        sub["leading_constituent_pt_fraction"],
        s=10,
        alpha=0.5,
        label=f"true {label}"
    )

plt.xlabel("Jet width")
plt.ylabel("Leading constituent pT fraction")
plt.title("Jet width vs leading constituent pT fraction, colored by truth")
plt.legend()
plt.tight_layout()

path = os.path.join(output_dir, "jet_width_vs_leading_pt_fraction_truth.png")
plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)


# Plot correct vs wrong for best model
plt.figure(figsize=(8, 6))

correct = best_df[best_df["correct"] == True]
wrong = best_df[best_df["correct"] == False]

plt.scatter(
    correct["jet_width"],
    correct["leading_constituent_pt_fraction"],
    s=10,
    alpha=0.35,
    label="correct"
)

plt.scatter(
    wrong["jet_width"],
    wrong["leading_constituent_pt_fraction"],
    s=14,
    alpha=0.75,
    label="wrong"
)

plt.xlabel("Jet width")
plt.ylabel("Leading constituent pT fraction")
plt.title(f"Correct vs wrong predictions: {best_model_name}")
plt.legend()
plt.tight_layout()

path = os.path.join(output_dir, f"{best_model_name}_correct_vs_wrong_width_leading_fraction.png")
plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)


print("\nDone.")
print("Results folder:", output_dir)
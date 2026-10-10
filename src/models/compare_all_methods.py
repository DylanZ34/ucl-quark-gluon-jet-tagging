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
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ============================================================
# SETTINGS
# ============================================================

# Step 3 update:
# jet_eta is now included as an input variable.
# I also included jet_pt because jet kinematics are useful for comparison.
# If your professor only wants eta, you can remove "jet_pt".
FEATURE_COLUMNS = [
    "jet_pt",
    "jet_eta",
    "jet_mass",
    "multiplicity",
    "ptD",
    "N95",
    "jet_width",
    "leading_constituent_pt_fraction",
    "charged_multiplicity",
    "neutral_multiplicity",
    "mean_constituent_pt",
    "max_constituent_pt"
]

OUTPUT_DIR = "method_comparison_results_with_eta"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42


# ============================================================
# LOAD EXTENDED DATA
# ============================================================

feature_files = sorted(
    glob.glob("features_*_extended/jet_substructure_features_extended_quark_gluon_only.csv"),
    key=os.path.getmtime
)

if len(feature_files) == 0:
    raise FileNotFoundError(
        "No extended feature file found. Run calculate_extended_features.py first."
    )

feature_file = feature_files[-1]
print("Using feature file:", feature_file)

df = pd.read_csv(feature_file)

print("\nOriginal rows:", len(df))
print("Truth label counts:")
print(df["truth_label"].value_counts())

required_columns = FEATURE_COLUMNS + ["truth_label"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Missing column: {col}\n"
            "Check that your extended feature file still contains jet_pt and jet_eta."
        )

df = df.dropna(subset=required_columns).copy()
df = df[df["truth_label"].isin(["quark", "gluon"])].copy()

df["label"] = df["truth_label"].map({
    "quark": 0,
    "gluon": 1
})

X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
y = df["label"].to_numpy(dtype=int)

print("\nUsable rows:", len(df))
print("\nFeatures used:")
for col in FEATURE_COLUMNS:
    print(" -", col)

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
    random_state=RANDOM_SEED,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

print("\nTraining labels:")
print(df_train["truth_label"].value_counts())

print("\nTesting labels:")
print(df_test["truth_label"].value_counts())


# ============================================================
# HELPER FUNCTIONS
# ============================================================

summary_rows = []
roc_data = {}


def save_result(model_name, y_true, y_pred, y_score=None):
    accuracy = accuracy_score(y_true, y_pred)

    if y_score is not None:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = np.nan

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print("Test accuracy:", accuracy)
    print("Test AUC:", auc)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["quark", "gluon"]
    ))

    summary_rows.append({
        "method": model_name,
        "test_accuracy": accuracy,
        "test_auc": auc
    })

    if y_score is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        roc_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc
        }


# ============================================================
# 1. UNSUPERVISED K-MEANS
# ============================================================

print("\nRunning K-means...")

scaler_kmeans = StandardScaler()
X_train_scaled = scaler_kmeans.fit_transform(X_train)
X_test_scaled = scaler_kmeans.transform(X_test)

kmeans = KMeans(
    n_clusters=2,
    n_init=50,
    random_state=RANDOM_SEED
)

kmeans.fit(X_train_scaled)

train_clusters = kmeans.predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

# Decide which cluster is more gluon-like using TRAINING labels only.
cluster_gluon_fraction = {}

for cluster_id in [0, 1]:
    y_in_cluster = y_train[train_clusters == cluster_id]

    if len(y_in_cluster) == 0:
        cluster_gluon_fraction[cluster_id] = 0.0
    else:
        cluster_gluon_fraction[cluster_id] = np.mean(y_in_cluster == 1)

gluon_cluster = max(cluster_gluon_fraction, key=cluster_gluon_fraction.get)
quark_cluster = 1 - gluon_cluster

print("\nK-means cluster gluon fractions from TRAIN set:")
print(cluster_gluon_fraction)
print("Quark-like cluster:", quark_cluster)
print("Gluon-like cluster:", gluon_cluster)

# Convert clusters into quark/gluon predictions
kmeans_pred = np.where(test_clusters == gluon_cluster, 1, 0)

# Distance-based score for ROC/AUC:
# higher score = more gluon-like
distances = kmeans.transform(X_test_scaled)
kmeans_score = distances[:, quark_cluster] - distances[:, gluon_cluster]

save_result(
    model_name="K-means",
    y_true=y_test,
    y_pred=kmeans_pred,
    y_score=kmeans_score
)


# ============================================================
# 2. LOGISTIC REGRESSION
# ============================================================

print("\nRunning logistic regression...")

logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
)

logistic.fit(X_train, y_train)

logistic_pred = logistic.predict(X_test)
logistic_score = logistic.predict_proba(X_test)[:, 1]

save_result(
    model_name="Logistic regression",
    y_true=y_test,
    y_pred=logistic_pred,
    y_score=logistic_score
)


# ============================================================
# 3. RANDOM FOREST
# ============================================================

print("\nRunning random forest...")

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    random_state=RANDOM_SEED,
    class_weight="balanced"
)

random_forest.fit(X_train, y_train)

rf_pred = random_forest.predict(X_test)
rf_score = random_forest.predict_proba(X_test)[:, 1]

save_result(
    model_name="Random forest",
    y_true=y_test,
    y_pred=rf_pred,
    y_score=rf_score
)


# ============================================================
# 4. FEED-FORWARD NEURAL NETWORK ARCHITECTURES
# ============================================================

# These are the architectures your professor suggested.
MLP_ARCHITECTURES = {
    "Feed-forward NN 32-32": (32, 32),
    "Feed-forward NN 64-64": (64, 64),
    "Feed-forward NN 10-10-10-10": (10, 10, 10, 10)
}

for model_name, architecture in MLP_ARCHITECTURES.items():
    print(f"\nRunning {model_name}...")

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=architecture,
            activation="relu",
            solver="adam",
            alpha=0.0001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_SEED
        )
    )

    mlp.fit(X_train, y_train)

    mlp_pred = mlp.predict(X_test)
    mlp_score = mlp.predict_proba(X_test)[:, 1]

    save_result(
        model_name=model_name,
        y_true=y_test,
        y_pred=mlp_pred,
        y_score=mlp_score
    )


# ============================================================
# 5. TRANSFORMER CONSTITUENT MODEL
# ============================================================

print("\nLooking for transformer results...")

transformer_pred_files = sorted(
    glob.glob("transformer_constituent_results/transformer_test_predictions.csv"),
    key=os.path.getmtime
)

transformer_summary_files = sorted(
    glob.glob("transformer_constituent_results/transformer_summary.csv"),
    key=os.path.getmtime
)

if len(transformer_pred_files) > 0:
    transformer_pred_file = transformer_pred_files[-1]
    print("Using transformer prediction file:", transformer_pred_file)

    transformer_df = pd.read_csv(transformer_pred_file)

    needed_cols = [
        "true_label_number",
        "predicted_label_number",
        "predicted_gluon_probability"
    ]

    for col in needed_cols:
        if col not in transformer_df.columns:
            raise ValueError(
                "Transformer prediction file is missing column: " + col
            )

    transformer_y_true = transformer_df["true_label_number"].to_numpy(dtype=int)
    transformer_y_pred = transformer_df["predicted_label_number"].to_numpy(dtype=int)
    transformer_score = transformer_df["predicted_gluon_probability"].to_numpy(dtype=float)

    save_result(
        model_name="Transformer constituents",
        y_true=transformer_y_true,
        y_pred=transformer_y_pred,
        y_score=transformer_score
    )

elif len(transformer_summary_files) > 0:
    transformer_summary_file = transformer_summary_files[-1]
    print("Using transformer summary file:", transformer_summary_file)

    transformer_summary = pd.read_csv(transformer_summary_file)
    row = transformer_summary.iloc[0]

    method_name = "Transformer constituents"

    if "method" in transformer_summary.columns:
        method_name = row["method"]

    summary_rows.append({
        "method": method_name,
        "test_accuracy": row["test_accuracy"],
        "test_auc": row["test_auc"]
    })

    print("Added transformer summary, but no ROC curve because prediction file was not found.")

else:
    print(
        "No transformer results found. "
        "Run train_transformer_constituents.py first if you want transformer included."
    )


# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary = pd.DataFrame(summary_rows)
summary = summary.sort_values("test_auc", ascending=False)

summary_path = os.path.join(OUTPUT_DIR, "all_methods_summary_with_eta.csv")
summary.to_csv(summary_path, index=False)

print("\n" + "=" * 70)
print("ALL METHODS SUMMARY")
print("=" * 70)
print(summary)
print("\nSaved:", summary_path)


# ============================================================
# PLOT 1: TEST ACCURACY COMPARISON
# ============================================================

plot_df = summary.sort_values("test_accuracy", ascending=True)

plt.figure(figsize=(11, 7))
plt.barh(plot_df["method"], plot_df["test_accuracy"])

for i, value in enumerate(plot_df["test_accuracy"]):
    plt.text(value + 0.005, i, f"{value:.3f}", va="center")

plt.xlabel("Test accuracy")
plt.ylabel("Method")
plt.title("Quark/gluon classification: test accuracy comparison with jet eta")
plt.xlim(0, 1)
plt.tight_layout()

accuracy_plot = os.path.join(OUTPUT_DIR, "all_methods_test_accuracy_comparison_with_eta.png")
plt.savefig(accuracy_plot, dpi=200)
plt.close()

print("Saved:", accuracy_plot)


# ============================================================
# PLOT 2: TEST AUC COMPARISON
# ============================================================

plot_df = summary.sort_values("test_auc", ascending=True)

plt.figure(figsize=(11, 7))
plt.barh(plot_df["method"], plot_df["test_auc"])

for i, value in enumerate(plot_df["test_auc"]):
    plt.text(value + 0.005, i, f"{value:.3f}", va="center")

plt.xlabel("Test AUC")
plt.ylabel("Method")
plt.title("Quark/gluon classification: test AUC comparison with jet eta")
plt.xlim(0, 1)
plt.tight_layout()

auc_plot = os.path.join(OUTPUT_DIR, "all_methods_test_auc_comparison_with_eta.png")
plt.savefig(auc_plot, dpi=200)
plt.close()

print("Saved:", auc_plot)


# ============================================================
# PLOT 3: COMBINED ROC CURVES
# ============================================================

if len(roc_data) > 0:
    plt.figure(figsize=(9, 7))

    for method_name, values in roc_data.items():
        plt.plot(
            values["fpr"],
            values["tpr"],
            label=f"{method_name} AUC={values['auc']:.3f}"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="random guessing")

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for all methods with jet eta")
    plt.legend()
    plt.tight_layout()

    roc_plot = os.path.join(OUTPUT_DIR, "all_methods_roc_curves_with_eta.png")
    plt.savefig(roc_plot, dpi=200)
    plt.close()

    print("Saved:", roc_plot)


# ============================================================
# PLOT 4: ACCURACY AND AUC SIDE-BY-SIDE
# ============================================================

plot_df = summary.sort_values("test_auc", ascending=False).copy()

x = np.arange(len(plot_df))
width = 0.35

plt.figure(figsize=(12, 7))

plt.bar(x - width / 2, plot_df["test_accuracy"], width, label="test accuracy")
plt.bar(x + width / 2, plot_df["test_auc"], width, label="test AUC")

plt.xticks(x, plot_df["method"], rotation=25, ha="right")
plt.ylabel("Score")
plt.title("Comparison of methods with jet eta included")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

combined_plot = os.path.join(OUTPUT_DIR, "all_methods_accuracy_and_auc_side_by_side_with_eta.png")
plt.savefig(combined_plot, dpi=200)
plt.close()

print("Saved:", combined_plot)


print("\nDone.")
print("Results saved in:", OUTPUT_DIR)
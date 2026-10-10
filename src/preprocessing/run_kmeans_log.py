import os
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

K = 2
RANDOM_SEED = 42
MAX_ITER = 300

# These are the transformed variables used for k-means.
# We use log(1+x) for variables with long tails.
FEATURE_COLUMNS = [
    "log_jet_mass",
    "log_multiplicity",
    "ptD",
    "log_N95"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_log1p(x):
    """
    Computes log(1+x), but clips negative values to 0 first.
    This avoids problems if a tiny numerical negative value appears.
    """
    return np.log1p(np.clip(x, 0, None))


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std[std == 0] = 1.0

    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def run_kmeans(X, k=2, max_iter=300, seed=42):
    rng = np.random.RandomState(seed)

    initial_indices = rng.choice(len(X), size=k, replace=False)
    centers = X[initial_indices].copy()

    for iteration in range(max_iter):
        distances = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        new_centers = np.zeros_like(centers)

        for cluster_id in range(k):
            points = X[labels == cluster_id]

            if len(points) == 0:
                new_centers[cluster_id] = X[rng.randint(len(X))]
            else:
                new_centers[cluster_id] = points.mean(axis=0)

        if np.allclose(centers, new_centers):
            print("K-means converged after", iteration + 1, "iterations")
            centers = new_centers
            break

        centers = new_centers

    return labels, centers


def pca_2d(X):
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_pca = X_centered @ Vt[:2].T

    explained_variance = (S ** 2) / (len(X) - 1)
    explained_ratio = explained_variance / explained_variance.sum()

    return X_pca, explained_ratio[:2]


# ============================================================
# LOAD NEWEST FEATURE FILE
# ============================================================

feature_files = sorted(
    glob.glob("/home/cmsusr/features_*/jet_substructure_features_quark_gluon_only.csv"),
    key=os.path.getmtime
)

if len(feature_files) == 0:
    raise FileNotFoundError(
        "No jet_substructure_features_quark_gluon_only.csv file found. "
        "Run calculate_substructure_features.py first."
    )

feature_file = feature_files[-1]
print("Using feature file:", feature_file)

df = pd.read_csv(feature_file)

print("Number of jets in feature file:", len(df))
print("Truth label counts:")
print(df["truth_label"].value_counts())


# ============================================================
# CREATE LOG-TRANSFORMED VARIABLES
# ============================================================

required_columns = [
    "jet_mass",
    "multiplicity",
    "ptD",
    "N95",
    "truth_label"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError("Missing column: " + col)

df = df.dropna(subset=required_columns).copy()
df = df[df["truth_label"].isin(["quark", "gluon"])].copy()

df["log_jet_mass"] = safe_log1p(df["jet_mass"].to_numpy(dtype=float))
df["log_multiplicity"] = safe_log1p(df["multiplicity"].to_numpy(dtype=float))
df["log_N95"] = safe_log1p(df["N95"].to_numpy(dtype=float))

print("\nUsing these k-means features:")
print(FEATURE_COLUMNS)

print("\nAfter cleaning:")
print("Number of usable jets:", len(df))
print(df["truth_label"].value_counts())

X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

# Standardize after log-transforming
X_scaled, feature_mean, feature_std = standardize(X)


# ============================================================
# RUN K-MEANS
# ============================================================

cluster_labels, centers_scaled = run_kmeans(
    X_scaled,
    k=K,
    max_iter=MAX_ITER,
    seed=RANDOM_SEED
)

df["kmeans_cluster"] = cluster_labels

print("\nCluster counts:")
print(df["kmeans_cluster"].value_counts().sort_index())


# ============================================================
# COMPARE CLUSTERS TO TRUTH LABELS
# ============================================================

comparison = pd.crosstab(df["kmeans_cluster"], df["truth_label"])

print("\nCluster vs truth label table:")
print(comparison)

cluster_majority_label = {}

for cluster_id in sorted(df["kmeans_cluster"].unique()):
    subset = df[df["kmeans_cluster"] == cluster_id]
    majority = subset["truth_label"].value_counts().idxmax()
    cluster_majority_label[cluster_id] = majority

df["cluster_majority_label"] = df["kmeans_cluster"].map(cluster_majority_label)
df["cluster_matches_truth"] = df["cluster_majority_label"] == df["truth_label"]

accuracy = df["cluster_matches_truth"].mean()

print("\nCluster majority labels:")
print(cluster_majority_label)

print("\nApproximate cluster agreement with truth:")
print(accuracy)


# ============================================================
# OUTPUT FOLDER
# ============================================================

base_dir = os.path.dirname(feature_file)
output_dir = os.path.join(base_dir, "kmeans_log_results")
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(output_dir, "kmeans_log_clustered_jets.csv")
df.to_csv(output_csv, index=False)

print("\nSaved clustered table:")
print(output_csv)


# ============================================================
# PCA VISUALIZATION
# ============================================================

X_pca, explained = pca_2d(X_scaled)

df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

print("\nPCA explained variance ratios:")
print("PCA 1:", explained[0])
print("PCA 2:", explained[1])


# Plot 1: PCA colored by k-means cluster
plt.figure(figsize=(8, 6))

for cluster_id in sorted(df["kmeans_cluster"].unique()):
    sub = df[df["kmeans_cluster"] == cluster_id]
    plt.scatter(
        sub["pca1"],
        sub["pca2"],
        s=10,
        alpha=0.6,
        label="cluster " + str(cluster_id)
    )

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("K-means with log-transformed variables")
plt.legend()
plt.tight_layout()

cluster_plot = os.path.join(output_dir, "pca_colored_by_kmeans_log_cluster.png")
plt.savefig(cluster_plot, dpi=200)
plt.close()

print("Saved:", cluster_plot)


# Plot 2: PCA colored by truth label
plt.figure(figsize=(8, 6))

for label in ["quark", "gluon"]:
    sub = df[df["truth_label"] == label]
    plt.scatter(
        sub["pca1"],
        sub["pca2"],
        s=10,
        alpha=0.6,
        label=label
    )

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Truth labels in log-transformed PCA space")
plt.legend()
plt.tight_layout()

truth_plot = os.path.join(output_dir, "pca_colored_by_truth_label_log.png")
plt.savefig(truth_plot, dpi=200)
plt.close()

print("Saved:", truth_plot)


# Plot 3: log multiplicity vs ptD, colored by truth
plt.figure(figsize=(8, 6))

for label in ["quark", "gluon"]:
    sub = df[df["truth_label"] == label]
    plt.scatter(
        sub["log_multiplicity"],
        sub["ptD"],
        s=10,
        alpha=0.6,
        label=label
    )

plt.xlabel("log(1 + multiplicity)")
plt.ylabel("pT_D")
plt.title("log multiplicity vs pT_D, colored by truth")
plt.legend()
plt.tight_layout()

simple_truth_plot = os.path.join(output_dir, "log_multiplicity_vs_ptD_truth.png")
plt.savefig(simple_truth_plot, dpi=200)
plt.close()

print("Saved:", simple_truth_plot)


# Plot 4: log multiplicity vs ptD, colored by k-means cluster
plt.figure(figsize=(8, 6))

for cluster_id in sorted(df["kmeans_cluster"].unique()):
    sub = df[df["kmeans_cluster"] == cluster_id]
    plt.scatter(
        sub["log_multiplicity"],
        sub["ptD"],
        s=10,
        alpha=0.6,
        label="cluster " + str(cluster_id)
    )

plt.xlabel("log(1 + multiplicity)")
plt.ylabel("pT_D")
plt.title("log multiplicity vs pT_D, colored by k-means")
plt.legend()
plt.tight_layout()

simple_cluster_plot = os.path.join(output_dir, "log_multiplicity_vs_ptD_kmeans.png")
plt.savefig(simple_cluster_plot, dpi=200)
plt.close()

print("Saved:", simple_cluster_plot)


# ============================================================
# SAVE PCA VERSION
# ============================================================

output_pca_csv = os.path.join(output_dir, "kmeans_log_clustered_jets_with_pca.csv")
df.to_csv(output_pca_csv, index=False)

print("Saved:", output_pca_csv)


# ============================================================
# SAVE CLUSTER CENTERS
# ============================================================

centers_log_units = centers_scaled * feature_std + feature_mean

centers_df = pd.DataFrame(
    centers_log_units,
    columns=FEATURE_COLUMNS
)

centers_df["cluster"] = range(K)
centers_df = centers_df[["cluster"] + FEATURE_COLUMNS]

centers_file = os.path.join(output_dir, "kmeans_log_cluster_centers.csv")
centers_df.to_csv(centers_file, index=False)

print("\nCluster centers in transformed feature space:")
print(centers_df)
print("Saved:", centers_file)


# Convert log centers back to approximate original units
centers_original = centers_df.copy()
centers_original["approx_jet_mass"] = np.expm1(centers_original["log_jet_mass"])
centers_original["approx_multiplicity"] = np.expm1(centers_original["log_multiplicity"])
centers_original["approx_N95"] = np.expm1(centers_original["log_N95"])
centers_original["approx_ptD"] = centers_original["ptD"]

centers_original = centers_original[
    ["cluster", "approx_jet_mass", "approx_multiplicity", "approx_ptD", "approx_N95"]
]

centers_original_file = os.path.join(output_dir, "kmeans_log_cluster_centers_approx_original_units.csv")
centers_original.to_csv(centers_original_file, index=False)

print("\nApproximate cluster centers in original units:")
print(centers_original)
print("Saved:", centers_original_file)

print("\nDone.")
print("Log-transformed k-means results folder:")
print(output_dir)

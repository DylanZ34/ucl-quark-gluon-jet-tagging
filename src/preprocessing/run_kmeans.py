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
FEATURE_COLUMNS = ["jet_mass", "multiplicity", "ptD", "N95"]
RANDOM_SEED = 42
MAX_ITER = 300


# ============================================================
# BASIC K-MEANS IMPLEMENTATION
# This avoids depending on scikit-learn inside the CMS container.
# ============================================================

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # avoid division by zero
    std[std == 0] = 1.0

    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def run_kmeans(X, k=2, max_iter=300, seed=42):
    rng = np.random.RandomState(seed)

    # randomly choose initial centers from data points
    initial_indices = rng.choice(len(X), size=k, replace=False)
    centers = X[initial_indices].copy()

    for iteration in range(max_iter):
        # distance from each point to each center
        distances = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))

        # assign each point to nearest center
        labels = np.argmin(distances, axis=1)

        # recompute centers
        new_centers = np.zeros_like(centers)
        for cluster_id in range(k):
            points = X[labels == cluster_id]

            if len(points) == 0:
                # if a cluster becomes empty, reinitialize it randomly
                new_centers[cluster_id] = X[rng.randint(len(X))]
            else:
                new_centers[cluster_id] = points.mean(axis=0)

        # stop if centers stop moving
        if np.allclose(centers, new_centers):
            print("K-means converged after", iteration + 1, "iterations")
            break

        centers = new_centers

    return labels, centers


def pca_2d(X):
    """
    Simple PCA using SVD.
    Returns the first two principal components.
    """
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
# CLEAN DATA
# ============================================================

needed_columns = FEATURE_COLUMNS + ["truth_label"]

for col in needed_columns:
    if col not in df.columns:
        raise ValueError("Missing column: " + col)

df = df.dropna(subset=FEATURE_COLUMNS + ["truth_label"]).copy()

# Keep only quark and gluon labels
df = df[df["truth_label"].isin(["quark", "gluon"])].copy()

print("\nAfter cleaning:")
print("Number of usable jets:", len(df))
print(df["truth_label"].value_counts())

X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

# Standardize features so mass, multiplicity, ptD, and N95 are on comparable scales
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
# This is evaluation only. The truth labels were NOT used by k-means.
# ============================================================

comparison = pd.crosstab(df["kmeans_cluster"], df["truth_label"])

print("\nCluster vs truth label table:")
print(comparison)

# Assign each cluster a majority label
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
# MAKE OUTPUT FOLDER
# ============================================================

base_dir = os.path.dirname(feature_file)
output_dir = os.path.join(base_dir, "kmeans_results")
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(output_dir, "kmeans_clustered_jets.csv")
df.to_csv(output_csv, index=False)

print("\nSaved clustered feature table:")
print(output_csv)


# ============================================================
# PCA VISUALIZATION
# ============================================================

X_pca, explained = pca_2d(X_scaled)

df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

# Plot 1: PCA colored by k-means cluster
plt.figure(figsize=(8, 6))

for cluster_id in sorted(df["kmeans_cluster"].unique()):
    sub = df[df["kmeans_cluster"] == cluster_id]
    plt.scatter(
        sub["pca1"],
        sub["pca2"],
        s=10,
        alpha=0.6,
        label=f"cluster {cluster_id}"
    )

plt.xlabel(f"PCA 1")
plt.ylabel(f"PCA 2")
plt.title("K-means clusters in PCA space")
plt.legend()
plt.tight_layout()

cluster_plot = os.path.join(output_dir, "pca_colored_by_kmeans_cluster.png")
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
plt.title("Truth labels in PCA space")
plt.legend()
plt.tight_layout()

truth_plot = os.path.join(output_dir, "pca_colored_by_truth_label.png")
plt.savefig(truth_plot, dpi=200)
plt.close()

print("Saved:", truth_plot)


# Plot 3: multiplicity vs ptD colored by truth
plt.figure(figsize=(8, 6))

for label in ["quark", "gluon"]:
    sub = df[df["truth_label"] == label]
    plt.scatter(
        sub["multiplicity"],
        sub["ptD"],
        s=10,
        alpha=0.6,
        label=label
    )

plt.xlabel("Multiplicity")
plt.ylabel("pT_D")
plt.title("Multiplicity vs pT_D, colored by truth")
plt.legend()
plt.tight_layout()

simple_truth_plot = os.path.join(output_dir, "multiplicity_vs_ptD_truth.png")
plt.savefig(simple_truth_plot, dpi=200)
plt.close()

print("Saved:", simple_truth_plot)


# Plot 4: multiplicity vs ptD colored by k-means cluster
plt.figure(figsize=(8, 6))

for cluster_id in sorted(df["kmeans_cluster"].unique()):
    sub = df[df["kmeans_cluster"] == cluster_id]
    plt.scatter(
        sub["multiplicity"],
        sub["ptD"],
        s=10,
        alpha=0.6,
        label=f"cluster {cluster_id}"
    )

plt.xlabel("Multiplicity")
plt.ylabel("pT_D")
plt.title("Multiplicity vs pT_D, colored by k-means cluster")
plt.legend()
plt.tight_layout()

simple_cluster_plot = os.path.join(output_dir, "multiplicity_vs_ptD_kmeans.png")
plt.savefig(simple_cluster_plot, dpi=200)
plt.close()

print("Saved:", simple_cluster_plot)


# ============================================================
# SAVE PCA VERSION TOO
# ============================================================

output_pca_csv = os.path.join(output_dir, "kmeans_clustered_jets_with_pca.csv")
df.to_csv(output_pca_csv, index=False)

print("Saved:", output_pca_csv)


# ============================================================
# PRINT FEATURE CENTERS
# ============================================================

centers_original_units = centers_scaled * feature_std + feature_mean

centers_df = pd.DataFrame(
    centers_original_units,
    columns=FEATURE_COLUMNS
)

centers_df["cluster"] = range(K)
centers_df = centers_df[["cluster"] + FEATURE_COLUMNS]

centers_file = os.path.join(output_dir, "kmeans_cluster_centers.csv")
centers_df.to_csv(centers_file, index=False)

print("\nCluster centers in original feature units:")
print(centers_df)

print("Saved:", centers_file)

print("\nDone.")
print("K-means results folder:", output_dir)

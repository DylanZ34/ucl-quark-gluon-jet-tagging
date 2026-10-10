import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD NEWEST EXTENDED FEATURE FILE
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
print("Using extended feature file:", feature_file)

df = pd.read_csv(feature_file)

print("Rows:", len(df))
print("Truth label counts:")
print(df["truth_label"].value_counts())


# ============================================================
# VARIABLES TO PLOT
# ============================================================

VARIABLES = [
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

LABELS = {
    "jet_mass": "Jet mass [GeV]",
    "multiplicity": "Multiplicity",
    "ptD": "pT_D",
    "N95": "N95",
    "jet_width": "Jet width / girth",
    "leading_constituent_pt_fraction": "Leading constituent pT fraction",
    "charged_multiplicity": "Charged multiplicity",
    "neutral_multiplicity": "Neutral multiplicity",
    "mean_constituent_pt": "Mean constituent pT [GeV]",
    "max_constituent_pt": "Max constituent pT [GeV]"
}

for var in VARIABLES:
    if var not in df.columns:
        raise ValueError(f"Missing variable: {var}")

df = df.dropna(subset=VARIABLES + ["truth_label"]).copy()
df = df[df["truth_label"].isin(["quark", "gluon"])].copy()


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_dir = "extended_feature_plots"
os.makedirs(output_dir, exist_ok=True)

print("Saving plots to:", output_dir)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reasonable_limits(data, low=0.01, high=0.99):
    data = pd.Series(data).dropna()
    if len(data) == 0:
        return None, None
    q_low = data.quantile(low)
    q_high = data.quantile(high)
    if q_low == q_high:
        return None, None
    return q_low, q_high


def save_histogram_by_truth(variable):
    plt.figure(figsize=(8, 6))

    q = df[df["truth_label"] == "quark"][variable].dropna()
    g = df[df["truth_label"] == "gluon"][variable].dropna()

    xmin, xmax = reasonable_limits(df[variable])

    if xmin is not None and xmax is not None:
        q_plot = q[(q >= xmin) & (q <= xmax)]
        g_plot = g[(g >= xmin) & (g <= xmax)]
        bins = np.linspace(xmin, xmax, 60)
    else:
        q_plot = q
        g_plot = g
        bins = 60

    # density=True compares shapes instead of raw counts
    plt.hist(q_plot, bins=bins, alpha=0.6, density=True, label="quark")
    plt.hist(g_plot, bins=bins, alpha=0.6, density=True, label="gluon")

    plt.xlabel(LABELS.get(variable, variable))
    plt.ylabel("Normalized density")
    plt.title(f"{LABELS.get(variable, variable)}: quark vs gluon")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, f"hist_{variable}_quark_vs_gluon.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


def save_scatter_by_truth(xvar, yvar, filename):
    plt.figure(figsize=(8, 6))

    # sample for readability if there are many points
    plot_df = df.copy()
    if len(plot_df) > 25000:
        plot_df = plot_df.sample(25000, random_state=42)

    for label in ["quark", "gluon"]:
        sub = plot_df[plot_df["truth_label"] == label]
        plt.scatter(
            sub[xvar],
            sub[yvar],
            s=8,
            alpha=0.45,
            label=label
        )

    xmin, xmax = reasonable_limits(plot_df[xvar])
    ymin, ymax = reasonable_limits(plot_df[yvar])

    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)

    plt.xlabel(LABELS.get(xvar, xvar))
    plt.ylabel(LABELS.get(yvar, yvar))
    plt.title(f"{LABELS.get(yvar, yvar)} vs {LABELS.get(xvar, xvar)}")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


# ============================================================
# 1. HISTOGRAMS FOR ALL VARIABLES
# ============================================================

for variable in VARIABLES:
    save_histogram_by_truth(variable)


# ============================================================
# 2. USEFUL 2D SCATTER PLOTS
# ============================================================

scatter_pairs = [
    ("multiplicity", "ptD", "scatter_multiplicity_vs_ptD_truth.png"),
    ("jet_width", "leading_constituent_pt_fraction", "scatter_jet_width_vs_leading_pt_fraction_truth.png"),
    ("charged_multiplicity", "neutral_multiplicity", "scatter_charged_vs_neutral_multiplicity_truth.png"),
    ("jet_width", "jet_mass", "scatter_jet_width_vs_jet_mass_truth.png"),
    ("jet_width", "N95", "scatter_jet_width_vs_N95_truth.png"),
    ("mean_constituent_pt", "max_constituent_pt", "scatter_mean_vs_max_constituent_pt_truth.png"),
    ("charged_multiplicity", "ptD", "scatter_charged_multiplicity_vs_ptD_truth.png"),
    ("neutral_multiplicity", "ptD", "scatter_neutral_multiplicity_vs_ptD_truth.png"),
]

for xvar, yvar, filename in scatter_pairs:
    save_scatter_by_truth(xvar, yvar, filename)


# ============================================================
# 3. CORRELATION MATRIX
# ============================================================

corr = df[VARIABLES].corr()

plt.figure(figsize=(10, 8))
im = plt.imshow(corr, vmin=-1, vmax=1)

plt.colorbar(im, label="Correlation")
plt.xticks(range(len(VARIABLES)), VARIABLES, rotation=90)
plt.yticks(range(len(VARIABLES)), VARIABLES)

for i in range(len(VARIABLES)):
    for j in range(len(VARIABLES)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Correlation matrix of extended jet variables")
plt.tight_layout()

path = os.path.join(output_dir, "correlation_matrix_extended_variables.png")
plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)


# ============================================================
# 4. SIMPLE BOXPLOTS BY TRUTH LABEL
# ============================================================

for variable in VARIABLES:
    q = df[df["truth_label"] == "quark"][variable].dropna()
    g = df[df["truth_label"] == "gluon"][variable].dropna()

    xmin, xmax = reasonable_limits(df[variable])
    if xmin is not None and xmax is not None:
        q = q[(q >= xmin) & (q <= xmax)]
        g = g[(g >= xmin) & (g <= xmax)]

    plt.figure(figsize=(7, 6))
    plt.boxplot([q, g], labels=["quark", "gluon"], showfliers=False)

    plt.ylabel(LABELS.get(variable, variable))
    plt.title(f"{LABELS.get(variable, variable)} by truth label")
    plt.tight_layout()

    path = os.path.join(output_dir, f"boxplot_{variable}_by_truth.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


print("\nDone.")
print("All plots saved in:", output_dir)
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

# Start with paper-like robust track-based features:
# charged constituents only, with pT > 1 GeV.
#
# If you want all constituents later, change:
# USE_CHARGED_ONLY = False
# CONST_PT_MIN = 0.0
USE_CHARGED_ONLY = True
CONST_PT_MIN = 1.0

# Jet image settings for N95
JET_RADIUS = 0.4
IMAGE_PIXELS = 33

KEYS = ["run", "lumi", "event", "jet_index"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_delta_phi(dphi):
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def compute_ptD(pt_values):
    total_pt = np.sum(pt_values)
    if total_pt <= 0:
        return np.nan
    return np.sqrt(np.sum(pt_values ** 2)) / total_pt


def compute_N95_from_jet_image(deta, dphi, pt_values, R=0.4, pixels=33):
    """
    Paper-like N95:
    Make a jet image from -R to +R in delta_eta and delta_phi.
    Fill each pixel with constituent pT.
    N95 = minimum number of highest-pT pixels needed to reach 95% of total pT.
    """

    total_pt = np.sum(pt_values)
    if total_pt <= 0:
        return np.nan

    # Keep only constituents inside the square jet image region
    mask = (deta >= -R) & (deta <= R) & (dphi >= -R) & (dphi <= R)
    deta = deta[mask]
    dphi = dphi[mask]
    pt_values = pt_values[mask]

    if len(pt_values) == 0:
        return np.nan

    image = np.zeros((pixels, pixels))

    # Convert delta eta/phi into pixel indices
    eta_bin = ((deta + R) / (2 * R) * pixels).astype(int)
    phi_bin = ((dphi + R) / (2 * R) * pixels).astype(int)

    # Avoid edge overflow
    eta_bin = np.clip(eta_bin, 0, pixels - 1)
    phi_bin = np.clip(phi_bin, 0, pixels - 1)

    for i in range(len(pt_values)):
        image[eta_bin[i], phi_bin[i]] += pt_values[i]

    pixel_pts = np.sort(image.flatten())[::-1]
    cumulative = np.cumsum(pixel_pts)

    return int(np.searchsorted(cumulative, 0.95 * total_pt) + 1)


# ============================================================
# LOAD NEWEST CSV FILES
# ============================================================

jet_files = sorted(glob.glob("/home/cmsusr/jets_*.csv"), key=os.path.getmtime)
const_files = sorted(glob.glob("/home/cmsusr/constituents_*.csv"), key=os.path.getmtime)

if len(jet_files) == 0:
    raise FileNotFoundError("No jets_*.csv files found in /home/cmsusr")

if len(const_files) == 0:
    raise FileNotFoundError("No constituents_*.csv files found in /home/cmsusr")

jet_file = jet_files[-1]
const_file = const_files[-1]

print("Using jet file:", jet_file)
print("Using constituent file:", const_file)

jets = pd.read_csv(jet_file)
constituents = pd.read_csv(const_file)

timestamp = os.path.basename(jet_file).replace("jets_", "").replace(".csv", "")
output_dir = f"/home/cmsusr/features_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("Output directory:", output_dir)


# ============================================================
# CHECK FOR TRUTH LABELS
# ============================================================

required_jet_columns = set(KEYS + ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "truth_label"])
missing_jet_columns = required_jet_columns - set(jets.columns)

if missing_jet_columns:
    raise ValueError(
        "Your jets CSV is missing these columns: "
        + str(missing_jet_columns)
        + "\nRun the updated run_analysis.sh first so jets.csv includes truth_label."
    )

required_const_columns = set(KEYS + [
    "constituent_pt",
    "constituent_eta",
    "constituent_phi",
    "constituent_mass",
    "charge"
])

missing_const_columns = required_const_columns - set(constituents.columns)

if missing_const_columns:
    raise ValueError(
        "Your constituents CSV is missing these columns: "
        + str(missing_const_columns)
    )


# ============================================================
# SELECT CONSTITUENTS
# ============================================================

selected_constituents = constituents.copy()

if CONST_PT_MIN is not None:
    selected_constituents = selected_constituents[
        selected_constituents["constituent_pt"] > CONST_PT_MIN
    ]

if USE_CHARGED_ONLY:
    selected_constituents = selected_constituents[
        selected_constituents["charge"] != 0
    ]

print("Original number of constituents:", len(constituents))
print("Selected number of constituents:", len(selected_constituents))

if USE_CHARGED_ONLY:
    print("Using charged constituents only.")
else:
    print("Using charged + neutral constituents.")

print("Constituent pT cut:", CONST_PT_MIN)


# ============================================================
# MERGE JET AXIS INFORMATION INTO CONSTITUENTS
# ============================================================

# The updated constituents.csv may already contain truth-label columns.
# Drop them here so pandas does not create duplicate _x/_y columns during merge.
duplicate_columns = [
    "jet_parton_flavour",
    "jet_hadron_flavour",
    "truth_label"
]

selected_constituents = selected_constituents.drop(
    columns=[c for c in duplicate_columns if c in selected_constituents.columns]
)

jet_info = jets[KEYS + [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_mass",
    "jet_parton_flavour",
    "jet_hadron_flavour",
    "truth_label"
]].copy()

merged = selected_constituents.merge(
    jet_info,
    on=KEYS,
    how="inner"
)

merged["delta_eta"] = merged["constituent_eta"] - merged["jet_eta"]
merged["delta_phi"] = wrap_delta_phi(
    merged["constituent_phi"] - merged["jet_phi"]
)


# ============================================================
# CALCULATE FEATURES PER JET
# ============================================================

feature_rows = []

for key, group in merged.groupby(KEYS, sort=False):
    pt = group["constituent_pt"].to_numpy(dtype=float)
    deta = group["delta_eta"].to_numpy(dtype=float)
    dphi = group["delta_phi"].to_numpy(dtype=float)

    if len(pt) == 0:
        continue

    first = group.iloc[0]

    multiplicity = len(group)
    ptD = compute_ptD(pt)
    N95 = compute_N95_from_jet_image(
        deta=deta,
        dphi=dphi,
        pt_values=pt,
        R=JET_RADIUS,
        pixels=IMAGE_PIXELS
    )

    feature_rows.append({
        "run": key[0],
        "lumi": key[1],
        "event": key[2],
        "jet_index": key[3],

        "jet_pt": first["jet_pt"],
        "jet_eta": first["jet_eta"],
        "jet_phi": first["jet_phi"],
        "jet_mass": first["jet_mass"],

        "jet_parton_flavour": first["jet_parton_flavour"],
        "jet_hadron_flavour": first["jet_hadron_flavour"],
        "truth_label": first["truth_label"],

        "multiplicity": multiplicity,
        "ptD": ptD,
        "N95": N95
    })

features = pd.DataFrame(feature_rows)

# Remove unknown labels for later quark/gluon ML studies
features_known = features[features["truth_label"].isin(["quark", "gluon"])].copy()

features_file = os.path.join(output_dir, "jet_substructure_features.csv")
features_known_file = os.path.join(output_dir, "jet_substructure_features_quark_gluon_only.csv")

features.to_csv(features_file, index=False)
features_known.to_csv(features_known_file, index=False)

print("Saved:", features_file)
print("Saved:", features_known_file)

print("\nAll truth label counts:")
print(features["truth_label"].value_counts(dropna=False))

print("\nKnown quark/gluon truth label counts:")
print(features_known["truth_label"].value_counts(dropna=False))

print("\nPreview:")
print(features_known.head())


# ============================================================
# MAKE HISTOGRAMS BY TRUTH LABEL
# ============================================================

def plot_variable_by_truth(df, column, xlabel, filename, bins=50, log_y=False):
    plt.figure(figsize=(8, 6))

    q = df[df["truth_label"] == "quark"][column].dropna()
    g = df[df["truth_label"] == "gluon"][column].dropna()

    plt.hist(q, bins=bins, alpha=0.6, label="quark")
    plt.hist(g, bins=bins, alpha=0.6, label="gluon")

    if log_y:
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(f"{xlabel}: quark vs gluon")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


# Choose bins carefully
plot_variable_by_truth(
    features_known,
    "jet_mass",
    "Jet mass [GeV]",
    "hist_jet_mass_quark_vs_gluon.png",
    bins=60,
    log_y=True
)

plot_variable_by_truth(
    features_known,
    "multiplicity",
    "Selected constituent multiplicity",
    "hist_multiplicity_quark_vs_gluon.png",
    bins=range(
        int(features_known["multiplicity"].min()),
        int(features_known["multiplicity"].max()) + 2
    ),
    log_y=False
)

plot_variable_by_truth(
    features_known,
    "ptD",
    "pT_D",
    "hist_ptD_quark_vs_gluon.png",
    bins=60,
    log_y=False
)

plot_variable_by_truth(
    features_known,
    "N95",
    "N95",
    "hist_N95_quark_vs_gluon.png",
    bins=range(
        int(features_known["N95"].min()),
        int(features_known["N95"].max()) + 2
    ),
    log_y=False
)


# ============================================================
# MAKE ONE COMBINED 2x2 FIGURE
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
variables = [
    ("jet_mass", "Jet mass [GeV]", 60, True),
    ("multiplicity", "Selected constituent multiplicity", None, False),
    ("ptD", "pT_D", 60, False),
    ("N95", "N95", None, False),
]

for ax, (column, xlabel, bins, log_y) in zip(axes.flatten(), variables):
    q = features_known[features_known["truth_label"] == "quark"][column].dropna()
    g = features_known[features_known["truth_label"] == "gluon"][column].dropna()

    if bins is None:
        min_bin = int(features_known[column].min())
        max_bin = int(features_known[column].max())
        bins = range(min_bin, max_bin + 2)

    ax.hist(q, bins=bins, alpha=0.6, label="quark")
    ax.hist(g, bins=bins, alpha=0.6, label="gluon")

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(xlabel)
    ax.legend()

plt.tight_layout()
combined_path = os.path.join(output_dir, "combined_features_quark_vs_gluon.png")
plt.savefig(combined_path, dpi=200)
plt.close()

print("Saved:", combined_path)

print("\nDone.")
print("Feature table:", features_known_file)
print("Plots folder:", output_dir)

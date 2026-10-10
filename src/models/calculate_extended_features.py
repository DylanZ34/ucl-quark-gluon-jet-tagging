import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

KEYS = ["run", "lumi", "event", "jet_index"]

# Use pT > 1 GeV, similar to your earlier feature calculation.
# This keeps very soft detector-level particles from dominating.
CONST_PT_MIN = 1.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_delta_phi(dphi):
    return np.arctan2(np.sin(dphi), np.cos(dphi))


# ============================================================
# LOAD NEWEST FILES
# ============================================================

feature_files = sorted(
    glob.glob("features_*/jet_substructure_features_quark_gluon_only.csv"),
    key=os.path.getmtime
)

jet_files = sorted(
    glob.glob("jets_*.csv"),
    key=os.path.getmtime
)

constituent_files = sorted(
    glob.glob("constituents_*.csv"),
    key=os.path.getmtime
)

if len(feature_files) == 0:
    raise FileNotFoundError("No feature file found. Run your feature calculation first.")

if len(jet_files) == 0:
    raise FileNotFoundError("No jets_*.csv file found.")

if len(constituent_files) == 0:
    raise FileNotFoundError("No constituents_*.csv file found.")

feature_file = feature_files[-1]
jet_file = jet_files[-1]
constituent_file = constituent_files[-1]

print("Using feature file:", feature_file)
print("Using jet file:", jet_file)
print("Using constituent file:", constituent_file)

features = pd.read_csv(feature_file)
jets = pd.read_csv(jet_file)
constituents = pd.read_csv(constituent_file)

print("\nOriginal feature rows:", len(features))
print("Original constituent rows:", len(constituents))


# ============================================================
# CHECK COLUMNS
# ============================================================

required_feature_columns = set(KEYS + [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_mass",
    "jet_parton_flavour",
    "truth_label",
    "multiplicity",
    "ptD",
    "N95"
])

required_constituent_columns = set(KEYS + [
    "constituent_pt",
    "constituent_eta",
    "constituent_phi",
    "charge"
])

missing_feature_columns = required_feature_columns - set(features.columns)
missing_constituent_columns = required_constituent_columns - set(constituents.columns)

if missing_feature_columns:
    raise ValueError("Feature file is missing columns: " + str(missing_feature_columns))

if missing_constituent_columns:
    raise ValueError("Constituent file is missing columns: " + str(missing_constituent_columns))


# ============================================================
# SELECT CONSTITUENTS
# ============================================================

selected = constituents.copy()
selected = selected[selected["constituent_pt"] > CONST_PT_MIN].copy()

print("\nConstituents after pT cut:", len(selected))
print("Constituent pT cut:", CONST_PT_MIN, "GeV")


# ============================================================
# MERGE JET AXIS INTO CONSTITUENTS
# ============================================================

jet_axis = features[KEYS + ["jet_pt", "jet_eta", "jet_phi"]].copy()

merged = selected.merge(
    jet_axis,
    on=KEYS,
    how="inner"
)

print("Merged selected constituents:", len(merged))


# ============================================================
# CALCULATE DELTA ETA, DELTA PHI, DELTA R
# ============================================================

merged["delta_eta"] = merged["constituent_eta"] - merged["jet_eta"]
merged["delta_phi"] = wrap_delta_phi(merged["constituent_phi"] - merged["jet_phi"])
merged["delta_R"] = np.sqrt(merged["delta_eta"]**2 + merged["delta_phi"]**2)

merged["is_charged"] = merged["charge"] != 0
merged["is_neutral"] = merged["charge"] == 0
merged["pt_times_delta_R"] = merged["constituent_pt"] * merged["delta_R"]


# ============================================================
# COMPUTE NEW VARIABLES PER JET
# ============================================================

grouped = merged.groupby(KEYS)

new_features = grouped.agg(
    total_selected_multiplicity=("constituent_pt", "size"),
    charged_multiplicity=("is_charged", "sum"),
    neutral_multiplicity=("is_neutral", "sum"),
    sum_constituent_pt=("constituent_pt", "sum"),
    mean_constituent_pt=("constituent_pt", "mean"),
    max_constituent_pt=("constituent_pt", "max"),
    sum_pt_times_delta_R=("pt_times_delta_R", "sum")
).reset_index()

new_features["jet_width"] = (
    new_features["sum_pt_times_delta_R"] / new_features["sum_constituent_pt"]
)

new_features["leading_constituent_pt_fraction"] = (
    new_features["max_constituent_pt"] / new_features["sum_constituent_pt"]
)

# Clean up helper column
new_features = new_features.drop(columns=["sum_pt_times_delta_R"])

print("\nNew feature rows:", len(new_features))
print("\nNew feature preview:")
print(new_features.head())


# ============================================================
# MERGE NEW VARIABLES INTO OLD FEATURE TABLE
# ============================================================

extended = features.merge(
    new_features,
    on=KEYS,
    how="left"
)

# Remove jets where the pT cut left no selected constituents
new_variable_columns = [
    "jet_width",
    "leading_constituent_pt_fraction",
    "charged_multiplicity",
    "neutral_multiplicity",
    "mean_constituent_pt",
    "max_constituent_pt",
    "sum_constituent_pt",
    "total_selected_multiplicity"
]

extended = extended.dropna(subset=new_variable_columns).copy()

# Keep only quark/gluon labels
extended_qg = extended[extended["truth_label"].isin(["quark", "gluon"])].copy()


# ============================================================
# SAVE OUTPUT
# ============================================================

timestamp = os.path.basename(feature_file).split("_")[1]
# fallback: create readable name from parent folder
parent_name = os.path.basename(os.path.dirname(feature_file))

output_dir = f"{parent_name}_extended"
os.makedirs(output_dir, exist_ok=True)

extended_file = os.path.join(output_dir, "jet_substructure_features_extended.csv")
extended_qg_file = os.path.join(output_dir, "jet_substructure_features_extended_quark_gluon_only.csv")

extended.to_csv(extended_file, index=False)
extended_qg.to_csv(extended_qg_file, index=False)

print("\nSaved:", extended_file)
print("Saved:", extended_qg_file)

print("\nExtended rows:", len(extended))
print("Extended quark/gluon rows:", len(extended_qg))

print("\nTruth label counts:")
print(extended_qg["truth_label"].value_counts())

print("\nColumns now available:")
print(extended_qg.columns.tolist())


# ============================================================
# MAKE QUICK HISTOGRAMS OF NEW VARIABLES
# ============================================================

plot_vars = [
    ("jet_width", "Jet width / girth"),
    ("leading_constituent_pt_fraction", "Leading constituent pT fraction"),
    ("charged_multiplicity", "Charged multiplicity"),
    ("neutral_multiplicity", "Neutral multiplicity"),
    ("mean_constituent_pt", "Mean constituent pT [GeV]"),
    ("max_constituent_pt", "Max constituent pT [GeV]")
]

for column, xlabel in plot_vars:
    plt.figure(figsize=(8, 6))

    q = extended_qg[extended_qg["truth_label"] == "quark"][column].dropna()
    g = extended_qg[extended_qg["truth_label"] == "gluon"][column].dropna()

    plt.hist(q, bins=60, alpha=0.6, label="quark")
    plt.hist(g, bins=60, alpha=0.6, label="gluon")

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(f"{xlabel}: quark vs gluon")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, f"hist_{column}_quark_vs_gluon.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


print("\nDone.")
print("Use this file for the next supervised ML run:")
print(extended_qg_file)
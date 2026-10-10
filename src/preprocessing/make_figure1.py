import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

JET_FILE = "jets_20260810_203552.csv"
CONSTITUENT_FILE = "constituents_20260810_203552.csv"

OUTPUT_PNG = "figure1_quark_gluon_substructure.png"
OUTPUT_PDF = "figure1_quark_gluon_substructure.pdf"
OUTPUT_CSV = "figure1_quark_gluon_data.csv"

# Common jet kinematic region.
# This reduces the chance that the plot is mainly showing differences
# in the quark/gluon jet-pT distributions rather than substructure.
JET_PT_MIN = 80.0
JET_PT_MAX = 120.0
JET_ABS_ETA_MAX = 2.4

# Match the charged-track-like constituent selection used in the analysis.
CONSTITUENT_PT_MIN = 1.0

# Read the large constituent file in chunks.
CHUNK_SIZE = 500_000

KEYS = ["run", "lumi", "event", "jet_index"]


# ============================================================
# HELPERS
# ============================================================

def wrap_delta_phi(dphi):
    """
    Wrap angular differences into [-pi, pi].
    """
    return np.arctan2(np.sin(dphi), np.cos(dphi))


# ============================================================
# LOAD AND SELECT JETS
# ============================================================

print("Loading jet table...")

jet_columns = [
    "run",
    "lumi",
    "event",
    "jet_index",
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_parton_flavour",
    "truth_label",
]

jets = pd.read_csv(
    JET_FILE,
    usecols=jet_columns
)

# Keep only quark/gluon-labeled jets.
jets = jets[
    jets["truth_label"].isin(["quark", "gluon"])
].copy()

# Restrict to a common kinematic region.
jets = jets[
    (jets["jet_pt"] >= JET_PT_MIN)
    & (jets["jet_pt"] < JET_PT_MAX)
    & (np.abs(jets["jet_eta"]) < JET_ABS_ETA_MAX)
].copy()

print()
print("Jets after selections:")
print(jets["truth_label"].value_counts())
print("Total selected jets:", len(jets))

if len(jets) == 0:
    raise RuntimeError(
        "No jets passed the selections. "
        "Check the pT and eta limits."
    )


# Keep only what needs to be merged into constituent chunks.
selected_jet_info = jets[
    KEYS + ["jet_eta", "jet_phi", "truth_label"]
].copy()


# ============================================================
# PROCESS CONSTITUENTS IN CHUNKS
# ============================================================

print()
print("Processing constituent file in chunks...")

constituent_columns = [
    "run",
    "lumi",
    "event",
    "jet_index",
    "constituent_pt",
    "constituent_eta",
    "constituent_phi",
    "charge",
]

partial_results = []

chunk_number = 0

for chunk in pd.read_csv(
    CONSTITUENT_FILE,
    usecols=constituent_columns,
    chunksize=CHUNK_SIZE
):
    chunk_number += 1

    # Charged constituents only.
    chunk = chunk[
        (chunk["charge"] != 0)
        & (chunk["constituent_pt"] > CONSTITUENT_PT_MIN)
    ].copy()

    if len(chunk) == 0:
        continue

    # Keep only constituents belonging to our selected jets.
    chunk = chunk.merge(
        selected_jet_info,
        on=KEYS,
        how="inner"
    )

    if len(chunk) == 0:
        continue

    # Distance from the jet axis.
    chunk["delta_eta"] = (
        chunk["constituent_eta"] - chunk["jet_eta"]
    )

    chunk["delta_phi"] = wrap_delta_phi(
        chunk["constituent_phi"] - chunk["jet_phi"]
    )

    chunk["delta_R"] = np.sqrt(
        chunk["delta_eta"] ** 2
        + chunk["delta_phi"] ** 2
    )

    chunk["pt_times_delta_R"] = (
        chunk["constituent_pt"] * chunk["delta_R"]
    )

    # Compute partial sums for this chunk.
    partial = (
        chunk
        .groupby(KEYS + ["truth_label"], as_index=False)
        .agg(
            multiplicity=("constituent_pt", "size"),
            sum_pt=("constituent_pt", "sum"),
            sum_pt_deltaR=("pt_times_delta_R", "sum"),
        )
    )

    partial_results.append(partial)

    if chunk_number % 10 == 0:
        print(f"  processed {chunk_number} chunks")


if not partial_results:
    raise RuntimeError(
        "No selected constituents were found. "
        "Check the input files and selections."
    )


# ============================================================
# COMBINE PARTIAL CHUNK RESULTS
# ============================================================

print()
print("Combining chunk results...")

combined = pd.concat(
    partial_results,
    ignore_index=True
)

# The same jet may appear in multiple file chunks,
# so add the partial sums together.
features = (
    combined
    .groupby(KEYS + ["truth_label"], as_index=False)
    .agg(
        multiplicity=("multiplicity", "sum"),
        sum_pt=("sum_pt", "sum"),
        sum_pt_deltaR=("sum_pt_deltaR", "sum"),
    )
)

features["jet_width"] = (
    features["sum_pt_deltaR"] / features["sum_pt"]
)

# Remove any pathological values.
features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

features = features.dropna(
    subset=["multiplicity", "jet_width"]
).copy()

features.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("Final jets used in Figure 1:")
print(features["truth_label"].value_counts())

print()
print("Mean values:")
print(
    features.groupby("truth_label")[
        ["multiplicity", "jet_width"]
    ].mean()
)

print()
print("Median values:")
print(
    features.groupby("truth_label")[
        ["multiplicity", "jet_width"]
    ].median()
)


# ============================================================
# SEPARATE QUARK AND GLUON SAMPLES
# ============================================================

quark = features[
    features["truth_label"] == "quark"
].copy()

gluon = features[
    features["truth_label"] == "gluon"
].copy()


# ============================================================
# CHOOSE PLOT BINS
# ============================================================

# Multiplicity is discrete, so center bins on integers.
min_mult = int(features["multiplicity"].min())
max_mult = int(features["multiplicity"].max())

multiplicity_bins = np.arange(
    min_mult - 0.5,
    max_mult + 1.5,
    1
)

# Width can have a few extreme tails.
# Use the 99.5th percentile only to keep the figure readable.
width_max = features["jet_width"].quantile(0.995)

width_bins = np.linspace(
    0,
    width_max,
    45
)


# ============================================================
# MAKE PUBLICATION-STYLE FIGURE
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.4)
)


# ------------------------------------------------------------
# Panel (a): multiplicity
# ------------------------------------------------------------

axes[0].hist(
    quark["multiplicity"],
    bins=multiplicity_bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Quark"
)

axes[0].hist(
    gluon["multiplicity"],
    bins=multiplicity_bins,
    density=True,
    histtype="step",
    linewidth=2,
    linestyle="--",
    label="Gluon"
)

axes[0].set_xlabel(
    r"Selected charged-constituent multiplicity"
)
axes[0].set_ylabel(
    "Normalized density"
)
axes[0].legend(
    frameon=False
)

axes[0].text(
    0.03,
    0.95,
    "(a)",
    transform=axes[0].transAxes,
    ha="left",
    va="top",
    fontsize=13
)


# ------------------------------------------------------------
# Panel (b): jet width
# ------------------------------------------------------------

axes[1].hist(
    quark["jet_width"],
    bins=width_bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Quark"
)

axes[1].hist(
    gluon["jet_width"],
    bins=width_bins,
    density=True,
    histtype="step",
    linewidth=2,
    linestyle="--",
    label="Gluon"
)

axes[1].set_xlabel(
    r"Jet width $w$"
)
axes[1].set_ylabel(
    "Normalized density"
)
axes[1].legend(
    frameon=False
)

axes[1].text(
    0.03,
    0.95,
    "(b)",
    transform=axes[1].transAxes,
    ha="left",
    va="top",
    fontsize=13
)


# ------------------------------------------------------------
# Shared annotation
# ------------------------------------------------------------

selection_text = (
    rf"${JET_PT_MIN:.0f} < p_T^{{\rm jet}} < "
    rf"{JET_PT_MAX:.0f}$ GeV"
    "\n"
    rf"$|\eta^{{\rm jet}}| < {JET_ABS_ETA_MAX}$"
    "\n"
    rf"charged constituents, $p_T>{CONSTITUENT_PT_MIN:.0f}$ GeV"
)

axes[1].text(
    0.97,
    0.95,
    selection_text,
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    fontsize=9
)


# Remove unnecessary top/right borders.
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)

plt.show()


print()
print("Done.")
print("Saved:")
print(" ", OUTPUT_PNG)
print(" ", OUTPUT_PDF)
print(" ", OUTPUT_CSV)
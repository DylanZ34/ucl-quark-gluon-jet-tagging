import os
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

# Find newest timestamped jet and constituent CSV files
jet_files = sorted(glob.glob("/home/cmsusr/jets_*.csv"), key=os.path.getmtime)
constituent_files = sorted(glob.glob("/home/cmsusr/constituents_*.csv"), key=os.path.getmtime)

if len(jet_files) == 0:
    raise FileNotFoundError("No jets_*.csv files found in /home/cmsusr")

if len(constituent_files) == 0:
    raise FileNotFoundError("No constituents_*.csv files found in /home/cmsusr")

jet_file = jet_files[-1]
constituent_file = constituent_files[-1]

print("Using jet file:", jet_file)
print("Using constituent file:", constituent_file)

jets = pd.read_csv(jet_file)
constituents = pd.read_csv(constituent_file)

base = os.path.basename(jet_file).replace("jets_", "").replace(".csv", "")
plot_dir = "/home/cmsusr/plots_" + base
os.makedirs(plot_dir, exist_ok=True)

print("Saving plots to:", plot_dir)

def make_histogram(data, column, title, xlabel, output_name, bins=80, use_percentile_range=True, log_y=False):
    values = data[column].dropna()

    if use_percentile_range:
        low = values.quantile(0.001)
        high = values.quantile(0.999)
        values = values[(values >= low) & (values <= high)]

    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins)

    if log_y:
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()

    output_path = os.path.join(plot_dir, output_name)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved:", output_path)

# Explicit angular bins for phi
# This avoids the alternating-bin artifact from automatic integer binning.
phi_bins = np.linspace(-np.pi, np.pi, 241)

# Eta bins
eta_bins = np.linspace(-5.5, 5.5, 121)

# Jet-level histograms
make_histogram(
    jets,
    "jet_pt",
    "Jet pT Distribution",
    "Jet pT [GeV]",
    "jet_pt_hist.png",
    bins=100,
    log_y=True
)

make_histogram(
    jets,
    "jet_mass",
    "Jet Mass Distribution",
    "Jet Mass [GeV]",
    "jet_mass_hist.png",
    bins=100,
    log_y=True
)

make_histogram(
    jets,
    "jet_phi",
    "Jet Phi Distribution",
    "Jet phi [radians]",
    "jet_phi_hist.png",
    bins=phi_bins,
    use_percentile_range=False
)

make_histogram(
    jets,
    "jet_eta",
    "Jet Eta Distribution",
    "Jet eta",
    "jet_eta_hist.png",
    bins=eta_bins,
    use_percentile_range=False
)

# Constituent-level histograms
make_histogram(
    constituents,
    "constituent_pt",
    "Constituent pT Distribution",
    "Constituent pT [GeV]",
    "constituent_pt_hist.png",
    bins=100,
    log_y=True
)

make_histogram(
    constituents,
    "constituent_mass",
    "Constituent Mass Distribution",
    "Constituent Mass [GeV]",
    "constituent_mass_hist.png",
    bins=100,
    log_y=True
)

make_histogram(
    constituents,
    "constituent_phi",
    "Constituent Phi Distribution",
    "Constituent phi [radians]",
    "constituent_phi_hist.png",
    bins=phi_bins,
    use_percentile_range=False
)

make_histogram(
    constituents,
    "constituent_eta",
    "Constituent Eta Distribution",
    "Constituent eta",
    "constituent_eta_hist.png",
    bins=eta_bins,
    use_percentile_range=False
)

print("Done making histograms.")
print("Number of jets:", len(jets))
print("Number of constituents:", len(constituents))
print("Plots saved in:", plot_dir)

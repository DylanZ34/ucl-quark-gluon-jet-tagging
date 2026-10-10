import os
import glob
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# USER OPTIONS
# =========================================================

# If you want to be closer to the paper, keep these as True / 1.0
USE_CHARGED_ONLY = False
CONST_PT_MIN = 0.0

# Optional jet selections (set to None to disable)
JET_PT_MIN = None
JET_PT_MAX = None
JET_ABS_ETA_MAX = None

# Jet radius for zg soft-drop declustering
JET_RADIUS = 0.4
SOFTDROP_ZCUT = 0.1
SOFTDROP_BETA = 0.0   # standard zg choice

# =========================================================
# HELPERS
# =========================================================

JET_KEYS = ["run", "lumi", "event", "jet_index"]


def wrap_dphi(dphi):
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + wrap_dphi(phi1 - phi2)**2)


def pt_eta_phi_m_to_p4(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p2 = px**2 + py**2 + pz**2
    E = np.sqrt(np.maximum(p2 + mass**2, 0.0))
    return E, px, py, pz


def p4_to_pt_eta_phi_m(E, px, py, pz):
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)
    p = np.sqrt(np.maximum(px**2 + py**2 + pz**2, 0.0))

    # Safe eta
    if p != abs(pz):
        eta = 0.5 * np.log((p + pz) / (p - pz))
    else:
        eta = np.sign(pz) * 1e6

    m2 = E**2 - p**2
    if m2 < 0:
        m2 = 0.0
    mass = np.sqrt(m2)

    return pt, eta, phi, mass


def compute_jet_mass_from_constituents(pt, eta, phi, mass):
    E, px, py, pz = pt_eta_phi_m_to_p4(pt, eta, phi, mass)
    Etot = np.sum(E)
    pxtot = np.sum(px)
    pytot = np.sum(py)
    pztot = np.sum(pz)
    _, _, _, m = p4_to_pt_eta_phi_m(Etot, pxtot, pytot, pztot)
    return m


def compute_ptd(pt):
    s1 = np.sum(pt)
    if s1 <= 0:
        return np.nan
    s2 = np.sum(pt**2)
    return np.sqrt(s2) / s1


def compute_n95(pt):
    total = np.sum(pt)
    if total <= 0:
        return np.nan
    pts = np.sort(pt)[::-1]
    csum = np.cumsum(pts)
    return int(np.searchsorted(csum, 0.95 * total) + 1)


def compute_d2(pt, eta, phi, beta=1.0):
    n = len(pt)
    if n < 3:
        return np.nan

    pt_sum = np.sum(pt)
    if pt_sum <= 0:
        return np.nan

    z = pt / pt_sum

    e2 = 0.0
    e3 = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            dr_ij = delta_r(eta[i], phi[i], eta[j], phi[j])
            e2 += z[i] * z[j] * (dr_ij ** beta)

    if e2 <= 0:
        return np.nan

    for i in range(n):
        for j in range(i + 1, n):
            dr_ij = delta_r(eta[i], phi[i], eta[j], phi[j])
            for k in range(j + 1, n):
                dr_ik = delta_r(eta[i], phi[i], eta[k], phi[k])
                dr_jk = delta_r(eta[j], phi[j], eta[k], phi[k])
                e3 += (
                    z[i] * z[j] * z[k]
                    * (dr_ij ** beta)
                    * (dr_ik ** beta)
                    * (dr_jk ** beta)
                )

    return e3 / (e2 ** 3)


# ---------------------------------------------------------
# Minimal Cambridge/Aachen reclustering tree for zg
# ---------------------------------------------------------

class Node:
    def __init__(self, E, px, py, pz, left=None, right=None):
        self.E = E
        self.px = px
        self.py = py
        self.pz = pz
        self.left = left
        self.right = right

    @property
    def pt(self):
        return float(np.hypot(self.px, self.py))

    @property
    def phi(self):
        return float(np.arctan2(self.py, self.px))

    @property
    def p(self):
        return float(np.sqrt(max(self.px**2 + self.py**2 + self.pz**2, 0.0)))

    @property
    def eta(self):
        p = self.p
        if p != abs(self.pz):
            return float(0.5 * np.log((p + self.pz) / (p - self.pz)))
        return float(np.sign(self.pz) * 1e6)

    def is_leaf(self):
        return self.left is None and self.right is None


def build_ca_tree(pt, eta, phi, mass):
    # Build leaf nodes
    active = []
    for i in range(len(pt)):
        E, px, py, pz = pt_eta_phi_m_to_p4(pt[i], eta[i], phi[i], mass[i])
        active.append(Node(float(E), float(px), float(py), float(pz)))

    if len(active) == 0:
        return None
    if len(active) == 1:
        return active[0]

    # Repeatedly merge closest pair in angle
    while len(active) > 1:
        best_i = None
        best_j = None
        best_dr2 = None

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                dr = delta_r(active[i].eta, active[i].phi, active[j].eta, active[j].phi)
                dr2 = dr * dr
                if best_dr2 is None or dr2 < best_dr2:
                    best_dr2 = dr2
                    best_i = i
                    best_j = j

        a = active[best_i]
        b = active[best_j]

        new_node = Node(
            a.E + b.E,
            a.px + b.px,
            a.py + b.py,
            a.pz + b.pz,
            left=a,
            right=b
        )

        # Remove j first, then i
        active.pop(best_j)
        active.pop(best_i)
        active.append(new_node)

    return active[0]


def compute_zg(pt, eta, phi, mass, zcut=0.1, beta=0.0, R0=0.4):
    if len(pt) < 2:
        return np.nan

    root = build_ca_tree(pt, eta, phi, mass)
    if root is None:
        return np.nan

    node = root

    while not node.is_leaf():
        a = node.left
        b = node.right

        # Order so that a is the harder branch
        if a.pt < b.pt:
            a, b = b, a

        pt_sum = a.pt + b.pt
        if pt_sum <= 0:
            return np.nan

        z = b.pt / pt_sum
        dr = delta_r(a.eta, a.phi, b.eta, b.phi)

        condition = z > zcut * ((dr / R0) ** beta)

        if condition:
            return z

        node = a

    return np.nan


def clip_to_percentiles(series, low=0.001, high=0.999):
    x = series.dropna().values
    if len(x) == 0:
        return x
    lo = np.quantile(x, low)
    hi = np.quantile(x, high)
    return x[(x >= lo) & (x <= hi)]


# =========================================================
# LOAD NEWEST CSV FILES
# =========================================================

jet_files = sorted(glob.glob("/home/cmsusr/jets_*.csv"), key=os.path.getmtime)
const_files = sorted(glob.glob("/home/cmsusr/constituents_*.csv"), key=os.path.getmtime)

if len(jet_files) == 0:
    raise FileNotFoundError("No jets_*.csv found in /home/cmsusr")
if len(const_files) == 0:
    raise FileNotFoundError("No constituents_*.csv found in /home/cmsusr")

jet_file = jet_files[-1]
const_file = const_files[-1]

print("Using jet file:", jet_file)
print("Using constituent file:", const_file)

jets = pd.read_csv(jet_file)
constituents = pd.read_csv(const_file)

timestamp = os.path.basename(jet_file).replace("jets_", "").replace(".csv", "")
plot_dir = f"/home/cmsusr/substructure_plots_{timestamp}"
os.makedirs(plot_dir, exist_ok=True)

print("Saving plots to:", plot_dir)

# =========================================================
# OPTIONAL JET SELECTIONS
# =========================================================

if JET_PT_MIN is not None:
    jets = jets[jets["jet_pt"] >= JET_PT_MIN]
if JET_PT_MAX is not None:
    jets = jets[jets["jet_pt"] <= JET_PT_MAX]
if JET_ABS_ETA_MAX is not None:
    jets = jets[np.abs(jets["jet_eta"]) <= JET_ABS_ETA_MAX]

# Keep only constituents belonging to selected jets
constituents = constituents.merge(
    jets[JET_KEYS],
    on=JET_KEYS,
    how="inner"
)

# =========================================================
# CONSTITUENT SELECTIONS
# =========================================================

if CONST_PT_MIN is not None:
    constituents = constituents[constituents["constituent_pt"] > CONST_PT_MIN]

if USE_CHARGED_ONLY:
    constituents = constituents[constituents["charge"] != 0]

print("Number of selected jets (pre-substructure):", len(jets))
print("Number of selected constituents:", len(constituents))

# Jets that still have at least 1 selected constituent
jets_with_const = constituents[JET_KEYS].drop_duplicates()
jets = jets.merge(jets_with_const, on=JET_KEYS, how="inner")

print("Number of jets after constituent selection:", len(jets))

# =========================================================
# CONSTITUENT SUMMARY FIGURE
# =========================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# constituent pT
x = clip_to_percentiles(constituents["constituent_pt"], 0.0, 0.999)
axes[0, 0].hist(x, bins=80)
axes[0, 0].set_xlabel("Constituent pT [GeV]")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Constituent pT")
axes[0, 0].set_yscale("log")

# constituent mass
x = clip_to_percentiles(constituents["constituent_mass"], 0.0, 0.999)
axes[0, 1].hist(x, bins=80)
axes[0, 1].set_xlabel("Constituent mass [GeV]")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Constituent mass")
axes[0, 1].set_yscale("log")

# constituent eta
x = constituents["constituent_eta"].dropna().values
axes[1, 0].hist(x, bins=np.linspace(-5.5, 5.5, 80))
axes[1, 0].set_xlabel("Constituent eta")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Constituent eta")

# constituent phi
x = constituents["constituent_phi"].dropna().values
axes[1, 1].hist(x, bins=np.linspace(-np.pi, np.pi, 72))
axes[1, 1].set_xlabel("Constituent phi [rad]")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("Constituent phi")

plt.tight_layout()
constituent_plot = os.path.join(plot_dir, "constituent_summary.png")
plt.savefig(constituent_plot, dpi=200)
plt.close()
print("Saved:", constituent_plot)

# =========================================================
# COMPUTE JET SUBSTRUCTURE VARIABLES
# =========================================================

rows = []

grouped = constituents.groupby(JET_KEYS, sort=False)

for jet_key, g in grouped:
    pt = g["constituent_pt"].to_numpy(dtype=float)
    eta = g["constituent_eta"].to_numpy(dtype=float)
    phi = g["constituent_phi"].to_numpy(dtype=float)
    mass = g["constituent_mass"].to_numpy(dtype=float)

    if len(pt) == 0:
        continue

    jet_mass = compute_jet_mass_from_constituents(pt, eta, phi, mass)
    multiplicity = len(pt)
    ptd = compute_ptd(pt)
    n95 = compute_n95(pt)
    zg = compute_zg(pt, eta, phi, mass, zcut=SOFTDROP_ZCUT, beta=SOFTDROP_BETA, R0=JET_RADIUS)
    d2 = compute_d2(pt, eta, phi, beta=1.0)

    # Grab original jet kinematics from jets df
    jet_row = jets[
        (jets["run"] == jet_key[0]) &
        (jets["lumi"] == jet_key[1]) &
        (jets["event"] == jet_key[2]) &
        (jets["jet_index"] == jet_key[3])
    ].iloc[0]

    rows.append({
        "run": jet_key[0],
        "lumi": jet_key[1],
        "event": jet_key[2],
        "jet_index": jet_key[3],
        "original_jet_pt": jet_row["jet_pt"],
        "original_jet_eta": jet_row["jet_eta"],
        "original_jet_phi": jet_row["jet_phi"],
        "original_jet_mass": jet_row["jet_mass"],
        "substructure_mass": jet_mass,
        "multiplicity": multiplicity,
        "ptD": ptd,
        "N95": n95,
        "zg": zg,
        "D2": d2
    })

sub = pd.DataFrame(rows)

sub_csv = os.path.join(plot_dir, "jet_substructure_variables.csv")
sub.to_csv(sub_csv, index=False)
print("Saved:", sub_csv)

print("Number of jets with substructure variables:", len(sub))

# =========================================================
# JET SUBSTRUCTURE FIGURE
# =========================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 1. mass
x = clip_to_percentiles(sub["substructure_mass"], 0.0, 0.999)
axes[0].hist(x, bins=80)
axes[0].set_xlabel("Jet mass [GeV]")
axes[0].set_ylabel("Count")
axes[0].set_title("Jet mass")
axes[0].set_yscale("log")

# 2. multiplicity
x = sub["multiplicity"].dropna().values
if len(x) > 0:
    bins = np.arange(np.min(x), np.max(x) + 2) - 0.5
    axes[1].hist(x, bins=bins)
axes[1].set_xlabel("Multiplicity")
axes[1].set_ylabel("Count")
axes[1].set_title("Jet constituent multiplicity")

# 3. pT_D
x = sub["ptD"].dropna().values
axes[2].hist(x, bins=80)
axes[2].set_xlabel("pT^D")
axes[2].set_ylabel("Count")
axes[2].set_title("Transverse momentum dispersion")

# 4. N95
x = sub["N95"].dropna().values
if len(x) > 0:
    bins = np.arange(np.min(x), np.max(x) + 2) - 0.5
    axes[3].hist(x, bins=bins)
axes[3].set_xlabel("N95")
axes[3].set_ylabel("Count")
axes[3].set_title("N95")

# 5. zg
x = sub["zg"].dropna().values
axes[4].hist(x, bins=np.linspace(0, 0.5, 60))
axes[4].set_xlabel("z_g")
axes[4].set_ylabel("Count")
axes[4].set_title("Groomed momentum fraction")

# 6. D2
x = clip_to_percentiles(sub["D2"], 0.0, 0.995)
axes[5].hist(x, bins=80)
axes[5].set_xlabel("D2")
axes[5].set_ylabel("Count")
axes[5].set_title("D2")

plt.tight_layout()
jet_plot = os.path.join(plot_dir, "jet_substructure.png")
plt.savefig(jet_plot, dpi=200)
plt.close()
print("Saved:", jet_plot)

print("Done.")
print("Output directory:", plot_dir)

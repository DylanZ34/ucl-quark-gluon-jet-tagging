import os
import glob
import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

# Transformer hyperparameter grid
D_MODEL_OPTIONS = [32, 64]
N_HEADS_OPTIONS = [2, 4]
N_LAYERS_OPTIONS = [1, 2]
MAX_CONSTITUENTS_OPTIONS = [30, 50]

# Constituent settings
CONSTITUENT_PT_MIN = 1.0
USE_CHARGED_ONLY = True

# Training settings
BATCH_SIZE = 256
EPOCHS = 12
LEARNING_RATE = 1e-3

# Set this to a number like 10000 if you want a quick test first.
# Set it back to None for your real final scan.
MAX_JETS_FOR_SCAN = None

KEYS = ["run", "lumi", "event", "jet_index"]

SCAN_OUTPUT_DIR = "transformer_hyperparameter_scan"
BEST_OUTPUT_DIR = "transformer_constituent_results"

os.makedirs(SCAN_OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_delta_phi(dphi):
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))


# ============================================================
# LOAD DATA
# ============================================================

feature_files = sorted(
    glob.glob("features_*_extended/jet_substructure_features_extended_quark_gluon_only.csv"),
    key=os.path.getmtime
)

constituent_files = sorted(
    glob.glob("constituents_*.csv"),
    key=os.path.getmtime
)

if len(feature_files) == 0:
    raise FileNotFoundError(
        "No extended feature file found. Run calculate_extended_features.py first."
    )

if len(constituent_files) == 0:
    raise FileNotFoundError("No constituents_*.csv file found.")

feature_file = feature_files[-1]
constituent_file = constituent_files[-1]

print("Using feature file:", feature_file)
print("Using constituent file:", constituent_file)

jets = pd.read_csv(feature_file)
constituents = pd.read_csv(constituent_file)

jets = jets[jets["truth_label"].isin(["quark", "gluon"])].copy()
jets = jets.dropna(subset=["truth_label", "jet_eta", "jet_phi"]).copy()

jets["label"] = jets["truth_label"].map({
    "quark": 0,
    "gluon": 1
})

jets = jets.sort_values(KEYS).reset_index(drop=True)

if MAX_JETS_FOR_SCAN is not None and len(jets) > MAX_JETS_FOR_SCAN:
    jets = jets.sample(MAX_JETS_FOR_SCAN, random_state=RANDOM_SEED).copy()
    jets = jets.sort_values(KEYS).reset_index(drop=True)

print("\nJets:")
print("Rows:", len(jets))
print(jets["truth_label"].value_counts())

print("\nConstituents:")
print("Rows:", len(constituents))


# ============================================================
# SELECT CONSTITUENTS / TRACKS
# ============================================================

selected = constituents.copy()

selected = selected[selected["constituent_pt"] > CONSTITUENT_PT_MIN].copy()

if USE_CHARGED_ONLY:
    selected = selected[selected["charge"] != 0].copy()

print("\nSelected constituents:")
print("Rows:", len(selected))
print("pT cut:", CONSTITUENT_PT_MIN)
print("Charged only:", USE_CHARGED_ONLY)


# ============================================================
# MERGE JET AXIS INTO CONSTITUENTS
# ============================================================

jet_axis = jets[KEYS + ["jet_eta", "jet_phi"]].copy()

merged = selected.merge(
    jet_axis,
    on=KEYS,
    how="inner"
)

merged["delta_eta"] = merged["constituent_eta"] - merged["jet_eta"]
merged["delta_phi"] = wrap_delta_phi(
    merged["constituent_phi"] - merged["jet_phi"]
)

print("\nMerged selected constituents:", len(merged))


# ============================================================
# BUILD BASE ARRAYS WITH MAX 50 CONSTITUENTS
# ============================================================

MAX_CONSTITUENTS_BASE = max(MAX_CONSTITUENTS_OPTIONS)

CONSTITUENT_FEATURES = [
    "log_pt",
    "delta_eta",
    "delta_phi",
    "log_mass",
    "charge",
    "pt_fraction"
]

N_FEATURES = len(CONSTITUENT_FEATURES)

X_all = np.zeros(
    (len(jets), MAX_CONSTITUENTS_BASE, N_FEATURES),
    dtype=np.float32
)

mask_all = np.zeros(
    (len(jets), MAX_CONSTITUENTS_BASE),
    dtype=np.float32
)

y_all = jets["label"].to_numpy(dtype=np.int64)

jet_key_to_index = {
    tuple(row[KEYS]): idx
    for idx, row in jets[KEYS].iterrows()
}

grouped = merged.groupby(KEYS)

filled_jets = 0

for key, group in grouped:
    if key not in jet_key_to_index:
        continue

    jet_idx = jet_key_to_index[key]

    group = group.sort_values("constituent_pt", ascending=False)
    group = group.head(MAX_CONSTITUENTS_BASE)

    pt = group["constituent_pt"].to_numpy(dtype=float)
    total_pt = np.sum(pt)

    if total_pt <= 0:
        continue

    n = len(group)

    X_all[jet_idx, :n, 0] = safe_log1p(pt)
    X_all[jet_idx, :n, 1] = group["delta_eta"].to_numpy(dtype=float)
    X_all[jet_idx, :n, 2] = group["delta_phi"].to_numpy(dtype=float)
    X_all[jet_idx, :n, 3] = safe_log1p(
        group["constituent_mass"].to_numpy(dtype=float)
    )
    X_all[jet_idx, :n, 4] = group["charge"].to_numpy(dtype=float)
    X_all[jet_idx, :n, 5] = pt / total_pt

    mask_all[jet_idx, :n] = 1.0
    filled_jets += 1

print("\nBuilt base transformer arrays:")
print("X_all shape:", X_all.shape)
print("mask_all shape:", mask_all.shape)
print("y_all shape:", y_all.shape)
print("Jets with at least one selected constituent:", filled_jets)

# Remove jets with no selected constituents
has_constituents = mask_all.sum(axis=1) > 0

X_all = X_all[has_constituents]
mask_all = mask_all[has_constituents]
y_all = y_all[has_constituents]
jets_used = jets[has_constituents].copy().reset_index(drop=True)

print("\nAfter removing empty jets:")
print("X_all shape:", X_all.shape)
print("Rows:", len(jets_used))
print(jets_used["truth_label"].value_counts())


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = np.arange(len(jets_used))

trainval_idx, test_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y_all
)

train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=y_all[trainval_idx]
)

print("\nSplit sizes:")
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))

print("\nTrain labels:")
print(jets_used.iloc[train_idx]["truth_label"].value_counts())

print("\nValidation labels:")
print(jets_used.iloc[val_idx]["truth_label"].value_counts())

print("\nTest labels:")
print(jets_used.iloc[test_idx]["truth_label"].value_counts())


# ============================================================
# MODEL DEFINITION
# ============================================================

class JetTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        max_constituents,
        d_model,
        n_heads,
        n_layers,
        dim_feedforward,
        dropout=0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(n_features, d_model)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_constituents, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, mask):
        # x shape: batch, max_constituents, n_features
        # mask shape: batch, max_constituents
        # mask = 1 for real constituent, 0 for padding

        x = self.input_projection(x)
        x = x + self.position_embedding[:, :x.shape[1], :]

        padding_mask = mask == 0

        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1)
        x = x * mask_expanded

        pooled = x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)

        logits = self.classifier(pooled)
        return logits


# ============================================================
# TRAINING / EVALUATION FUNCTIONS
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)


def make_dataloaders(max_constituents):
    X_cfg = X_all[:, :max_constituents, :].copy()
    mask_cfg = mask_all[:, :max_constituents].copy()

    X_train = X_cfg[train_idx]
    X_val = X_cfg[val_idx]
    X_test = X_cfg[test_idx]

    mask_train = mask_cfg[train_idx]
    mask_val = mask_cfg[val_idx]
    mask_test = mask_cfg[test_idx]

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    # Standardize using training constituents only
    valid_train = mask_train == 1

    feature_mean = X_train[valid_train].mean(axis=0)
    feature_std = X_train[valid_train].std(axis=0)
    feature_std[feature_std == 0] = 1.0

    def apply_standardization(X_array, mask_array):
        X_scaled = X_array.copy()
        valid = mask_array == 1
        X_scaled[valid] = (X_scaled[valid] - feature_mean) / feature_std
        X_scaled[mask_array == 0] = 0.0
        return X_scaled

    X_train = apply_standardization(X_train, mask_train)
    X_val = apply_standardization(X_val, mask_val)
    X_test = apply_standardization(X_test, mask_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(mask_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(mask_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def evaluate_model(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for xb, mb, yb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)

            logits = model(xb, mb)
            loss = criterion(logits, yb)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * len(yb)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    avg_loss = total_loss / len(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score
    }


def train_one_config(config, config_number):
    d_model = config["d_model"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    max_constituents = config["max_constituents"]

    dim_feedforward = 2 * d_model

    print("\n" + "=" * 80)
    print(f"Config {config_number}: {config['name']}")
    print("=" * 80)
    print(config)

    torch.manual_seed(RANDOM_SEED + config_number)

    train_loader, val_loader, test_loader = make_dataloaders(max_constituents)

    model = JetTransformer(
        n_features=N_FEATURES,
        max_constituents=max_constituents,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_auc = -1
    best_epoch = -1
    best_state = None

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        train_true = []
        train_pred = []
        train_score = []

        for xb, mb, yb in train_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb, mb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            total_train_loss += loss.item() * len(yb)

            train_true.extend(yb.cpu().numpy())
            train_pred.extend(preds.detach().cpu().numpy())
            train_score.extend(probs.detach().cpu().numpy())

        train_true = np.array(train_true)
        train_pred = np.array(train_pred)
        train_score = np.array(train_score)

        train_loss = total_train_loss / len(train_true)
        train_acc = accuracy_score(train_true, train_pred)
        train_auc = roc_auc_score(train_true, train_score)

        val_results = evaluate_model(model, val_loader, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "val_loss": val_results["loss"],
            "val_accuracy": val_results["accuracy"],
            "val_auc": val_results["auc"]
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train acc={train_acc:.4f} auc={train_auc:.4f} | "
            f"val acc={val_results['accuracy']:.4f} auc={val_results['auc']:.4f}"
        )

        if val_results["auc"] > best_val_auc:
            best_val_auc = val_results["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Restore best validation model
    model.load_state_dict(best_state)

    train_final = evaluate_model(model, train_loader, criterion)
    val_final = evaluate_model(model, val_loader, criterion)
    test_final = evaluate_model(model, test_loader, criterion)

    print("\nBest epoch:", best_epoch)
    print("Final train AUC:", train_final["auc"])
    print("Final val AUC:", val_final["auc"])
    print("Final test AUC:", test_final["auc"])
    print("Final test accuracy:", test_final["accuracy"])

    history_df = pd.DataFrame(history)
    history_path = os.path.join(
        SCAN_OUTPUT_DIR,
        f"{config['name']}_history.csv"
    )
    history_df.to_csv(history_path, index=False)

    result_row = {
        "name": config["name"],
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "max_constituents": max_constituents,
        "dim_feedforward": dim_feedforward,
        "best_epoch": best_epoch,
        "train_accuracy": train_final["accuracy"],
        "train_auc": train_final["auc"],
        "val_accuracy": val_final["accuracy"],
        "val_auc": val_final["auc"],
        "test_accuracy": test_final["accuracy"],
        "test_auc": test_final["auc"]
    }

    return result_row, model, test_final


# ============================================================
# CREATE CONFIGURATIONS
# ============================================================

configs = []

for d_model, n_heads, n_layers, max_constituents in itertools.product(
    D_MODEL_OPTIONS,
    N_HEADS_OPTIONS,
    N_LAYERS_OPTIONS,
    MAX_CONSTITUENTS_OPTIONS
):
    if d_model % n_heads != 0:
        continue

    name = (
        f"T_d{d_model}_h{n_heads}_l{n_layers}_c{max_constituents}"
    )

    configs.append({
        "name": name,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "max_constituents": max_constituents
    })

print("\nTransformer configurations to test:")
for c in configs:
    print(c["name"], c)

print("\nTotal configurations:", len(configs))


# ============================================================
# RUN SCAN
# ============================================================

all_results = []
best_model = None
best_test_results = None
best_config = None
best_val_auc_seen = -1

for i, config in enumerate(configs, start=1):
    result_row, model, test_results = train_one_config(config, i)

    all_results.append(result_row)

    if result_row["val_auc"] > best_val_auc_seen:
        best_val_auc_seen = result_row["val_auc"]
        best_model = copy.deepcopy(model)
        best_test_results = test_results
        best_config = result_row.copy()

    results_so_far = pd.DataFrame(all_results)
    results_so_far.to_csv(
        os.path.join(SCAN_OUTPUT_DIR, "transformer_scan_results_so_far.csv"),
        index=False
    )


# ============================================================
# SAVE FULL SCAN SUMMARY
# ============================================================

summary = pd.DataFrame(all_results)
summary_sorted = summary.sort_values("val_auc", ascending=False)

summary_path = os.path.join(SCAN_OUTPUT_DIR, "transformer_scan_summary.csv")
summary_sorted_path = os.path.join(SCAN_OUTPUT_DIR, "transformer_scan_summary_sorted_by_val_auc.csv")

summary.to_csv(summary_path, index=False)
summary_sorted.to_csv(summary_sorted_path, index=False)

print("\n" + "=" * 80)
print("TRANSFORMER HYPERPARAMETER SCAN SUMMARY")
print("=" * 80)
print(summary_sorted)

print("\nSaved:", summary_path)
print("Saved:", summary_sorted_path)


# ============================================================
# SAVE BEST TRANSFORMER IN STANDARD OUTPUT FOLDER
# ============================================================

best_summary = pd.DataFrame([{
    "method": "Transformer constituents",
    "selected_config": best_config["name"],
    "selection_metric": "validation AUC",
    "d_model": best_config["d_model"],
    "n_heads": best_config["n_heads"],
    "n_layers": best_config["n_layers"],
    "max_constituents": best_config["max_constituents"],
    "best_epoch": best_config["best_epoch"],
    "train_accuracy": best_config["train_accuracy"],
    "train_auc": best_config["train_auc"],
    "val_accuracy": best_config["val_accuracy"],
    "val_auc": best_config["val_auc"],
    "test_accuracy": best_config["test_accuracy"],
    "test_auc": best_config["test_auc"],
    "constituent_pt_min": CONSTITUENT_PT_MIN,
    "charged_only": USE_CHARGED_ONLY
}])

best_summary_scan_path = os.path.join(
    SCAN_OUTPUT_DIR,
    "best_transformer_summary.csv"
)

best_summary_standard_path = os.path.join(
    BEST_OUTPUT_DIR,
    "transformer_summary.csv"
)

best_summary.to_csv(best_summary_scan_path, index=False)
best_summary.to_csv(best_summary_standard_path, index=False)

print("\nBest transformer selected by validation AUC:")
print(best_summary)

print("\nSaved:", best_summary_scan_path)
print("Saved:", best_summary_standard_path)

# Save best model
model_path = os.path.join(BEST_OUTPUT_DIR, "transformer_model.pt")
torch.save(best_model.state_dict(), model_path)
print("Saved:", model_path)

# Save best test predictions
jets_test = jets_used.iloc[test_idx].copy().reset_index(drop=True)

test_output = jets_test.copy()
test_output["true_label_number"] = best_test_results["y_true"]
test_output["predicted_label_number"] = best_test_results["y_pred"]
test_output["predicted_label"] = np.where(
    best_test_results["y_pred"] == 1,
    "gluon",
    "quark"
)
test_output["predicted_gluon_probability"] = best_test_results["y_score"]
test_output["correct"] = test_output["truth_label"] == test_output["predicted_label"]

prediction_scan_path = os.path.join(
    SCAN_OUTPUT_DIR,
    "best_transformer_test_predictions.csv"
)

prediction_standard_path = os.path.join(
    BEST_OUTPUT_DIR,
    "transformer_test_predictions.csv"
)

test_output.to_csv(prediction_scan_path, index=False)
test_output.to_csv(prediction_standard_path, index=False)

print("Saved:", prediction_scan_path)
print("Saved:", prediction_standard_path)


# ============================================================
# PLOTS
# ============================================================

# Plot validation AUC for all configs
plot_df = summary_sorted.sort_values("val_auc", ascending=True)

plt.figure(figsize=(11, 8))
plt.barh(plot_df["name"], plot_df["val_auc"])

for i, value in enumerate(plot_df["val_auc"]):
    plt.text(value + 0.003, i, f"{value:.3f}", va="center")

plt.xlabel("Validation AUC")
plt.ylabel("Transformer configuration")
plt.title("Transformer hyperparameter scan: validation AUC")
plt.xlim(0, 1)
plt.tight_layout()

path = os.path.join(SCAN_OUTPUT_DIR, "transformer_validation_auc_comparison.png")
plt.savefig(path, dpi=200)
plt.close()
print("Saved:", path)

# Plot test AUC for all configs
plot_df = summary_sorted.sort_values("test_auc", ascending=True)

plt.figure(figsize=(11, 8))
plt.barh(plot_df["name"], plot_df["test_auc"])

for i, value in enumerate(plot_df["test_auc"]):
    plt.text(value + 0.003, i, f"{value:.3f}", va="center")

plt.xlabel("Test AUC")
plt.ylabel("Transformer configuration")
plt.title("Transformer hyperparameter scan: test AUC")
plt.xlim(0, 1)
plt.tight_layout()

path = os.path.join(SCAN_OUTPUT_DIR, "transformer_test_auc_comparison.png")
plt.savefig(path, dpi=200)
plt.close()
print("Saved:", path)

# Plot validation accuracy for all configs
plot_df = summary_sorted.sort_values("val_accuracy", ascending=True)

plt.figure(figsize=(11, 8))
plt.barh(plot_df["name"], plot_df["val_accuracy"])

for i, value in enumerate(plot_df["val_accuracy"]):
    plt.text(value + 0.003, i, f"{value:.3f}", va="center")

plt.xlabel("Validation accuracy")
plt.ylabel("Transformer configuration")
plt.title("Transformer hyperparameter scan: validation accuracy")
plt.xlim(0, 1)
plt.tight_layout()

path = os.path.join(SCAN_OUTPUT_DIR, "transformer_validation_accuracy_comparison.png")
plt.savefig(path, dpi=200)
plt.close()
print("Saved:", path)

# Best transformer ROC curve
RocCurveDisplay.from_predictions(
    best_test_results["y_true"],
    best_test_results["y_score"],
    name=f"Best transformer: {best_config['name']}"
)

plt.title("Best transformer ROC curve")
plt.tight_layout()

path = os.path.join(BEST_OUTPUT_DIR, "transformer_roc_curve.png")
plt.savefig(path, dpi=200)
plt.close()
print("Saved:", path)

# Confusion matrix for best transformer
cm = confusion_matrix(
    best_test_results["y_true"],
    best_test_results["y_pred"]
)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Best transformer confusion matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], ["quark", "gluon"])
plt.yticks([0, 1], ["quark", "gluon"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()

path = os.path.join(BEST_OUTPUT_DIR, "transformer_confusion_matrix.png")
plt.savefig(path, dpi=200)
plt.close()
print("Saved:", path)

print("\nBest transformer classification report:")
print(classification_report(
    best_test_results["y_true"],
    best_test_results["y_pred"],
    target_names=["quark", "gluon"]
))

print("\nDone.")
print("Full scan results folder:", SCAN_OUTPUT_DIR)
print("Best transformer results folder:", BEST_OUTPUT_DIR)
print("\nNow rerun compare_all_methods.py to include the best transformer.")
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
MAX_CONSTITUENTS = 50
CONSTITUENT_PT_MIN = 1.0
USE_CHARGED_ONLY = True   # "tracks" means charged constituents

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3

KEYS = ["run", "lumi", "event", "jet_index"]

OUTPUT_DIR = "transformer_constituent_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Sort so the train/test split is reproducible
jets = jets.sort_values(KEYS).reset_index(drop=True)

print("\nJets:")
print("Rows:", len(jets))
print(jets["truth_label"].value_counts())

print("\nConstituents:")
print("Rows:", len(constituents))


# ============================================================
# SELECT TRACKS / CONSTITUENTS
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
merged["delta_phi"] = wrap_delta_phi(merged["constituent_phi"] - merged["jet_phi"])

print("\nMerged selected constituents:", len(merged))


# ============================================================
# BUILD FIXED-SIZE SEQUENCE ARRAYS
# ============================================================

# Constituent-level input features:
# 1. log(1 + pT)
# 2. delta_eta relative to jet axis
# 3. delta_phi relative to jet axis
# 4. log(1 + mass)
# 5. charge
# 6. pT fraction inside selected constituents

CONSTITUENT_FEATURES = [
    "log_pt",
    "delta_eta",
    "delta_phi",
    "log_mass",
    "charge",
    "pt_fraction"
]

N_FEATURES = len(CONSTITUENT_FEATURES)

X = np.zeros((len(jets), MAX_CONSTITUENTS, N_FEATURES), dtype=np.float32)
mask = np.zeros((len(jets), MAX_CONSTITUENTS), dtype=np.float32)
y = jets["label"].to_numpy(dtype=np.int64)

# Create fast lookup from jet key to row number
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

    # Sort constituents by pT, highest first
    group = group.sort_values("constituent_pt", ascending=False).head(MAX_CONSTITUENTS)

    pt = group["constituent_pt"].to_numpy(dtype=float)
    total_pt = np.sum(pt)

    if total_pt <= 0:
        continue

    n = len(group)

    X[jet_idx, :n, 0] = safe_log1p(pt)
    X[jet_idx, :n, 1] = group["delta_eta"].to_numpy(dtype=float)
    X[jet_idx, :n, 2] = group["delta_phi"].to_numpy(dtype=float)
    X[jet_idx, :n, 3] = safe_log1p(group["constituent_mass"].to_numpy(dtype=float))
    X[jet_idx, :n, 4] = group["charge"].to_numpy(dtype=float)
    X[jet_idx, :n, 5] = pt / total_pt

    mask[jet_idx, :n] = 1.0
    filled_jets += 1

print("\nBuilt transformer arrays:")
print("X shape:", X.shape)
print("mask shape:", mask.shape)
print("y shape:", y.shape)
print("Jets with at least one selected constituent:", filled_jets)

# Remove jets with no selected constituents
has_constituents = mask.sum(axis=1) > 0

X = X[has_constituents]
mask = mask[has_constituents]
y = y[has_constituents]
jets_used = jets[has_constituents].copy().reset_index(drop=True)

print("\nAfter removing empty jets:")
print("X shape:", X.shape)
print("Rows:", len(jets_used))
print(jets_used["truth_label"].value_counts())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(len(jets_used))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y
)

X_train = X[train_idx]
X_test = X[test_idx]

mask_train = mask[train_idx]
mask_test = mask[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

jets_train = jets_used.iloc[train_idx].copy()
jets_test = jets_used.iloc[test_idx].copy()

print("\nTrain size:", len(y_train))
print("Test size:", len(y_test))

print("\nTraining label counts:")
print(jets_train["truth_label"].value_counts())

print("\nTesting label counts:")
print(jets_test["truth_label"].value_counts())


# ============================================================
# STANDARDIZE FEATURES USING TRAINING DATA ONLY
# ============================================================

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
X_test = apply_standardization(X_test, mask_test)

print("\nFeature standardization:")
for name, mean, std in zip(CONSTITUENT_FEATURES, feature_mean, feature_std):
    print(f"{name:15s} mean={mean: .4f} std={std: .4f}")


# ============================================================
# PYTORCH DATASETS
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

mask_train_t = torch.tensor(mask_train, dtype=torch.float32)
mask_test_t = torch.tensor(mask_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, mask_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, mask_test_t, y_test_t)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class JetTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        max_constituents,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
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

        # PyTorch wants True for padded positions
        padding_mask = mask == 0

        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # Masked mean pooling over real constituents only
        mask_expanded = mask.unsqueeze(-1)
        x = x * mask_expanded

        pooled = x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)

        logits = self.classifier(pooled)
        return logits


model = JetTransformer(
    n_features=N_FEATURES,
    max_constituents=MAX_CONSTITUENTS
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nModel:")
print(model)


# ============================================================
# TRAINING LOOP
# ============================================================

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(1, EPOCHS + 1):
    model.train()

    total_loss = 0.0
    train_preds = []
    train_true = []

    for xb, mb, yb in train_loader:
        xb = xb.to(device)
        mb = mb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb, mb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)

        preds = torch.argmax(logits, dim=1)

        train_preds.extend(preds.detach().cpu().numpy())
        train_true.extend(yb.detach().cpu().numpy())

    train_loss = total_loss / len(train_dataset)
    train_acc = accuracy_score(train_true, train_preds)

    model.eval()

    total_test_loss = 0.0
    test_preds = []
    test_true = []
    test_scores = []

    with torch.no_grad():
        for xb, mb, yb in test_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)

            logits = model(xb, mb)
            loss = criterion(logits, yb)

            total_test_loss += loss.item() * len(yb)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_true.extend(yb.cpu().numpy())
            test_scores.extend(probs.cpu().numpy())

    test_loss = total_test_loss / len(test_dataset)
    test_acc = accuracy_score(test_true, test_preds)
    test_auc = roc_auc_score(test_true, test_scores)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(
        f"Epoch {epoch:02d} | "
        f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
        f"test loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}"
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

model.eval()

final_preds = []
final_scores = []
final_true = []

with torch.no_grad():
    for xb, mb, yb in test_loader:
        xb = xb.to(device)
        mb = mb.to(device)

        logits = model(xb, mb)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        final_preds.extend(preds.cpu().numpy())
        final_scores.extend(probs.cpu().numpy())
        final_true.extend(yb.numpy())

final_preds = np.array(final_preds)
final_scores = np.array(final_scores)
final_true = np.array(final_true)

final_accuracy = accuracy_score(final_true, final_preds)
final_auc = roc_auc_score(final_true, final_scores)

print("\n" + "=" * 70)
print("FINAL TRANSFORMER RESULTS")
print("=" * 70)
print("Test accuracy:", final_accuracy)
print("Test AUC:", final_auc)

cm = confusion_matrix(final_true, final_preds)

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(
    final_true,
    final_preds,
    target_names=["quark", "gluon"]
))


# ============================================================
# SAVE OUTPUTS
# ============================================================

summary = pd.DataFrame([{
    "method": "Transformer constituents",
    "test_accuracy": final_accuracy,
    "test_auc": final_auc,
    "max_constituents": MAX_CONSTITUENTS,
    "constituent_pt_min": CONSTITUENT_PT_MIN,
    "charged_only": USE_CHARGED_ONLY,
    "epochs": EPOCHS
}])

summary_path = os.path.join(OUTPUT_DIR, "transformer_summary.csv")
summary.to_csv(summary_path, index=False)

print("\nSaved:", summary_path)

# Save test predictions
test_output = jets_test.copy().reset_index(drop=True)
test_output["true_label_number"] = final_true
test_output["predicted_label_number"] = final_preds
test_output["predicted_label"] = np.where(final_preds == 1, "gluon", "quark")
test_output["predicted_gluon_probability"] = final_scores
test_output["correct"] = test_output["truth_label"] == test_output["predicted_label"]

pred_path = os.path.join(OUTPUT_DIR, "transformer_test_predictions.csv")
test_output.to_csv(pred_path, index=False)

print("Saved:", pred_path)

# Save model
model_path = os.path.join(OUTPUT_DIR, "transformer_model.pt")
torch.save(model.state_dict(), model_path)

print("Saved:", model_path)


# ============================================================
# PLOTS
# ============================================================

# Training curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, label="train loss")
plt.plot(range(1, EPOCHS + 1), test_losses, label="test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer training and test loss")
plt.legend()
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "transformer_loss_curve.png")
plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

plt.figure(figsize=(8, 6))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label="train accuracy")
plt.plot(range(1, EPOCHS + 1), test_accuracies, label="test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transformer training and test accuracy")
plt.legend()
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "transformer_accuracy_curve.png")
plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

# Confusion matrix
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Transformer confusion matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], ["quark", "gluon"])
plt.yticks([0, 1], ["quark", "gluon"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "transformer_confusion_matrix.png")
plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

# ROC curve
RocCurveDisplay.from_predictions(final_true, final_scores)
plt.title("Transformer ROC curve")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "transformer_roc_curve.png")
plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

print("\nDone.")
print("Transformer results folder:", OUTPUT_DIR)
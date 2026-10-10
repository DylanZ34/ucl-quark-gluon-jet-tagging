import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

RESULTS_DIR = "supervised_baseline_results"

MODEL_FILES = {
    "logistic_regression": "logistic_regression_test_predictions.csv",
    "random_forest": "random_forest_test_predictions.csv",
    "simple_neural_network": "simple_neural_network_test_predictions.csv",
}

OUTPUT_DIR = os.path.join(RESULTS_DIR, "supervised_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_multiplicity_vs_ptD_by_prediction(df, model_name):
    plt.figure(figsize=(8, 6))

    for label in ["quark", "gluon"]:
        sub = df[df["predicted_label"] == label]
        plt.scatter(
            sub["multiplicity"],
            sub["ptD"],
            s=10,
            alpha=0.5,
            label=f"predicted {label}"
        )

    plt.xlabel("Multiplicity")
    plt.ylabel("pT_D")
    plt.title(f"Multiplicity vs pT_D, colored by {model_name} prediction")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{model_name}_multiplicity_vs_ptD_prediction.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


def plot_multiplicity_vs_ptD_by_truth(df, model_name):
    plt.figure(figsize=(8, 6))

    for label in ["quark", "gluon"]:
        sub = df[df["truth_label"] == label]
        plt.scatter(
            sub["multiplicity"],
            sub["ptD"],
            s=10,
            alpha=0.5,
            label=f"true {label}"
        )

    plt.xlabel("Multiplicity")
    plt.ylabel("pT_D")
    plt.title(f"Multiplicity vs pT_D, colored by truth label")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{model_name}_multiplicity_vs_ptD_truth.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


def plot_multiplicity_vs_ptD_correct_vs_wrong(df, model_name):
    df = df.copy()
    df["correct"] = df["truth_label"] == df["predicted_label"]

    plt.figure(figsize=(8, 6))

    correct = df[df["correct"] == True]
    wrong = df[df["correct"] == False]

    plt.scatter(
        correct["multiplicity"],
        correct["ptD"],
        s=10,
        alpha=0.35,
        label="correct"
    )

    plt.scatter(
        wrong["multiplicity"],
        wrong["ptD"],
        s=14,
        alpha=0.75,
        label="wrong"
    )

    plt.xlabel("Multiplicity")
    plt.ylabel("pT_D")
    plt.title(f"Multiplicity vs pT_D, correct vs wrong: {model_name}")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{model_name}_multiplicity_vs_ptD_correct_vs_wrong.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


def plot_multiplicity_vs_ptD_probability(df, model_name):
    if "predicted_gluon_probability" not in df.columns:
        print(f"Skipping probability plot for {model_name}: no probability column.")
        return

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        df["multiplicity"],
        df["ptD"],
        c=df["predicted_gluon_probability"],
        s=10,
        alpha=0.6
    )

    plt.xlabel("Multiplicity")
    plt.ylabel("pT_D")
    plt.title(f"Multiplicity vs pT_D, gluon probability: {model_name}")
    plt.colorbar(scatter, label="Predicted gluon probability")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{model_name}_multiplicity_vs_ptD_gluon_probability.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


def plot_mass_vs_multiplicity_by_prediction(df, model_name):
    plt.figure(figsize=(8, 6))

    for label in ["quark", "gluon"]:
        sub = df[df["predicted_label"] == label]
        plt.scatter(
            sub["multiplicity"],
            sub["jet_mass"],
            s=10,
            alpha=0.5,
            label=f"predicted {label}"
        )

    plt.xlabel("Multiplicity")
    plt.ylabel("Jet mass [GeV]")
    plt.title(f"Jet mass vs multiplicity, colored by {model_name} prediction")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{model_name}_mass_vs_multiplicity_prediction.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


def plot_N95_vs_ptD_by_prediction(df, model_name):
    plt.figure(figsize=(8, 6))

    for label in ["quark", "gluon"]:
        sub = df[df["predicted_label"] == label]
        plt.scatter(
            sub["N95"],
            sub["ptD"],
            s=10,
            alpha=0.5,
            label=f"predicted {label}"
        )

    plt.xlabel("N95")
    plt.ylabel("pT_D")
    plt.title(f"N95 vs pT_D, colored by {model_name} prediction")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{model_name}_N95_vs_ptD_prediction.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved:", path)


# ============================================================
# MAIN SCRIPT
# ============================================================

for model_name, filename in MODEL_FILES.items():
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        print("Missing file:", path)
        continue

    print("\n" + "=" * 70)
    print("Making plots for:", model_name)
    print("=" * 70)

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Truth labels:")
    print(df["truth_label"].value_counts())
    print("Predicted labels:")
    print(df["predicted_label"].value_counts())

    plot_multiplicity_vs_ptD_by_prediction(df, model_name)
    plot_multiplicity_vs_ptD_by_truth(df, model_name)
    plot_multiplicity_vs_ptD_correct_vs_wrong(df, model_name)
    plot_multiplicity_vs_ptD_probability(df, model_name)
    plot_mass_vs_multiplicity_by_prediction(df, model_name)
    plot_N95_vs_ptD_by_prediction(df, model_name)

print("\nDone.")
print("Plots saved in:", OUTPUT_DIR)
import glob
import numpy as np
import pandas as pd
import sklearn
import matplotlib

print("Python environment works.")
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)

feature_files = glob.glob("features_*/jet_substructure_features_quark_gluon_only.csv")

print("\nFeature files found:")
for f in feature_files:
    print(f)

if feature_files:
    feature_file = sorted(feature_files)[-1]
    df = pd.read_csv(feature_file)

    print("\nLoaded:", feature_file)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    print("\nTruth labels:")
    print(df["truth_label"].value_counts())
else:
    print("No feature file found. Check that the tar file was unpacked correctly.")
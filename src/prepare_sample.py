"""Stage the small supplied sample in the directory expected by the original scripts."""
from pathlib import Path
import shutil
root = Path(__file__).resolve().parents[1]
dest = root / 'work' / 'features_sample_extended'
dest.mkdir(parents=True, exist_ok=True)
shutil.copy2(root / 'data' / 'feature_sample.csv', dest / 'jet_substructure_features_extended_quark_gluon_only.csv')
print('Sample staged in work/. Run scripts from that directory; sample scores are not paper results.')

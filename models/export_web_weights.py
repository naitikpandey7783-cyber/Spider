"""
Exports the trained stylometric logistic-regression weights (Signal Engine B)
to a small JSON blob so the published browser demo can run real trained
coefficients client-side with zero server dependency. The full TF-IDF
vocabulary and n-gram LM are too large to ship in a lean single-page demo,
so the live artifact showcases the stylometric engine; the complete 3-engine
ensemble runs server-side via api/main.py.
"""
import json
import joblib
import numpy as np
import sys
sys.path.insert(0, "/home/claude/spidersense/models")
from features import FEATURE_FUNCS

ARTIFACTS = "/home/claude/spidersense/models/artifacts"
lr_style = joblib.load(f"{ARTIFACTS}/lr_stylometric.joblib")
mean = np.load(f"{ARTIFACTS}/style_norm_mean.npy")
std = np.load(f"{ARTIFACTS}/style_norm_std.npy")

weights = {
    "feature_names": list(FEATURE_FUNCS.keys()),
    "coef": lr_style.coef_[0].tolist(),
    "intercept": float(lr_style.intercept_[0]),
    "mean": mean.tolist(),
    "std": std.tolist(),
}
with open("/home/claude/spidersense/web/stylometric_weights.json", "w") as f:
    json.dump(weights, f, indent=2)
print(json.dumps(weights, indent=2))

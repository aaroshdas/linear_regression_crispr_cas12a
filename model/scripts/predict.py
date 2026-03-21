import argparse
import pickle
import numpy as np
from feature_engineering import build_features

MODEL_PATH = "../linear_results/ridge_model.pkl"
MEAN_PATH = "../weights/target_mean.npy"
STD_PATH = "../weights/target_std.npy"


def filter_sequence(seq: str) -> str:
    if len(seq) > 34:
        filtered = seq[:34]
        return filtered
    return seq

def predict(sequence: str):
    sequence = sequence.upper().strip()
    sequence = filter_sequence(sequence)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    t_mean = float(np.load(MEAN_PATH))
    t_std  = float(np.load(STD_PATH))

    features = build_features([sequence], include_one_hot=True).values.astype(np.float32)
    features = np.nan_to_num(features)

    pred_norm = model.predict(features)[0]
    pred_pct  = pred_norm * t_std + t_mean

    return pred_pct

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sequence", type=str)
    args = p.parse_args()

    result = predict(args.sequence)
    print(f"Predicted indel frequency: {result:.2f}%")
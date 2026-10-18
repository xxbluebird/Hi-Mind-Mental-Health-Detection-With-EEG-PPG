#!/usr/bin/env python3
# =============================================================================
# RASPBERRY PI INFERENCE — cuML SVM Boundary Manual CPU
# =============================================================================
# Cara pakai:
#
# 1) Prediksi dari CSV fitur:
#    python raspi_inference_stress.py \
#        --artifact cuml_svm_stress_poly_cpu_boundary_UPDATED_THRESHOLD.pkl \
#        --csv fitur_baru.csv \
#        --out prediksi_stress.csv
#
# CSV fitur_baru.csv wajib punya kolom fitur sesuai feature_order di artifact.
#
# 2) Output:
#    prob_positive, decision_score, y_pred, label_pred
# =============================================================================

import argparse
import pickle
import numpy as np
import pandas as pd


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def kernel_cpu(X, SV, config):
    X = np.asarray(X, dtype=np.float64)
    SV = np.asarray(SV, dtype=np.float64)

    kernel = str(config["kernel"]).lower()

    if kernel == "linear":
        return X @ SV.T

    if kernel == "rbf":
        gamma = float(config["gamma"])
        x_norm = np.sum(X ** 2, axis=1)[:, None]
        sv_norm = np.sum(SV ** 2, axis=1)[None, :]
        sq_dist = x_norm + sv_norm - 2.0 * (X @ SV.T)
        sq_dist = np.maximum(sq_dist, 0.0)
        return np.exp(-gamma * sq_dist)

    if kernel == "poly":
        gamma = float(config["gamma"])
        coef0 = float(config.get("coef0", 0.0))
        degree = int(config.get("degree", 3))
        return (gamma * (X @ SV.T) + coef0) ** degree

    if kernel == "sigmoid":
        gamma = float(config["gamma"])
        coef0 = float(config.get("coef0", 0.0))
        return np.tanh(gamma * (X @ SV.T) + coef0)

    raise ValueError(f"Kernel tidak didukung: {kernel}")


def portable_transform(X, imputer_statistics, scaler_info):
    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    X = np.clip(X, -1e6, 1e6)

    stats = np.asarray(imputer_statistics, dtype=np.float64)
    X = np.where(np.isnan(X), stats, X)

    scaler_type = scaler_info["type"]

    if scaler_type == "robust":
        center = np.asarray(scaler_info["center"], dtype=np.float64)
        scale = np.asarray(scaler_info["scale"], dtype=np.float64)
        return (X - center) / scale

    if scaler_type == "standard":
        mean = np.asarray(scaler_info["mean"], dtype=np.float64)
        scale = np.asarray(scaler_info["scale"], dtype=np.float64)
        return (X - mean) / scale

    if scaler_type == "minmax":
        scale = np.asarray(scaler_info["scale"], dtype=np.float64)
        min_ = np.asarray(scaler_info["min"], dtype=np.float64)
        return X * scale + min_

    raise ValueError(f"Scaler type tidak dikenali: {scaler_type}")


def predict_from_features_array(X_raw, artifact):
    feature_order = artifact["feature_order"]
    config = artifact["config"]

    imputer_statistics = artifact["imputer"]["statistics"]
    scaler_info = artifact["scaler"]

    support_vectors = artifact["support_vectors"]
    dual_coef = artifact["dual_coef"]
    intercept = artifact["intercept"]

    decision_sign = float(artifact.get("decision_sign", 1.0))
    A = float(artifact.get("platt_A", 1.0))
    B = float(artifact.get("platt_B", 0.0))

    X_sc = portable_transform(
        X_raw,
        imputer_statistics=imputer_statistics,
        scaler_info=scaler_info,
    )

    K = kernel_cpu(X_sc, support_vectors, config)
    decision = K @ dual_coef + float(intercept[0])
    decision = decision_sign * decision

    prob_positive = sigmoid(A * decision + B)

    threshold_mode = artifact.get("deployment_threshold_mode", "probability")

    if threshold_mode == "decision_score":
        threshold = float(artifact["deployment_decision_threshold"])
        direction = artifact.get("deployment_threshold_direction", "<=")

        if direction == "<=":
            y_pred = (decision <= threshold).astype(int)
        elif direction == ">=":
            y_pred = (decision >= threshold).astype(int)
        else:
            raise ValueError(f"Direction tidak dikenali: {direction}")

    else:
        threshold = float(artifact["segment_threshold"])
        direction = ">="
        y_pred = (prob_positive >= threshold).astype(int)

    label_pred = np.where(y_pred == 1, "Tidak normal", "Normal")

    return {
        "decision_score": decision,
        "prob_positive": prob_positive,
        "y_pred": y_pred,
        "label_pred": label_pred,
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        "direction": direction,
    }
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="Path artifact .pkl")
    parser.add_argument("--csv", required=True, help="CSV fitur input")
    parser.add_argument("--out", default="prediksi_stress.csv", help="CSV output")
    args = parser.parse_args()

    with open(args.artifact, "rb") as f:
        artifact = pickle.load(f)

    feature_order = artifact["feature_order"]

    df = pd.read_csv(args.csv)

    missing = [f for f in feature_order if f not in df.columns]
    if missing:
        raise ValueError(
            "CSV input tidak punya fitur berikut:\n" + "\n".join(missing)
        )

    X_raw = df[feature_order].copy()
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    X_raw = X_raw.values.astype(float)

    pred = predict_from_features_array(X_raw, artifact)

    df_out = df.copy()
    df_out["decision_score"] = pred["decision_score"]
    df_out["prob_positive"] = pred["prob_positive"]
    df_out["y_pred"] = pred["y_pred"]
    df_out["label_pred"] = pred["label_pred"]
    
    # Tambahan audit threshold deployment
    df_out["threshold_mode"] = pred["threshold_mode"]
    df_out["threshold"] = pred["threshold"]
    df_out["direction"] = pred["direction"]
    
    df_out.to_csv(args.out, index=False)


    df_out.to_csv(args.out, index=False)

    print("Selesai.")
    print("Output:", args.out)
    print(df_out[["decision_score", "prob_positive", "y_pred", "label_pred"]].head())


if __name__ == "__main__":
    main()

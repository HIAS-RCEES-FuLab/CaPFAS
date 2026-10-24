import os
import ast
import time
import gc
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# =========================
# Parameter
# =========================
test_dir = r"MS2_dataset_split_SMILES_type5_12_to_1\test"
model_dir = r"traditional_ML_results"
output_dir = os.path.join(model_dir, "traditional_test_results")
os.makedirs(output_dir, exist_ok=True)

bin_size = 0.1
base_bin_size = 0.01
msms_min_mz, msms_max_mz = 50, 1000
nl_min_mz, nl_max_mz = 12, 1000
feature_type = "Combined"
knn_predict_batch_size = 500
experiment_name = "MS2_dataset_split_SMILES_type5_12_to_1_bin0p1_P_N"

print("=" * 70)
print("Traditional Machine Learning - Test Prediction")
print(f"Test directory: {test_dir}")
print(f"Model directory: {model_dir}")
print(f"Output directory: {output_dir}")
print(f"Bin size: {bin_size}")
print(f"Ion modes: P + N")
print("=" * 70)

def get_bin_parameters(bin_size):
    return int(msms_min_mz / bin_size), int(msms_max_mz / bin_size), int(nl_min_mz / bin_size), int(nl_max_mz / bin_size)

def get_test_files(test_dir):
    file_list = []
    for mode in ["P", "N"]:
        mode_dir = os.path.join(test_dir, mode)
        for root, dirs, files in os.walk(mode_dir):
            for file in files:
                if file.lower().endswith(".csv"):
                    file_list.append((os.path.join(root, file), mode))
    return file_list

def process_msms_features(feature_string, bin_size, max_msms_bin):
    msms_x = np.zeros(max_msms_bin, dtype=np.float32)
    if pd.isna(feature_string):
        return msms_x
    try:
        feats = ast.literal_eval(feature_string)
    except Exception:
        return msms_x
    for bin_idx, inten in feats:
        try:
            mz = int(bin_idx) * base_bin_size
            inten = float(inten)
            if mz < msms_min_mz or mz >= msms_max_mz:
                continue
            new_bin = int(np.floor(mz / bin_size))
            if 0 <= new_bin < max_msms_bin and inten > msms_x[new_bin]:
                msms_x[new_bin] = inten
        except Exception:
            continue
    return msms_x

def process_nl_features(feature_string, bin_size, max_nl_bin):
    nl_x = np.zeros(max_nl_bin, dtype=np.float32)
    if pd.isna(feature_string):
        return nl_x
    try:
        nl_list = ast.literal_eval(feature_string)
    except Exception:
        return nl_x
    for bin_idx in nl_list:
        try:
            mz = int(bin_idx) * base_bin_size
            if mz < nl_min_mz or mz >= nl_max_mz:
                continue
            new_bin = int(np.floor(mz / bin_size))
            if 0 <= new_bin < max_nl_bin:
                nl_x[new_bin] = 1.0
        except Exception:
            continue
    return nl_x

def load_test_dataset(file_list, bin_size, max_msms_bin, max_nl_bin):
    X, y, metadata = [], [], []
    failed = 0
    for fpath, mode in file_list:
        print(f"Read: {fpath}")
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"Read fail: {fpath}\n{e}")
            continue
        ion_feature = 1.0 if mode == "P" else 0.0
        for row_idx, row in df.iterrows():
            try:
                label = int(row["CF2"])
                if label not in [0, 1]:
                    failed += 1
                    continue
                msms_x = process_msms_features(row["Feature_Mass_spectral"], bin_size, max_msms_bin) if "Feature_Mass_spectral" in df.columns else np.zeros(max_msms_bin, dtype=np.float32)
                mass_diff_x = process_nl_features(row["Feature_Mass_difference"], bin_size, max_nl_bin) if "Feature_Mass_difference" in df.columns else np.zeros(max_nl_bin, dtype=np.float32)
                neutral_loss_x = process_nl_features(row["Feature_Neutral_loss"], bin_size, max_nl_bin) if "Feature_Neutral_loss" in df.columns else np.zeros(max_nl_bin, dtype=np.float32)
                if feature_type == "Mass_difference":
                    nl_x = mass_diff_x
                elif feature_type == "Neutral_loss":
                    nl_x = neutral_loss_x
                elif feature_type == "Combined":
                    nl_x = np.maximum(mass_diff_x, neutral_loss_x)
                else:
                    raise ValueError(f"Unknown feature_type: {feature_type}")
                if "Feature_Exact_mass" in df.columns and "Feature_KMD_norm" in df.columns:
                    precursor_x = np.array([float(row["Feature_Exact_mass"]) / 1000.0, float(row["Feature_KMD_norm"]), ion_feature], dtype=np.float32)
                else:
                    precursor_x = np.zeros(3, dtype=np.float32)
                X.append(np.concatenate([msms_x, nl_x, precursor_x]))
                y.append(label)
                metadata.append({"Source_File": fpath, "Mode": mode, "Row_Index": row_idx})
            except Exception:
                failed += 1
                continue
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    metadata = pd.DataFrame(metadata)
    print(f"Loaded samples: {len(y)}")
    if len(y) > 0:
        print(f"PFAS: {(y == 1).sum()}, Non-PFAS: {(y == 0).sum()}")
    if X.ndim == 2:
        print(f"Feature dimension: {X.shape[1]}")
    print(f"Failed samples: {failed}")
    return X, y, metadata

def predict_model(model, X, model_name, batch_size=500):
    if model_name != "KNN":
        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
        return pred, prob
    pred_list, prob_list = [], []
    n_samples = len(X)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        print(f"KNN prediction: {start}/{n_samples}", end="\r")
        X_batch = X[start:end]
        pred_list.append(model.predict(X_batch))
        prob_list.append(model.predict_proba(X_batch)[:, 1])
        del X_batch
        gc.collect()
    print()
    return np.concatenate(pred_list), np.concatenate(prob_list)

def evaluate_model(y_true, pred, prob, model_name):
    acc = accuracy_score(y_true, pred)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(y_true, prob)
    pr_auc = average_precision_score(y_true, prob)
    print(f"{model_name}: Accuracy={acc:.4f} | ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f}")
    print(f"Test: TP={tp}, FN={fn}, TN={tn}, FP={fp}")
    return {"Model": model_name, "Test_Accuracy": acc, "Test_ROC_AUC": roc_auc, "Test_PR_AUC": pr_auc, "Test_TP": tp, "Test_FN": fn, "Test_TN": tn, "Test_FP": fp}

# =========================
# Main
# =========================
overall_start_time = time.time()
min_msms_bin, max_msms_bin, min_nl_bin, max_nl_bin = get_bin_parameters(bin_size)
print(f"MS2 dimension: {max_msms_bin}")
print(f"NL dimension: {max_nl_bin}")
print(f"Total feature dimension: {max_msms_bin + max_nl_bin + 3}")

test_files = get_test_files(test_dir)
print(f"Test CSV files: {len(test_files)}")

X_test, y_test, metadata = load_test_dataset(test_files, bin_size, max_msms_bin, max_nl_bin)
print(f"Test shape: {X_test.shape}")

model_patterns = [
    ("RF", f"{experiment_name}_RF.joblib"),
    ("SVM", f"{experiment_name}_SVM.joblib"),
    ("LOG", f"{experiment_name}_LOG.joblib"),
    ("KNN", f"{experiment_name}_KNN.joblib"),
    ("XGB", f"{experiment_name}_XGB.joblib")
]

all_results = []

for model_name, model_filename in model_patterns:
    model_file = os.path.join(model_dir, model_filename)
    print("\n" + "-" * 70)
    print(f"Loading: {model_name}")
    try:
        model = joblib.load(model_file)
        if hasattr(model, "n_features_in_"):
            print(f"Model features: {model.n_features_in_}, Test features: {X_test.shape[1]}")
        prediction_start = time.time()
        test_pred, test_prob = predict_model(model, X_test, model_name, knn_predict_batch_size)
        prediction_time = time.time() - prediction_start
        result = evaluate_model(y_test, test_pred, test_prob, model_name)
        result["Prediction_Time_seconds"] = prediction_time
        result["Model_File"] = model_file
        all_results.append(result)
        prediction_df = metadata.copy()
        prediction_df["True_Label"] = y_test
        prediction_df["Prediction"] = test_pred
        prediction_df["Probability"] = test_prob
        prediction_file = os.path.join(output_dir, f"{experiment_name}_{model_name}_test_predictions.csv")
        prediction_df.to_csv(prediction_file, index=False, encoding="utf-8-sig")
        print(f"Prediction saved: {prediction_file}")
        print(f"Prediction time: {prediction_time:.2f} s")
        del model, test_pred, test_prob, prediction_df
        gc.collect()
    except Exception as e:
        print(f"{model_name} predict fail: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()

results_df = pd.DataFrame(all_results)
summary_file = os.path.join(output_dir, "all_traditional_ML_test_results.csv")
results_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

test_metadata = metadata.copy()
test_metadata["True_Label"] = y_test
test_metadata_file = os.path.join(output_dir, "test_samples_metadata.csv")
test_metadata.to_csv(test_metadata_file, index=False, encoding="utf-8-sig")

overall_time = time.time() - overall_start_time
print("\n" + "=" * 70)
print(f"Test samples: {len(y_test)}")
print(f"Successful models: {len(all_results)}")
print(f"Total time: {overall_time:.2f} s ({overall_time / 60:.2f} min)")
print(f"Summary: {summary_file}")
print("=" * 70)
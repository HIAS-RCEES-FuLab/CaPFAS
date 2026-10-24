import os
import ast
import time
import gc
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
XGB_AVAILABLE = True

warnings.filterwarnings("ignore")

# =========================
# Parameter
# =========================
base_dirs = [r"MS2_dataset_split_SMILES_type5_12_to_1"]
ion_mode_sets = [["P", "N"]]
bin_sizes = [0.1]
base_bin_size = 0.01
msms_min_mz, msms_max_mz = 50, 1000
nl_min_mz, nl_max_mz = 12, 1000
feature_type = "Combined"
random_state = 42
model_dir = r"traditional_ML_results"
os.makedirs(model_dir, exist_ok=True)
knn_predict_batch_size = 500

print("=" * 70)
print("Traditional Machine Learning")
print(f"Datasets: {len(base_dirs)}")
print(f"Bin sizes: {bin_sizes}")
print(f"Ion mode sets: {ion_mode_sets}")
print("=" * 70)

def get_bin_parameters(bin_size):
    return int(msms_min_mz / bin_size), int(msms_max_mz / bin_size), int(nl_min_mz / bin_size), int(nl_max_mz / bin_size)

def get_files(split, ion_modes, base_dir):
    file_list = []
    for mode in ion_modes:
        split_dir = os.path.join(base_dir, split, mode)
        for root, dirs, files in os.walk(split_dir):
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

def load_dataset(file_list, bin_size, max_msms_bin, max_nl_bin):
    X, y = [], []
    failed = 0
    for fpath, mode in file_list:
        print(fpath)
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"Read error: {fpath}")
            print(e)
            continue
        ion_feature = 1.0 if mode == "P" else 0.0
        for _, row in df.iterrows():
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
            except Exception:
                failed += 1
                continue

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    print(f"Loaded samples: {len(y)}")
    if len(y) > 0:
        print(f"PFAS: {(y == 1).sum()}, Non-PFAS: {(y == 0).sum()}")
    if X.ndim == 2:
        print(f"Feature dimension: {X.shape[1]}")
    print(f"Failed samples: {failed}")
    return X, y

def get_models():
    models = {
        "RF": RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=random_state),
        "SVM": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=random_state))
        ]),
        "LOG": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", solver="saga", max_iter=100, tol=1e-3, random_state=random_state, n_jobs=-1))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1))
        ])
    }
    if XGB_AVAILABLE:
        models["XGB"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.2, min_child_weight=1, objective="binary:logistic", eval_metric="logloss", random_state=random_state, n_jobs=-1, tree_method="hist")
    return models

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

def train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val, experiment_name):
    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training finished: {training_time:.2f} s")

    model_file = os.path.join(model_dir, f"{experiment_name}_{model_name}.joblib")
    joblib.dump(model, model_file)
    print(f"Model saved: {model_file}")

    prediction_start = time.time()
    train_pred, train_prob = predict_model(model, X_train, model_name, knn_predict_batch_size)
    val_pred, val_prob = predict_model(model, X_val, model_name, knn_predict_batch_size)
    prediction_time = time.time() - prediction_start

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train, train_pred, labels=[0, 1]).ravel()
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()
    train_roc_auc = roc_auc_score(y_train, train_prob)
    val_roc_auc = roc_auc_score(y_val, val_prob)
    train_pr_auc = average_precision_score(y_train, train_prob)
    val_pr_auc = average_precision_score(y_val, val_prob)

    result = {
        "Model": model_name,
        "Experiment": experiment_name,
        "Train_Accuracy": train_acc,
        "Train_ROC_AUC": train_roc_auc,
        "Train_PR_AUC": train_pr_auc,
        "Train_TP": train_tp,
        "Train_FN": train_fn,
        "Train_TN": train_tn,
        "Train_FP": train_fp,
        "Val_Accuracy": val_acc,
        "Val_ROC_AUC": val_roc_auc,
        "Val_PR_AUC": val_pr_auc,
        "Val_TP": val_tp,
        "Val_FN": val_fn,
        "Val_TN": val_tn,
        "Val_FP": val_fp,
        "Training_Time_seconds": training_time,
        "Prediction_Time_seconds": prediction_time,
        "Model_File": model_file
    }

    print(f"Train Accuracy: {train_acc:.4f} | ROC-AUC: {train_roc_auc:.4f} | PR-AUC: {train_pr_auc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f} | ROC-AUC: {val_roc_auc:.4f} | PR-AUC: {val_pr_auc:.4f}")
    print(f"Val: TP={val_tp}, FN={val_fn}, TN={val_tn}, FP={val_fp}")
    print(f"Training time: {training_time:.2f} s")
    print(f"Prediction time: {prediction_time:.2f} s")

    prediction_df = pd.DataFrame({"True_Label": y_val, "Prediction": val_pred, "Probability": val_prob})
    prediction_file = os.path.join(model_dir, f"{experiment_name}_{model_name}_predictions.csv")
    prediction_df.to_csv(prediction_file, index=False)
    print(f"Prediction saved: {prediction_file}")

    del train_pred, train_prob, val_pred, val_prob
    gc.collect()
    return result

# =========================
# Main
# =========================
overall_start_time = time.time()
all_results = []
total_experiments = len(base_dirs) * len(bin_sizes) * len(ion_mode_sets)
experiment_count = 0

for base_dir in base_dirs:
    for bin_size in bin_sizes:
        for ion_modes in ion_mode_sets:
            experiment_count += 1
            db_name = os.path.basename(os.path.normpath(base_dir))
            ion_name = "_".join(ion_modes)
            bin_name = str(bin_size).replace(".", "p")
            experiment_name = f"{db_name}_bin{bin_name}_{ion_name}"

            print("\n" + "=" * 70)
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Dataset: {db_name}")
            print(f"Bin size: {bin_size}")
            print(f"Ion modes: {ion_modes}")
            print("=" * 70)

            min_msms_bin, max_msms_bin, min_nl_bin, max_nl_bin = get_bin_parameters(bin_size)
            print(f"MS2 dimension: {max_msms_bin}")
            print(f"NL dimension: {max_nl_bin}")
            print(f"Total feature dimension: {max_msms_bin + max_nl_bin + 3}")

            train_files = get_files("train", ion_modes, base_dir)
            val_files = get_files("val", ion_modes, base_dir)
            print(f"Train CSV: {len(train_files)}")
            print(f"Val CSV: {len(val_files)}")

            X_train, y_train = load_dataset(train_files, bin_size, max_msms_bin, max_nl_bin)
            X_val, y_val = load_dataset(val_files, bin_size, max_msms_bin, max_nl_bin)

            print(f"Train shape: {X_train.shape}")
            print(f"Val shape: {X_val.shape}")

            models = get_models()
            print("\nModels: " + ", ".join(models.keys()))

            for model_name, model in models.items():
                try:
                    result = train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val, experiment_name)
                    all_results.append(result)
                    del model
                    gc.collect()
                except Exception as e:
                    print(f"\n{model_name} fail:")
                    print(e)
                    import traceback
                    traceback.print_exc()
                    gc.collect()

            del X_train, X_val, y_train, y_val
            gc.collect()

results_df = pd.DataFrame(all_results)
summary_file = os.path.join(model_dir, "all_traditional_ML_results.csv")
results_df.to_csv(summary_file, index=False)
overall_time = time.time() - overall_start_time

print("\n" + "=" * 70)
print(f"Total experiments: {total_experiments}")
print(f"Models per experiment: {len(get_models())}")
print(f"Successful models: {len(all_results)}")
print(f"Total time: {overall_time:.2f} s ({overall_time / 60:.2f} min)")
print(f"Results: {summary_file}")
print("=" * 70)
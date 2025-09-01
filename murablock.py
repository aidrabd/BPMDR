import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")


def calculate_descriptors(smiles_list, top_morgan_bits=25):
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    feature_names = []

    rdkit_desc_names = [x[0] for x in Descriptors._descList]
    lipinski_names = [
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "NumHeteroatoms",
    ]
    feature_names.extend(rdkit_desc_names)
    feature_names.extend(lipinski_names)

    n_morgan_bits = 1024
    maccs_names = [f"MACCS_{i}" for i in range(167)]

    descriptors_list = []
    morgan_fps_list_full = []
    maccs_fps_list = []
    estate_fps_list = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            descriptors_list.append([np.nan] * len(rdkit_desc_names + lipinski_names))
            morgan_fps_list_full.append([0] * n_morgan_bits)
            maccs_fps_list.append([0] * 167)
            estate_fps_list.append([np.nan] * 79)
            continue

        desc = descriptor_calculator.CalcDescriptors(mol)
        desc = [float(x) if x is not None else np.nan for x in desc]

        lipinski_desc = [
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.NumAliphaticRings(mol),
            Lipinski.NumSaturatedRings(mol),
            Lipinski.NumHeteroatoms(mol),
        ]
        lipinski_desc = [float(x) for x in lipinski_desc]

        descriptors_list.append(desc + lipinski_desc)

        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_morgan_bits)
        morgan_fps_list_full.append(list(morgan_fp))

        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_fps_list.append(list(maccs_fp))

        estate_fp = Fingerprinter.FingerprintMol(mol)[0]
        estate_fp = [float(x) if x is not None else np.nan for x in estate_fp]
        estate_fps_list.append(estate_fp)

    estate_names = [f"Estate_{i}" for i in range(len(estate_fps_list[0]))]

    morgan_array_full = np.array(morgan_fps_list_full, dtype=np.float64)

    variances = np.var(morgan_array_full, axis=0)
    top_morgan_indices = np.argsort(variances)[::-1][:top_morgan_bits]
    top_morgan_indices = np.sort(top_morgan_indices)

    morgan_array = morgan_array_full[:, top_morgan_indices]
    morgan_names = [f"Morgan_{i}" for i in top_morgan_indices]

    feature_names.extend(morgan_names)
    feature_names.extend(maccs_names)
    feature_names.extend(estate_names)

    descriptors_array = np.array(descriptors_list, dtype=np.float64)
    maccs_array = np.array(maccs_fps_list, dtype=np.float64)
    estate_array = np.array(estate_fps_list, dtype=np.float64)

    X = np.hstack([descriptors_array, morgan_array, maccs_array, estate_array])

    return X, feature_names


def main():
    print("=== QSAR Training: murablock.py ===")
    csv_path = input("Enter path to your CSV file with 'SMILES' and 'pIC50' columns: ")
    df = pd.read_csv(csv_path, encoding="latin1")

    if "SMILES" not in df.columns or "pIC50" not in df.columns:
        print("Input CSV must contain 'SMILES' and 'pIC50' columns.")
        return

    smiles = df["SMILES"].tolist()
    y = df["pIC50"].values.astype(float)

    print("Calculating descriptors and fingerprints for molecules (top 25 Morgan bits)...")
    X, feature_names = calculate_descriptors(smiles, top_morgan_bits=25)

    X[np.isinf(X)] = np.nan

    nan_cols = np.isnan(X).any(axis=0)
    nan_mask = ~nan_cols
    X = X[:, nan_mask]
    feature_names = [f for f, keep in zip(feature_names, nan_mask) if keep]
    print(f"Removed {np.sum(nan_cols)} columns with NaNs.")

    vt = VarianceThreshold()
    X = vt.fit_transform(X)
    variance_mask = vt.get_support()
    feature_names = [f for f, keep in zip(feature_names, variance_mask) if keep]
    print(f"Removed {np.sum(~variance_mask)} zero variance columns.")

    final_feature_mask = np.zeros(nan_mask.shape, dtype=bool)
    final_feature_mask[nan_mask] = variance_mask

    output_dir = "qsar_models"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(final_feature_mask, os.path.join(output_dir, "final_feature_mask.joblib"))

    # Split dataset 65:20:15
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    val_ratio_adjusted = 0.20 / (1 - 0.15)  # 0.20 / 0.85 ~ 0.235294
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=42
    )

    print(f"Data split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    f_scores = f_regression(X_train, y_train)[0]
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    f_norm = (f_scores - np.min(f_scores)) / (np.max(f_scores) - np.min(f_scores) + 1e-10)
    mi_norm = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-10)
    combined_scores = 0.5 * f_norm + 0.5 * mi_norm

    top_k = 50
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    X_train_sel = X_train[:, top_indices]
    X_val_sel = X_val[:, top_indices]
    X_test_sel = X_test[:, top_indices]

    selected_features = [feature_names[i] for i in top_indices]

    joblib.dump(top_indices, os.path.join(output_dir, "selected_feature_indices.joblib"))
    joblib.dump(selected_features, os.path.join(output_dir, "selected_feature_names.joblib"))

    print(f"Selected top {top_k} features.")

    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_sel, y_train)
        joblib.dump(model, os.path.join(output_dir, f"{name}_model.joblib"))

    print(f"All models trained and saved to '{output_dir}/'")

    def save_preds(X_sel, y_true, set_name):
        start_idx = 0
        if set_name == "Validation":
            start_idx = len(X_train)
        elif set_name == "Test":
            start_idx = len(X_train) + len(X_val)

        df_preds = pd.DataFrame(
            {
                "SMILES": np.array(smiles)[start_idx : start_idx + len(X_sel)],
                "Actual_pIC50": y_true,
            }
        )

        for name, model in models.items():
            preds = model.predict(X_sel)
            df_preds[f"Pred_{name}"] = preds

        df_preds["Set"] = set_name
        df_preds.to_csv(os.path.join(output_dir, f"{set_name.lower()}_predictions.csv"), index=False)

    save_preds(X_train_sel, y_train, "Train")
    save_preds(X_val_sel, y_val, "Validation")
    save_preds(X_test_sel, y_test, "Test")

    print("Prediction CSVs for train, validation, and test sets saved.")


if __name__ == "__main__":
    main()
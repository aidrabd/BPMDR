import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter
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


def predict_new(csv_path, output_file):
    print(f"Loading new data from {csv_path}...")
    df = pd.read_csv(csv_path, encoding="latin1")
    if "SMILES" not in df.columns:
        print("Input CSV must contain a 'SMILES' column.")
        return

    smiles = df["SMILES"].tolist()

    print("Calculating descriptors for new molecules (top 25 Morgan bits)...")
    X, feature_names = calculate_descriptors(smiles, top_morgan_bits=25)

    output_dir = "qsar_models"
    scaler = joblib.load(os.path.join(output_dir, "scaler.joblib"))
    top_indices = joblib.load(os.path.join(output_dir, "selected_feature_indices.joblib"))
    final_feature_mask = joblib.load(os.path.join(output_dir, "final_feature_mask.joblib"))

    X[np.isinf(X)] = np.nan

    if final_feature_mask.shape[0] != X.shape[1]:
        print(
            f"Error: Feature mask length {final_feature_mask.shape[0]} does not match descriptor count {X.shape[1]}."
        )
        print("Make sure the descriptor calculation code is consistent between training and prediction.")
        return

    X = X[:, final_feature_mask]

    X_scaled_full = scaler.transform(X)

    if max(top_indices) >= X_scaled_full.shape[1]:
        print("Warning: Feature indices exceed feature matrix size. Predictions may be inaccurate.")
        top_indices = [i for i in top_indices if i < X_scaled_full.shape[1]]

    X_selected = X_scaled_full[:, top_indices]

    model_files = [f for f in os.listdir(output_dir) if f.endswith("_model.joblib")]
    models = {}
    for mf in model_files:
        name = mf.replace("_model.joblib", "")
        if name in {"random_forest", "gradient_boosting"}:
            models[name] = joblib.load(os.path.join(output_dir, mf))

    print("Predicting pIC50 values with selected models...")

    for name, model in models.items():
        preds = model.predict(X_selected)
        df[f"Pred_pIC50_{name}"] = preds

    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    print("=== QSAR Prediction: predict.py ===")
    input_csv = input("Enter path to CSV file with SMILES to predict: ")
    output_filename = input("Enter output CSV filename for predictions: ")
    predict_new(input_csv, output_filename)
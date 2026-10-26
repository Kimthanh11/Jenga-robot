
import csv
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from train_jenga_presentation import (
    JengaCNN,
    JengaDataset,
    denormalize_com,
)

# IMPORTANT: Import the ORIGINAL random generator,
# not the balanced generator.
from generate_towers import (
    generate_tower,
    tower_to_row,
    MAX_LAYERS,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "jenga_cnn.pth"
TEST_CSV = "towers_random_test.csv"
OUTPUT_CSV = "jenga_random_test_predictions.csv"

NUMBER_TEST_TOWERS = 10000
BATCH_SIZE = 128
SEED = 2026

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 1. GENERATE INDEPENDENT RANDOM TEST DATA
# ============================================================

def generate_random_test_dataset():

    random.seed(SEED)

    fieldnames = (
        ["Tower_ID"]
        + [f"Layer_{i}" for i in range(1, MAX_LAYERS + 1)]
        + ["Stable", "CoM"]
    )

    stable_count = 0
    unstable_count = 0

    with open(
        TEST_CSV, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()

        for tower_id in range(1, NUMBER_TEST_TOWERS + 1):

            # Original random generation only
            blocks, blocks_per_layer = generate_tower(
                mode="random"
            )

            row = tower_to_row(
                tower_id,
                blocks,
                blocks_per_layer
            )

            writer.writerow(row)

            if row["Stable"] == 1:
                stable_count += 1
            else:
                unstable_count += 1

    print("\nRandom test dataset generated")
    print("Total:", NUMBER_TEST_TOWERS)
    print("Stable:", stable_count)
    print("Unstable:", unstable_count)


# ============================================================
# 2. LOAD TRAINED CNN
# ============================================================

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = JengaCNN().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nLoaded trained model:", MODEL_PATH)

    return model


# ============================================================
# 3. EVALUATE
# ============================================================

def test_model(model):

    dataset = JengaDataset(TEST_CSV)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_com_true = []
    all_com_pred = []

    all_stable_true = []
    all_stable_prob = []

    with torch.no_grad():

        for inputs, com, stable in loader:

            inputs = inputs.to(DEVICE)

            pred_com, pred_logits = model(inputs)

            probabilities = torch.sigmoid(
                pred_logits.squeeze(1)
            )

            all_com_true.append(com.numpy())
            all_com_pred.append(
                pred_com.cpu().numpy()
            )

            all_stable_true.append(stable.numpy())
            all_stable_prob.append(
                probabilities.cpu().numpy()
            )

    com_true = np.concatenate(all_com_true)
    com_pred = np.concatenate(all_com_pred)

    stable_true = np.concatenate(
        all_stable_true
    ).astype(int)

    stable_prob = np.concatenate(
        all_stable_prob
    )

    stable_pred = (stable_prob >= 0.5).astype(int)

    # Convert CoM predictions to meters
    com_true_m = denormalize_com(com_true)
    com_pred_m = denormalize_com(com_pred)

    # ========================================================
    # STABILITY RESULTS
    # ========================================================

    print("\n========== STABILITY RESULTS ==========")

    accuracy = accuracy_score(
        stable_true,
        stable_pred
    )

    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(
        stable_true,
        stable_pred,
        labels=[0, 1],
        target_names=["Unstable", "Stable"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        stable_true,
        stable_pred,
        labels=[0, 1]
    ))

    if len(np.unique(stable_true)) == 2:
        auc = roc_auc_score(
            stable_true,
            stable_prob
        )
        print("ROC-AUC:", auc)

    # ========================================================
    # COM RESULTS
    # ========================================================

    print("\n========== CoM RESULTS ==========")

    mae = mean_absolute_error(
        com_true_m,
        com_pred_m,
        multioutput="raw_values"
    )

    rmse = np.sqrt(mean_squared_error(
        com_true_m,
        com_pred_m,
        multioutput="raw_values"
    ))

    print("CoM MAE [x, y, z] (meters):", mae)
    print("CoM RMSE [x, y, z] (meters):", rmse)

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    results = pd.DataFrame({
        "Tower_ID": dataset.tower_ids,

        "Actual_CoM_X": com_true_m[:, 0],
        "Actual_CoM_Y": com_true_m[:, 1],
        "Actual_CoM_Z": com_true_m[:, 2],

        "Pred_CoM_X": com_pred_m[:, 0],
        "Pred_CoM_Y": com_pred_m[:, 1],
        "Pred_CoM_Z": com_pred_m[:, 2],

        "Actual_Stable": stable_true,
        "Pred_Stable": stable_pred,
        "Stable_Probability": stable_prob,
    })

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nPredictions saved to:", OUTPUT_CSV)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_random_test_dataset()

    model = load_model()

    test_model(model)

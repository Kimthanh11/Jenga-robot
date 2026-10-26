
import ast
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

CSV_PATH = "towers_dataset.csv"
MODEL_PATH = "jenga_cnn.pth"
PREDICTIONS_PATH = "jenga_test_predictions.csv"

NUM_LAYERS = 27
MAX_BLOCKS = 3
FEATURES_PER_LAYER = 7

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
SEED = 42

COM_WEIGHT = 1.0
STABLE_WEIGHT = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# 2. POSITION NORMALIZATION
# ============================================================

# Fixed geometric center, NOT the ground-truth CoM.
# The XY scale covers the expected block centers.
XY_CENTER = 0.0
XY_SCALE = 0.08

LAYER_SPACING = 0.0305
Z_MIN = 0.0
Z_MAX = NUM_LAYERS * LAYER_SPACING

Z_CENTER = (Z_MIN + Z_MAX) / 2.0
Z_SCALE = (Z_MAX - Z_MIN) / 2.0

COM_SCALE = np.array(
    [XY_SCALE, XY_SCALE, Z_SCALE],
    dtype=np.float32
)

COM_CENTER = np.array(
    [XY_CENTER, XY_CENTER, Z_CENTER],
    dtype=np.float32
)


def normalize_position(x, y, z):
    return (
        (x - XY_CENTER) / XY_SCALE,
        (y - XY_CENTER) / XY_SCALE,
        (z - Z_CENTER) / Z_SCALE,
    )


def normalize_com(com):
    return (com - COM_CENTER) / COM_SCALE


def denormalize_com(com):
    return com * COM_SCALE + COM_CENTER


# ============================================================
# 3. DATASET
# ============================================================

class JengaDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Shuffle all rows before preprocessing
        df = df.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        inputs = []
        com_targets = []
        stable_targets = []
        tower_ids = []

        for _, row in df.iterrows():

            # 27 layers, 7 features per layer
            tower = np.zeros(
                (NUM_LAYERS, FEATURES_PER_LAYER),
                dtype=np.float32
            )

            for layer_idx in range(NUM_LAYERS):
                column = f"Layer_{layer_idx + 1}"
                cell = row[column]

                if pd.isna(cell) or str(cell).strip() == "":
                    continue

                blocks = ast.literal_eval(str(cell))

                if not blocks:
                    continue

                # Sort blocks consistently along their
                # layer's primary horizontal axis.
                axis = 0 if layer_idx % 2 == 0 else 1
                blocks = sorted(
                    blocks,
                    key=lambda block: block[axis]
                )

                # [z, x1, y1, x2, y2, x3, y3]
                _, _, z_norm = normalize_position(
                    0.0, 0.0, blocks[0][2]
                )

                tower[layer_idx, 0] = z_norm

                for block_idx, block in enumerate(blocks[:MAX_BLOCKS]):
                    x, y, z, yaw = block

                    x_norm, y_norm, _ = normalize_position(
                        x, y, z
                    )

                    tower[layer_idx, 1 + block_idx * 2] = x_norm
                    tower[layer_idx, 2 + block_idx * 2] = y_norm

            com = np.array(
                ast.literal_eval(str(row["CoM"])),
                dtype=np.float32
            )

            stable = float(row["Stable"])

            if stable not in (0.0, 1.0):
                raise ValueError(
                    f"Invalid stability label: {stable}"
                )

            inputs.append(tower)
            com_targets.append(normalize_com(com))
            stable_targets.append(stable)
            tower_ids.append(row["Tower_ID"])

        self.inputs = torch.tensor(
            np.stack(inputs),
            dtype=torch.float32
        )

        self.coms = torch.tensor(
            np.stack(com_targets),
            dtype=torch.float32
        )

        self.stables = torch.tensor(
            stable_targets,
            dtype=torch.float32
        )

        self.tower_ids = tower_ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.coms[idx],
            self.stables[idx]
        )


# ============================================================
# 4. CNN MODEL
# ============================================================

class JengaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=2),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=2),
            nn.ReLU(),

            nn.Flatten(),

            # 27 -> 26 -> 25 layers
            nn.Linear(64 * 25, 32),
            nn.ReLU(),
        )

        # Predict normalized CoM: x, y, z
        self.com_head = nn.Linear(32, 3)

        # Predict stability logit
        self.stable_head = nn.Linear(32, 1)

    def forward(self, x):
        # Input: (batch, 27, 7)
        x = x.transpose(1, 2)

        # Shared CNN features: (batch, 32)
        features = self.cnn(x)

        com = self.com_head(features)
        stable_logits = self.stable_head(features)

        return com, stable_logits


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate(model, loader, com_criterion, stable_criterion):
    model.eval()

    total_loss = 0.0
    all_com_true = []
    all_com_pred = []
    all_stable_true = []
    all_stable_prob = []

    with torch.no_grad():
        for inputs, com, stable in loader:
            inputs = inputs.to(DEVICE)
            com = com.to(DEVICE)
            stable = stable.to(DEVICE)

            pred_com, pred_logits = model(inputs)
            pred_logits = pred_logits.squeeze(1)

            loss_com = com_criterion(pred_com, com)
            loss_stable = stable_criterion(
                pred_logits, stable
            )

            loss = (
                COM_WEIGHT * loss_com
                + STABLE_WEIGHT * loss_stable
            )

            total_loss += loss.item() * len(inputs)

            all_com_true.append(com.cpu().numpy())
            all_com_pred.append(pred_com.cpu().numpy())
            all_stable_true.append(stable.cpu().numpy())
            all_stable_prob.append(
                torch.sigmoid(pred_logits).cpu().numpy()
            )

    com_true = np.concatenate(all_com_true)
    com_pred = np.concatenate(all_com_pred)
    stable_true = np.concatenate(all_stable_true).astype(int)
    stable_prob = np.concatenate(all_stable_prob)
    stable_pred = (stable_prob >= 0.5).astype(int)

    # Convert CoM back into meters
    com_true_m = denormalize_com(com_true)
    com_pred_m = denormalize_com(com_pred)

    com_mae = mean_absolute_error(
        com_true_m,
        com_pred_m
    )

    stable_acc = accuracy_score(
        stable_true,
        stable_pred
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "com_mae": com_mae,
        "stable_acc": stable_acc,
        "com_true": com_true_m,
        "com_pred": com_pred_m,
        "stable_true": stable_true,
        "stable_pred": stable_pred,
        "stable_prob": stable_prob,
    }


# ============================================================
# 6. TRAINING
# ============================================================

def main():
    print("Device:", DEVICE)

    dataset = JengaDataset(CSV_PATH)

    print("Number of towers:", len(dataset))
    print("Input shape:", dataset.inputs.shape)

    unique, counts = np.unique(
        dataset.stables.numpy(),
        return_counts=True
    )
    print("Stability distribution:", dict(zip(unique, counts)))

    if len(dataset) < 3:
        raise ValueError("At least 3 towers are required.")

    n_train = int(0.70 * len(dataset))
    n_val = int(0.15 * len(dataset))
    n_test = len(dataset) - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE
    )

    model = JengaCNN().to(DEVICE)

    com_criterion = nn.MSELoss()

    # Compensate for class imbalance using training labels.
    train_labels = dataset.stables[train_set.indices]
    n_positive = train_labels.sum().item()
    n_negative = len(train_labels) - n_positive

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "Training data must contain both Stable=0 and Stable=1."
        )

    stable_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for inputs, com, stable in train_loader:
            inputs = inputs.to(DEVICE)
            com = com.to(DEVICE)
            stable = stable.to(DEVICE)

            optimizer.zero_grad()

            pred_com, pred_logits = model(inputs)
            pred_logits = pred_logits.squeeze(1)

            loss_com = com_criterion(pred_com, com)
            loss_stable = stable_criterion(
                pred_logits, stable
            )

            loss = (
                COM_WEIGHT * loss_com
                + STABLE_WEIGHT * loss_stable
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(inputs)

        train_loss = total_train_loss / len(train_set)

        val_results = evaluate(
            model,
            val_loader,
            com_criterion,
            stable_criterion
        )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_results['loss']:.5f} | "
            f"CoM MAE: {val_results['com_mae']:.6f} m | "
            f"Stable Acc: {val_results['stable_acc']:.4f}"
        )

    # ========================================================
    # 7. SAVE BEST MODEL
    # ========================================================

    model.load_state_dict(best_state)

    torch.save({
        "model_state_dict": best_state,
        "num_layers": NUM_LAYERS,
        "features_per_layer": FEATURES_PER_LAYER,
        "com_center": COM_CENTER.tolist(),
        "com_scale": COM_SCALE.tolist(),
    }, MODEL_PATH)

    print(f"\nBest model saved to {MODEL_PATH}")

    # ========================================================
    # 8. FINAL TEST
    # ========================================================

    results = evaluate(
        model,
        test_loader,
        com_criterion,
        stable_criterion
    )

    print("\n========== TEST RESULTS ==========")

    print("CoM MAE (meters):", results["com_mae"])

    print("CoM RMSE (meters):")
    print(np.sqrt(mean_squared_error(
        results["com_true"],
        results["com_pred"],
        multioutput="raw_values"
    )))

    print("\nStability Accuracy:", results["stable_acc"])

    print("\nClassification Report:")
    print(classification_report(
        results["stable_true"],
        results["stable_pred"],
        labels=[0, 1],
        target_names=["Unstable", "Stable"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        results["stable_true"],
        results["stable_pred"],
        labels=[0, 1]
    ))

    # ========================================================
    # 9. EXPORT TEST PREDICTIONS
    # ========================================================

    predictions = pd.DataFrame({
        "Tower_ID": [
            dataset.tower_ids[i] for i in test_set.indices
        ],
        "Actual_CoM_X": results["com_true"][:, 0],
        "Actual_CoM_Y": results["com_true"][:, 1],
        "Actual_CoM_Z": results["com_true"][:, 2],
        "Pred_CoM_X": results["com_pred"][:, 0],
        "Pred_CoM_Y": results["com_pred"][:, 1],
        "Pred_CoM_Z": results["com_pred"][:, 2],
        "Actual_Stable": results["stable_true"],
        "Pred_Stable": results["stable_pred"],
        "Stable_Probability": results["stable_prob"],
    })

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Predictions saved to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

# Jenga Tower Dataset, CNN Training, and MuJoCo Test

This project generates 50,000 Jenga towers, stores them in a CSV dataset, trains a CNN using PyTorch to predict tower stability and Center of Mass (CoM), and converts selected towers into MuJoCo XML scenes for testing.

## Files

| File | Purpose |
|---|---|
| `generate_towers.py` | Generates 50,000 towers with 50% stable and 50% unstable configurations. |
| `towers_dataset.csv` | Stores tower positions, stability labels, and CoM. Each tower has a unique `Tower_ID`. |
| `train_jenga_cnn.py` | Trains a 1D CNN using PyTorch to predict stability and CoM. |
| `jenga_cnn.pth` | Saved trained CNN model. |
| `jenga_test_predictions.csv` | CNN predictions on the test dataset. |
| `test_jenga_cnn.py` | Evaluates the trained CNN on 10,000 independently generated random towers. |
| `jenga_random_test_predictions.csv` | Predictions from the independent random test. |
| `generate_xml.py` | Converts a selected tower from the CSV into a MuJoCo XML scene. |
| `jenga_test.xml` | Generated MuJoCo scene for the selected tower. |

## Workflow

### 1. Generate the dataset

Run:

```bash
python generate_towers.py
```

Each tower contains 27 blocks with randomly generated positions and rotations.

Initially, the dataset was highly imbalanced, with approximately 95% unstable and 5% stable towers. The generator was modified to produce more diverse configurations and a balanced dataset:

- 25,000 stable towers (50%)
- 25,000 unstable towers (50%)

Stability is estimated using the CoM and the supporting area of each layer.

### 2. Train the CNN

Run:

```bash
python train_jenga_cnn.py
```

A 1D CNN is implemented using PyTorch.

- **Input:** 27 layers × 7 features (Z and up to three X/Y block positions per layer).
- **Output:** Stability (stable/unstable) and CoM (X, Y, Z).
- **Architecture:** Two convolutional layers and a fully connected layer with separate prediction heads.
- **Dataset split:** 70% training, 15% validation, 15% testing.

The trained model is saved as `jenga_cnn.pth`.

### 3. Test the CNN

Run:

```bash
python test_jenga_cnn.py
```

The model is evaluated on 10,000 independently generated random towers to check whether it can generalize beyond the balanced training dataset.

**Results:**

| Metric | Balanced Test | Random Test |
|---|---:|---:|
| Stability accuracy | 99.11% | 98.70% |
| Stable precision | 98.75% | 82.17% |
| Stable recall | 99.50% | 96.63% |
| ROC-AUC | — | 0.9965 |

**CoM prediction error on the random test (MAE):**

| Coordinate | Error |
|---|---:|
| X | 0.342 mm |
| Y | 0.380 mm |
| Z | 5.040 mm |

The CNN correctly identified 516 out of 534 stable towers in the random test.

Predictions are saved in `jenga_random_test_predictions.csv`.

### 4. Choose a tower

Open `towers_dataset.csv` and choose the `Tower_ID` you want to test in MuJoCo.

### 5. Generate the MuJoCo XML

Set the selected `Tower_ID` in `generate_xml.py`, then run:

```bash
python generate_xml.py
```

The script reads the tower's block positions from the CSV and generates `jenga_test.xml`.

### 6. Test the tower in MuJoCo

Open `jenga_test.xml` in MuJoCo or use it as input to a MuJoCo test script.

The XML contains only the selected tower, not all 50,000 towers.

## Notes

- If you change the selected `Tower_ID`, run `generate_xml.py` again.
- Make sure the scripts point to the correct CSV and model files.
- The dataset stability labels are based on a geometric approximation, not a full physics simulation.
- Actual stability in MuJoCo depends on contact forces, friction, gravity, and simulation duration.
- The pretrained CNN can later be used as a feature extractor for the Jenga reinforcement learning policy.

## Next Steps

- Try increasing the dataset to 100,000 towers.
- Reduce false-stable predictions.
- Compare CNN stability predictions with MuJoCo simulation results.
- Integrate the pretrained CNN into the PPO training pipeline.
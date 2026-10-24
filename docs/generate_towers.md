# Jenga Tower Dataset and MuJoCo Test

Generating 50,000 Jenga towers, stores them in a CSV dataset, and convert one selected tower into a MuJoCo XML scene for testing.

## Files

| File | Purpose |
|---|---|
| `generate_towers.py` | Generates 50,000 towers. |
| `towers_dataset.csv` | Stores the generated towers and their data. Each tower is identified by a `Tower_ID`. |
| `generate_xml.py` | Reads one selected tower from the CSV and converts it into a MuJoCo XML scene. |
| `jenga_test.xml` | The generated MuJoCo scene for the selected tower. |

## Workflow

### 1. Generate the dataset

Run `generate_towers.py`. It creates `towers_dataset.csv` with the generated towers.

```bash
python generate_towers.py
```

### 2. Choose a tower
Open towers_dataset.csv and choose the Tower_ID you want to test.

### 3. Generate the MuJoCo XML
Set the chosen Tower_ID in generate_xml.py, then run the script. It reads that tower’s block positions from the CSV and writes jenga_test.xml.
python generate_xml.py

### 4. Test the tower
Open jenga_test.xml in MuJoCo or use it as the input to your MuJoCo test script. The XML contains the selected tower only; it does not contain all 50,000 towers.

Notes
- If you change the selected Tower_ID, run generate_xml.py again before testing.
- Check that generate_xml.py points to the correct CSV file.
- A stability result depends on the simulation settings and test duration. It describes the tower’s behavior under that test, not whether it will remain standing forever.
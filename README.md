# PA01 - Data Stream Mining: Lisboa vs Porto

**Assignment code:** PA01
**Student:** Louis Silva
**City A:** Lisboa
**City B:** Porto
**Business scenario:** Urban Walking Tour Operator.
**Prediction target:** T1 — binary rain event, with $y_t = 1$ when hourly precipitation is at least 0.1 mm, and $y_t = 0$ otherwise.
**Historical period:** 1 January 2020 – 31 December 2025 (six calendar years)
**Random seed:** `42` (defined in `config.yaml`)

## Python and Package Versions

This project uses **Python 3.12.3**.

| Package | Version |
| --- | --- |
| `river` | `0.26.1` |
| `pandas` | `3.0.6` |
| `numpy` | `2.5.3` |
| `matplotlib` | `3.11.2` |
| `requests` | `2.34.2` |
| `PyYAML` | `6.0.3` |
| `ipykernel` | `7.3.0` |
| `scipy` | `1.18.1` |
| `seaborn` | `0.13.2` |

The full dependency list for reproduction are specified in `requirements.txt`.

## Project Structure

```text
.
├── data/
│   └── raw/
│       ├── Lisboa_openmeteo_raw.json
│       └── Porto_openmeteo_raw.json
├── report/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── src/
├── practical_assignment.ipynb
├── config.yaml
├── README.md
└── requirements.txt
```

- **`data/`** - Raw Open-Meteo JSON weather observations for Lisboa and Porto.
- **`report/`** - LaTeX report source, references, and figures used in the report.
- **`src/`** - Python modules imported by `practical_assignment.ipynb` for loading, cleaning, preprocessing, modelling, evaluation, and plotting.
- **`practical_assignment.ipynb`** - Main notebook containing exploratory city analyses, comparative data analysis, the data stream mining experiment, and code to generate report figures.
- **`config.yaml`** - Open-Meteo API parameters, data paths, cleaning rules, feature settings, model parameters, evaluation settings, plotting configuration, and random seed.
- **`README.md`** - Project overview and reproduction instructions.
- **`requirements.txt`** - Python package dependencies with pinned versions.

## Environment Setup

You only need to do steps 1 and 2 per machine.

## Step 1 - Create and activate the virtual environment

**macOS / Linux**
```bash
python3 -m venv venv           
source venv/bin/activate
pip install -r requirements.txt
```

**Windows -- PowerShell**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows -- cmd.exe**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

## Step 2 - Register the Jupyter kernel

To make the Jupyter kernel use the activated environment, do:

```bash
python -m ipykernel install --user --name PA01 --display-name "Python (PA01)"
```

In your notebook application, select the kernel `Python (PA01)`.

## Step 3 - Reopen the project

Whenever you return to the project, activate the environment from the project root:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows -- PowerShell**
```powershell
venv\Scripts\Activate.ps1
```

**Windows -- cmd.exe**
```cmd
venv\Scripts\activate.bat
```

Open `practical_assignment.ipynb` in JupyterLab, Jupyter Notebook, VS Code, or another compatible notebook application. If JupyterLab is installed, you can start it with:

```bash
jupyter-lab practical_assignment.ipynb
```

## Reproducing the Main Experiment

1. Activate the virtual environment, open `practical_assignment.ipynb`, and select the registered project kernel.
2. Run the first cell in the notebook.
3. From the project root, verify that `data/raw/Lisboa_openmeteo_raw.json` and `data/raw/Porto_openmeteo_raw.json` are present. If not, run the second cell to download the data.
4. Go to the **Data Stream Mining Experiment** section and run the first cell. The results of the experiment gets saved to `results/report.txt`

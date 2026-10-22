# QuantumHealth Optimizer

HEALTH OPTIMISER — synthetic-data educational demonstration.

An educational Streamlit prototype demonstrating a pipeline from synthetic patient profile through illustrative machine learning, patient-specific constraints, binary optimization, and independent validation. It is not clinically validated, does not diagnose, and does not provide treatment or medication advice. All patient profiles and option attributes are synthetic and fictional.

## Requirements

- Python 3.13 recommended for deployment
- VS Code with the Microsoft Python and Pylance extensions (optional)

## Install and run (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Use the sidebar to enter a synthetic profile, then select **Run / Re-optimize**. Changing profile values and submitting again reruns the workflow; **Reset demonstration inputs** restores the defaults. Qiskit can be left unchecked for the exact classical baseline. Checking the local QAOA experiment runs seeded trials using Qiskit's `StatevectorSampler`, a statevector simulator on the local classical computer. Failed quantum attempts are labeled and fall back to the classical baseline with the error detail shown.

## Project structure

```text
app.py                  Streamlit dashboard
data/sample_patients.csv Example synthetic records (not loaded as real patient data)
src/data_processing.py  Input validation
src/ml_model.py         Synthetic illustrative risk model
src/treatment_candidates.py Fictional options and restrictions
src/optimizer.py        QUBO, exact baseline, optional Qiskit QAOA
src/validation.py       Independent feasibility checks
src/explainability.py   Result explanation
tests/test_optimizer.py Core behavior tests
```

## Run tests

```powershell
python -m pytest -q
```

Tests cover profile validation, reproducibility of the artificial labels, the Random Forest prediction pipeline, explicit candidate exclusions, resource-constrained optimization, QUBO conversion, a local QAOA run, and independent eligibility/resource checks. The simulator demonstrates the QAOA algorithm workflow, not quantum-hardware execution or quantum speedup.

The dashboard shows a synthetic held-out model-metric chart, candidate attribute comparisons, classical/QAOA trial objectives and execution times, the QAOA candidate bit string and configuration, each selected candidate's contribution (`benefit − risk_weight × risk − cost_weight × cost`), independent eligibility/duplicate/resource checks, and the Qiskit-converted QUBO. Re-submitting changed synthetic inputs re-runs preprocessing, model inference, and optimization and compares the new result with the previous run. Candidate properties are fictional and do not change based on profile or simulated outcomes.

## Deploy a public demo on Streamlit Community Cloud

1. Create a dedicated GitHub repository for this project and push the project folder contents to its `main` branch. Do not publish the enclosing `C:\Users\dell` folder.
2. Confirm `.gitignore` is in the repository and that `.streamlit/secrets.toml`, `.env`, virtual environments, and caches are not tracked. This app requires no secrets.
3. In Streamlit Community Cloud, create a new app and select the GitHub repository, `main` branch, and `app.py` as the main file. Select Python 3.13 in the advanced settings if offered.
4. Deploy. The app installs its pinned runtime packages from `requirements.txt`; `requirements-dev.txt` adds pytest for local testing and CI.
5. Check the deployed page and the GitHub Actions **Tests** workflow after each change. Update the app by pushing tested changes to `main`.

This is a demonstration deployment, not a production clinical service. Use synthetic data only. Do not enter, upload, or store real patient information, protected health information, credentials, or other sensitive data. The app has no user authentication, clinical validation, regulated-data controls, or service-level guarantees.

The data-processing module requires numeric `age`, `bmi`, `heart_rate`, `systolic_bp`, and `symptom_score` values. It rejects missing, nonnumeric, infinite, and out-of-demo-range inputs, then returns a one-row Pandas DataFrame. Bounds are software-demo limits only and are not medical reference ranges. The dashboard's demonstration budget is kept separate from this feature DataFrame.

## Demonstration / viva outline

1. Explain that every data point and option property is synthetic and the system is not for clinical use.
2. Enter a profile and describe the validation and illustrative model score.
3. Exclude one or more hypothetical options and change the maximum resource units.
4. Explain the binary decision variables and objective: maximize illustrative benefit after weighted risk and cost, subject to the resource limit.
5. Show independent feasibility checks after the solver and explain the Qiskit attempt/fallback status.
6. Update the profile and rerun to demonstrate the full pipeline.

## Limitations

The generated training set, artificial labels, option benefits, risks, and constraints are invented solely for software demonstration. The reported test accuracy only measures agreement with the artificial label. The predicted positive-class probability is not a disease probability or validated risk score. No real prescribing guidance, patient records, clinical validation, or quantum hardware execution is included. QAOA support depends on installed Qiskit package compatibility and an available sampler implementation.

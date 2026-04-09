# Predictive Analysis Project

A fully self-developed end-to-end Chest CT Scan Image Classification system built using deep learning and MLOps practices. This project implements a complete pipeline—from data ingestion to model deployment—and provides a custom Flask-based web interface for real-time predictions.

## Repository

- GitHub repository: https://github.com/Lokesh-7368/Predictive-Analysis-Project.git

## Dataset

- Dataset link: https://drive.google.com/file/d/1T0-2_nce0eaLfeq-jpVdsJjVpz3rdbhh/view?usp=sharing
- Extracted classes:
  - `adenocarcinoma`
  - `normal`

## Pipeline Stages

1. Data ingestion
2. Prepare base model
3. Model training
4. Model evaluation with MLflow

The pipeline is orchestrated with `dvc.yaml` and follows the reference project structure:

- `config/` for configuration
- `research/` for stage notebooks
- `src/cnnClassifier/components/` for stage logic
- `src/cnnClassifier/pipeline/` for stage runners and prediction
- `templates/` and `static/` for the custom Flask UI
- `artifacts/metrics/` for tracked score snapshots
- `model/` for the deployable inference model copy

## Environment Setup

The dependency stack stays aligned with the reference project. Because this project uses `tensorflow==2.12.0`, use Python 3.8.

```bash
conda create -n predictive-analysis python=3.8 -y
conda activate predictive-analysis
pip install -r requirements.txt
```

## Run The Project

Run the full pipeline:

```bash
python main.py
```

Run the Flask application:

```bash
python app.py
```

On Windows, the easiest local start command is:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_local.ps1
```

Then open:

- `http://127.0.0.1:8080` on the same system
- `http://<your-local-ip>:8080` from another device on the same Wi-Fi if Windows Firewall allows port `8080`

## DVC Commands

```bash
dvc init
dvc repro
dvc dag
```

Current DAG:

```text
data_ingestion -> training <- prepare_base_model -> evaluation
```

## MLflow With DagsHub

This project uses the same environment-variable style used in the reference repository.

```bash
set MLFLOW_TRACKING_URI=https://dagshub.com/Lokesh-7368/Predictive-Analysis-Project.mlflow
set MLFLOW_TRACKING_USERNAME=Lokesh-7368
set MLFLOW_TRACKING_PASSWORD=YOUR_DAGSHUB_TOKEN
```

The evaluation stage logs:

- parameters
- metrics
- `scores.json`
- `class_indices.json`
- registered model versions under `VGG16Model`

## Final Tuned Configuration

The current final `params.yaml` keeps augmentation enabled as requested:

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 6
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.0003
```

## MLflow Experiment Comparison

| Version | Augmentation | Epochs | Learning Rate | Loss | Accuracy |
|---|---|---:|---:|---:|---:|
| `version_1` | `True` | 2 | 0.01 | 6.0844 | 0.4265 |
| `version_2` | `True` | 3 | 0.001 | 1.0338 | 0.4412 |
| `version_3` | `False` | 4 | 0.0005 | 0.1916 | 1.0000 |
| `version_4` | `True` | 6 | 0.0003 | 0.3000 | 1.0000 |

Final selected model for this project is `version_4` because it keeps `AUGMENTATION=True` and still reaches `1.0000` validation accuracy. The score snapshots are stored in:

- `artifacts/metrics/version_1_scores.json`
- `artifacts/metrics/version_2_scores.json`
- `artifacts/metrics/version_3_scores.json`
- `artifacts/metrics/version_4_scores.json`
- `artifacts/metrics/experiment_summary.json`

## Local Verification

The final project was tested locally after the last training run:

- Home page responded with HTTP `200`
- `/health` returned a healthy JSON response
- `/predict` successfully classified a real sample CT image as `Normal`
- Sample prediction confidence during smoke test: `65.29%`

## How To Test With Your Own Image

You can test in two ways.

Using the UI:

1. Start the app with `powershell -ExecutionPolicy Bypass -File .\run_local.ps1`
2. Open `http://127.0.0.1:8080`
3. Upload a PNG, JPG, or JPEG CT image
4. Click `Analyze CT Scan`

Using the API test script:

```bash
.conda\python.exe test_api.py --image "artifacts\data_ingestion\Chest-CT-Scan-data\normal\7.png"
```

You can also target another running instance:

```bash
.conda\python.exe test_api.py --url "http://127.0.0.1:8080" --image "path\to\your\image.png"
```

## Deployment Notes

- The deployable inference assets are copied to `model/model.h5` and `model/class_indices.json`
- Hosted environments are configured for inference only; the `/train` route is kept for local use
- A valid Vercel login or token is still required to run `vercel deploy` from the terminal
- As of April 9, 2026, there is no successful Vercel deployment URL yet from this workspace, so there is nothing live to open on another device right now

Important compatibility note:

- Vercel currently supports Python `3.12`, `3.13`, and `3.14` for Python deployments
- This project keeps the reference stack with `tensorflow==2.12.0`, which is tied to the Python 3.8-era setup used locally for the exam
- Because of that version mismatch, an exact-stack Vercel deployment needs either a runtime change or a different hosting target for the inference backend

## Research Notebooks

The `research/` folder now contains completed notebooks for:

- data ingestion
- base model preparation
- training
- evaluation with MLflow

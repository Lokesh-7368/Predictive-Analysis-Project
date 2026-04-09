import os
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.utils.common import create_directories, decodeImage

app = Flask(__name__)
CORS(app)

PREDICTION_DIR = Path("artifacts/prediction")
INPUT_IMAGE_PATH = PREDICTION_DIR / "inputImage.jpg"


class ClientApp:
    def __init__(self):
        create_directories([PREDICTION_DIR])
        self.filename = str(INPUT_IMAGE_PATH)
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


def is_training_enabled():
    return os.getenv("DISABLE_SERVER_TRAINING", "false").lower() != "true" and not os.getenv("VERCEL")


def build_dataset_summary():
    data_root = Path("artifacts/data_ingestion/Chest-CT-Scan-data")
    summary = []

    if not data_root.exists():
        return summary

    for class_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        image_count = sum(1 for item in class_dir.iterdir() if item.is_file())
        summary.append(
            {
                "label": class_dir.name.replace("_", " ").title(),
                "slug": class_dir.name,
                "count": image_count,
            }
        )

    return summary


def ensure_dvc_initialized():
    if Path(".dvc").exists():
        return ""

    dvc_command = [sys.executable, "-m", "dvc", "init"]
    if not Path(".git").exists():
        dvc_command.append("--no-scm")

    init_process = subprocess.run(
        dvc_command,
        capture_output=True,
        text=True,
        check=False,
    )

    if init_process.returncode != 0:
        raise RuntimeError(init_process.stderr or init_process.stdout or "DVC initialization failed.")

    return init_process.stdout.strip()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    config = ConfigurationManager()
    return render_template(
        "index.html",
        dataset_summary=build_dataset_summary(),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", config.get_evaluation_config().mlflow_uri),
        training_enabled=is_training_enabled(),
    )


@app.route("/health", methods=["GET"])
@cross_origin()
def health():
    return jsonify({"status": "ok", "app": "Predictive Analysis Project"})


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    try:
        if not is_training_enabled():
            return (
                jsonify(
                    {
                        "success": False,
                        "message": (
                            "Training is available only in local development. "
                            "Hosted deployments are configured for inference only."
                        ),
                        "log": "",
                    }
                ),
                403,
            )

        init_log = ensure_dvc_initialized()
        process = subprocess.run(
            [sys.executable, "-m", "dvc", "repro"],
            capture_output=True,
            text=True,
            check=False,
        )

        combined_log = "\n".join(filter(None, [init_log, process.stdout, process.stderr])).strip()

        if process.returncode != 0:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Training pipeline failed.",
                        "log": combined_log,
                    }
                ),
                500,
            )

        return jsonify(
            {
                "success": True,
                "message": "Training pipeline completed successfully.",
                "log": combined_log,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "log": ""}), 500


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        payload = request.get_json(silent=True) or {}
        image = payload.get("image")

        if not image:
            return jsonify({"success": False, "message": "No image payload received."}), 400

        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify({"success": True, "result": result})
    except FileNotFoundError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

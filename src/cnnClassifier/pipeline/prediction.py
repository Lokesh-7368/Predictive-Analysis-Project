import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path

from cnnClassifier.constants import PARAMS_FILE_PATH
from cnnClassifier.utils.common import load_json, read_yaml


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def _resolve_model_path() -> Path:
        candidate_paths = [
            Path("artifacts/training/model.h5"),
            Path("model/model.h5"),
        ]

        for path in candidate_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Trained model not found. Run the training pipeline before requesting predictions."
        )

    @staticmethod
    def _load_class_labels():
        candidate_paths = [
            Path("artifacts/training/class_indices.json"),
            Path("model/class_indices.json"),
        ]

        for class_indices_path in candidate_paths:
            if class_indices_path.exists():
                content = load_json(class_indices_path).to_dict()
                index_to_class = content.get("index_to_class", {})
                return [
                    index_to_class[str(index)]
                    for index in sorted(map(int, index_to_class.keys()))
                ]

        training_data_dir = Path("artifacts/data_ingestion/Chest-CT-Scan-data")
        if training_data_dir.exists():
            return sorted([path.name for path in training_data_dir.iterdir() if path.is_dir()])

        return ["adenocarcinoma", "normal"]

    @staticmethod
    def _friendly_name(label: str) -> str:
        mapping = {
            "adenocarcinoma": "Adenocarcinoma Cancer",
            "normal": "Normal",
        }
        return mapping.get(label, label.replace("_", " ").title())

    @staticmethod
    def _description(label: str) -> str:
        mapping = {
            "adenocarcinoma": "Model sees stronger visual evidence of lung adenocarcinoma patterns.",
            "normal": "Model sees stronger visual evidence of a normal chest CT pattern.",
        }
        return mapping.get(label, "Prediction generated from the trained chest CT classifier.")

    def predict(self):
        model = load_model(self._resolve_model_path())
        params = read_yaml(PARAMS_FILE_PATH)
        class_labels = self._load_class_labels()

        target_size = tuple(params.IMAGE_SIZE[:2])
        test_image = image.load_img(self.filename, target_size=target_size)
        test_image = image.img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        probabilities = model.predict(test_image, verbose=0)[0]

        if len(class_labels) < len(probabilities):
            class_labels.extend(
                [f"class_{index}" for index in range(len(class_labels), len(probabilities))]
            )
        elif len(class_labels) > len(probabilities):
            class_labels = class_labels[: len(probabilities)]

        predicted_index = int(np.argmax(probabilities))
        predicted_label = class_labels[predicted_index]

        probability_rows = []
        for index, label in enumerate(class_labels):
            probability_rows.append(
                {
                    "label": self._friendly_name(label),
                    "slug": label,
                    "probability": round(float(probabilities[index]) * 100, 2),
                }
            )

        probability_rows.sort(key=lambda item: item["probability"], reverse=True)

        return {
            "label": predicted_label,
            "display_label": self._friendly_name(predicted_label),
            "description": self._description(predicted_label),
            "confidence": round(float(probabilities[predicted_index]) * 100, 2),
            "probabilities": probability_rows,
        }

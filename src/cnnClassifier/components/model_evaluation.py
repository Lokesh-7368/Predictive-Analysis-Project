from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.keras
import tensorflow as tf

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _build_run_name(self) -> str:
        params = self.config.all_params
        return (
            f"vgg16-aug-{str(params.get('AUGMENTATION', False)).lower()}-"
            f"epochs-{params.get('EPOCHS')}-"
            f"lr-{params.get('LEARNING_RATE')}"
        )

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20,
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator, verbose=0)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=self.config.score_file, data=scores)

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name=self._build_run_name()):
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            mlflow.set_tags(
                {
                    "project": "Predictive Analysis Project",
                    "model_family": "VGG16",
                    "augmentation": str(self.config.all_params.get("AUGMENTATION", False)).lower(),
                }
            )

            if self.config.class_indices_path.exists():
                mlflow.log_artifact(str(self.config.class_indices_path))
            if self.config.score_file.exists():
                mlflow.log_artifact(str(self.config.score_file))

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="VGG16Model",
                )
            else:
                mlflow.keras.log_model(self.model, "model")

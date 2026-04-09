import base64
import json
import os
from pathlib import Path
from typing import Any

import joblib
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from cnnClassifier import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return a ConfigBox."""

    try:
        with open(path_to_yaml, encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info("yaml file: %s loaded successfully", path_to_yaml)
            return ConfigBox(content)
    except BoxValueError as exc:
        raise ValueError("yaml file is empty") from exc
    except Exception as exc:
        raise exc


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create a list of directories."""

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info("created directory at: %s", path)


@ensure_annotations
def save_json(path: Path, data: dict):
    """Save json data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4)

    logger.info("json file saved at: %s", path)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load json data."""

    with open(path, encoding="utf-8") as file_obj:
        content = json.load(file_obj)

    logger.info("json file loaded successfully from: %s", path)
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save binary data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value=data, filename=path)
    logger.info("binary file saved at: %s", path)


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary data."""

    data = joblib.load(path)
    logger.info("binary file loaded from: %s", path)
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Get size in KB."""

    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as file_obj:
        file_obj.write(imgdata)


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as file_obj:
        return base64.b64encode(file_obj.read())

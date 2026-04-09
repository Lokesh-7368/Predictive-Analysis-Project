import os
import zipfile
from pathlib import Path

import gdown

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the url.
        """

        zip_download_dir = Path(self.config.local_data_file)
        zip_download_dir.parent.mkdir(parents=True, exist_ok=True)

        if zip_download_dir.exists() and zip_download_dir.stat().st_size > 0:
            logger.info(
                "File already exists at %s with size %s",
                zip_download_dir,
                get_size(zip_download_dir),
            )
            return str(zip_download_dir)

        dataset_url = self.config.source_URL
        logger.info("Downloading data from %s into file %s", dataset_url, zip_download_dir)
        gdown.download(url=dataset_url, output=str(zip_download_dir), fuzzy=True, quiet=False)
        logger.info("Downloaded data into file %s", zip_download_dir)

        return str(zip_download_dir)

    def extract_zip_file(self):
        """
        Extract the zip file into the data directory.
        """

        unzip_path = Path(self.config.unzip_dir)
        extracted_data_dir = unzip_path / "Chest-CT-Scan-data"

        if extracted_data_dir.exists() and any(extracted_data_dir.iterdir()):
            logger.info("Data already extracted at %s", extracted_data_dir)
            return

        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info("Zip file extracted to %s", unzip_path)

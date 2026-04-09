from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline


STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare base model"
try:
    logger.info("*******************")
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"
try:
    logger.info("*******************")
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Evaluation stage"
try:
    logger.info("*******************")
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e

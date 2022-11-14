import typer
import logging
import datetime
import json
import os

from feature_extraction.feature_extractor import FeatureExtractor
from document_classification.document_classifier import DocumentClassifier
from document_classification.firstpage_classifier import FirstpageClassifier
from preprocessing import convert_to_img
from prediction.predict import predict_from_directory
from utils.logger import LOGGING_FILENAME, LOGGING_FORMAT, LOGGING_LEVELS


cli = typer.Typer()
logger = logging.getLogger(__name__)


def train_document_classifier(
    model_name: str = "EfficientNetB4", 
    verbose: bool = False,
    yes_to_all_user_input: bool = False,
) -> None:
    """
    Train the document classifier with OCCRP data using the specifications in the config.py file.
    """
    document_classifier = DocumentClassifier(model_name, verbose, yes_to_all_user_input)
    document_classifier.mlflow_train_pipeline()


def train_feature_extraction(model_name: str = "AlexNet", verbose: bool = False) -> None:
    """
    Train the document feature extractor model with a dataset such as RVL-CDIP.
    """

    feature_extractor = FeatureExtractor(model_name, verbose)
    feature_extractor.mlflow_train_pipeline()


def train_firstpage_classifier(
    model_name: str = "EfficientNetB4",
    verbose: bool = False,
    yes_to_all_user_input: bool = False,
) -> None:
    """
    Train the first page classifier.
    """

    firstpage_classifier = FirstpageClassifier(model_name, verbose, yes_to_all_user_input)
    firstpage_classifier.mlflow_train_pipeline()


def predict(input_path: str, output_path: str, model_name: str = "EfficientNetB4") -> None:
    """Predict the documents contained in the INPUT_PATH using the model MODEL_NAME
    and outputs the prediction in OUTPUT_PATH in a json format
    """
    prediction = predict_from_directory(input_path, model_name)
    prediction_filename = datetime.datetime.now().strftime(r"prediction__%Y_%m_%d_%H_%M_%S.json")
    with open(os.path.join(output_path, prediction_filename), "w") as prediction_file:
        json.dump(prediction, prediction_file, ensure_ascii=False, indent=4)


def convert_all_to_jpg(
    input_path: str, output_path: str, skip_converted_files: bool = False, only_first_page: bool = False
) -> None:
    """Looks for all .pdf, .tif, .doc, .docx, .jpg in the INPUT_PATH and its subdirectories and
    converts them to jpgs in the OUTPUT_PATH directory.

    This is a preprocessing step used in the document classifier.
    """

    print(f"Starting conversion of documents in {input_path} to {output_path}")
    convert_to_img.convert_all_to_jpg(input_path, output_path, skip_converted_files, only_first_page)


@cli.command()
def train_full_classifier(
    model_name: str = typer.Option("EfficientNetB4", help="The classifier model to be trained"),
    verbose: bool = False,
) -> None:
    """Train the multiclass and the binary classifier using the MODEL_NAME architecture."""
    logger.info("Training multiclass document classifier")
    document_classifier = DocumentClassifier(model_name, verbose)
    document_classifier.mlflow_train_pipeline()

    logger.info("Training binary first page classifier")
    firstpage_classifier = FirstpageClassifier(model_name, verbose)
    firstpage_classifier.mlflow_train_pipeline()


def logging_level(log: str = "INFO") -> None:
    """Set logging level"""

    logging.basicConfig(
        format=LOGGING_FORMAT,
        level=LOGGING_LEVELS[log.upper()],
        handlers=[logging.FileHandler(LOGGING_FILENAME), logging.StreamHandler()],
    )


if __name__ == "__main__":
    logging_level("INFO")
    # train_document_classifier(yes_to_all_user_input=True)
    train_firstpage_classifier(yes_to_all_user_input=True)
   

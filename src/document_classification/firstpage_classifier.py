import logging
from typing import List

from config import settings
from document_classification.document_classifier import DocumentClassifier

logger = logging.getLogger(__name__)


class FirstpageClassifier(DocumentClassifier):
    """Firstpage classifier classifies if a document belongst to first pages or middle/end pages."""

    # ------------------------------------------------------- #
    # ------------------ Initialization:
    # ------------------------------------------------------- #

    def __init__(self, model_name: str, verbose: bool = False, yes_to_all_user_input: bool = False) -> None:
        """Function that initializes the parameters of an object of the class FirstpageClassifier.
            It uses the initialization function of the class DocumentClassifier, with the specific
            parameters in the config.py file associated with the first-page classification training task.

        Args:
            model_name (str): string containing the model name matching the dictionary of models provided. This parameter
            is usually provided via CLI.
            verbose (bool, optional): Defaults to False.
        """
        super().__init__(model_name, verbose, yes_to_all_user_input, output_path=settings.OUTPUT_FIRSTPAGE_CLASSIFICATION)
        
        # overwrites the values in the parent class
        self.mlflow_experiment_name = "First page classifier"
        self.train_only_with_first_pages = False
        self.LABELS_FILTER: List[str] = settings.LABELS_FILTER_FIRSTPAGES
        self.RAW_DOCUMENTS_TRANSFER_LEARNING = ""
        self.input_images_path = settings.INPUT_IMAGES_FIRSTPAGE_CLASSIFICATION
        self.ENABLE_CATEGORY_OTHERS = False

import os
import random

import pytest

from utils_for_tests import get_directory_file_paths

TEST_JPG_DIRECTORY = "tests/test_data/prediction/jpg"
TEST_PDF_DIRECTORY = "tests/test_data/prediction/pdf"
TEST_TIFF_DIRECTORY = "tests/test_data/prediction/tiff"
TEST_CORRUPT_DIRECTORY = "tests/test_data/prediction/corrupt"


@pytest.fixture
def pdf_paths():
    return get_directory_file_paths(TEST_PDF_DIRECTORY)


@pytest.fixture
def jpg_paths():
    return get_directory_file_paths(TEST_JPG_DIRECTORY)


@pytest.fixture
def tiff_paths():
    return get_directory_file_paths(TEST_TIFF_DIRECTORY)


@pytest.fixture
def corrupt_pdf_path():
    return os.path.join(TEST_CORRUPT_DIRECTORY, "corrupt.pdf")


@pytest.fixture
def corrupt_jpg_path():
    return os.path.join(TEST_CORRUPT_DIRECTORY, "corrupt.jpg")


@pytest.fixture
def corrupt_tiff_path():
    return os.path.join(TEST_CORRUPT_DIRECTORY, "corrupt.tiff")


@pytest.fixture
def document_paths(jpg_paths, tiff_paths, pdf_paths):
    random.seed(100)
    paths = jpg_paths + tiff_paths + pdf_paths
    return random.sample(paths, len(paths))


@pytest.fixture
def corrupt_document_paths(corrupt_jpg_path, corrupt_pdf_path, corrupt_tiff_path):
    random.seed(100)
    paths = [corrupt_jpg_path, corrupt_pdf_path, corrupt_tiff_path]
    return random.sample(paths, len(paths))

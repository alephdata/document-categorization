import cv2
import numpy as np
import pytest

from utils_for_tests import get_directory_file_paths

TEST_IMAGES_DIRECTORY = "tests/test_data/prediction/jpg"
TEST_MEAN_DIRECTORY = "tests/test_data/prediction/mean"
TEST_STD_DIRECTORY = "tests/test_data/prediction/std"
IMG_SIZE = 227


@pytest.fixture(params=get_directory_file_paths(TEST_MEAN_DIRECTORY))
def mean_array(request):
    return np.load(request.param)


@pytest.fixture(params=get_directory_file_paths(TEST_STD_DIRECTORY))
def std_array(request):
    return np.load(request.param)


@pytest.fixture(params=get_directory_file_paths(TEST_IMAGES_DIRECTORY))
def img_array(request):
    return np.array(cv2.resize(cv2.imread(request.param), (IMG_SIZE, IMG_SIZE))).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

import pytest
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder

from preprocessing.import_images import count_rows_csv
from preprocessing.import_images import count_rows_txt
from preprocessing.import_images import reduce_dataset
from preprocessing.import_images import import_images
from preprocessing.import_images import process_images
from preprocessing.import_images import encode_labels
from preprocessing.import_images import transform_to_tensorflow_format


@pytest.fixture
def path_to_test_folder():
    return "tests/test_feature_extraction/test_data/"


@pytest.fixture
def images_dir():
    return "tests/test_feature_extraction/test_data/test_images/"


##############################################################
##############################################################
## Test count_rows_csv


@pytest.mark.parametrize(
    "table_name, true_n_rows, error_message,",
    [
        ("small_table_noheader.csv", 4, "Problems with csv's with no header."),
        ("small_table_header.csv", 4, "Problems with csv's with header."),
        ("small_table_commas_noheader.txt", 4, "Problems with comma sep. txt's with no header."),
        ("small_table_commas_header.txt", 4, "Problems with comma sep. txt's with header."),
        ("small_table_spaces_noheader.txt", 4, "Problems with space sep. txt's with no header."),
        ("small_table_spaces_header.txt", 4, "Problems with space sep. txt's with header."),
    ],
)
def test_count_rows_csv(path_to_test_folder, table_name, true_n_rows, error_message):
    """Test to verify if counting works on no-header csv"""
    num_rows = count_rows_csv(os.path.join(path_to_test_folder, table_name))
    assert num_rows == true_n_rows, error_message


def test_count_rows_csv_empty():
    """Test to verify what happens if we give an empty imput"""
    with pytest.raises(FileNotFoundError):
        count_rows_csv("")


def test_count_rows_csv_emptyfile(path_to_test_folder):
    """Test to verify what happens if we give an empty imput"""
    with pytest.raises(Exception):
        count_rows_csv(os.path.join(path_to_test_folder, "empty.csv"))


##############################################################
##############################################################
## Test count_rows_txt


@pytest.mark.parametrize(
    "table_name, true_n_rows, error_message,",
    [
        ("small_table_noheader.csv", 4, "Problems with csv's with no header."),
        ("small_table_header.csv", 4, "Problems with csv's with header."),
        ("small_table_commas_noheader.txt", 4, "Problems with comma sep. txt's with no header."),
        ("small_table_commas_header.txt", 4, "Problems with comma sep. txt's with header."),
        ("small_table_spaces_noheader.txt", 4, "Problems with space sep. txt's with no header."),
        ("small_table_spaces_header.txt", 4, "Problems with space sep. txt's with header."),
    ],
)
def test_count_rows_txt(path_to_test_folder, table_name, true_n_rows, error_message):
    """Test to verify if counting works on no-header csv"""
    num_rows = count_rows_txt(os.path.join(path_to_test_folder, table_name))
    assert num_rows == true_n_rows, error_message


def test_count_rows_txt_noinput():
    """Test to verify what happens if we give an empty imput"""
    with pytest.raises(FileNotFoundError):
        count_rows_txt("")


def test_count_rows_txt_emptyfile(path_to_test_folder):
    """Test to verify what happens if we give an empty imput"""
    with pytest.raises(Exception):
        count_rows_txt(os.path.join(path_to_test_folder, "empty.txt"))


##############################################################
##############################################################
## Testing reducedDataset


@pytest.mark.parametrize(
    "file_name, indexChoice, new_index, name, error_message,",
    [
        ("small_table_header.csv", range(4), 1, "JM", "Add exception if table has header"),
        ("small_table_noheader.csv", range(4), 1, "JM", "Problem with re-indexing/no header."),
    ],
)
def test_reducedDataset_header(path_to_test_folder, file_name, indexChoice, new_index, name, error_message):
    """Test if it imports the correct rows in the table if the table has a header."""
    fileDir = os.path.join(path_to_test_folder, file_name)
    reduced_data = reducedDataset(fileDir, indexChoice)
    assert reduced_data["doc type"][new_index] == name, error_message


@pytest.mark.parametrize(
    "file_name, indexChoice, axis, expected_size, error_message,",
    [
        ("small_table_header.csv", range(1, 3), 0, 2, "Not subsetting correctly"),
        ("small_table_header.csv", range(1, 3), 1, 2, "Not subsetting correctly"),
        ("small_table_noheader.csv", range(1, 3), 0, 2, "Problem with re-indexing/no header."),
        ("small_table_noheader.csv", range(1, 3), 1, 2, "Problem with re-indexing/no header."),
    ],
)
def test_reducedDataset_shape(path_to_test_folder, file_name, indexChoice, axis, expected_size, error_message):
    fileDir = os.path.join(path_to_test_folder, file_name)
    reduced_data = reducedDataset(fileDir, indexChoice)
    assert reduced_data.shape[axis] == expected_size, error_message


def test_reducedDataset_outofbound(path_to_test_folder):
    """Verify"""
    fileDir = os.path.join(path_to_test_folder, "small_table_noheader.csv")
    indexChoice = range(1, 10)
    with pytest.raises(Exception):
        reduced_data = reducedDataset(fileDir, indexChoice)


##############################################################
##############################################################
## Testing importImages


def test_importImages_labels(images_dir):
    names_table = reducedDataset(fileDir=os.path.join(images_dir, "images_table.txt"), indexChoice=range(5))
    images, labels, faulty = importImages(images_dir=images_dir, names=names_table, im_size=100)
    assert labels[0] == "0", "incorrect label loaded"
    assert labels[1] == "1", "incorrect label loaded"
    assert labels[2] == "0", "incorrect label loaded"


def test_importImages_faulty(images_dir):
    names_table = reducedDataset(fileDir=os.path.join(images_dir, "images_table.txt"), indexChoice=range(5))
    images, labels, faulty = importImages(images_dir=images_dir, names=names_table, im_size=100)
    assert len(faulty) == 2, "incorrect number of corrupted files"
    assert faulty[0] == 2, "incorrect handling of empty files"
    assert faulty[1] == 3, "incorrect handling of txt files"


def test_importImages_images(images_dir):
    names_table = reducedDataset(fileDir=os.path.join(images_dir, "images_table.txt"), indexChoice=range(5))
    images, labels, faulty = importImages(images_dir=images_dir, names=names_table, im_size=100)
    assert images.shape[0] == 3, "incorrect number of imported files"
    assert images.shape[1] == 100, "incorrect width of images"
    assert images.shape[2] == 100, "incorrect height of images"
    assert images.shape[3] == 3, "incorrect number of color channels"


def test_importImages_bounds(images_dir):
    names_table = reducedDataset(fileDir=os.path.join(images_dir, "images_table.txt"), indexChoice=range(5))
    images, labels, faulty = importImages(images_dir=images_dir, names=names_table, im_size=100)
    bounded = True
    for i in range(40, 60):
        for j in range(40, 60):
            for k in range(3):
                if images[2, i, j, k] > 1 or images[2, i, j, k] < 0:
                    bounded = False

    assert bounded, "The images are not between 0 and 1."


##############################################################
##############################################################
## Testing Encoders on numbers:


@pytest.mark.parametrize(
    "labels, error_message",
    [
        (range(10), "Ex 1: Encoder does not preserve order range(10)."),
        (range(100), "Ex 2: Encoder does not preserve order range(100)."),
        (range(200), "Ex 3: Encoder does not preserve order range(200)."),
        ([3, 2, 1, 0], "Ex 3: Encoder does not preserve order [3,2,1,0]."),
        (range(100, 200), "Ex 4: Encoder does not preserve order range(100,200)."),
    ],
)
def test_LabelEncoder_preserves_int_order(labels, error_message):
    an_encoder = LabelEncoder()
    an_encoder.fit(labels)
    y_encoded = an_encoder.transform(labels)

    test_result = 0
    for i in range(len(labels) - 1):
        if y_encoded[i] >= y_encoded[i + 1]:
            test_result += 1

    assert test_result == 0, error_message


##############################################################
##############################################################
## Testing Encoders on numbers:


def test_process_images_centered(images_dir):
    names_table = reducedDataset(fileDir=os.path.join(images_dir, "images_table.txt"), indexChoice=range(5))
    images, labels, faulty = importImages(images_dir=images_dir, names=names_table, im_size=100)
    an_encoder = LabelEncoder()
    an_encoder.fit(["0", "1"])
    labels_encoded = ourLabelEncoding(labels, an_encoder)
    mean_images = np.mean(images, axis=0)

    standard_images, labels = process_images(images, labels_encoded)

    centered = True
    for image_index in range(3):
        image_mean = np.mean(standard_images[image_index, :, :, :])
        print("Mean of image:", image_mean)
        print("Abs mean of image:", round(abs(image_mean), 5))

        print(centered)
        if abs(image_mean) > 10 ** (-5):
            centered = False

    assert centered, "The dataset is not centered."


def test_process_images_stdev(images_dir):
    names_table = reducedDataset(fileDir=os.path.join(images_dir, "images_table.txt"), indexChoice=range(5))
    images, labels, faulty = importImages(images_dir=images_dir, names=names_table, im_size=100)
    an_encoder = LabelEncoder()
    an_encoder.fit(["0", "1"])
    labels_encoded = ourLabelEncoding(labels, an_encoder)
    mean_images = np.mean(images, axis=0)

    standard_images, labels = process_images(images, labels_encoded)

    scaled = True
    for image_index in range(3):
        image_stdev = np.std(standard_images[image_index, :, :, :])
        # mean_image[image_index] = np.mean(standard_images[image_index,:,:,:])
        # stdev_image[image_index] = np.std(standard_images[image_index,:,:,:])
        print("Mean of image:", image_stdev)
        print("Abs mean of image:", round(abs(image_stdev), 5))

        print(scaled)
        if abs(image_stdev - 1) > 10 ** (-5):
            scaled = False

    assert scaled, "The dataset is not centered."


##############################################################
##############################################################
## Testing Encoders on numbers:


def test_tensorflowFormat_num_splits(images_dir):
    names_table = reducedDataset(fileDir=os.path.join(images_dir, "images_table.txt"), indexChoice=range(5))
    images, labels, faulty = importImages(images_dir=images_dir, names=names_table, im_size=100)
    an_encoder = LabelEncoder()
    an_encoder.fit(["0", "1"])
    labels_encoded = ourLabelEncoding(labels, an_encoder)

    example_ds = tensorflowFormat(images, labels_encoded, sub_batch_size=2)
    assert len(list(example_ds.as_numpy_iterator())) == 2, "Batch eliminates obs. Set drop_remainder=False"

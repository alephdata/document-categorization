# import all packages
import pytest
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from PIL import Image
from os.path import exists as file_exists
import math

from preprocessing.import_images import count_rows_txt
from feature_extraction.models.AlexNet_model import create_model_AlexNet
from feature_extraction.models.VGG16_model import create_model_VGG16
from document_classification.models.transfer_AlexNet_model import create_transfer_model_AlexNet
from document_classification.models.transfer_VGG16_model import create_transfer_model_VGG16
from preprocessing.standardization import reduce_dataset
from preprocessing.standardization import save_batched_center
from preprocessing.standardization import save_batched_stdev
from preprocessing.standardization import import_images


##############################################
# create path fixtures
##############################################


@pytest.fixture
def AlexNet_model_path():
    return "/data/dssg/occrp/data/weights/featureSelection/AlexNet_2022_07_06-18_12_50/weights/epoch7"


@pytest.fixture
def VGG16_model_path():
    return "/data/dssg/occrp/data/weights/featureSelection/VGG16_224_2022_07_06-18_12_54/weights/epoch0"


@pytest.fixture
def table_path():
    return "/data/dssg/occrp/data/train_validation_test/2022-07-14_v3/"


@pytest.fixture
def test_data_path():
    return "/home/thenn/dssgxdfki2022-occrp/tests/test_data/"


##############################################
# test inital AlexNet model
##############################################


def test_create_model_AlexNet_correct_layers():
    """create model and check if layers match, should be 19
    input: initial model
    output: number of layers"""

    model_AlexNet = create_model_AlexNet()
    layer_amount_AlexNet = len(model_AlexNet.layers)
    print(layer_amount_AlexNet)
    assert layer_amount_AlexNet == 19


def test_create_model_AlexNet_incorrect_layers():
    """create model and check if layers match, should be 19
    input: initial model
    output: number of layers"""

    model_AlexNet = create_model_AlexNet()
    layer_amount_AlexNet = len(model_AlexNet.layers)
    print(layer_amount_AlexNet)
    assert layer_amount_AlexNet == 20


def test_create_model_AlexNet_correct_inputshape():
    """model should have 256 filters in layer 4
    input: initial model
    output: input shapes"""

    model_AlexNet = create_model_AlexNet()
    weights_AlexNet = len(model_AlexNet.layers[4].get_weights()[1])
    print(weights_AlexNet)
    assert weights_AlexNet == 256


def test_create_model_AlexNet_wrong_inputshape():
    """model should have 256 filters in layer 4
    input: initial model
    output: input shapes"""

    model_AlexNet = create_model_AlexNet()
    weights_AlexNet = len(model_AlexNet.layers[4].get_weights()[1])
    print(weights_AlexNet)
    assert weights_AlexNet == 96


##############################################
# test inital VGG16 model
##############################################


def test_create_model_VGG16_correct_layers():
    """create model and check if layers match, should be 22
    input: initial model
    output: number of layers"""

    model_VGG16 = create_model_VGG16()
    layer_amount_VGG16 = len(model_VGG16.layers)
    print(layer_amount_VGG16)
    assert (
        layer_amount_VGG16 == 22
    )  # why does model do not have 22 layers? because it does not start with a Sequential model?


def test_create_model_VGG16_incorrect_layers():
    """create model and check if layers match, should be 22
    input: initial model
    output: number of layers"""

    model_VGG16 = create_model_VGG16()
    layer_amount_VGG16 = len(model_VGG16.layers)
    print(layer_amount_VGG16)
    assert layer_amount_VGG16 == 20


def test_create_model_VGG16_correct_inputshape():
    """model should have 128 filters in layer 4
    input: initial model
    output: input shapes"""

    model_VGG16 = create_model_VGG16()
    weights_VGG16 = len(model_VGG16.layers[4].get_weights()[1])
    print(weights_VGG16)
    assert weights_VGG16 == 128


def test_create_model_VGG16_wrong_inputshape():
    """model should have 128 filters in layer 4
    input: initial model
    output: input shapes"""

    model_VGG16 = create_model_VGG16()
    weights_VGG16 = len(model_VGG16.layers[4].get_weights()[1])
    print(weights_VGG16)
    assert weights_VGG16 == 96


##############################################
# test transfer AlexNet model
##############################################


def test_remove_two_layers_AlexNet(AlexNet_model_path):
    """check whether layers get removed, should then be 17 layers
    input: inital model
    output: model minus two layers"""

    initial_model_AlexNet = create_model_AlexNet()
    initial_model_AlexNet.load_weights(AlexNet_model_path)
    transfer_model_AlexNet = Sequential(initial_model_AlexNet.layers[:-2])
    layer_amount_AlexNet = len(transfer_model_AlexNet.layers)
    assert layer_amount_AlexNet == 17


def test_create_transfer_model_AlexNet(AlexNet_model_path):
    """check whether transfer model has same amount of layers as inital mdel
    input: initial model
    output: transfer model"""

    labels = 3
    model_AlexNet = create_model_AlexNet(AlexNet_model_path)
    layer_amount_AlexNet = len(model_AlexNet.layers)
    transfer_model_AlexNet_reduced = create_transfer_model_AlexNet(saved_weights_dir=AlexNet_model_path, labels_num=labels)
    layer_amount_AlexNet_reduced = len(transfer_model_AlexNet_reduced.layers)
    assert layer_amount_AlexNet == layer_amount_AlexNet_reduced


##############################################
# test transfer VGG16 model
##############################################


def test_remove_two_layers_VGG16(VGG16_model_path):
    """check whether layers get removed, should then be 20 layers
    input: initial model
    output: initial model minus two layers"""

    initial_model_VGG16 = create_model_VGG16()
    initial_model_VGG16.load_weights(VGG16_model_path)
    transfer_model_VGG16 = Sequential(initial_model_VGG16.layers[:-2])
    layer_amount_VGG16 = len(transfer_model_VGG16.layers)
    assert layer_amount_VGG16 == 20


def test_create_transfer_model_VGG16(VGG16_model_path):
    """check whether transfer model has same amount of layers as inital mdel
    input: initial model
    output: transformed model"""

    labels = 3
    model_VGG16 = create_model_VGG16()
    layer_amount_VGG16 = len(model_VGG16.layers)
    transfer_model_VGG16_reduced = create_transfer_model_VGG16(saved_weights_dir=VGG16_model_path, labels_num=labels)
    layer_amount_VGG16_reduced = len(transfer_model_VGG16_reduced.layers)
    assert layer_amount_VGG16 == layer_amount_VGG16_reduced


##############################################
# test standardization script
##############################################


def test_reduced_dataset_file(table_path):
    """check whether output is really dataframe and if rows match length of input indexes
    input: table and indexes
    output: dataframe with matching indexes"""

    num_val = count_rows_txt(os.path.join(table_path, "val.txt"))
    print("Number of val documents: ", num_val)
    val_names = reduce_dataset(os.path.join(table_path, "val.txt"), range(num_val))
    assert isinstance(val_names, pd.DataFrame)
    assert len(val_names.index) == num_val


def test_incorrect_reduced_dataset_file(table_path):
    """check whether output is really dataframe and if rows match length of input indexes
    input: table and indexes
    output: dataframe with matching indexes"""

    num_val = count_rows_txt(os.path.join(table_path, "val.txt"))
    num_val_reduced = num_val - 1
    print("Number of val documents: ", num_val)
    val_names = reduce_dataset(os.path.join(table_path, "val.txt"), range(num_val))
    assert isinstance(val_names, pd.DataFrame)
    assert len(val_names.index) == num_val_reduced


##########################################################################################


def test_save_center_file(test_data_path):
    """check whether function outputs an np array
    input: image
    output: mean np array"""

    img = Image.open(os.path.join(test_data_path, "test.jpeg"))
    tf_image = np.array(img)
    save_center(tf_image, test_data_path, "test_mean.npy")
    assert file_exists(os.path.join(test_data_path, "test_mean.npy"))


def test_incorrect_save_center_file(test_data_path):
    """check whether function outputs an np array
    input: corrupted image
    output: no mean np array"""

    img = Image.open(os.path.join(test_data_path, "corrupted_test.jpeg"))
    tf_image = np.array(img)
    save_center(tf_image, test_data_path, "corrupted_test_mean.npy")
    assert file_exists(os.path.join(test_data_path, "corrupted_test_mean.npy"))


def test_save_correct_center_shape(test_data_path):
    """check whether correct center is saved
    input: provide image diectory and calculate mean of those images
    output: saved numpy array which contains the mean of the images analzyed and is of shape 227,227,3"""

    test_names = reduce_dataset("/data/dssg/occrp/data/train_validation_test/2022-07-14_v3/" + "test.txt"), range(820)
    test_image_scaled, test_labels, test_faulty = import_images("", test_names, 227)
    save_center(image_data=test_image_scaled, save_dir=test_data_path, file_name="center.npy")
    loaded_center = np.load(os.path.join(test_data_path, "center.npy"))
    manual_array = np.arange(154587).reshape(227, 227, 3)
    assert manual_array.shape == loaded_center.shape


def test_save_incorrect_center_shape(test_data_path):
    """check whether correct center is saved
    input: provide image diectory and calculate mean of those images
    output: wrong numpy array which contains the mean of the images analzyed with different shape then 227,227,3"""

    test_names = reduce_dataset("/data/dssg/occrp/data/train_validation_test/2022-07-14_v3/" + "test.txt", range(820))
    test_image_scaled, test_labels, test_faulty = import_images("", test_names, 227)
    save_center(image_data=test_image_scaled, save_dir=test_data_path, file_name="center.npy")
    loaded_center = np.load(os.path.join(test_data_path, "center.npy"))
    manual_array = np.arange(154587).reshape(220, 220, 3)
    assert manual_array.shape == loaded_center.shape


##########################################################################################


def test_save_stdev_file(test_data_path):
    """check whether function outputs an np array
    input: image
    output: stdev np array"""
    img = Image.open(os.path.join(test_data_path, "test.jpeg"))
    tf_image = np.array(img)
    save_stdev(tf_image, test_data_path, "test_stdev.npy")
    assert file_exists(os.path.join(test_data_path, "test_stdev.npy"))


def test_incorrect_save_stdev_file(test_data_path):
    """check whether function outputs an np array
    input: corrupted image
    output: no stdev np array"""

    img = Image.open(os.path.join(test_data_path, "corrupted_test.jpeg"))
    tf_image = np.array(img)
    save_stdev(tf_image, test_data_path, "corrupted_test_stdev.npy")
    assert file_exists(os.path.join(test_data_path, "corrupted_test_stdev.npy"))


def test_save_correct_stdev_shape(test_data_path):
    """check whether correct stdev is saved
    input: provide image diectory and calculate stdev of those images
    output: saved numpy array which contains the stdev of the images analzyed and is of shape 227,227,3"""

    test_names = reduced_dataset("/data/dssg/occrp/data/train_validation_test/2022-07-14_v3/" + "test.txt", range(820))
    test_image_scaled, test_labels, test_faulty = import_images("", test_names, 227)
    save_stdev(image_data=test_image_scaled, save_dir=test_data_path, file_name="stdev.npy")
    loaded_stdev = np.load(os.path.join(test_data_path, "stdev.npy"))
    print("The size of the stdev tensor is:", loaded_stdev.shape)
    manual_array = np.arange(154587).reshape(227, 227, 3)
    assert manual_array.shape == loaded_stdev.shape


def test_save_incorrect_stdev_shape(test_data_path):
    """check whether correct stdev is saved
    input: provide image diectory and calculate stdev of those images
    output: wrong numpy array which contains the stdev of the images analzyed and is not of shape 227,227,3"""

    test_names = reduced_dataset("/data/dssg/occrp/data/train_validation_test/2022-07-14_v3/" + "test.txt", range(820))
    test_image_scaled, test_labels, test_faulty = import_images("", test_names, 227)
    save_stdev(image_data=test_image_scaled, save_dir=test_data_path, file_name="stdev.npy")
    loaded_stdev = np.load(os.path.join(test_data_path, "stdev.npy"))
    print("The size of the stdev tensor is:", loaded_stdev.shape)
    manual_array = np.arange(154587).reshape(220, 220, 3)
    assert manual_array.shape == loaded_stdev.shape


##########################################################################################


def test_save_batched_center_file(table_path, test_data_path):
    """check whether function outputs an np array
    input: Directory of data table, directory of data directory, size of importing batches,target saving directory, name of saving file
    output:  Saved .npy file containing the image center tensor"""

    save_batched_center(os.path.join(table_path, "train.txt"), "", 20, 227, test_data_path, "test_batch_mean.npy")
    assert file_exists(os.path.join(test_data_path, "test_batch_mean.npy"))


def test_incorrect_save_batched_center_file(table_path, test_data_path):
    """check whether function outputs an np array
    input: Corrupt directory of data table, directory of data directory, size of importing batches,target saving directory, name of saving file
    output: no saved .npy file containing the image center tensor"""

    save_batched_center(os.path.join(table_path, "train.txt"), "", 20, 227, test_data_path, "corrupt_test_batch_mean.npy")
    assert file_exists(test_data_path + "test_batch_mean.npy")


def test_batched_nonbatched_mean_similarity(table_path, test_data_path):
    """check whether calculated means of entire dataset and batches match
    input:
    output:"""
    save_batched_center(os.path.join(table_path, "train.txt"), "", 20, 227, test_data_path, "test_batch_mean.npy")
    loaded_batched_center = np.load(os.path.join(test_data_path, "test_batch_mean.npy"))
    loaded_center = np.load(os.path.join(test_data_path, "test_batch_mean.npy"))
    assert math.isclose(
        loaded_batched_center[100, 100, 1], loaded_center[100, 100, 1], abs_tol=1e-3
    )  # tolerance for varaition due to different loading of images slight variation happens


##########################################################################################


def test_save_batched_stdev_file(table_path, test_data_path):
    """check whether function outputs an np array
    input: Directory of data table, directory of data directory, size of importing batches, saved center directory, saved center filename, target saving directory, name of saving file.
    output: Saved .npy file containing the image stdev tensor."""

    save_batched_stdev(
        os.path.join(table_path, "train.txt"),
        "",
        20,
        227,
        test_data_path,
        "test_batch_mean.npy",
        test_data_path,
        "test_batch_stdev.npy",
    )
    assert file_exists(os.path.join(test_data_path, "test_batch_stdev.npy"))


def test_incorrect_save_batched_stdev_file(table_path, test_data_path):
    """check whether function outputs an np array
    input: Corrupt directory of data table, directory of data directory, size of importing batches, saved center directory, saved center filename, target saving directory, name of saving file.
    output: no saved .npy file containing the image stdev tensor"""

    save_batched_stdev(
        os.path.join(table_path, "train.txt"),
        "",
        20,
        227,
        test_data_path,
        "test_batch_mean.npy",
        test_data_path,
        "test_batch_stdev.npy",
    )
    assert file_exists(os.path.join(test_data_path, "test_batch_stdev.npy"))

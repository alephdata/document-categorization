import logging
import os

import cv2
from cv2 import error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from typing import Tuple, List

logger = logging.getLogger(__name__)


# TODO count_rows_csv() and count_rows_txt() should be one function, which checks which file type it has to deal with
# Step 0: Importing the label-to-number labels.
def count_rows_csv(csv_table_dir: str) -> int:
    """
    Receives:
        csv table path

    Returns:
        the total number of rows in the document
    """

    df: pd.DataFrame = pd.read_csv(csv_table_dir, header=0)
    shape: Tuple[int, int] = df.shape
    num_obs: int = shape[0]

    return num_obs


def count_rows_txt(txt_table_dir: str) -> int:
    """Importing the label-to-number labels.
    Receives:
        txt table path

    Returns:
        total number of rows in the document
    """

    df = pd.read_csv(txt_table_dir, delimiter=" ")
    df = pd.DataFrame(df)

    shape: Tuple[int, int] = df.shape
    num_obs: int = shape[0]

    return num_obs


# TODO rename this function
# TODO refactor, delete this function
def reduce_dataset(file_dir: str, index_selection: List[int], verbose: bool = False) -> pd.DataFrame:
    """Read given randomized rows of the training table within a given range.

    Receives:
        csv table path with columns "file_path" (str) and "true_index" (num), a set of indexes

    Args:
        fileDir (_type_): _description_
        indexChoice (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame with only the selected indexes.
    """

    # Import dataset.
    full_names_db: pd.DataFrame = pd.read_csv(file_dir, delimiter=" ", names=["file_path", "true_index"], dtype=str)

    if verbose:
        print(full_names_db.head)
        print(full_names_db.shape)

    # Reduced to random sample on given range.
    reduced_names: pd.DataFrame = full_names_db.iloc[index_selection]
    reduced_names = reduced_names.reset_index(drop=True)

    return reduced_names


def import_images(
    images_dir: str, names: pd.DataFrame, im_size: int, verbose: bool = False, as_gray: bool = False
) -> Tuple[np.ndarray, List[int], List]:
    """Given file names, create reduced dataset containing images of interest/ labels of interest.
    Receives:
        Directory of images, names of images relative paths, image size

    Returns:
        np.array with image tensor, image labels, list of faulty files.
    """

    # TODO check if images_dir is valid directory, else error and exit

    n_subset: int = names.shape[0]

    images: List[np.ndarray] = []
    labels: List[int] = []
    faulty: List[int] = []
    faulty_counter: int = 0
    file_counter: int = 0

    as_color: int = 0
    if not as_gray:
        as_color = 1

    if verbose:
        print(f"The shape of names table is: {names.shape}")
        print("Importing dataset ...")
    # In this loop, we extract all document images,
    # and resize them to 227x227x3 (color image)
    for i in range(n_subset):
        data_path = os.path.join(images_dir, names["file_path"][i])
        try:
            im: np.ndarray = cv2.imread(data_path, as_color)
            img: np.ndarray = np.array(im)  # reading image as array  # TODO isn't this already an ndarray?
            if type(img) is np.ndarray:  # TODO Can this be False? If False, catch exception?
                # print(f," : ", img.shape[0], " x ", img.shape[1])
                img_resized: np.ndarray = cv2.resize(img, (im_size, im_size))
                images.append(img_resized)
                labels.append(int(names["true_index"][i]))
        except error as e:
            # Save the positions of damaged files:
            faulty_counter = faulty_counter + 1
            faulty.append(file_counter)
            logger.error(f"Faulty file in {data_path}, error: {e}")
        file_counter = file_counter + 1

    if verbose:
        print(f"There are a total of {faulty_counter} faulty files in the data.")

    # Turn into np-array and normalize.
    images_scaled: np.ndarray = np.array(images).astype("float32") / 255.0
    if as_gray:
        images_scaled = np.expand_dims(images_scaled, axis=3)
    if verbose:
        print(f"Size of image tensor is: {images_scaled.shape}")

    return images_scaled, labels, faulty


def encode_labels(labels: List[int], label_encoder: LabelEncoder) -> np.ndarray:
    """
    Receives:
        Labels to encode, initialized encoder.

    Returns:
        Encoded labels.
    """

    labels_encoded: np.ndarray = label_encoder.transform(labels)
    labels_encoded = labels_encoded.reshape(-1, 1)

    return labels_encoded


def process_images(image: np.ndarray, label: list) -> tuple:
    """
    Receives:
        np.array image tensor, labels

    Returns:
        np.array of normalized images, labels.
    """

    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    return image, label


def transform_to_tensorflow_format(
    images_scaled: np.ndarray, labels_encoded: np.ndarray, batch_size: int
) -> tf.data.Dataset:
    """
    Receives:
        np.array of image tensors, array of encoded labels, size of batch

    Returns:
        tf.data dataframe containing both normalized images and labels.
    """

    data_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((images_scaled, labels_encoded))

    data_ds_size = tf.data.experimental.cardinality(data_ds).numpy()

    data_ds = (
        data_ds.map(process_images)
        .shuffle(buffer_size=data_ds_size)
        .batch(batch_size=batch_size, drop_remainder=False)  # Used to be "True", but "False" is better.
    )

    return data_ds

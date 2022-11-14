import pytest

from prediction.data_load import documents_to_array, jpgs_to_array, pdfs_to_array, tiffs_to_array


IMG_SIZE: int = 227


def test_jpgs_to_array(jpg_paths):
    img_array = jpgs_to_array(jpg_paths)
    assert img_array.shape == (3, IMG_SIZE, IMG_SIZE, 3)


def test_tiffs_to_array(tiff_paths):
    img_array = tiffs_to_array(tiff_paths)
    assert img_array.shape == (2, IMG_SIZE, IMG_SIZE, 3)


def test_pdfs_to_array(pdf_paths):
    img_array = pdfs_to_array(pdf_paths)
    assert img_array.shape == (3, IMG_SIZE, IMG_SIZE, 3)


def test_documents_to_array(document_paths):
    img_array = documents_to_array(document_paths)
    assert img_array.shape == (8, IMG_SIZE, IMG_SIZE, 3)


def test_corrupt_jpgs_to_array(jpg_paths, corrupt_jpg_path):
    with pytest.raises(Exception):
        jpgs_to_array(jpg_paths + [corrupt_jpg_path])


def test_corrupt_tiffs_to_array(tiff_paths, corrupt_tiff_path):
    with pytest.raises(Exception):
        jpgs_to_array(tiff_paths + [corrupt_tiff_path])


def test_corrupt_pdfs_to_array(pdf_paths, corrupt_pdf_path):
    with pytest.raises(Exception):
        pdfs_to_array(pdf_paths + [corrupt_pdf_path])


def test_corrupt_documents_to_array(corrupt_document_paths):
    with pytest.raises(Exception):
        documents_to_array(corrupt_document_paths)


def test_empty_jpgs_to_array():
    with pytest.raises(Exception):
        jpgs_to_array([])


def test_empty_tiffs_to_array():
    with pytest.raises(Exception):
        tiffs_to_array([])


def test_empty_pdfs_to_array():
    with pytest.raises(Exception):
        pdfs_to_array([])


def test_empty_documents_to_array():
    with pytest.raises(Exception):
        documents_to_array([])

import pytest

# here include within fixtures which we need for all tests, such as paths etc
# then via importing pytest in the test modules this gets run automatically
# see https://docs.pytest.org/en/6.2.x/example/special.html?highlight=conftest
# see def bad_images_array() function in tests/prediction/test_predict_labels.py how to implement this


@pytest.fixture
def variables() -> dict:

    VARIABLES: dict = {"IMG_SIZE": 227}

    return VARIABLES

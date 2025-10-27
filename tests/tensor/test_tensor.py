import pytest
import torch

import jetdl

from ..utils import PyTestAsserts, generate_random_data


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (50,),
        (500,),
        (3, 4),
        (30, 40),
        (300, 400),
        (2, 3, 5),
        (2, 30, 50),
        (20, 30, 50),
        (1, 5),
        (1, 50),
        (1, 500),
        (5, 1),
        (50, 1),
        (500, 1),
    ],
)
def test_tensor_metadata(shape):
    """
    Tests that the metadata (shape, ndim, size) of a jetdl.Tensor matches
    the metadata of a torch.Tensor for various tensor shapes.
    """
    # Create a numpy array with the given shape
    data = generate_random_data(shape)

    # Create jetdl and torch tensors
    jetdl_tensor = jetdl.tensor(data)
    torch_tensor = torch.tensor(data)

    # Compare metadata
    assert_object = PyTestAsserts(jetdl_tensor, torch_tensor)
    assert (
        assert_object.check_basic_metadata()
    ), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(), assert_object.results_error_output()


@pytest.mark.parametrize(
    "data",
    [
        ([{2, 3}, {4, 5}]),
        ([{1, 2, 3}, (5, 6, 7)]),
        ([set([1, 2]), set([3, 4])]),
    ],
)
def test_tensor_incorrect_inputs_dtype(data):
    with pytest.raises(RuntimeError) as err:
        jetdl_tensor = jetdl.tensor(data)
    assert "could not infer dtype" in str(err.value)

import pytest
import torch

import jetdl

from ..utils import SEED, generate_random_data

torch.manual_seed(SEED)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (5,),
        (3, 4),
        (2, 3, 5),
        (2, 3, 4, 5),
    ],
)
def test_tensor_str(shape):
    data = generate_random_data(shape)
    j_tensor = jetdl.tensor(data)
    t_tensor = torch.tensor(data)

    # Basic check: does not raise error and is not empty
    j_str = str(j_tensor)
    assert isinstance(j_str, str)
    assert len(j_str) > 0

    # A simple heuristic to check if the string representation is reasonable
    # It should contain numbers and brackets
    assert any(char.isdigit() for char in j_str)
    assert '[' in j_str or ']' in j_str or j_tensor.ndim == 0

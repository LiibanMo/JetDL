import pytest

from tensorlite import (
    tensor,
    Tensor,
    outer,
)


def test_outer():
    tensorA = tensor([1, 1, 1])
    tensorB = tensor([1, 1])
    result = outer(tensorA, tensorB)

    assert isinstance(result, Tensor)
    assert result.shape == (3, 2)
    assert result.flatten()._data == pytest.approx([1] * 6)
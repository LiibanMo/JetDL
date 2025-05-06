import pytest

from tensorlite import (
    tensor,
    Tensor,
    ones,
    exp,
    log,
)

def test_ones():
    tensor = ones(5)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (5,)
    assert tensor._data == [1, 1, 1, 1, 1]

    tensor = ones((2, 3))
    assert tensor.shape == (2, 3)
    assert tensor.flatten()._data == pytest.approx([1] * 6)

    with pytest.raises(TypeError):
        ones("invalid_shape")


def test_exp():
    input_tensor = tensor([0, 1, 2])
    result = exp(input_tensor)

    assert isinstance(result, Tensor)
    assert result.shape == input_tensor.shape
    assert result._data == pytest.approx([1, 2.718281828459045, 7.38905609893065])


def test_log():
    input_tensor = tensor([1, 2.718281828459045, 7.38905609893065])
    result = log(input_tensor)

    assert isinstance(result, Tensor)
    assert result.shape == input_tensor.shape
    assert result._data == pytest.approx([0, 1, 2])

    with pytest.raises(ValueError):
        log(Tensor([-1, 0]))  # Logarithm of non-positive numbers should raise an error
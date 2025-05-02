import pytest
from tensorlite.routines import ones, outer, exp, log, no_grad, GradMode
from tensorlite.tensor import Tensor

def test_ones():
    tensor = ones(5)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (5,)
    assert tensor.data == [1, 1, 1, 1, 1]

    tensor = ones((2, 3))
    assert tensor.shape == (2, 3)
    assert tensor.flatten().data == pytest.approx([1] * 6)

    with pytest.raises(TypeError):
        ones("invalid_shape")


def test_outer():
    tensorA = ones(3)
    tensorB = ones(2)
    result = outer(tensorA, tensorB)

    assert isinstance(result, Tensor)
    assert result.shape == (3, 2)
    assert result.flatten().data == pytest.approx([1] * 6)


def test_exp():
    tensor = Tensor([0, 1, 2])
    result = exp(tensor)

    assert isinstance(result, Tensor)
    assert result.shape == tensor.shape
    assert result.data == pytest.approx([1, 2.718281828459045, 7.38905609893065])


def test_log():
    tensor = Tensor([1, 2.718281828459045, 7.38905609893065])
    result = log(tensor)

    assert isinstance(result, Tensor)
    assert result.shape == tensor.shape
    assert result.data == pytest.approx([0, 1, 2])

    with pytest.raises(ValueError):
        log(Tensor([-1, 0]))  # Logarithm of non-positive numbers should raise an error


def test_no_grad():
    assert GradMode.is_enabled()

    with no_grad():
        assert not GradMode.is_enabled()

    assert GradMode.is_enabled()

    with no_grad():
        with no_grad():
            assert not GradMode.is_enabled()

    assert GradMode.is_enabled()
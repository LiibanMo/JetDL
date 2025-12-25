import pytest

import jetdl


# Test device property on CPU tensors

def test_cpu_tensor_device_default():
    """CPU tensors should have device='cpu' by default."""
    t = jetdl.tensor([1, 2, 3])
    assert t.device == "cpu"


def test_cpu_tensor_device_explicit():
    """Tensors created with device='cpu' should have device='cpu'."""
    t = jetdl.tensor([1, 2, 3], device="cpu")
    assert t.device == "cpu"


def test_zeros_cpu_device():
    """zeros() with device='cpu' should create CPU tensor."""
    t = jetdl.zeros((3, 3), device="cpu")
    assert t.device == "cpu"


def test_ones_cpu_device():
    """ones() with device='cpu' should create CPU tensor."""
    t = jetdl.ones((3, 3), device="cpu")
    assert t.device == "cpu"


def test_fill_cpu_device():
    """fill() with device='cpu' should create CPU tensor."""
    t = jetdl.fill((3, 3), 5.0, device="cpu")
    assert t.device == "cpu"


# Test is_cpu and is_cuda properties

def test_cpu_tensor_is_cpu():
    """CPU tensor should have is_cpu=True."""
    t = jetdl.tensor([1, 2, 3])
    assert t.is_cpu is True


def test_cpu_tensor_is_not_cuda():
    """CPU tensor should have is_cuda=False."""
    t = jetdl.tensor([1, 2, 3])
    assert t.is_cuda is False


# Test CUDA availability functions

def test_cuda_is_available_returns_bool():
    """cuda.is_available() should return a boolean."""
    result = jetdl.cuda.is_available()
    assert isinstance(result, bool)


def test_cuda_device_count_returns_int():
    """cuda.device_count() should return an integer >= 0."""
    result = jetdl.cuda.device_count()
    assert isinstance(result, int)
    assert result >= 0

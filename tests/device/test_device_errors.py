import pytest

import jetdl


# Invalid device string tests

invalid_device_strings = [
    "gpu",
    "CPU",
    "CUDA",
    "cuda:",
    "cuda:-1",
    "cuda:abc",
    "mps",
    "tpu",
]


@pytest.mark.parametrize("invalid_device", invalid_device_strings)
def test_invalid_device_string_tensor(invalid_device):
    """Creating tensor with invalid device string should raise error."""
    with pytest.raises(Exception) as exc_info:
        jetdl.tensor([1, 2, 3], device=invalid_device)
    assert "invalid" in str(exc_info.value).lower() or "Invalid" in str(exc_info.value)


@pytest.mark.parametrize("invalid_device", invalid_device_strings)
def test_invalid_device_string_zeros(invalid_device):
    """zeros() with invalid device string should raise error."""
    with pytest.raises(Exception) as exc_info:
        jetdl.zeros((3, 3), device=invalid_device)
    assert "invalid" in str(exc_info.value).lower() or "Invalid" in str(exc_info.value)


@pytest.mark.parametrize("invalid_device", invalid_device_strings)
def test_invalid_device_string_ones(invalid_device):
    """ones() with invalid device string should raise error."""
    with pytest.raises(Exception) as exc_info:
        jetdl.ones((3, 3), device=invalid_device)
    assert "invalid" in str(exc_info.value).lower() or "Invalid" in str(exc_info.value)


@pytest.mark.parametrize("invalid_device", invalid_device_strings)
def test_invalid_device_string_fill(invalid_device):
    """fill() with invalid device string should raise error."""
    with pytest.raises(Exception) as exc_info:
        jetdl.fill((3, 3), 5.0, device=invalid_device)
    assert "invalid" in str(exc_info.value).lower() or "Invalid" in str(exc_info.value)


# Device mismatch error tests (only run if CUDA available)

requires_cuda = pytest.mark.skipif(
    not jetdl.cuda.is_available(),
    reason="CUDA is not available"
)


@requires_cuda
def test_device_mismatch_add():
    """Adding CPU and CUDA tensors should raise device mismatch error."""
    cpu_tensor = jetdl.tensor([1, 2, 3])
    cuda_tensor = jetdl.tensor([4, 5, 6]).cuda()

    with pytest.raises(Exception) as exc_info:
        _ = cpu_tensor + cuda_tensor
    assert "device" in str(exc_info.value).lower()


@requires_cuda
def test_device_mismatch_sub():
    """Subtracting CPU and CUDA tensors should raise device mismatch error."""
    cpu_tensor = jetdl.tensor([1, 2, 3])
    cuda_tensor = jetdl.tensor([4, 5, 6]).cuda()

    with pytest.raises(Exception) as exc_info:
        _ = cpu_tensor - cuda_tensor
    assert "device" in str(exc_info.value).lower()


@requires_cuda
def test_device_mismatch_mul():
    """Multiplying CPU and CUDA tensors should raise device mismatch error."""
    cpu_tensor = jetdl.tensor([1, 2, 3])
    cuda_tensor = jetdl.tensor([4, 5, 6]).cuda()

    with pytest.raises(Exception) as exc_info:
        _ = cpu_tensor * cuda_tensor
    assert "device" in str(exc_info.value).lower()


@requires_cuda
def test_device_mismatch_div():
    """Dividing CPU and CUDA tensors should raise device mismatch error."""
    cpu_tensor = jetdl.tensor([1.0, 2.0, 3.0])
    cuda_tensor = jetdl.tensor([4.0, 5.0, 6.0]).cuda()

    with pytest.raises(Exception) as exc_info:
        _ = cpu_tensor / cuda_tensor
    assert "device" in str(exc_info.value).lower()


@requires_cuda
def test_device_mismatch_matmul():
    """Matmul between CPU and CUDA tensors should raise device mismatch error."""
    cpu_tensor = jetdl.tensor([[1, 2], [3, 4]])
    cuda_tensor = jetdl.tensor([[5, 6], [7, 8]]).cuda()

    with pytest.raises(Exception) as exc_info:
        _ = jetdl.linalg.matmul(cpu_tensor, cuda_tensor)
    assert "device" in str(exc_info.value).lower()

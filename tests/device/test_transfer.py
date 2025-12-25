import pytest

import jetdl

# Skip CUDA tests if CUDA is not available
requires_cuda = pytest.mark.skipif(
    not jetdl.cuda.is_available(),
    reason="CUDA is not available"
)


# Test .to() method on CPU

def test_to_cpu_from_cpu():
    """Calling .to('cpu') on CPU tensor should return tensor on CPU."""
    t = jetdl.tensor([1, 2, 3])
    t_cpu = t.to("cpu")
    assert t_cpu.device == "cpu"
    assert t_cpu.is_cpu is True


def test_to_cpu_preserves_data():
    """Calling .to('cpu') should preserve tensor data."""
    t = jetdl.tensor([[1.0, 2.0], [3.0, 4.0]])
    t_cpu = t.to("cpu")
    assert list(t_cpu.shape) == [2, 2]


def test_to_cpu_preserves_requires_grad():
    """Calling .to('cpu') should preserve requires_grad."""
    t = jetdl.tensor([1.0, 2.0, 3.0], requires_grad=True)
    t_cpu = t.to("cpu")
    assert t_cpu.requires_grad is True


# Test .cpu() method

def test_cpu_method_on_cpu_tensor():
    """Calling .cpu() on CPU tensor should return tensor on CPU."""
    t = jetdl.tensor([1, 2, 3])
    t_cpu = t.cpu()
    assert t_cpu.device == "cpu"
    assert t_cpu.is_cpu is True


# Test Module.to(), .cuda(), .cpu() methods

def test_module_to_cpu():
    """Module.to('cpu') should keep parameters on CPU."""
    model = jetdl.nn.Linear(10, 5)
    model.to("cpu")
    assert model.weight.device == "cpu"
    assert model.bias.device == "cpu"


def test_module_cpu_method():
    """Module.cpu() should keep parameters on CPU."""
    model = jetdl.nn.Linear(10, 5)
    model.cpu()
    assert model.weight.device == "cpu"
    assert model.bias.device == "cpu"


# CUDA transfer tests (only run if CUDA available)

@requires_cuda
def test_to_cuda_from_cpu():
    """Calling .to('cuda') on CPU tensor should transfer to CUDA."""
    t = jetdl.tensor([1, 2, 3])
    t_cuda = t.to("cuda")
    assert t_cuda.is_cuda is True
    assert "cuda" in t_cuda.device


@requires_cuda
def test_cuda_method():
    """Calling .cuda() should transfer tensor to CUDA."""
    t = jetdl.tensor([1, 2, 3])
    t_cuda = t.cuda()
    assert t_cuda.is_cuda is True


@requires_cuda
def test_to_cpu_from_cuda():
    """Calling .to('cpu') on CUDA tensor should transfer to CPU."""
    t = jetdl.tensor([1, 2, 3])
    t_cuda = t.cuda()
    t_cpu = t_cuda.to("cpu")
    assert t_cpu.is_cpu is True
    assert t_cpu.device == "cpu"


@requires_cuda
def test_cpu_method_from_cuda():
    """Calling .cpu() on CUDA tensor should transfer to CPU."""
    t = jetdl.tensor([1, 2, 3])
    t_cuda = t.cuda()
    t_cpu = t_cuda.cpu()
    assert t_cpu.is_cpu is True


@requires_cuda
def test_roundtrip_cpu_cuda_cpu():
    """Data should be preserved after CPU -> CUDA -> CPU transfer."""
    import torch

    data = [[1.0, 2.0], [3.0, 4.0]]
    t = jetdl.tensor(data)
    t_cuda = t.cuda()
    t_cpu = t_cuda.cpu()

    # Compare with original
    t_torch = torch.tensor(data)

    from ..utils import PyTestAsserts
    assert_obj = PyTestAsserts(t_cpu, t_torch)
    assert assert_obj.check_results(), assert_obj.results_error_output()


@requires_cuda
def test_module_to_cuda():
    """Module.to('cuda') should move parameters to CUDA."""
    model = jetdl.nn.Linear(10, 5)
    model.to("cuda")
    assert model.weight.is_cuda is True
    assert model.bias.is_cuda is True


@requires_cuda
def test_module_cuda_method():
    """Module.cuda() should move parameters to CUDA."""
    model = jetdl.nn.Linear(10, 5)
    model.cuda()
    assert model.weight.is_cuda is True
    assert model.bias.is_cuda is True


@requires_cuda
def test_module_roundtrip():
    """Module parameters should be preserved after CPU -> CUDA -> CPU."""
    model = jetdl.nn.Linear(4, 2)

    # Get original weight shape
    orig_shape = list(model.weight.shape)

    # Move to CUDA and back
    model.cuda()
    model.cpu()

    # Shape should be preserved
    assert list(model.weight.shape) == orig_shape
    assert model.weight.is_cpu is True

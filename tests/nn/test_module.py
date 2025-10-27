import pytest

import jetdl.nn as nn
from jetdl import tensor
from jetdl.nn import Parameter

# nn.Module


def test_empty_module():
    """
    Tests that a bare nn.Module has no parameters.
    """
    module = nn.Module()
    assert len(list(module.parameters())) == 0


def test_module_with_parameters():
    """
    Tests that Tensors with requires_grad=True and explicit nn.Parameters
    are registered correctly.
    """

    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = Parameter([4, 5])
            self.t1 = tensor([2.0], requires_grad=True)
            self.not_a_param1 = tensor([3.0], requires_grad=False)
            self.not_a_param2 = [Parameter([2, 3])]  # In a list, should not be found

    m = TestModule()
    params = list(m.parameters())
    assert len(params) == 1


def test_nested_module():
    """
    Tests that parameters from submodules are collected.
    """

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = Parameter([2, 2])

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.p2 = Parameter([2])
            self.inner = Inner()
            self.inner_list = [Inner()]  # Modules in a list are not registered

    m = Outer()
    params = list(m.parameters())
    assert len(params) == 2


def test_forward_not_implemented():
    """
    Tests that the base Module's forward method raises NotImplementedError.
    """
    m = nn.Module()
    with pytest.raises(NotImplementedError):
        m.forward(1)


def test_call_invokes_forward():
    """
    Tests that calling a module instance invokes its forward method.
    """

    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_called = False

        def forward(self, x):
            self.forward_called = True
            return x

    m = TestModule()
    assert not m.forward_called
    inp = tensor([1.0])
    out = m(inp)  # invokes __call__
    assert m.forward_called
    assert out is inp

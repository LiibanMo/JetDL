import pytest
from jetdl import tensor
from jetdl.linalg import matmul


def test_matmul_square_2x2():
    """Test matrix multiplication with square 2x2 matrices."""
    a = tensor([[1, 2], [3, 4]], requires_grad=False)
    b = tensor([[5, 6], [7, 8]], requires_grad=False)
    
    result = matmul(a, b)
    
    # Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
    # Expected: [[19, 22], [43, 50]]
    # Flattened: [19, 22, 43, 50]
    assert result.shape == (2, 2)
    assert result._data[0] == pytest.approx(19)
    assert result._data[1] == pytest.approx(22)
    assert result._data[2] == pytest.approx(43)
    assert result._data[3] == pytest.approx(50)


def test_matmul_non_square_2x3_3x2():
    """Test matrix multiplication with non-square matrices: 2x3 @ 3x2 = 2x2."""
    a = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=False)  # 2x3
    b = tensor([[7, 8], [9, 10], [11, 12]], requires_grad=False)  # 3x2
    
    result = matmul(a, b)
    
    # Expected: [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12], [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
    # Expected: [[58, 64], [139, 154]]
    # Flattened: [58, 64, 139, 154]
    assert result.shape == (2, 2)
    assert result._data[0] == pytest.approx(58)
    assert result._data[1] == pytest.approx(64)
    assert result._data[2] == pytest.approx(139)
    assert result._data[3] == pytest.approx(154)


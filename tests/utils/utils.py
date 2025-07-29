import torch

SEED = 123
ERR = 1e-6

def generate_random_data(shape1, shape2=None):
    if shape2 is None:
        return torch.rand(shape1).tolist()
    return torch.rand(shape1).tolist(), torch.rand(shape2).tolist()


def generate_shape_ids(shapes) -> str:
    return f" {shapes} "


class PyTestAsserts:
    def __init__(self, result, expected):
        self.j = result  # jetdl
        self.t = expected  # torch

    def check_shapes(self) -> bool:
        return self.j.shape == self.t.shape

    def shapes_error_output(self) -> str:
        return f"Expected shapes to match: {self.j.shape} vs {self.t.shape}"

    def check_results(self, err:float=ERR) -> bool:
        return torch.allclose(self.j, self.t, err)

    def results_error_output(self) -> str:
        return f"Expected tensors to be close: {self.j} vs {self.t}"

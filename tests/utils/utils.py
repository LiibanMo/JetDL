import jetdl
import torch

SEED = 123
ERR = 1e-6

def generate_random_data(shape1, shape2=None):
    if shape2 is None:
        return torch.rand(shape1).tolist()
    return torch.rand(shape1).tolist(), torch.rand(shape2).tolist()

def generate_shape_ids(shapes) -> str:
    return f" {shapes} "

def obtain_result_tensors(data1, data2, operation:str):
    jetdl_op, torch_op = operation_registry[operation]

    j1 = jetdl.tensor(data1)
    j2 = jetdl.tensor(data2)
    j3 = jetdl_op(j1, j2)

    t1 = torch.tensor(data1)
    t2 = torch.tensor(data2)
    expected_tensor = torch_op(t1, t2)

    return j3, expected_tensor

operation_registry = {
    "ADD": (jetdl.add, torch.add),
    "SUB": (jetdl.sub, torch.sub),
    "MUL": (jetdl.mul, torch.mul),
    "DIV": (jetdl.div, torch.div),
    "MATMUL": (jetdl.matmul, torch.matmul),
}

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
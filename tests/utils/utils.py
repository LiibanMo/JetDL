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
        self.j: jetdl.Tensor = result  
        self.t: torch.Tensor = expected  

    def check_shapes(self) -> bool:
        return self.j.shape == self.t.shape
    
    def check_ndims(self) -> bool:
        return self.j.ndim == self.t.ndim
    
    def check_sizes(self) -> bool:
        return self.j.size == self.t.numel()
    
    def check_strides(self) -> bool:
        return self.j.strides == self.t.stride()
    
    def check_is_contiguous(self) -> bool:
        return self.j.is_contiguous == self.t.is_contiguous()
    
    def check_basic_metadata(self) -> bool:
        return (
            self.check_shapes() and 
            self.check_ndims() and 
            self.check_sizes() and 
            self.check_strides() and
            self.check_is_contiguous()
        )

    def shapes_error_output(self) -> str:
        return f"Expected shapes to match: {self.j.shape} vs {self.t.shape}"
    
    def ndim_error_output(self) -> str:
        return f"Expected ndim to match: {self.j.ndim} vs {self.t.ndim}"
   
    def size_error_output(self) -> str:
        return f"Expected sizes to match: {self.j.size} vs {self.t.numel()}"
   
    def strides_error_output(self) -> str:
        return f"Expected strides to match: {self.j.strides} vs {self.t.stride()}"
    
    def is_contiguous_error_output(self) -> str:
        return f"Expected is_contiguous to match: {self.j.is_contiguous} vs {self.t.is_contiguous()}"

    def basic_metadata_error_output(self) -> str:
        if not self.check_shapes():
            return self.shapes_error_output()
        
        elif not self.check_ndims():
            return self.ndim_error_output()
        
        elif not self.check_sizes():
            return self.size_error_output()
        
        elif not self.check_strides():
            return self.strides_error_output()
        
        elif not self.check_is_contiguous():
            return self.is_contiguous_error_output()

    def check_results(self, err:float=ERR) -> bool:
        j_torch_version = torch.tensor(self.j._data).reshape(self.j.shape)
        return torch.allclose(j_torch_version, self.t, err)

    def results_error_output(self) -> str:
        return f"Expected tensors to be close: {self.j._data} vs {self.t}"
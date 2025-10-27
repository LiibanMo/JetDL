import pytest
import torch

import jetdl
import jetdl.nn as nn
from jetdl.nn import Parameter 

from ..utils import PyTestAsserts, generate_random_data, generate_shape_ids

# nn.Linear

def test_linear_initialization():
    in_features, out_features = 10, 5
    linear_layer = nn.Linear(in_features, out_features)

    assert hasattr(linear_layer, 'weight')
    assert hasattr(linear_layer, 'bias')
    assert isinstance(linear_layer.weight, Parameter)
    assert isinstance(linear_layer.bias, Parameter)
    
    assert linear_layer.weight.shape == (out_features, in_features)
    assert linear_layer.bias.shape == (out_features,)

    params = list(linear_layer.parameters())
    assert len(params) == 2
    param_shapes = [p.shape for p in params]
    assert (out_features, in_features) in param_shapes
    assert (out_features,) in param_shapes

@pytest.mark.parametrize("batch_size, in_features, out_features", [
    (1, 10, 5),
    (8, 20, 10),
], ids=generate_shape_ids)
def test_linear_forward(batch_size, in_features, out_features):
    j_linear = nn.Linear(in_features, out_features )
    t_linear = torch.nn.Linear(in_features, out_features)

    with torch.no_grad():
        t_linear.weight.copy_(torch.asarray(j_linear.weight).reshape(j_linear.weight.shape))
        t_linear.bias.copy_(torch.asarray(j_linear.bias).reshape(j_linear.bias.shape))

    input_data = generate_random_data((batch_size, in_features))
    j_input = jetdl.tensor(input_data)
    t_input = torch.tensor(input_data, dtype=torch.float32)

    j_output = j_linear(j_input)
    print(j_output)
    t_output = t_linear(t_input)
    print(t_output)

    assert_object = PyTestAsserts(j_output, t_output)
    assert assert_object.check_basic_metadata(), assert_object.basic_metadata_error_output()
    assert assert_object.check_results(err=1e-3), assert_object.results_error_output()
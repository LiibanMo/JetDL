# JetDL

JetDL is a deep learning library built from scratch in C++ with a Python interface designed to be familiar to PyTorch users.

## Installation

To install the library and its dependencies, run the following command in your terminal:

```bash
pip install jetdl
```

## Usage

Here are some examples of how to use JetDL for basic tensor operations and for building and training a simple neural network.

### Tensor Initialization

You can initialize tensors in various ways:

```python
import jetdl

# Create a tensor from a list
a = jetdl.tensor([[1, 2], [3, 4]])
print(a)

# Create a tensor of zeros
b = jetdl.zeros((2, 3))
print(b)

# Create a tensor of ones
c = jetdl.ones((3, 2))
print(c)
```

### Tensor Arithmetic

JetDL supports common arithmetic operations, which are performed element-wise.

```python
import jetdl

a = jetdl.tensor([[1, 2], [3, 4]])
b = jetdl.tensor([[5, 6], [7, 8]])

# Addition
c = a + b
print(c)

# Subtraction
d = a - b
print(d)

# Element-wise multiplication
e = a * b
print(e)

# Element-wise division
f = a / b
print(f)
```

### Tensor Matrix Multiplication

You can perform matrix multiplication using `jetdl.matmul`.

```python
import jetdl

# Create two tensors with compatible shapes for matrix multiplication
a = jetdl.tensor([[1, 2], [3, 4]])
b = jetdl.tensor([[5, 6], [7, 8]])

# Perform matrix multiplication
c = jetdl.matmul(a, b)
print(c)
print(c.shape)
```

### Training a Neural Network

Here is an example of how to define a simple neural network with a Linear and a ReLU layer and train it on random data.

```python
import jetdl
import jetdl.nn as nn

# 1. Define the model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        return x

model = SimpleNet()

# 2. Create dummy data
X_train = jetdl.rand(100, 10)
y_train = jetdl.rand(100, 20)

# 3. Define loss and optimizer
criterion = nn.MSELoss()
optimizer = jetdl.optim.SGD(model.parameters(), lr=0.01)

# 4. Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

```

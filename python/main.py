from tensor import Tensor

tensorA = Tensor([[1,2], 
                  [3,4], 
                  [5,6]])

tensorB = Tensor([[1,1],
                  [1,1],
                  [1,1]])

tensorC = tensorA - tensorB

print("Successful")
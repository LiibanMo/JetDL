def _obtain_broadcast_booleans(tensorA, tensorB):
    from ..tensor._C import _TensorBase

    if not isinstance(tensorB, _TensorBase):
        broadcasted = True
        B_is_tensor = False
    elif tensorA.shape != tensorB.shape:
        broadcasted = True
        B_is_tensor = True
    else:
        broadcasted = False
        B_is_tensor = True
    return broadcasted, B_is_tensor

def _obtain_broadcasted_batch_dims(tensorA, tensorB):
    max_ndim = max(tensorA.ndim, tensorB.ndim)
    
    broadcasted_batch_dimsA, broadcasted_batch_dimsB = [], []
    
    diff_in_dims = abs(tensorA.ndim - tensorB.ndim)

    if tensorA.ndim < max_ndim:
        for idx in range(diff_in_dims):
            broadcasted_batch_dimsA.append(idx)
    elif tensorB.ndim < max_ndim:
        for idx in range(diff_in_dims):
            broadcasted_batch_dimsB.append(idx)
   
    return (broadcasted_batch_dimsA, broadcasted_batch_dimsB)
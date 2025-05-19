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

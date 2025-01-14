def _calculate_strides(shape_:list) -> list:
    strides = [1]
    shape = shape_.copy()
    shape.reverse()
    for idx, dim in enumerate(shape[:-1]):
        strides.append(strides[idx] * dim)
    strides.reverse()
    return strides
class GradMode:
    _enabled = True
    _no_grad_depth = 0

    @classmethod
    def is_enabled(cls):
        return cls.is_enabled and cls._no_grad_depth == 0


class no_grad:
    def __enter__(self):
        GradMode._no_grad_depth += 1

    def __exit__(self, exec_type, exec_value, traceback):
        GradMode._no_grad_depth -= 1

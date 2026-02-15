"""Fix Triton compatibility on Jetson (PyTorch 2.5.0a0 + Triton 3.5.0)."""
import triton.testing, inspect

_orig = triton.testing.do_bench
_valid_params = set(inspect.signature(_orig).parameters.keys())

def _patched(*args, **kwargs):
    if 'percentiles' in kwargs:
        kwargs['quantiles'] = kwargs.pop('percentiles')
    filtered = {k: v for k, v in kwargs.items() if k in _valid_params}
    return _orig(*args, **filtered)

triton.testing.do_bench = _patched

import torch._inductor.runtime.runtime_utils as rutils
rutils.triton_do_bench = _patched

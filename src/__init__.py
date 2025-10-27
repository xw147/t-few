# PyTorch 2.6 compatibility fix for skip_code import error
import torch

try:
    # Try to import skip_code - if it fails, create a dummy function
    from torch._C._dynamo.eval_frame import skip_code
except (ImportError, AttributeError):
    # Create a compatibility layer for older/newer PyTorch versions
    from types import ModuleType
    
    # Create dummy module structure if it doesn't exist
    if not hasattr(torch._C, '_dynamo'):
        torch._C._dynamo = ModuleType('_dynamo')
    if not hasattr(torch._C._dynamo, 'eval_frame'):
        torch._C._dynamo.eval_frame = ModuleType('eval_frame')
    
    # Create dummy skip_code function
    def skip_code(x):
        return x
    
    torch._C._dynamo.eval_frame.skip_code = skip_code
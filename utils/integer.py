import torch

def quantize_int_weight(module):
    """
    get weight of type 'uint8' of a quantized module.
    Bias are not quantized and you can use raw bias.
    """
    assert hasattr(module, 'weight'), f"module {module} does not have weight"
    assert module.w_bit == 8, f"module {module}'s weight is quantized with {module.w_bit} bits"

    w_int = (module.weight/module.w_interval).round_().clamp_(-module.w_qmax, module.w_qmax-1)
    w_int = w_int.cpu().detach().to(torch.int8)
    return w_int

def get_model_int_weight(wrapped_modules):
    """
    Get quantized weights (in int8) of a model.

    Return:
        A dict, with modules' names as keys, and int weights as values.
    """

    int_weights = {}

    for name, m in wrapped_modules.items():
        try:
            int_weights[name] = quantize_int_weight(m)
        except:
            pass
    
    return int_weights

import torch.nn as nn
from utils.models import MatMul



def wrap_modules_in_net(net,cfg):
    wrapped_modules={}
    module_dict={}
    module_types = {"qkv":"qlinear_qkv", "proj":'qlinear_proj', 'fc1':'qlinear_MLP_1', 'fc2':"qlinear_MLP_2", 'head':'qlinear_classifier','matmul1':"qmatmul_qk", 'matmul2':"qmatmul_scorev", "reduction": "qlinear_reduction"}
    
    it=[(name,m) for name,m in net.named_modules()]
    for name,m in it:
        module_dict[name]=m
        idx=name.rfind('.')
        if idx==-1:
            idx=0
        father_name=name[:idx]
        if father_name in module_dict:
            father_module=module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        if isinstance(m,nn.Conv2d):
            # Embedding Layer
            idx = idx+1 if idx != 0 else idx
            new_m=cfg.get_module("qconv",m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,m.padding_mode)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
        elif isinstance(m,nn.Linear):
            # Linear Layer
            idx = idx+1 if idx != 0 else idx
            new_m = cfg.get_module(module_types[name[idx:]],m.in_features,m.out_features)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
        elif isinstance(m,MatMul):
            # Matmul Layer
            idx = idx+1 if idx != 0 else idx
            new_m = cfg.get_module(module_types[name[idx:]])
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
    print("Completed net wrap.")
    return wrapped_modules
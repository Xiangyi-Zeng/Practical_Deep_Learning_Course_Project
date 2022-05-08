from cgi import test
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import torch
from tqdm import tqdm
from importlib import reload,import_module
from itertools import product
import os
import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.quant_calib import HessianQuantCalibrator
from utils.models import get_net
from collections import defaultdict
from utils.integer import get_model_int_weight


def test_classification(net,test_loader,max_iteration=None, description=None):
    pos=0
    tot=0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q=tqdm(test_loader, desc=description)
        for inp,target in q:
            i+=1
            inp=inp.cuda()
            target=target.cuda()
            out=net(inp)
            pos_num=torch.sum(out.argmax(1)==target).item()
            pos+=pos_num
            tot+=inp.size(0)
            q.set_postfix({"acc":pos/tot})
            if i >= max_iteration:
                break
    return pos/tot
    
def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("./configs"))
    if config_name+".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg

def experiment(net='vit_tiny_patch16_224', config="Basic"):
    """
    A basic testbench.
    """
    quant_cfg = init_config(config)
    net = get_net(net)

    wrapped_modules = net_wrap.wrap_modules_in_net(net,quant_cfg)
    
    g=datasets.ViTImageNetLoaderGenerator('./data/imagenet','imagenet',32,32,12,kwargs={"model":net})
    test_loader=g.test_loader()
    calib_loader=g.calib_loader(num=32)
    
    quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) 
    quant_calibrator.batching_quant_calib()

    test_acc = test_classification(net,test_loader)
    print('quantized test acc for {} in setting {}'.format(net,config))
    print(test_acc)

    quantized_model_weights = get_model_int_weight(wrapped_modules)
    quantized_model_size = get_model_size(quantized_model_weights)
    print('quantized model size for {} in setting {}'.format(net,config))
    print("%.2f MB" %(quantized_model_size))
    return test_acc,quantized_model_size


def get_model_size(mdl):
    torch.save(mdl, "tmp.pt")
    model_size = os.path.getsize("tmp.pt")/1e6
    
    os.remove('tmp.pt')

    return model_size

def get_baseline(net):
    model = get_net(net)
    g=datasets.ViTImageNetLoaderGenerator('./data/imagenet','imagenet',32,32,12,kwargs={"model":model})
    test_loader=g.test_loader()
    test_acc = test_classification(model,test_loader)
    model_size = get_model_size(model.state_dict())
    
    return test_acc,model_size

if __name__=='__main__':
    nets = [
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",

        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224"
        ]

    # get baseline data for ViT models   
    test_acc_dict = {}
    model_size_dict = {}
    for net in nets:
        test_acc,model_size = get_baseline(net)
        test_acc_dict[net] = test_acc
        model_size_dict[net] = model_size
    torch.save(test_acc_dict,'test_acc.pt')
    torch.save(model_size_dict,'model_size.pt')

    # get test_acc and model size for quant ViT models
    # in two settings
    cfg_list = []

    configs= ['Basic','PTQ4ViT']
    quantized_test_acc_dict = defaultdict(list)
    quantized_model_size_dict = defaultdict(list)
   
    cfg_list = [{
        "net":net,
        "config":config,
        }
        for net, config in product(nets, configs) 
    ]

    for cfg in cfg_list:
        test_acc,model_size= experiment(**cfg)
        quantized_test_acc_dict[cfg['net']].append(test_acc)
        quantized_model_size_dict[cfg['net']].append(model_size)

    torch.save(quantized_test_acc_dict,'PTQ_test_acc.pt')
    torch.save(quantized_model_size_dict,'PTQ_model_size.pt')
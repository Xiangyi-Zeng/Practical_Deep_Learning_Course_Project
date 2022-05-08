import torch
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import timm
from tqdm import tqdm
import utils.datasets as datasets
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import torchvision.transforms as transforms
import requests
from collections import defaultdict
import os
from utils.models import get_net

def get_model_size(mdl):
    torch.save(mdl, "tmp.pt")
    model_size = os.path.getsize("tmp.pt")/1e6
    
    os.remove('tmp.pt')

    return model_size

def get_image():
    transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

    img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
    img = transform(img)[None,]

    return img

def get_quantized_model(net_name='vit_base_patch16_224'):
    model = timm.create_model(net_name, pretrained=True)
    model.eval()

    backend = "fbgemm"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear,torch.nn.Conv2d}, dtype=torch.qint8)

    return quantized_model

    
def get_inference_time(net_name,img):
    model = timm.create_model(net_name, pretrained=True)
    quantized_model = get_quantized_model(net_name)
    with torch.autograd.profiler.profile(use_cuda=False) as prof1:
        out = model(img)
    with torch.autograd.profiler.profile(use_cuda=False) as prof2:
        out = quantized_model(img)
    infer_time_1 = prof1.self_cpu_time_total/1000
    infer_time_2 = prof2.self_cpu_time_total/1000
    return infer_time_1,infer_time_2

def test_classification(net,test_loader,max_iteration=None, description=None):
    pos=0
    tot=0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q=tqdm(test_loader, desc=description)
        for inp,target in q:
            i+=1
            inp=inp
            target=target
            out=net(inp)
            pos_num=torch.sum(out.argmax(1)==target).item()
            pos+=pos_num
            tot+=inp.size(0)
            q.set_postfix({"acc":pos/tot})
            if i >= max_iteration:
                break
    return pos/tot

if __name__ == '__main__':
    nets = [
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",

        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224"
        ]
    infer_time_dict = defaultdict(list)
    test_acc_dict = {}
    model_size_dict ={}
    img = get_image()
    for net_name in nets:
        quantized_model = get_quantized_model(net_name)
        # #model size
        # model_size = get_model_size(quantized_model.state_dict())
        # model_size_dict[net_name] = model_size
        #inference time
        infer_time_1,infer_time_2 = get_inference_time(net_name,img)
        infer_time_dict[net_name]=[infer_time_1,infer_time_2]
        # #test acc
        # net = timm.create_model(net_name, pretrained=True)
        # g=datasets.ViTImageNetLoaderGenerator('./data/imagenet','imagenet',32,32,2,kwargs={"model":net})
        # test_loader=g.test_loader()
        # test_acc =test_classification(quantized_model,test_loader)
        # test_acc_dict[net_name] = test_acc

    torch.save(infer_time_dict,'experimental_results/infer_time_pytorch.pt')
    # torch.save(test_acc_dict,'experimental_results/Pytorch_test_acc.pt')
    # torch.save(model_size_dict,'experimental_results/Pytorch_model_size.pt')
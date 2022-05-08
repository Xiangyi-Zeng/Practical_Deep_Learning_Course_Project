from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import torch

PTQ_model_size = torch.load('experimental_results/PTQ_model_size.pt')
PTQ_test_acc = torch.load('experimental_results/PTQ_test_acc.pt')
b_models_size = torch.load('experimental_results/model_size.pt')
b_test_acc = torch.load('experimental_results/test_acc.pt')
pytorch_model_size = torch.load('experimental_results/Pytorch_model_size.pt')
Pytorch_test_acc = torch.load('experimental_results/Pytorch_test_acc.pt')
nets = [
        "vit_t_16",
        "vit_s_16",
        "vit_b_16",

        "deit_t_16",
        "deit_s_16",
        "deit_b_16"
        ]
w = 0.2
model_size = np.array(list(PTQ_model_size.values()))
test_acc =  np.array(list(PTQ_test_acc.values()))
basic_model_size = model_size[:,0]
PTQ4ViT_model_size = model_size[:,1]
basic_test_acc = test_acc[:,0]*100
PTQ4ViT_test_acc = test_acc[:,1]*100
base_model_size = np.array(list(b_models_size.values()))
base_test_acc = np.array(list(b_test_acc.values()))*100
p_model_size = np.array(list(pytorch_model_size.values()))
p_test_acc = np.array(list(Pytorch_test_acc.values()))*100


bar1 = np.arange(len(nets))
bar2 = [i+w for i in bar1]
bar3 = [i+w for i in bar2]
bar4 = [i+w for i in bar3]

#plot model size
plt.figure(1)
plt.bar(bar1,base_model_size,w,label='baseline')
plt.bar(bar2,basic_model_size,w,label='Basic')
plt.bar(bar3,PTQ4ViT_model_size,w,label='PTQ4ViT')
plt.bar(bar4,p_model_size,w,label='DQ-Pytorch')
plt.xticks(bar2,nets)
plt.xlabel('ViT models')
plt.ylabel('Model size /MB')
plt.title('Model size of ViT models w/ and w/o quantization')
plt.legend()

#plot test acc
plt.figure(2)
plt.bar(bar1,base_test_acc,w,label='baseline')
plt.bar(bar2,basic_test_acc,w,label='Basic')
plt.bar(bar3,PTQ4ViT_test_acc,w,label='PTQ4ViT')
plt.bar(bar4,p_test_acc,w,label='DQ-Pytorch')
plt.xticks(bar2,nets)
plt.xlabel('ViT models')
plt.ylabel('Test accuracy %')
plt.ylim(50,100)
plt.title('Test accuracy of ViT models w/ and w/o quantization')
plt.legend()

#plot model size reduction
basic_reduce = 100*(base_model_size-basic_model_size)/base_model_size
PTQ4ViT_reduce = 100*(base_model_size-PTQ4ViT_model_size)/base_model_size
p_reduce = 100*(base_model_size-p_model_size)/base_model_size

plt.figure(3)
plt.bar(bar1,basic_reduce,w,label='Basic')
plt.bar(bar2,PTQ4ViT_reduce,w,label='PTQ4ViT')
plt.bar(bar3,p_reduce,w,label='DQ-Pytorch')
plt.xticks(bar2,nets)
plt.xlabel('ViT models')
plt.ylabel('Model size reduction %')
plt.ylim(50,100)
plt.title('Model size reduction of ViT models w/ quantization')
plt.legend()


#plot test acc loss
basic_loss = 100*(base_test_acc-basic_test_acc)/base_test_acc
PTQ4ViT_loss = 100*(base_test_acc-PTQ4ViT_test_acc)/base_test_acc
p_loss = 100*(base_test_acc-p_test_acc)/base_test_acc

plt.figure(4)
plt.bar(bar1,basic_loss,w,label='Basic')
plt.bar(bar2,PTQ4ViT_loss,w,label='PTQ4ViT')
plt.bar(bar3,p_loss,w,label='DQ-Pytorch')
plt.xticks(bar2,nets)
plt.xlabel('ViT models')
plt.ylabel('Test accuracy loss %')
plt.ylim(0,10)
plt.title('Test accuracy loss of ViT models w/ quantization')
plt.legend()


#plot throughput
infer_time = torch.load('experimental_results/infer_time_pytorch.pt')
thp = np.array(list(infer_time.values()))
thp_base = 1000/thp[:,0]
thp_p = 1000/thp[:,1]

plt.figure(5)
plt.bar(bar1,thp_base,w,label='baseline')
plt.bar(bar2,thp_p,w,label='DQ-pytorch')
plt.xticks(bar1,nets)
plt.xlabel('ViT models')
plt.ylabel('Throughput images/s')
plt.ylim(0,60)
plt.title('Throughput of ViT models w/ and w/o quantization')
plt.legend()
plt.show()


"""
Reuse version v4
Author: Hahn Yuan
"""
import torch
import numpy as np
import os
import copy
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.utils.data
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class LoaderGenerator():
    """
    """
    def __init__(self,root,dataset_name,train_batch_size=1,test_batch_size=1,num_workers=0,kwargs={}):
        self.root=root
        self.dataset_name=str.lower(dataset_name)
        self.train_batch_size=train_batch_size
        self.test_batch_size=test_batch_size
        self.num_workers=num_workers
        self.kwargs=kwargs
        self.items=[]
        self._train_set=None
        self._test_set=None
        self._calib_set=None
        self.train_transform=None
        self.test_transform=None
        self.train_loader_kwargs = {
            'num_workers': self.num_workers ,
            'pin_memory': kwargs.get('pin_memory',True),
            'drop_last':kwargs.get('drop_last',False)
            }
        self.test_loader_kwargs=self.train_loader_kwargs.copy()
        self.load()
    
    @property
    def train_set(self):
        pass
    
    @property
    def test_set(self):
        pass
    
    def load(self):
        pass
    
    def train_loader(self):
        assert self.train_set is not None
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True,  **self.train_loader_kwargs)
    
    def test_loader(self,shuffle=False,batch_size=None):
        assert self.test_set is not None
        if batch_size is None:
            batch_size=self.test_batch_size
        return torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle,  **self.test_loader_kwargs)
    
    def val_loader(self):
        assert self.val_set is not None
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.test_batch_size, shuffle=False,  **self.test_loader_kwargs)
    
    def trainval_loader(self):
        assert self.trainval_set is not None
        return torch.utils.data.DataLoader(self.trainval_set, batch_size=self.train_batch_size, shuffle=True,  **self.train_loader_kwargs)

    def calib_loader(self,num=1024,seed=3):
        if self._calib_set is None:
            np.random.seed(seed)
            # inds=np.random.permutation(len(self.train_set))[:num]
            # self._calib_set=torch.utils.data.Subset(copy.deepcopy(self.train_set),inds)
            inds=np.random.permutation(len(self.test_set))[:num]
            self._calib_set=torch.utils.data.Subset(copy.deepcopy(self.test_set),inds)
            self._calib_set.dataset.transform=self.test_transform
        return torch.utils.data.DataLoader(self._calib_set, batch_size=num, shuffle=False,  **self.train_loader_kwargs)
        
class ImageNetLoaderGenerator(LoaderGenerator):
    def load(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set=ImageFolder(os.path.join(self.root,'train'), self.train_transform)
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set=ImageFolder(os.path.join(self.root,'val'), self.test_transform)
        return self._test_set

       
class ViTImageNetLoaderGenerator(ImageNetLoaderGenerator):
    """
    DataLoader for Vision Transformer. 
    To comply with timm's framework, we use the model's corresponding transform.
    """
    def __init__(self, root, dataset_name, train_batch_size, test_batch_size, num_workers, kwargs={}):
        kwargs.update({"pin_memory":False})
        super().__init__(root, dataset_name, train_batch_size=train_batch_size, test_batch_size=test_batch_size, num_workers=num_workers, kwargs=kwargs)

    def load(self):
        model = self.kwargs.get("model", None)
        assert model != None, f"No model in ViTImageNetLoaderGenerator!"

        config = resolve_data_config({}, model=model)
        self.train_transform = create_transform(**config, is_training=True)
        self.test_transform = create_transform(**config)


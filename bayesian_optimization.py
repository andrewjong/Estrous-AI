
# coding: utf-8

# In[1]:


from torchvision import models as tvm
import pretrainedmodels as ptm
models = (
    tvm.resnet101, 
    tvm.resnet152, 
    
    ptm.inceptionv4, 
    ptm.inceptionresnetv2
)

DATA_DIR = 'data/4_class_11'

domain = [
    {'name': 'model_num', 'type': 'discrete', 'domain': range(len(models))},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (1, 4, 16, 24, 32, 48, 64)},
    {'name': 'adam_lr', 'type': 'continuous', 'domain': (0.001, 0.005, 0.01, 0.05, 0.1, 0.2)},
    {'name': 'adam_beta1', 'type': 'continuous', 'domain': (0.8, 1.)},
    {'name': 'adam_beta2', 'type': 'continuous', 'domain': (0.8, 1.)},
    {'name': 'adam_wtdecay', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'epochs', 'type': 'discrete', 'domain': (10, 20, 30, 40, 50)}
]


# In[2]:


def f(x):
    """ Value function to minimize for bayesian optimization """
    val_acc = train(
        model_num=int(x[:,0]),
        batch_size=int(x[:,1]),
        adam_lr=float(x[:,2]),
        adam_b1=float(x[:,3]),
        adam_b2=float(x[:,4]),
        adam_wtdecay=float(x[:,5]),
        epochs=int(x[:,6])
    )
    
    return -val_acc


NUM_CLASSES = 4


def prepare_model(model_num):
    if model_num <= 1:
        # torchvision models
        pretrained = True
        last = 'fc'
        image_size = 224
    else:
        # pretrainedmodels package specific differences
        pretrained = 'imagenet'
        last = 'last_linear'
        image_size = 299
    
    model = models[model_num](pretrained=pretrained)
    num_in = getattr(model, last).in_features
    setattr(model, last, torch.nn.Linear(num_in, NUM_CLASSES))
    return model, image_size


import torch
from torch.optim import Adam
from torch.nn import  CrossEntropyLoss
from src.trainable import Trainable
from src.utils import get_datasets_and_loaders

def train(model_num, batch_size, adam_lr, adam_b1, adam_b2, adam_wtdecay, epochs):
    print(f'model_num: {model_num}, batch_size: {batch_size}, adam_lr: {adam_lr}, adam_b1: {adam_b1}, adam_b2: {adam_b2}, adam_wtdecay: {adam_wtdecay}, epochs: {epochs}')
    model, image_size = prepare_model(model_num)
    
    _, dataloaders = get_datasets_and_loaders(DATA_DIR, 'train', 'val', 
                                              image_size=image_size)
    
    optimizer = Adam(model.parameters(), adam_lr, (adam_b1, adam_b2), adam_wtdecay)
    criterion = CrossEntropyLoss()
    
    trainable = Trainable(model, criterion, optimizer)
    val_acc = trainable.train(dataloaders, epochs)
    return val_acc
    


# In[3]:


# # TEST CODE
# train(model, 4, 1e-3, 0.9, 0.999, 0, 50)


# In[ ]:


from GPyOpt.methods import BayesianOptimization

problem = BayesianOptimization(
    f=f,
    domain=domain
)
problem.run_optimization(max_iter=10)
problem.plot_acquisition()


# In[ ]:


problem.plot_convergence


# # tried given BO code, out of memory issues. chased this around for a long time
# 
# 
# # found pretrainedmodels package, want to use inceptionresnetv2
# 
# 
# # right now trying to implement in PyTorch
# 
# 

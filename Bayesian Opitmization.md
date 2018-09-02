

```python
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
```


```python
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

iteration = 0
def train(model_num, batch_size, adam_lr, adam_b1, adam_b2, adam_wtdecay, epochs):
    global iteration
    iteration += 1
    print("ITERATION", iteration)
    print(f'model_num: {model_num}, batch_size: {batch_size}, epochs: {epochs}')
    print(f'adam_lr: {adam_lr}, adam_b1: {adam_b1}, adam_b2: {adam_b2}, adam_wtdecay: {adam_wtdecay}')

    model, image_size = prepare_model(model_num)
    
    _, dataloaders = get_datasets_and_loaders(DATA_DIR, 'train', 'val', 
                                              image_size=image_size)
    
    optimizer = Adam(model.parameters(), adam_lr, (adam_b1, adam_b2), adam_wtdecay)
    criterion = CrossEntropyLoss()
    
    trainable = Trainable(model, criterion, optimizer)
    val_acc = trainable.train(dataloaders, epochs)
    return val_acc
    
```


```python
# # TEST CODE
# train(model, 4, 1e-3, 0.9, 0.999, 0, 50)
```


```python
from GPyOpt.methods import BayesianOptimization

problem = BayesianOptimization(
    f=f,
    domain=domain
)
problem.run_optimization(max_iter=10)
problem.plot_acquisition()
```

    ITERATION 1
    model_num: 1, batch_size: 1, epochs: 20
    adam_lr: 0.001427697690696603, adam_b1: 0.9242185691433029, adam_b2: 0.9704897877665405, adam_wtdecay: 0.4858382667441655


    Train:   0%|          | 0/988 [00:00<?, ?images/s]

    Epoch 1/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.9218, Acc: 0.6437


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.2992, Acc: 0.6038
    
    Epoch 2/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.7840, Acc: 0.6812


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.1745, Acc: 0.6604
    
    Epoch 3/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.7189, Acc: 0.7176


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 5.7423, Acc: 0.6368
    
    Epoch 4/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.7062, Acc: 0.7379


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.0778, Acc: 0.6085
    
    Epoch 5/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6788, Acc: 0.7510


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.9380, Acc: 0.6321
    
    Epoch 6/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6334, Acc: 0.7510


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.8784, Acc: 0.7170
    
    Epoch 7/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6634, Acc: 0.7551


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.8127, Acc: 0.7028
    
    Epoch 8/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6224, Acc: 0.7723


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6787, Acc: 0.6981
    
    Epoch 9/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6078, Acc: 0.7733


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.7488, Acc: 0.6981
    
    Epoch 10/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6194, Acc: 0.7834


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.9128, Acc: 0.6415
    
    Epoch 11/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5882, Acc: 0.7702


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.7833, Acc: 0.6651
    
    Epoch 12/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6192, Acc: 0.7632


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6787, Acc: 0.7358
    
    Epoch 13/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6078, Acc: 0.7814


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.7277, Acc: 0.7264
    
    Epoch 14/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5856, Acc: 0.7733


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.7223, Acc: 0.6698
    
    Epoch 15/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5462, Acc: 0.7996


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6386, Acc: 0.7217
    
    Epoch 16/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5271, Acc: 0.7986


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6147, Acc: 0.7689
    
    Epoch 17/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5525, Acc: 0.8006


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6634, Acc: 0.7217
    
    Epoch 18/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5329, Acc: 0.8047


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.8097, Acc: 0.6934
    
    Epoch 19/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.4639, Acc: 0.8310


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.7169, Acc: 0.7406
    
    Epoch 20/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5241, Acc: 0.8097


                                                              

    Val Loss: 0.9772, Acc: 0.6745
    
    Training completed in 12m 48s
    Best validation accuracy: 0.7689
    Associated train accuracy: 0.7986
    Associated train loss: 0.5271
    ITERATION 2
    model_num: 3, batch_size: 48, epochs: 20
    adam_lr: 0.004531646808602268, adam_b1: 0.9759809061088922, adam_b2: 0.894015438386755, adam_wtdecay: 0.405696165546497


    Train:   0%|          | 0/988 [00:00<?, ?images/s]

    Epoch 1/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.9783, Acc: 0.6174


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.9200, Acc: 0.6840
    
    Epoch 2/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.7550, Acc: 0.7176


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.7622, Acc: 0.7264
    
    Epoch 3/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.7217, Acc: 0.7318


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.7162, Acc: 0.6792
    
    Epoch 4/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6747, Acc: 0.7399


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.1444, Acc: 0.7783
    
    Epoch 5/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6135, Acc: 0.7713


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.7801, Acc: 0.7453
    
    Epoch 6/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6248, Acc: 0.7682


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6769, Acc: 0.7830
    
    Epoch 7/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5854, Acc: 0.7794


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.8939, Acc: 0.7311
    
    Epoch 8/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5539, Acc: 0.7834


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6953, Acc: 0.7972
    
    Epoch 9/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5842, Acc: 0.7682


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.5679, Acc: 0.7783
    
    Epoch 10/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5528, Acc: 0.7996


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6703, Acc: 0.7217
    
    Epoch 11/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5171, Acc: 0.7996


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.5841, Acc: 0.7453
    
    Epoch 12/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5349, Acc: 0.8057


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.5453, Acc: 0.7736
    
    Epoch 13/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5108, Acc: 0.8158


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.0908, Acc: 0.6792
    
    Epoch 14/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5011, Acc: 0.8128


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.7055, Acc: 0.7358
    
    Epoch 15/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.4968, Acc: 0.8067


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.5942, Acc: 0.7736
    
    Epoch 16/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.4797, Acc: 0.8128


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6712, Acc: 0.7264
    
    Epoch 17/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5344, Acc: 0.7905


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.8095, Acc: 0.7123
    
    Epoch 18/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5080, Acc: 0.8036


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.6701, Acc: 0.7406
    
    Epoch 19/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.4452, Acc: 0.8279


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.5374, Acc: 0.7736
    
    Epoch 20/20
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.4659, Acc: 0.8401


                                                              

    Val Loss: 0.5668, Acc: 0.7830
    
    Training completed in 17m 56s
    Best validation accuracy: 0.7972
    Associated train accuracy: 0.7834
    Associated train loss: 0.5539
    ITERATION 3
    model_num: 3, batch_size: 4, epochs: 30
    adam_lr: 0.00327263357324869, adam_b1: 0.849819351502919, adam_b2: 0.9775856470467268, adam_wtdecay: 0.8052200030010476


    Train:   0%|          | 0/988 [00:00<?, ?images/s]

    Epoch 1/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.9877, Acc: 0.6113


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.7243, Acc: 0.6557
    
    Epoch 2/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.8361, Acc: 0.6802


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.5431, Acc: 0.6509
    
    Epoch 3/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.8150, Acc: 0.6771


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 2.0840, Acc: 0.6698
    
    Epoch 4/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.7804, Acc: 0.7186


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 19.6260, Acc: 0.7028
    
    Epoch 5/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.7222, Acc: 0.7318


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 2.3894, Acc: 0.7311
    
    Epoch 6/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6889, Acc: 0.7348


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 27.0650, Acc: 0.6887
    
    Epoch 7/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6577, Acc: 0.7429


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 2.3055, Acc: 0.7075
    
    Epoch 8/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6455, Acc: 0.7632


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.4365, Acc: 0.7075
    
    Epoch 9/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5931, Acc: 0.7733


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 2.9324, Acc: 0.7311
    
    Epoch 10/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5979, Acc: 0.7753


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 4.9416, Acc: 0.7217
    
    Epoch 11/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5744, Acc: 0.7824


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 2.7259, Acc: 0.7028
    
    Epoch 12/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.6025, Acc: 0.7713


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 3.7452, Acc: 0.6840
    
    Epoch 13/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5968, Acc: 0.7763


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.5292, Acc: 0.7217
    
    Epoch 14/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5548, Acc: 0.7895


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.5145, Acc: 0.7358
    
    Epoch 15/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5556, Acc: 0.7986


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 2.5361, Acc: 0.7264
    
    Epoch 16/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5437, Acc: 0.8016


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.1777, Acc: 0.7311
    
    Epoch 17/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5874, Acc: 0.7885


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 2.8956, Acc: 0.7217
    
    Epoch 18/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5281, Acc: 0.8067


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.6135, Acc: 0.7123
    
    Epoch 19/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5483, Acc: 0.8057


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.9884, Acc: 0.7264
    
    Epoch 20/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5345, Acc: 0.8047


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 2.1412, Acc: 0.7075
    
    Epoch 21/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5365, Acc: 0.7854


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.3685, Acc: 0.7453
    
    Epoch 22/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5055, Acc: 0.8087


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 3.9304, Acc: 0.7358
    
    Epoch 23/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.4906, Acc: 0.8239


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 0.9914, Acc: 0.7358
    
    Epoch 24/30
    ----------


    Val:   0%|          | 0/212 [00:00<?, ?images/s]            

    Train Loss: 0.5199, Acc: 0.8117


    Train:   0%|          | 0/988 [00:00<?, ?images/s]        

    Val Loss: 1.3743, Acc: 0.7453
    
    Epoch 25/30
    ----------


    Train:  59%|█████▉    | 584/988 [00:30<00:20, 19.84images/s]Process Process-516:
    Process Process-513:
    Process Process-515:
    Process Process-514:
    Traceback (most recent call last):
    Traceback (most recent call last):
    Traceback (most recent call last):
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
    Traceback (most recent call last):
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/process.py", line 93, in run
        self._target(*self._args, **self._kwargs)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 106, in _worker_loop
        samples = collate_fn([dataset[i] for i in batch_indices])
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/process.py", line 93, in run
        self._target(*self._args, **self._kwargs)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/process.py", line 93, in run
        self._target(*self._args, **self._kwargs)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/process.py", line 93, in run
        self._target(*self._args, **self._kwargs)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 106, in <listcomp>
        samples = collate_fn([dataset[i] for i in batch_indices])
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 96, in _worker_loop
        r = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/datasets/folder.py", line 101, in __getitem__
        sample = self.loader(path)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 96, in _worker_loop
        r = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/queues.py", line 104, in get
        if not self._poll(timeout):
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/datasets/folder.py", line 147, in default_loader
        return pil_loader(path)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/queues.py", line 104, in get
        if not self._poll(timeout):
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/datasets/folder.py", line 130, in pil_loader
        return img.convert('RGB')
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 96, in _worker_loop
        r = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/connection.py", line 257, in poll
        return self._poll(timeout)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/connection.py", line 257, in poll
        return self._poll(timeout)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/PIL/Image.py", line 892, in convert
        self.load()
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/queues.py", line 104, in get
        if not self._poll(timeout):
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/connection.py", line 414, in _poll
        r = wait([self], timeout)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/PIL/ImageFile.py", line 235, in load
        n, err_code = decoder.decode(b)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/connection.py", line 414, in _poll
        r = wait([self], timeout)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/connection.py", line 257, in poll
        return self._poll(timeout)
    KeyboardInterrupt
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/connection.py", line 911, in wait
        ready = selector.select(timeout)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/connection.py", line 911, in wait
        ready = selector.select(timeout)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/connection.py", line 414, in _poll
        r = wait([self], timeout)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/selectors.py", line 376, in select
        fd_event_list = self._poll.poll(timeout)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/selectors.py", line 376, in select
        fd_event_list = self._poll.poll(timeout)
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/multiprocessing/connection.py", line 911, in wait
        ready = selector.select(timeout)
    KeyboardInterrupt
    KeyboardInterrupt
      File "/home/alwood/anaconda3/envs/estrous-ai/lib/python3.6/selectors.py", line 376, in select
        fd_event_list = self._poll.poll(timeout)
    KeyboardInterrupt
                                                                


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-4-8655deade285> in <module>()
          3 problem = BayesianOptimization(
          4     f=f,
    ----> 5     domain=domain
          6 )
          7 problem.run_optimization(max_iter=10)


    ~/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/GPyOpt/methods/bayesian_optimization.py in __init__(self, f, domain, constraints, cost_withGradients, model_type, X, Y, initial_design_numdata, initial_design_type, acquisition_type, normalize_Y, exact_feval, acquisition_optimizer_type, model_update_interval, evaluator_type, batch_size, num_cores, verbosity, verbosity_model, maximize, de_duplication, **kwargs)
        116         self.initial_design_type  = initial_design_type
        117         self.initial_design_numdata = initial_design_numdata
    --> 118         self._init_design_chooser()
        119 
        120         # --- CHOOSE the model type. If an instance of a GPyOpt model is passed (possibly user defined), it is used.


    ~/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/GPyOpt/methods/bayesian_optimization.py in _init_design_chooser(self)
        191         if self.X is None:
        192             self.X = initial_design(self.initial_design_type, self.space, self.initial_design_numdata)
    --> 193             self.Y, _ = self.objective.evaluate(self.X)
        194         # Case 2
        195         elif self.X is not None and self.Y is None:


    ~/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/GPyOpt/core/task/objective.py in evaluate(self, x)
         48 
         49         if self.n_procs == 1:
    ---> 50             f_evals, cost_evals = self._eval_func(x)
         51         else:
         52             try:


    ~/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/GPyOpt/core/task/objective.py in _eval_func(self, x)
         72         for i in range(x.shape[0]):
         73             st_time    = time.time()
    ---> 74             rlt = self.func(np.atleast_2d(x[i]))
         75             f_evals     = np.vstack([f_evals,rlt])
         76             cost_evals += [time.time()-st_time]


    <ipython-input-2-421a18cf4ad1> in f(x)
          8         adam_b2=float(x[:,4]),
          9         adam_wtdecay=float(x[:,5]),
    ---> 10         epochs=int(x[:,6])
         11     )
         12 


    <ipython-input-2-421a18cf4ad1> in train(model_num, batch_size, adam_lr, adam_b1, adam_b2, adam_wtdecay, epochs)
         58 
         59     trainable = Trainable(model, criterion, optimizer)
    ---> 60     val_acc = trainable.train(dataloaders, epochs)
         61     return val_acc
         62 


    ~/Development/Estrous-AI/src/trainable.py in train(self, dataloaders, num_epochs, results_filepath)
        150                             outputs = model(inputs)
        151                             _, predictions = torch.max(outputs, 1)
    --> 152                             loss = criterion(outputs, labels)
        153 
        154                             # backprop and update weights during train


    ~/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torch/tensor.py in backward(self, gradient, retain_graph, create_graph)
         91                 products. Defaults to ``False``.
         92         """
    ---> 93         torch.autograd.backward(self, gradient, retain_graph, create_graph)
         94 
         95     def register_hook(self, hook):


    ~/anaconda3/envs/estrous-ai/lib/python3.6/site-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables)
         88     Variable._execution_engine.run_backward(
         89         tensors, grad_tensors, retain_graph, create_graph,
    ---> 90         allow_unreachable=True)  # allow_unreachable flag
         91 
         92 


    KeyboardInterrupt: 



```python
problem.plot_convergence
```

# Run again, but with more iterations

```python
%load_ext autoreload
%autoreload 2

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder


from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from IPython.display import clear_output
import os

from src.model import *
from src.utils import *
from src.train import *

models_path = './models'
if not os.path.exists(models_path):
    os.makedirs(models_path)
```

Perform data augmentation and take a look at the dimensions of the data


```python
#data augmentation and preprocessing
transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.CenterCrop(256), transforms.ToTensor()])

alz_dataset = ImageFolder("./Alzheimers-ADNI/train/", transform=transform)
train_dataset, val_dataset = torch.utils.data.random_split(alz_dataset, ( round(len(alz_dataset)*0.9), round(len(alz_dataset)*0.1) ))


#Datasets we are going to be working with
datasets = {'train': train_dataset, 'val': val_dataset}
test_dataset = ImageFolder("./Alzheimers-ADNI/test/", transform=transform)

#Output some infos about the data and datasets
print("Training data lenght: ", len(datasets["train"]))
print("Test data lenght: ", len(datasets["train"]))
print("Dimensions of an image: ", datasets["train"][0][0].shape)
```

    Training data lenght:  993
    Test data lenght:  993
    Dimensions of an image:  torch.Size([3, 256, 256])
    

# Take a look at the data, classes
There are 5 classes of patients:
- Alzheimer's Disease
- Early Mild Cognitive Impairment
- Late Mild Cognitive Impairment
- Mild Cognitive Impairment
- Control Normals


```python
show_examples(datasets["train"])
```


![png](output_files/output_4_0.png)


# Preparing some parameters for the training and evaluation process
We have to set following things in a configuration file, so at a later point we can use raytune for Hyperparameter search:


```python
config = {"batch_size":16, "lr":1e-3}
```


```python
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=config["batch_size"], shuffle=True, num_workers=8, drop_last=True) for x in ['train', 'val']}
test_loader =torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=8, drop_last=False)
```

# The model
The approach is simple: First a Convolutional autoencoder is trained to encode and decode the images, than the encoder part which gained a good performance in extracting the features from MRI's is saved and transferred to be used in combination with
2 linear layers which will do the classification. In the second step only these two last layers will be trained.


```python
autoencoder = Autoencoder()
```


```python
classifier = AlzheimerClassifier()
```

#### Move everything to the GPU if possible 


```python
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        autoencoder = nn.DataParallel(autoencoder)
        classifier = nn.DataParallel(classifier)
autoencoder.to(device)
classifier.to(device)

clear_output()
```

#### Optimizers
At this point only Adam will be used, no fancy stuff


```python
#Autoencoder
optim_a = torch.optim.Adam(autoencoder.parameters(), lr = 1e-3)
#Classifier
optim_b = torch.optim.Adam(classifier.parameters(), lr = 1e-3)
```

#### Loss functions
For the autoencoder the usual MSE Loss will be used, for the classifier (clearly) the Cross Entropy


```python
#Autoencoder
loss_a = nn.MSELoss()
#Classifier
loss_b = nn.CrossEntropyLoss()
```

##### Use a learning rate scheduler


```python
lambda_a = lambda epoch: 0.65 ** epoch
scheduler_a = torch.optim.lr_scheduler.LambdaLR(optim_a, lr_lambda=lambda_a)
scheduler_b =  torch.optim.lr_scheduler.LambdaLR(optim_b, lr_lambda=lambda_a)
```

### Lets train the Autoencoder
Of course we have to keep track of the whole progress using Tensorboard, and use self-defined callbacks like early stopping and occasional saving if the model performs well.


```python
#train_autoencoder(autoencoder, dataloaders,loss_a, optim_a, scheduler_a, device,  num_epochs=25, early_stop=5)
```

Test the autoencoder


```python
show_ae_results(test_loader, autoencoder)
```


![png](output_files/output_22_0.png)


#### Train the Classifier


```python
autoencoder.load_state_dict(torch.load(os.path.join(models_path, "autoencoder.pth")))
```




    <All keys matched successfully>




```python
torch.save(autoencoder.encoder.state_dict(), os.path.join(models_path, "encoder.pth"))
```


```python
classifier.encoder.load_state_dict(torch.load(os.path.join(models_path, "encoder.pth")))
```




    <All keys matched successfully>




```python
#train_classifier(classifier, dataloaders,loss_b, optim_b, scheduler_b, device,  num_epochs=25, early_stop=5)
```

    Epoch 24/24
    

    100%|██████████████████████████████████████████████████████████████████████████████████| 25/25 [17:57<00:00, 43.10s/it]

    train loss: 1.4528 Accurracy: 0.4502
    Training complete in 17m 58s
    Best validation accuracy: 0.3909
    

    
    




    AlzheimerClassifier(
      (encoder): Sequential(
        (0): Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU()
        (2): AdaptiveMaxPool2d(output_size=(200, 200))
        (3): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): ReLU()
        (6): AdaptiveMaxPool2d(output_size=(128, 128))
        (7): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (9): ReLU()
        (10): AdaptiveMaxPool2d(output_size=(64, 64))
        (11): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (12): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (13): ReLU()
        (14): AdaptiveMaxPool2d(output_size=(32, 32))
        (15): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (16): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): ReLU()
        (18): AdaptiveMaxPool2d(output_size=(16, 16))
        (19): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (20): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (21): ReLU()
        (22): AdaptiveMaxPool2d(output_size=(8, 8))
        (23): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (classifier): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Dropout(p=0.1, inplace=False)
        (2): Linear(in_features=16384, out_features=5000, bias=True)
        (3): ReLU()
        (4): Dropout(p=0.1, inplace=False)
        (5): Linear(in_features=5000, out_features=5, bias=True)
        (6): Softmax(dim=None)
      )
    )




```python
#Testing the model
classifier.to("cpu")
classifier.load_state_dict(torch.load("./models/classifier.pth"))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
test_model(test_loader, classifier)
print("Acurracy: ", test_acurracy(test_loader, classifier)*100, "%")
```

    Acurracy:  43.58974358974359 %
    


![png](output_files/output_28_1.png)


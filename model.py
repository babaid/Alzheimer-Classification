from torch import nn
import matplotlib.pyplot as plt
import torch
from PIL import Image

class AlzheimerClassifierNet(nn.Module):
    def __init__(self, hparams=None):
        super(AlzheimerClassifierNet, self).__init__()
        self.hparams = hparams
        
        self.model = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=4, bias=None), nn.ReLU(), nn.BatchNorm2d(32),#208x176
                                   nn.MaxPool2d(kernel_size=4, stride=2),#103x87
                                   nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=None), nn.ReLU(),nn.BatchNorm2d(64),
                                   nn.MaxPool2d(kernel_size=3, stride=2),#51x43
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=None), nn.ReLU(), nn.BatchNorm2d(128),
                                   nn.MaxPool2d(kernel_size=2, stride=2), #25x21
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=None), nn.ReLU(), nn.BatchNorm2d(256),
                                   nn.MaxPool2d(kernel_size=2, stride=2), #12x10
                                   nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=None), nn.ReLU(), nn.BatchNorm2d(512),
                                   nn.MaxPool2d(kernel_size=2, stride=2), #6x5
                                   nn.Flatten(),
                                   nn.Dropout(0.1),
                                   nn.Linear(512*6*5, 2000), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2000, 4)
                                  )
        
    def forward(self, x):
        return self.model(x)
    
    
"""
A Convolutional made out of Adaptive layers, for lazy testing
"""
class AdaptiveAlzheimerClassifierNet(nn.Module):
    def __init__(self):
        super(AdaptiveAlzheimerClassifierNet, self).__init__()
        
        
        self.model = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=4, bias=None), nn.ELU(), nn.BatchNorm2d(32),#208x176
                                   nn.AdaptiveMaxPool2d((150, 150)),
                                   nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=None), nn.ELU(),nn.BatchNorm2d(64),
                                   nn.AdaptiveMaxPool2d((100, 100)),
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=None), nn.ELU(), nn.BatchNorm2d(128),
                                   nn.AdaptiveMaxPool2d((50, 50)), 
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=None), nn.ELU(), nn.BatchNorm2d(256),
                                   nn.AdaptiveMaxPool2d((25, 25)), 
                                   nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=None), nn.ELU(), nn.BatchNorm2d(512),
                                   nn.AdaptiveMaxPool2d((15, 15)),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=None), nn.ELU(), nn.BatchNorm2d(512),
                                   nn.AdaptiveMaxPool2d((10, 10)),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=None), nn.ELU(), nn.BatchNorm2d(512),
                                   nn.AdaptiveMaxPool2d((5, 5)),
                                
                                   nn.Flatten(),
                                   nn.Dropout(0.2),
                                   nn.Linear(512*5*5, 2000), nn.ELU(), nn.Dropout(0.1), nn.Linear(2000, 4)
                                  )
        
    def forward(self, x):
        return self.model(x)

    
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.skip = nn.Identity()
        self.model = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride),
                                   nn.BatchNorm2d(channels),
                                   nn.ELU(), nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride))
    def forward(self, x):
        return self.model(x)+self.skip(x)
    
    
class AdaptiveResidualClassifier(nn.Module):
    #Somekind of resnet, almost 50
    def __init__(self):
        super(AdaptiveResidualClassifier, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=1), nn.ELU(), 
                                   nn.AdaptiveMaxPool2d((150, 150)),
                                   ResidualBlock(64), nn.ELU(),nn.BatchNorm2d(64),
                                   ResidualBlock(64), nn.ELU(),nn.BatchNorm2d(64),
                                   ResidualBlock(64), nn.ELU(),nn.BatchNorm2d(64),
                                   nn.AdaptiveMaxPool2d((100, 100)),#51x43
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ELU(), ,nn.BatchNorm2d(128),
                                   nn.AdaptiveMaxPool2d((50, 50)),
                                   ResidualBlock(128), nn.ELU(),nn.BatchNorm2d(128),
                                   ResidualBlock(128), nn.ELU(),nn.BatchNorm2d(128),
                                   ResidualBlock(128), nn.ELU(),nn.BatchNorm2d(128),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ELU(),,nn.BatchNorm2d(128),
                                   nn.AdaptiveMaxPool2d((25, 25)),
                                   ResidualBlock(256), nn.ELU(),nn.BatchNorm2d(256),
                                   ResidualBlock(256), nn.ELU(),nn.BatchNorm2d(256),
                                   ResidualBlock(256), nn.ELU(),nn.BatchNorm2d(256),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ELU(),,nn.BatchNorm2d(512),
                                   nn.AdaptiveMaxPool2d((5, 5)),
                                   ResidualBlock(512), nn.ELU(),nn.BatchNorm2d(512),
                                   ResidualBlock(512), nn.ELU(),nn.BatchNorm2d(512),
                                   ResidualBlock(512), nn.ELU(),nn.BatchNorm2d(512),
                                   nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nn.ELU(), ,nn.BatchNorm2d(256),
                                   nn.Flatten(),
                                   nn.Linear(256*5*5, 2000),nn.ELU(), nn.Linear(2000, 4)
                                   
                                  )
        
    def forward(self, x):
        return self.model(x)

"""
Tests a model, given a dataloader.
"""  
def test_model(dataloader, model):
    model.eval()
    img, label = next(iter(dataloader))["image"], next(iter(dataloader))["label"]
    fig, axs = plt.subplots(3, 1+img.size(0),figsize=(30, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 0.1, wspace=0.1)
    fig.suptitle("Some Examples")
    pred = model(img.cuda())
    softmax = nn.Softmax(pred)
    preds = torch.argmax(pred, dim = 1)
    classes = sorted(["Non demented", "Very Mildly Demented", "Mildly Demented", "ModerateDemented"])
    
    axs[0][0].text(0.5, 0.5, "Image", fontsize = 14, fontweight="bold")
    axs[1][0].text(0.5, 0.5, "Predicted Label", fontsize = 14,  fontweight="bold")
    axs[2][0].text(0.5, 0.5, "Ground Truth", fontsize = 14,  fontweight="bold")
    
    for i in range(1+img.size(0)):
        for j in range(3):
            axs[j][i].axis("off")
    for i, data in enumerate(img):
         
        axs[0][i+1].imshow(data.squeeze(0).numpy())
        
        axs[1][i+1].text(0.5, 0.5, classes[preds[i].item()],horizontalalignment='center', verticalalignment='center', fontsize=13)
        axs[2][i+1].text(0.5, 0.5, classes[label[i].item()],horizontalalignment='center', verticalalignment='center', fontsize=13)
        
def test_acurracy(loader, model):
    correct = 0
    all = 0
    model.eval()
    for i, data in enumerate(loader):
        img, label = data["image"].cuda(), data["label"].cuda()
        pred = torch.argmax(model(img), dim=1)
        for j in range(label.size(0)):
            all+=1
            
            if pred[j].item() == label[j].item():
                correct += 1
    return correct/all
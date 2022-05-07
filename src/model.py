from torch import nn
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

class Autoencoder(nn.Module):
    """
    Aautoencoder for transfer learning
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1, bias=None),nn.ReLU(), nn.AdaptiveMaxPool2d((200, 200)), nn.BatchNorm2d(8),
                                        nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((128, 128)), nn.BatchNorm2d(16),
                                        nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((64, 64)), nn.BatchNorm2d(32),
                                        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((32, 32)), nn.BatchNorm2d(64),
                                        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((16, 16)), nn.BatchNorm2d(128),
                                        nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((8, 8)), nn.BatchNorm2d(256)
                                         )
        self.decoder = nn.Sequential(nn.ConvTranspose2d(256,128, kernel_size=2, stride=2, padding=0, bias=None), nn.ELU(), nn.BatchNorm2d(128),#16
                                     nn.ConvTranspose2d(128,64, kernel_size=2, stride=2, padding=0, bias=None), nn.ELU(), nn.BatchNorm2d(64),#32
                                     nn.ConvTranspose2d(64,32, kernel_size=2, stride=2, padding=0, bias=None), nn.ELU(), nn.BatchNorm2d(32),#64
                                     nn.ConvTranspose2d(32,16, kernel_size=2, stride=2, padding=0, bias=None), nn.ELU(), nn.BatchNorm2d(16),#128   
                                     nn.ConvTranspose2d(16,3, kernel_size=2, stride=2, padding=0, bias=None), nn.Sigmoid())
        
        

    def forward(self, x):        
        return self.decoder(self.encoder(x))
        
        
        
        
        
        
class AlzheimerClassifier(nn.Module):
    def __init__(self, hparams=None):
        super(AlzheimerClassifier, self).__init__()
        
        self.hparams = hparams
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1, bias=None),nn.ReLU(), nn.AdaptiveMaxPool2d((200, 200)), nn.BatchNorm2d(8),
                                        nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((128, 128)), nn.BatchNorm2d(16),
                                        nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((64, 64)), nn.BatchNorm2d(32),
                                        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((32, 32)), nn.BatchNorm2d(64),
                                        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((16, 16)), nn.BatchNorm2d(128),
                                        nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=None), nn.ReLU(), nn.AdaptiveMaxPool2d((8, 8)), nn.BatchNorm2d(256)
                                         )

       
        
        self.classifier=nn.Sequential(nn.Flatten(), nn.Dropout(0.1),nn.Linear(256*8*8, 5000), nn.ReLU(), nn.Dropout(0.1), nn.Linear(5000, 5), nn.Softmax())
        
    def forward(self, x):

        with torch.no_grad():
            x = self.encoder(x)
            
        return self.classifier(x)
    
    
"""
A Convolutional made out of Adaptive layers, for lazy testing
"""
class AdaptiveAlzheimerClassifierNet(nn.Module):
    def __init__(self):
        super(AdaptiveAlzheimerClassifierNet, self).__init__()
        
        
        self.model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=4, bias=None), nn.ELU(), nn.BatchNorm2d(32),#208x176
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
                                   nn.Linear(512*5*5, 2000), nn.ELU(), nn.Dropout(0.1), nn.Linear(2000, 5)
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
        self.model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1), nn.ELU(), 
                                   nn.AdaptiveMaxPool2d((150, 150)),
                                   ResidualBlock(64), nn.ELU(),nn.BatchNorm2d(64),
                                   ResidualBlock(64), nn.ELU(),nn.BatchNorm2d(64),
                                   ResidualBlock(64), nn.ELU(),nn.BatchNorm2d(64),
                                   nn.AdaptiveMaxPool2d((100, 100)),#51x43
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ELU(), nn.BatchNorm2d(128),
                                   nn.AdaptiveMaxPool2d((50, 50)),
                                   ResidualBlock(128), nn.ELU(),nn.BatchNorm2d(128),
                                   ResidualBlock(128), nn.ELU(),nn.BatchNorm2d(128),
                                   ResidualBlock(128), nn.ELU(),nn.BatchNorm2d(128),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ELU(),nn.BatchNorm2d(256),
                                   nn.AdaptiveMaxPool2d((25, 25)),
                                   ResidualBlock(256), nn.ELU(),nn.BatchNorm2d(256),
                                   ResidualBlock(256), nn.ELU(),nn.BatchNorm2d(256),
                                   ResidualBlock(256), nn.ELU(),nn.BatchNorm2d(256),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ELU(),nn.BatchNorm2d(512),
                                   nn.AdaptiveMaxPool2d((5, 5)),
                                   ResidualBlock(512), nn.ELU(),nn.BatchNorm2d(512),
                                   ResidualBlock(512), nn.ELU(),nn.BatchNorm2d(512),
                                   ResidualBlock(512), nn.ELU(),nn.BatchNorm2d(512),
                                   nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nn.ELU(),nn.BatchNorm2d(256),
                                   nn.Flatten(),
                                   nn.Linear(256*5*5, 2000),nn.ELU(), nn.Linear(2000, 5)
                                   
                                  )
        
    def forward(self, x):
        return self.model(x)


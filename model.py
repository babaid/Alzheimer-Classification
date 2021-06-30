from torch import nn

class AlzheimerClassifierNet(nn.Module):
    def __init__(self, hparams=None):
        super(AlzheimerClassifierNet, self).__init__()
        self.hparams = hparams
        
        self.model = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=4, bias=None), nn.ReLU(True), nn.BatchNorm2d(32),#208x176
                                   nn.MaxPool2d(kernel_size=4, stride=2),#103x87
                                   nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=None), nn.ReLU(True), nn.BatchNorm2d(64),
                                   nn.MaxPool2d(kernel_size=3, stride=2),#51x43
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=None), nn.ReLU(True), nn.BatchNorm2d(128),
                                   nn.MaxPool2d(kernel_size=2, stride=2), #25x21
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=None), nn.ReLU(True), nn.BatchNorm2d(256),
                                   nn.MaxPool2d(kernel_size=2, stride=2), #12x10
                                   nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=None), nn.ReLU(True), nn.BatchNorm2d(512),
                                   nn.MaxPool2d(kernel_size=2, stride=2), #6x5
                                   nn.Flatten(),
                                   nn.Dropout(0.1),
                                   nn.Linear(512*6*5, 2000), nn.ReLU(True), nn.Dropout(0.1), nn.Linear(2000, 4)
                                  )
        
    def forward(self, x):
        return self.model(x)
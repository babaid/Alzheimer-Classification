import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
import os
from PIL import Image

class AlzheimerDataset(Dataset):
    """Alzheimers classification dataset."""
    def __init__(self, root_dir,transform=transforms.ToTensor()):
        self.root_dir= root_dir
        self.transform = transform
        self.classes, self.class_id = self._find_classes(self.root_dir)
        self.images, self.labels = self._make_dataset(self.root_dir, self.class_id)
        
    @staticmethod
    def _find_classes(directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        classes_id = {classes[i]:i for i in range(len(classes))}
        return classes, classes_id
    
    @staticmethod
    def _make_dataset(directory, class_id):
        images, labels = [], []
        for target in sorted(class_id.keys()):
            label = class_id[target]
            target_dir = os.path.join(directory, target)
            for root, _, filenames in sorted(os.walk(target_dir)):
                for filename in sorted(filenames):
                    images.append(os.path.join(target_dir, filename))
                    one_hot_label = torch.zeros(len(class_id), dtype=torch.long)
                    one_hot_label[label] = 1
                    labels.append(label)
        assert len(images)==len(labels)
        return images, labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        if self.transform:
            image = Image.open(self.images[index])
            return {"image":self.transform(image), "label":self.labels[index]}
        else:
            return {"image":self.transform(image), "label":self.labels[index]}
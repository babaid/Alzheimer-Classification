# Alzheimer-Classification
Using pytorch to classify different stages of Alzheimer

The Dataset has it's source on Kaggle.com (https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images).
I've tried different convolutional network architectures, experimenting with adaptive layers, skip connections, and reasonable data augmentations.
I've decided to use RMSprop as optimizer instead of other alternatives, it proved itself to be the fastest in terms of convergence for this task.

As far as accuracy goes percentage of around 75 percent can be achieved on weaker models, the literaly accuracy of a resnet-50 is around 85 percent. Higher accuracy could be achieved if the dataset would be larger.
I also have to say that using only MRI pictures isnt necessary the best way to approach the problem. Other imaging methods like fMRI or MEG show viable alternatives with better resolutions.

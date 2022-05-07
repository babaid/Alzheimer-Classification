import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch

def show_examples(dataset, num_images=5, classes=sorted(["AD", "CN", "EMCI", "LMCI", "MCI" ])):
    fig, ax = plt.subplots(2, num_images, figsize=(20, 8))
    for i in range(num_images):
        index = np.random.randint(0, len(dataset))
        ax[0][i].imshow(transforms.ToPILImage()(dataset[index][0]))
        ax[1][i].text(0.3, 0.5, classes[dataset[index][1]])

        ax[0][i].axis("off")
        ax[1][i].axis("off")
        
#Overfits the model to one image
def overfit_one(image, model, loss_fn, optim):
    model.train()
    optim.zero_grad()
    img, label = image[0].cuda(), image[1].cuda()
    pred = model(img)
    loss = loss_fn(pred, label)
    loss.backward()
    optim.step()
   
    return loss.item()

def train_tmp(dataloader, model, loss_fn, optim):
    model.train()
    run_loss = 0.0
    for i, data in enumerate(dataloader):
        optim.zero_grad()
        image, label = data[0].to("cuda"), data[1].to("cuda")
        pred = model(image)
        loss = loss_fn(pred, label)
        loss.backward()
        optim.step()
        run_loss += loss.detach().item()
       
    return run_loss/len(dataloader)


"""
Tests a model, given a dataloader.
"""  
def test_model(dataloader, model):
    model.eval()
    img, label = next(iter(dataloader))[0], next(iter(dataloader))[1]
    fig, axs = plt.subplots(3, 1+img.size(0),figsize=(30, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 0.1, wspace=0.1)
    fig.suptitle("Some Examples")
    pred = model(img)
    softmax = nn.Softmax(pred)
    preds = torch.argmax(pred, dim = 1)
    #classes = sorted(["Non demented", "Very Mildly Demented", "Mildly Demented", "ModerateDemented"])
    classes = sorted(["AD", "CN", "EMCI", "LMCI", "MCI" ])
    axs[0][0].text(0.5, 0.5, "Image", fontsize = 14, fontweight="bold")
    axs[1][0].text(0.5, 0.5, "Predicted Label", fontsize = 14,  fontweight="bold")
    axs[2][0].text(0.5, 0.5, "Ground Truth", fontsize = 14,  fontweight="bold")
    
    for i in range(1+img.size(0)):
        for j in range(3):
            axs[j][i].axis("off")
    for i, data in enumerate(img):
         
        axs[0][i+1].imshow(transforms.ToPILImage()(data))
        
        axs[1][i+1].text(0.5, 0.5, classes[preds[i].item()],horizontalalignment='center', verticalalignment='center', fontsize=13)
        axs[2][i+1].text(0.5, 0.5, classes[label[i].item()],horizontalalignment='center', verticalalignment='center', fontsize=13)
        
def test_acurracy(loader, model):
    correct = 0
    all = 0
    model.eval()
    for i, data in enumerate(loader):
        img, label = data[0], data[1]
        pred = torch.argmax(model(img), dim=1)
        for j in range(label.size(0)):
            all+=1
            
            if pred[j].item() == label[j].item():
                correct += 1
    return correct/all

def show_ae_results(dataloader, model):
    model.eval()
    model.to("cpu")
    img, label = next(iter(dataloader))[0], next(iter(dataloader))[1]
    
    fig, axs = plt.subplots(2, 1+img.size(0),figsize=(30, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 0.1, wspace=0.1)
    fig.suptitle("Some Examples")
    
    pred = model(img)
    
    
    classes = sorted(["AD", "CN", "EMCI", "LMCI", "MCI" ])
    
    axs[0][0].text(0.5, 0.5, "Image", fontsize = 14, fontweight="bold")
    axs[1][0].text(0.5, 0.5, "Reconstructed Image", fontsize = 14,  fontweight="bold")
    
    for i in range(1+img.size(0)):
        for j in range(2):
            axs[j][i].axis("off")
            
    for i, data in enumerate(img):

        axs[0][i+1].imshow(transforms.ToPILImage()(data))
        axs[1][i+1].imshow(transforms.ToPILImage()(pred[i]))
        
    
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from tqdm import tqdm
from IPython.display import clear_output
from torch.utils.tensorboard import SummaryWriter
import sys

def train_autoencoder(model, dataloaders, criterion, optimizer, scheduler, device,  num_epochs=25, early_stop=5):
    best_loss = sys.maxsize
    writer = SummaryWriter()
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in  ['train', 'val']}

    since = time.time()

    #save the best model
    best_model = copy.deepcopy(model.state_dict())
    phases =  ['train', 'val'] 
    cntr = 0
    for epoch in tqdm(range(num_epochs)):

        #print epochs, clear the output some times
        clear_output(wait=True)
        print(f'Epoch {epoch}/{num_epochs-1}')
    
        
        for phase in phases:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0


            #iterate trough datal
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                #actual training
                with torch.set_grad_enabled(phase=='train'):

                    outputs = model(inputs)                
                    loss = criterion(outputs, inputs)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                #benchmarking performance of model
                running_loss+=loss.item()




            if phase=='train':
                scheduler.step()

            #benchmarking performance  of model
            epoch_loss = running_loss/dataset_sizes[phase]


            writer.add_scalar("autoencoder/loss/{phase}".format(phase=phase), epoch_loss, epoch)


            print(f'{phase} loss: {epoch_loss:.4f}')

            #if model performs better, save it
            if phase == 'val':

                #tune.report(mean_accuracy=epoch_acc)

                if epoch_loss < best_loss:
                    cntr = 0
                    best_loss = epoch_loss

                    best_model = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "./models/autoencoder.pth")
                else:
                    cntr += 1
            if cntr > early_stop:
                break
    time_elapsed = time.time()-since
    
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
    

    model.load_state_dict(best_model)
    writer.close()
    return model







def train_classifier(model, dataloaders, criterion, optimizer, scheduler, device,  num_epochs=25, early_stop=5):
    
    
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in  ['train', 'val']}
    since = time.time()
    writer = SummaryWriter()
    #save the best model
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0
    cntr = 0
    for epoch in tqdm(range(num_epochs)):

        #print epochs, clear the output some times
        clear_output(wait=True)
        print(f'Epoch {epoch}/{num_epochs-1}')

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #iterate trough datal
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                #actual training
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                
                
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                #benchmarking performance of model
                running_loss+=loss.item()*inputs.size(0)
                running_corrects += sum(preds == labels.data)



            if phase=='train':
                scheduler.step()

            #benchmarking performance  of model
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double()/dataset_sizes[phase] 
            
            writer.add_scalar("classifier/loss/{phase}".format(phase=phase), epoch_loss, epoch)
            writer.add_scalar("classifier/acc/{phase}".format(phase=phase), epoch_acc, epoch)

            print(f'{phase} loss: {epoch_loss:.4f} Accurracy: {epoch_acc:.4f}')

            #if model performs better, save it
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc   
                    best_model = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "./models/classifier.pth")
                    cntr = 0
                else:
                    cntr += 1
            if cntr>early_stop:
                break
                
    time_elapsed = time.time()-since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model)
    writer.close()
    
    return model

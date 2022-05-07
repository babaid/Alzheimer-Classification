from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from dataset import AudioDataset
from model import LinearClassifier
import ray

def load_data(data_dir="../data"):

    data_dir = "C:\\Users\\Dezs\u0151Babai\\Desktop\\ml-syntegon\\knives\\data"
    audio_dataset = AudioDataset(data_dir)
    train_dataset, val_dataset = torch.utils.data.random_split(audio_dataset, ( round(len(audio_dataset)*0.8), round(len(audio_dataset)*0.2) ))
    datasets = {'train': train_dataset, 'val': val_dataset}
    return datasets

def train_audio(config, checkpoint_dir="../checkpoints", data_dir=None):


    datasets = load_data()
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=config["batch_size"], shuffle=True, num_workers=8, drop_last=True) for x in ['train', 'val']}

    model = LinearClassifier(input_size=120000, l1=config["l1"], l2=config["l2"], l3=config["l3"])

    # @parallelize and use gpu if available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    
 
    for epoch in range(10):
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(dataloaders["train"], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["audio"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs.squeeze().to(torch.float32), labels.squeeze().to(torch.float32))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 20 == 19:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(dataloaders["val"], 0):
            with torch.no_grad():
                inputs, labels = data["audio"], data["label"]
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs.squeeze().to(torch.float32), labels.squeeze().to(torch.float32))
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

    print("Finished Training")

def test_accuracy(model, device="cpu"):
    
    dataset= AudioDataset("C:\\Users\\Dezs\u0151Babai\\Desktop\\ml-syntegon\\knives\\data")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data["audio"], data["label"]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze().to(torch.float32)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):

    ray.init(object_store_memory=10**9)
   
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 11)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 10)),
        "l3": tune.sample_from(lambda _: 2 ** np.random.randint(2, 10)),
        "lr": tune.loguniform(1e-4, 5e-1),
        "batch_size": tune.choice([4, 8, 16, 32])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(

        train_audio,
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = LinearClassifier(best_trial.config["l1"], best_trial.config["l2"], best_trial.config["l3"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
   
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
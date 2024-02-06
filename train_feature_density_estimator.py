# Evan Cook, evan.cook@robotics.utias.utoronto.ca
# January 2024

from tqdm import tqdm
import pickle
import numpy as np
import torch
import random
import argparse
import sys
from datetime import datetime
import os

import normflows as nf # https://github.com/VincentStimper/normalizing-flows
import fde
import pickle as pk
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# To profile, measure start and end time using time.now()
time = datetime.utcnow()
def delta_t(a, b):
    return abs(a-b).total_seconds()

####################################################################################################
# Configuration (may be overridden by arguments)
####################################################################################################

# Project constants
config = {
          "flow_architecture": "Glow",
          "model name": "ResNet50",
          "dataset": "ImageNet1k",
          "output path": "checkpoint/flow_resnet50.pt",
          "try cuda": True,
          "epoch count": 1,
          "batch size": 250,
          "val data count": 50000,
          "learning rate": 1e-5,
          "seed": 42,
          "imagenet dir": os.path.join(os.environ['HOME'], "data", "ImageNet1k"),
         }

####################################################################################################
# Training Code
####################################################################################################

# Inner training loop for flow_model - run one epoch through the dataloader
def train(flow_model, classifier_model, device, dataloader, optimizer):
    flow_model.train()
    classifier_model.eval()
    losses = []

    progress_bar = tqdm(dataloader)
    for data, target in progress_bar:
        # Reset gradients
        optimizer.zero_grad()

        # Load the data and labels from the training dataset
        data, targets = data.to(device), target.to(device)

        # Run the data through our backbone so we can extract the feature representations
        with torch.no_grad():
            logits, features = fde.get_features(classifier_model, data, device)

        logprob = flow_model.log_prob(features)
        loss = -torch.mean(logprob)

        # Do backpropagation and perform gradient descent
        loss.backward()
        optimizer.step()

        losses.append(float(loss))
        progress_bar.set_description("Training, loss = {:.2f}".format(float(loss)))

    return float(np.mean(losses))

def run():
    print("Training {}...".format(config["model name"]))

    ################################################################################################
    # Environment setup
    ################################################################################################

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = config["try cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(">> Training device:", device)
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    ################################################################################################
    # Models
    ################################################################################################

    # Create our classifier backbone
    classifier_model = fde.create_classifier(config["model name"])
    if classifier_model is None:
        print("Unknown model name!")
        sys.exit(1)

    classifier_model = classifier_model.to(device)
    classifier_model.eval()

    # Create our flow model
    base = nf.distributions.base.DiagGaussian(classifier_model.feature_dimension, trainable=False)
    flows = fde.create_glow_flows(10, classifier_model.feature_dimension)
    flow_model = nf.NormalizingFlow(base, flows).to(device)

    ################################################################################################
    # Dataloader
    ################################################################################################

    train_data = datasets.ImageNet(root=config['imagenet dir'], split='train', transform=classifier_model.transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config["batch size"], shuffle=True, **kwargs)

    ################################################################################################
    # Training loop
    ################################################################################################

    optimizer = torch.optim.Adam(flow_model.parameters(), lr=config["learning rate"])

    for epoch in range(config["epoch count"]):
        tick = time.now()
        epoch_train_loss = train(flow_model, classifier_model, device, train_dataloader, optimizer)
        tock = time.now()
        print(">> Epoch {} complete, {:.2f} seconds elapsed.".format(epoch, delta_t(tick, tock)))
        torch.save(flow_model.state_dict(), config["output path"])

    print(">> Complete.")

def main():
    global config
    parser = argparse.ArgumentParser(description="Script to train a normalizing flow density estimation model on the feature representations of a pretrained ImageNet1k backbone")

    # Add command line arguments
    parser.add_argument('output_path', type=str, help='Output path of the normalizing flow model')
    parser.add_argument('model_name', type=str, help='Name of the pytorch pretrained ImageNet1k backbone model')
    parser.add_argument('--seed', type=int, default=config["seed"], help='Random seed')
    parser.add_argument('--epochs', type=int, default=config["epoch count"], help='Number of epochs to train the normalizing flow for (optional)')
    parser.add_argument('--batch_size', type=int, default=config["batch size"], help='Training batch size (optional)')
    args = parser.parse_args()

    config["output path"] = args.output_path
    config["model name"] = args.model_name
    config["epoch count"] = args.epochs
    config["batch size"] = args.batch_size
    config["seed"] = args.seed

    run()


if __name__ == "__main__":
    main()

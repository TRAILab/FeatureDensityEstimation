# Evan Cook, evan.cook@robotics.utias.utoronto.ca
# January 2024

from tqdm import tqdm
import pickle
import numpy as np
import torch
import random
import argparse
import sys
import csv
import math

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datetime import datetime
import os

from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import scipy
import pickle as pk

import normflows as nf # https://github.com/VincentStimper/normalizing-flows
import fde

from typing import Union, Optional, Tuple

####################################################################################################
# Configuration (may be overridden by arguments)
####################################################################################################

# Project constants
config = {
          "flow_architecture": "Glow",
          "model name": "ResNet50",
          "dataset": "ImageNet1k",
          "input path": "checkpoint/flow_resnet50.pt",
          "log path": "results/flow_reset50.csv",
          "try cuda": True,
          "batch size": 20,
          "seed": 42,
          "full eval": True,
          "imagenet dir": os.path.join(os.environ['HOME'], "data", "ImageNet1k"),
         }

class LogWriter:
    def __init__(self, filepath):
        self.filepath = filepath

        # Check if the file already exists
        file_exists = os.path.exists(filepath)

        # Open the file in append mode ('a') if it exists, otherwise in write mode ('w')
        self.file = open(filepath, mode='a' if file_exists else 'w', newline='')

        # Create a CSV writer
        self.writer = csv.writer(self.file)

        # Write the header row if it's a new file
        if not file_exists:
            self.writer.writerow(["Model", "Metric", "Value"])

    def log(self, model, metric, value):
        self.writer.writerow([model, metric, value])

    def close(self):
        self.file.close()

####################################################################################################
# Evaluation Code
####################################################################################################

# Adopted from https://github.com/deeplearning-wisc/react/blob/2aa35d9993ddb00409dc41824dbc008b5cc16e20/score.py#L22
def odin_score(inputs, model):
    # See https://github.com/deeplearning-wisc/react/blob/2aa35d9993ddb00409dc41824dbc008b5cc16e20/eval.py#L140
    temper = 1000.0
    noiseMagnitude1 = 0.005

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)
    outputs, _ = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    # outputs = model(Variable(tempInputs))
    with torch.no_grad():
        outputs, _ = model(tempInputs)
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores

def get_dataloader_odin_scores(dataloader, classifier_model, device):
    classifier_model.eval()
    odin_scores = []
    for data, target in tqdm(dataloader):
        data = data.to(device)
        odin_scores.append(odin_score(data, classifier_model))

    return np.concatenate(odin_scores)


def uniformity_computation(features: torch.Tensor,
                           multiplier: int = 2,
                           sample_size: Optional[int] = None) -> torch.Tensor:
    """Compute uniformity loss for a tensor of size B x dim.

    B = Batch size or number of points in the scene.
    dim = representation dimension.
    """
    features = F.normalize(features, dim=1)
    if sample_size:
        # Sample a subset of rows from x uniformly at random
        indices = torch.randperm(features.size(0))[:sample_size]
        features = features[indices]

    return torch.pdist(features, p=2).pow(2).mul(-multiplier).exp().mean().log()


def tolerance_computation(features: torch.Tensor,
                          labels: torch.Tensor,
                          sample_size: Optional[int] = None) -> dict:
    """Computes tolerance loss for a tensor of size B x dim."""
    features = F.normalize(features, dim=1)
    label_count = len(torch.unique(labels))
    tolerance = [0.] * label_count

    # loop from 0 to the number of classes
    for label in range(label_count):
        # Get the feature vectors for this label, with sampling if needed to reduce memory load.
        if sample_size:
            label_features = features[labels == label][:sample_size]
        else:
            label_features = features[labels == label]

        # Compute the pairwise dot product matrix
        # The dot product matrix will be of shape [num_features, num_features]
        dot_product_matrix = torch.matmul(label_features, label_features.t())

        # Since we want pairwise dot products, we need to exclude the diagonal elements
        # which represent the dot product of the vectors with themselves.
        # We use triu (upper triangle) with diagonal offset of 1 to exclude the diagonal.
        upper_triangular = dot_product_matrix.triu(diagonal=1)

        # Calculate the number of elements in the upper triangle.
        # Num elements = upper triangular without the diagonal = dim * (dim - 1) / 2
        num_elements = len(label_features) * (len(label_features) - 1) / 2

        # If there are elements, calculate the mean and add it to the list of class tolerances
        if len(label_features) > 1:
            tolerance[label] = upper_triangular.sum() / num_elements

    return sum(tolerance) / len(tolerance)

def get_dataloader_features(dataloader, classifier_model, device):
    classifier_model.eval()
    all_features = []
    all_labels = []
    for data, target in tqdm(dataloader):
        data = data.to(device)
        with torch.no_grad():
            logits, features = lde.get_features(classifier_model, data, device)
            all_features.append(features)
            all_labels.append(target)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    return all_features, all_labels

def run():
    global config
    print("Testing {}...".format(config["model name"]))

    ################################################################################################
    # Environment setup
    ################################################################################################

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(">> Training device:", device)
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    results = LogWriter(config["log path"])

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

    # Load our flow model
    base = nf.distributions.base.DiagGaussian(classifier_model.feature_dimension, trainable=False)
    flows = fde.create_glow_flows(10, classifier_model.feature_dimension)
    flow_model = nf.NormalizingFlow(base, flows).to(device)
    flow_model.load_state_dict(torch.load(config["input path"]))
    flow_model.to(device)
    flow_model.eval()

    ################################################################################################
    # Dataloader
    ################################################################################################

    val_data = datasets.ImageNet(root=config['imagenet dir'], split='val', transform=classifier_model.transform)
    val_name = "ImageNet1k"
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=config['batch size'], shuffle=False, **kwargs)

    ################################################################################################
    # In Distribution Evaluation
    ################################################################################################

    # ID data: grab a batch from the dataloader and feed it through the classifier
    id_features, id_logits, id_correct, id_labels = [], [], [], []
    for batch_idx, (data, target) in enumerate(tqdm(val_dataloader, desc=val_name)):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            logits, features = classifier_model(data)
            id_features.append(features)
            id_logits.append(logits)
            id_labels.append(target)
            
            # For ID data, count correct predictions
            _, pred = logits.max(1) # get the index of the max probability
            id_correct.append(pred.eq(target.view_as(pred)))

    id_features = torch.cat(id_features)
    id_correct = torch.cat(id_correct)
    id_logits = torch.cat(id_logits)
    id_labels = torch.cat(id_labels)

    # ReACT: Sec 4.1 of https://arxiv.org/pdf/2111.12797.pdf
    # Rectification threshold is 90% of unnormalized feature vector
    react_threshold = np.percentile(id_features.detach().to("cpu").view(-1), 90)
    id_features_react = torch.clamp(id_features, max=react_threshold)
    id_logits_react = classifier_model.linear(id_features_react)

    with torch.no_grad():
        id_msp = torch.max(F.softmax(id_logits, dim=1), dim=1)[0].cpu().numpy()
        id_energy = -torch.logsumexp((id_logits), dim=1).cpu().numpy()
        id_react = -torch.logsumexp((id_logits_react), dim=1).cpu().numpy()
        id_logprob = flow_model.log_prob(F.normalize(id_features, dim=1)).cpu().numpy()

        id_nan_count = len(id_logprob)-np.sum(np.isfinite(id_logprob))
        if id_nan_count > 0:
            print("Warning! Flow model is producing ID NaN probabilities: {} / {}".format(id_nan_count, len(id_logprob)))
            id_logprob = np.nan_to_num(id_logprob)

    results.log(model=config["model name"], metric="Classifier validation accuracy", value="{:.4f}".format(sum(id_correct)/len(id_correct)))

    # For a quick evaluation, skip ODIN, uniformity, and tolerance calculation
    if config["full eval"]:
        id_odin =  get_dataloader_odin_scores(val_dataloader, classifier_model, device)

        # Caluclate uniformity and tolerance of in-distribution features
        print("Calculating tolerance...")
        tolerance = tolerance_computation(id_features, id_labels)
        results.log(model=config["model name"], metric="Tolerance", value="{:.4f}".format(tolerance))

        print("Calculating uniformity...")
        uniformity = uniformity_computation(id_features)
        results.log(model=config["model name"], metric="Uniformity", value="{:.4f}".format(-uniformity))

    ################################################################################################
    # Out-of-Distribution Evaluation
    ################################################################################################

    # Loop through OOD datasets and evaluate AUROC vs in-distribution dataset
    ood_data = {
        "Textures"    : datasets.DTD(root='data', download=True, split='val', transform=classifier_model.transform),
        "iNaturalist" : datasets.ImageFolder("/home/evan/Research/data/ReACT/iNaturalist", transform=classifier_model.transform),
        "SUN"         : datasets.ImageFolder("/home/evan/Research/data/ReACT/SUN", transform=classifier_model.transform),
        "Places"      : datasets.ImageFolder("/home/evan/Research/data/ReACT/Places", transform=classifier_model.transform)
    }

    for ood_name in ood_data:
        ood_dataloader = torch.utils.data.DataLoader(ood_data[ood_name], batch_size=config["batch size"], shuffle=False, **kwargs)

        # OOD data: grab a batch from the dataloader and feed it through the classifier
        ood_features, ood_logits = [], []
        for batch_idx, (data, target) in enumerate(tqdm(ood_dataloader, desc=ood_name)):
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                logits, features = classifier_model(data)
                ood_features.append(features)
                ood_logits.append(logits)

        ood_features = torch.cat(ood_features)
        ood_logits = torch.cat(ood_logits)

        # ReACT: Sec 4.1 of https://arxiv.org/pdf/2111.12797.pdf
        ood_features_react = torch.clamp(ood_features, max=react_threshold)
        ood_logits_react = classifier_model.linear(ood_features_react)

        # Evaluate log probabilities of data
        with torch.no_grad():
            ood_msp = torch.max(F.softmax(ood_logits, dim=1), dim=1)[0].cpu().numpy()
            ood_energy = -torch.logsumexp((ood_logits), dim=1).cpu().numpy()
            ood_react = -torch.logsumexp((ood_logits_react), dim=1).cpu().numpy()
            ood_logprob = flow_model.log_prob(F.normalize(ood_features, dim=1)).cpu().numpy()

            ood_nan_count = len(ood_logprob)-np.sum(np.isfinite(ood_logprob))
            if ood_nan_count > 0:
                print("Warning! Flow model is producing OOD NaN probabilities: {} / {}".format(ood_nan_count, len(ood_logprob)))
                ood_logprob = np.nan_to_num(ood_logprob)

        # Calculate AUROC using Free Energy and normalizing flow log probability.
        auroc_energy = fde.auroc(id_energy, ood_energy)
        auroc_msp = fde.auroc(id_msp, ood_msp)
        auroc_flow = fde.auroc(id_logprob, ood_logprob)
        auroc_react = fde.auroc(id_react, ood_react)

        results.log(model=config["model name"], metric="{} / {} MSP AUROC".format(val_name, ood_name), value="{:.4f}".format(auroc_msp))
        results.log(model=config["model name"], metric="{} / {} Free Energy AUROC".format(val_name, ood_name), value="{:.4f}".format(auroc_energy))
        results.log(model=config["model name"], metric="{} / {} ReACT AUROC".format(val_name, ood_name), value="{:.4f}".format(auroc_react))
        results.log(model=config["model name"], metric="{} / {} Flow AUROC".format(val_name, ood_name), value="{:.4f}".format(auroc_flow))

        if config["full eval"]:
            ood_odin = get_dataloader_odin_scores(ood_dataloader, classifier_model, device)
            auroc_odin = fde.auroc(id_odin, ood_odin)
            results.log(model=config["model name"], metric="{} / {} ODIN AUROC".format(val_name, ood_name), value="{:.4f}".format(auroc_odin))

    results.close()


def main():
    global config

    # Create argument parser
    parser = argparse.ArgumentParser(description="Script to evaluate a normalizing flow density estimation model on the feature representations of a pretrained ImageNet1k backbone")

    # Add command line arguments
    parser.add_argument('input_path', type=str, help='Input path of the trained normalizing flow model')
    parser.add_argument('model_name', type=str, help='Name of the pytorch pretrained ImageNet1k backbone model')
    parser.add_argument('log_path', type=str, help='Path to log results to (CSV format)')
    parser.add_argument('--quick', action="store_true", help='Quick evaluation; skip calculating ODIN, Uniformity, Tolerance')
    args = parser.parse_args()

    config["input path"] = args.input_path
    config["log path"] = args.log_path
    config["model name"] = args.model_name
    config["full eval"] = not args.quick

    run()

if __name__ == "__main__":
    main()

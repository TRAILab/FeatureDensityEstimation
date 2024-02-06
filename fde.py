# Evan Cook, evan.cook@robotics.utias.utoronto.ca
# January 2024

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
import normflows as nf # https://github.com/VincentStimper/normalizing-flows
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import imagenet_models

####################################################################################################
# Utilities
####################################################################################################

# Helper to get the desired feature variables from the classifier model
def get_features(classifier_model, data, device):
    logits, features = classifier_model(data)

    # Normalize the feature variables
    feature_norms = torch.norm(features, dim=1).unsqueeze(-1)
    features = torch.div(features, feature_norms)
    return logits, features

# Calculate the AUROC metric for two 1D tensors of values.
# This first fits a logistic regression to separate the distributions, then calculates AUROC using these probabilities.
def auroc(x1, x2):
    x = np.concatenate((x1, x2))
    y_true = np.concatenate((np.zeros(len(x1)),
                             np.ones(len(x2))))

    logistic_regression = LogisticRegression(solver='liblinear', random_state=42)
    logistic_regression.fit(x.reshape(-1, 1), y_true)
    p_y = logistic_regression.predict_proba(x.reshape(-1, 1)).T[1, :]
    crossover = (-logistic_regression.intercept_ / logistic_regression.coef_).item()
    return roc_auc_score(y_true, p_y)

# Helper for plotting: returns useful min / max axis limits for plotting.
# Uses percentile limits to ignore outliers, then pads with some margin.
def plot_limits(x, percentile=2, margin=0.25):
    percentile = np.clip(0.0, percentile, 100.0)
    percentile = min(percentile, 100.0 - percentile)
    minval = np.percentile(x, percentile)
    maxval = np.percentile(x, 100.0 - percentile)
    delta = maxval - minval
    return [minval - (margin * delta),
            maxval + (margin * delta)]

####################################################################################################
# ImageNet1k Classifier Models
####################################################################################################

# Helper function to create a pretrained classifier backbone model given the model name
def create_classifier(model_name):
    classifier_model = None

    # Load our classifier model.
    config = imagenet_models.PRETRAINED_IMAGENET_MODELS.get(model_name, None)

    if config is not None:
        classifier_model = imagenet_models.ClassifierFactory(config['model'], config['weights'], config['linear_head_path'])

    return classifier_model

####################################################################################################
# CIFAR-10 Classifier Models
####################################################################################################

# Basic ResNet18 architecture that exposes the unmodified penultimate feature space (d=512).
# Modified for CIFAR-10
class ResNet18Cifar(nn.Module):
    def __init__(self):
        super(ResNet18Cifar, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.linear(features)
        return logits, features

####################################################################################################
# Normalizing Flow Models
####################################################################################################

# RealNVP flow example: https://github.com/VincentStimper/normalizing-flows/blob/master/example/real_nvp.ipynb
def create_realNVP_flows(K=4, feature_size=64):
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(feature_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([feature_size, feature_size, feature_size], init_zeros=True)
        t = nf.nets.MLP([feature_size, feature_size, feature_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(feature_size)]
    return flows

# A wrapper for Glow's Invertible1x1Conv, padding with a W and H dimension (since image format is assumed)
class InvertiblePermutation(nf.flows.Flow):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.transform = nf.flows.mixing.Invertible1x1Conv(num_channels, True)

    def forward(self, z):
        z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        z_, ldj = self.transform.forward(z)
        return z_[:, :, 0, 0], ldj

    def inverse(self, z):
        z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        z_, ldj = self.transform.inverse(z)
        return z_[:, :, 0, 0], ldj

# See https://arxiv.org/abs/1807.03039
def create_glow_flows(K=10, feature_size=2):
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(feature_size)]) # Can be set to [0 .. 1]
    flows = []
    for i in range(K):
        s = nf.nets.MLP([feature_size, feature_size, feature_size], init_zeros=True)
        t = nf.nets.MLP([feature_size, feature_size, feature_size], init_zeros=True)
        flows += [nf.flows.ActNorm(feature_size)]
        flows += [InvertiblePermutation(feature_size)]
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    return flows

# Residual flow example: https://github.com/VincentStimper/normalizing-flows/blob/master/examples/residual.ipynb
def create_residual_flows(K=4, feature_size=64):
    flows = []
    for i in range(K):
        net = nf.nets.LipschitzMLP([feature_size, feature_size, feature_size], init_zeros=True, lipschitz_const=0.9)
        flows += [nf.flows.Residual(net, reduce_memory=True)]
        flows += [nf.flows.ActNorm(feature_size)]
    return flows

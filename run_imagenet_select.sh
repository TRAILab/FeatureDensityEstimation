#!/bin/bash

# ResNet50_V1
# Acc@1: 76.13,	Acc@5: 92.862,	Params: 25.6M,	GFLOPS: 4.09
python train_feature_density_estimator.py "checkpoint/Flow_ResNet50_V1.pt" "ResNet50_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNet50_V1.pt" "ResNet50_V1" "results/imagenet_results_select.csv"

# MobileNet_V2_V1
# Acc@1: 71.878,	Acc@5: 90.286,	Params: 3.5M,	GFLOPS: 0.3
python train_feature_density_estimator.py "checkpoint/Flow_MobileNet_V2_V1.pt" "MobileNet_V2_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MobileNet_V2_V1.pt" "MobileNet_V2_V1" "results/imagenet_results_select.csv"

# Swin_T
# Acc@1: 81.474,	Acc@5: 95.776,	Params: 28.3M,	GFLOPS: 4.49
python train_feature_density_estimator.py "checkpoint/Flow_Swin_T.pt" "Swin_T" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Swin_T.pt" "Swin_T" "results/imagenet_results_select.csv"

# ViT_B_32
# Acc@1: 75.912,	Acc@5: 92.466,	Params: 88.2M,	GFLOPS: 4.41
python train_feature_density_estimator.py "checkpoint/Flow_ViT_B_32.pt" "ViT_B_32" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_B_32.pt" "ViT_B_32" "results/imagenet_results_select.csv"

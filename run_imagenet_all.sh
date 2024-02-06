#!/bin/bash

# AlexNet
# Acc@1: 56.522,	Acc@5: 79.066,	Params: 61.1M,	GFLOPS: 0.71
python train_feature_density_estimator.py "checkpoint/Flow_AlexNet.pt" "AlexNet" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_AlexNet.pt" "AlexNet" "results/imagenet_results.csv"

# ConvNeXt_Base
# Acc@1: 84.062,	Acc@5: 96.87,	Params: 88.6M,	GFLOPS: 15.36
python train_feature_density_estimator.py "checkpoint/Flow_ConvNeXt_Base.pt" "ConvNeXt_Base" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ConvNeXt_Base.pt" "ConvNeXt_Base" "results/imagenet_results.csv"

# ConvNeXt_Large
# Acc@1: 84.414,	Acc@5: 96.976,	Params: 197.8M,	GFLOPS: 34.36
python train_feature_density_estimator.py "checkpoint/Flow_ConvNeXt_Large.pt" "ConvNeXt_Large" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ConvNeXt_Large.pt" "ConvNeXt_Large" "results/imagenet_results.csv"

# ConvNeXt_Small
# Acc@1: 83.616,	Acc@5: 96.65,	Params: 50.2M,	GFLOPS: 8.68
python train_feature_density_estimator.py "checkpoint/Flow_ConvNeXt_Small.pt" "ConvNeXt_Small" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ConvNeXt_Small.pt" "ConvNeXt_Small" "results/imagenet_results.csv"

# ConvNeXt_Tiny
# Acc@1: 82.52,	Acc@5: 96.146,	Params: 28.6M,	GFLOPS: 4.46
python train_feature_density_estimator.py "checkpoint/Flow_ConvNeXt_Tiny.pt" "ConvNeXt_Tiny" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ConvNeXt_Tiny.pt" "ConvNeXt_Tiny" "results/imagenet_results.csv"

# DenseNet121
# Acc@1: 74.434,	Acc@5: 91.972,	Params: 8.0M,	GFLOPS: 2.83
python train_feature_density_estimator.py "checkpoint/Flow_DenseNet121.pt" "DenseNet121" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_DenseNet121.pt" "DenseNet121" "results/imagenet_results.csv"

# DenseNet161
# Acc@1: 77.138,	Acc@5: 93.56,	Params: 28.7M,	GFLOPS: 7.73
python train_feature_density_estimator.py "checkpoint/Flow_DenseNet161.pt" "DenseNet161" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_DenseNet161.pt" "DenseNet161" "results/imagenet_results.csv"

# DenseNet169
# Acc@1: 75.6,	Acc@5: 92.806,	Params: 14.1M,	GFLOPS: 3.36
python train_feature_density_estimator.py "checkpoint/Flow_DenseNet169.pt" "DenseNet169" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_DenseNet169.pt" "DenseNet169" "results/imagenet_results.csv"

# DenseNet201
# Acc@1: 76.896,	Acc@5: 93.37,	Params: 20.0M,	GFLOPS: 4.29
python train_feature_density_estimator.py "checkpoint/Flow_DenseNet201.pt" "DenseNet201" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_DenseNet201.pt" "DenseNet201" "results/imagenet_results.csv"

# EfficientNet_B0
# Acc@1: 77.692,	Acc@5: 93.532,	Params: 5.3M,	GFLOPS: 0.39
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B0.pt" "EfficientNet_B0" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B0.pt" "EfficientNet_B0" "results/imagenet_results.csv"

# EfficientNet_B1_V1
# Acc@1: 78.642,	Acc@5: 94.186,	Params: 7.8M,	GFLOPS: 0.69
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B1_V1.pt" "EfficientNet_B1_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B1_V1.pt" "EfficientNet_B1_V1" "results/imagenet_results.csv"

# EfficientNet_B1_V2
# Acc@1: 79.838,	Acc@5: 94.934,	Params: 7.8M,	GFLOPS: 0.69
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B1_V2.pt" "EfficientNet_B1_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B1_V2.pt" "EfficientNet_B1_V2" "results/imagenet_results.csv"

# EfficientNet_B2
# Acc@1: 80.608,	Acc@5: 95.31,	Params: 9.1M,	GFLOPS: 1.09
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B2.pt" "EfficientNet_B2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B2.pt" "EfficientNet_B2" "results/imagenet_results.csv"

# EfficientNet_B3
# Acc@1: 82.008,	Acc@5: 96.054,	Params: 12.2M,	GFLOPS: 1.83
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B3.pt" "EfficientNet_B3" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B3.pt" "EfficientNet_B3" "results/imagenet_results.csv"

# EfficientNet_B4
# Acc@1: 83.384,	Acc@5: 96.594,	Params: 19.3M,	GFLOPS: 4.39
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B4.pt" "EfficientNet_B4" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B4.pt" "EfficientNet_B4" "results/imagenet_results.csv"

# EfficientNet_B5
# Acc@1: 83.444,	Acc@5: 96.628,	Params: 30.4M,	GFLOPS: 10.27
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B5.pt" "EfficientNet_B5" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B5.pt" "EfficientNet_B5" "results/imagenet_results.csv"

# EfficientNet_B6
# Acc@1: 84.008,	Acc@5: 96.916,	Params: 43.0M,	GFLOPS: 19.07
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B6.pt" "EfficientNet_B6" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B6.pt" "EfficientNet_B6" "results/imagenet_results.csv"

# EfficientNet_B7
# Acc@1: 84.122,	Acc@5: 96.908,	Params: 66.3M,	GFLOPS: 37.75
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B7.pt" "EfficientNet_B7" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_B7.pt" "EfficientNet_B7" "results/imagenet_results.csv"

# EfficientNet_V2_L
# Acc@1: 85.808,	Acc@5: 97.788,	Params: 118.5M,	GFLOPS: 56.08
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_V2_L.pt" "EfficientNet_V2_L" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_V2_L.pt" "EfficientNet_V2_L" "results/imagenet_results.csv"

# EfficientNet_V2_M
# Acc@1: 85.112,	Acc@5: 97.156,	Params: 54.1M,	GFLOPS: 24.58
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_V2_M.pt" "EfficientNet_V2_M" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_V2_M.pt" "EfficientNet_V2_M" "results/imagenet_results.csv"

# EfficientNet_V2_S
# Acc@1: 84.228,	Acc@5: 96.878,	Params: 21.5M,	GFLOPS: 8.37
python train_feature_density_estimator.py "checkpoint/Flow_EfficientNet_V2_S.pt" "EfficientNet_V2_S" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_EfficientNet_V2_S.pt" "EfficientNet_V2_S" "results/imagenet_results.csv"

# GoogLeNet
# Acc@1: 69.778,	Acc@5: 89.53,	Params: 6.6M,	GFLOPS: 1.5
python train_feature_density_estimator.py "checkpoint/Flow_GoogLeNet.pt" "GoogLeNet" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_GoogLeNet.pt" "GoogLeNet" "results/imagenet_results.csv"

# Inception_V3
# Acc@1: 77.294,	Acc@5: 93.45,	Params: 27.2M,	GFLOPS: 5.71
python train_feature_density_estimator.py "checkpoint/Flow_Inception_V3.pt" "Inception_V3" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Inception_V3.pt" "Inception_V3" "results/imagenet_results.csv"

# MNASNet0_5
# Acc@1: 67.734,	Acc@5: 87.49,	Params: 2.2M,	GFLOPS: 0.1
python train_feature_density_estimator.py "checkpoint/Flow_MNASNet0_5.pt" "MNASNet0_5" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MNASNet0_5.pt" "MNASNet0_5" "results/imagenet_results.csv"

# MNASNet0_75
# Acc@1: 71.18,	Acc@5: 90.496,	Params: 3.2M,	GFLOPS: 0.21
python train_feature_density_estimator.py "checkpoint/Flow_MNASNet0_75.pt" "MNASNet0_75" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MNASNet0_75.pt" "MNASNet0_75" "results/imagenet_results.csv"

# MNASNet1_0
# Acc@1: 73.456,	Acc@5: 91.51,	Params: 4.4M,	GFLOPS: 0.31
python train_feature_density_estimator.py "checkpoint/Flow_MNASNet1_0.pt" "MNASNet1_0" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MNASNet1_0.pt" "MNASNet1_0" "results/imagenet_results.csv"

# MNASNet1_3
# Acc@1: 76.506,	Acc@5: 93.522,	Params: 6.3M,	GFLOPS: 0.53
python train_feature_density_estimator.py "checkpoint/Flow_MNASNet1_3.pt" "MNASNet1_3" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MNASNet1_3.pt" "MNASNet1_3" "results/imagenet_results.csv"

# MaxVit_T
# Acc@1: 83.7,	Acc@5: 96.722,	Params: 30.9M,	GFLOPS: 5.56
python train_feature_density_estimator.py "checkpoint/Flow_MaxVit_T.pt" "MaxVit_T" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MaxVit_T.pt" "MaxVit_T" "results/imagenet_results.csv"

# MobileNet_V2_V1
# Acc@1: 71.878,	Acc@5: 90.286,	Params: 3.5M,	GFLOPS: 0.3
python train_feature_density_estimator.py "checkpoint/Flow_MobileNet_V2_V1.pt" "MobileNet_V2_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MobileNet_V2_V1.pt" "MobileNet_V2_V1" "results/imagenet_results.csv"

# MobileNet_V2_V2
# Acc@1: 72.154,	Acc@5: 90.822,	Params: 3.5M,	GFLOPS: 0.3
python train_feature_density_estimator.py "checkpoint/Flow_MobileNet_V2_V2.pt" "MobileNet_V2_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MobileNet_V2_V2.pt" "MobileNet_V2_V2" "results/imagenet_results.csv"

# MobileNet_V3_Large_V1
# Acc@1: 74.042,	Acc@5: 91.34,	Params: 5.5M,	GFLOPS: 0.22
python train_feature_density_estimator.py "checkpoint/Flow_MobileNet_V3_Large_V1.pt" "MobileNet_V3_Large_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MobileNet_V3_Large_V1.pt" "MobileNet_V3_Large_V1" "results/imagenet_results.csv"

# MobileNet_V3_Large_V2
# Acc@1: 75.274,	Acc@5: 92.566,	Params: 5.5M,	GFLOPS: 0.22
python train_feature_density_estimator.py "checkpoint/Flow_MobileNet_V3_Large_V2.pt" "MobileNet_V3_Large_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MobileNet_V3_Large_V2.pt" "MobileNet_V3_Large_V2" "results/imagenet_results.csv"

# MobileNet_V3_Small
# Acc@1: 67.668,	Acc@5: 87.402,	Params: 2.5M,	GFLOPS: 0.06
python train_feature_density_estimator.py "checkpoint/Flow_MobileNet_V3_Small.pt" "MobileNet_V3_Small" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_MobileNet_V3_Small.pt" "MobileNet_V3_Small" "results/imagenet_results.csv"

# RegNet_X_16GF_V1
# Acc@1: 80.058,	Acc@5: 94.944,	Params: 54.3M,	GFLOPS: 15.94
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_16GF_V1.pt" "RegNet_X_16GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_16GF_V1.pt" "RegNet_X_16GF_V1" "results/imagenet_results.csv"

# RegNet_X_16GF_V2
# Acc@1: 82.716,	Acc@5: 96.196,	Params: 54.3M,	GFLOPS: 15.94
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_16GF_V2.pt" "RegNet_X_16GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_16GF_V2.pt" "RegNet_X_16GF_V2" "results/imagenet_results.csv"

# RegNet_X_1_6GF_V1
# Acc@1: 77.04,	Acc@5: 93.44,	Params: 9.2M,	GFLOPS: 1.6
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_1_6GF_V1.pt" "RegNet_X_1_6GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_1_6GF_V1.pt" "RegNet_X_1_6GF_V1" "results/imagenet_results.csv"

# RegNet_X_1_6GF_V2
# Acc@1: 79.668,	Acc@5: 94.922,	Params: 9.2M,	GFLOPS: 1.6
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_1_6GF_V2.pt" "RegNet_X_1_6GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_1_6GF_V2.pt" "RegNet_X_1_6GF_V2" "results/imagenet_results.csv"

# RegNet_X_32GF_V1
# Acc@1: 80.622,	Acc@5: 95.248,	Params: 107.8M,	GFLOPS: 31.74
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_32GF_V1.pt" "RegNet_X_32GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_32GF_V1.pt" "RegNet_X_32GF_V1" "results/imagenet_results.csv"

# RegNet_X_32GF_V2
# Acc@1: 83.014,	Acc@5: 96.288,	Params: 107.8M,	GFLOPS: 31.74
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_32GF_V2.pt" "RegNet_X_32GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_32GF_V2.pt" "RegNet_X_32GF_V2" "results/imagenet_results.csv"

# RegNet_X_3_2GF_V1
# Acc@1: 78.364,	Acc@5: 93.992,	Params: 15.3M,	GFLOPS: 3.18
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_3_2GF_V1.pt" "RegNet_X_3_2GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_3_2GF_V1.pt" "RegNet_X_3_2GF_V1" "results/imagenet_results.csv"

# RegNet_X_3_2GF_V2
# Acc@1: 81.196,	Acc@5: 95.43,	Params: 15.3M,	GFLOPS: 3.18
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_3_2GF_V2.pt" "RegNet_X_3_2GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_3_2GF_V2.pt" "RegNet_X_3_2GF_V2" "results/imagenet_results.csv"

# RegNet_X_400MF_V1
# Acc@1: 72.834,	Acc@5: 90.95,	Params: 5.5M,	GFLOPS: 0.41
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_400MF_V1.pt" "RegNet_X_400MF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_400MF_V1.pt" "RegNet_X_400MF_V1" "results/imagenet_results.csv"

# RegNet_X_400MF_V2
# Acc@1: 74.864,	Acc@5: 92.322,	Params: 5.5M,	GFLOPS: 0.41
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_400MF_V2.pt" "RegNet_X_400MF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_400MF_V2.pt" "RegNet_X_400MF_V2" "results/imagenet_results.csv"

# RegNet_X_800MF_V1
# Acc@1: 75.212,	Acc@5: 92.348,	Params: 7.3M,	GFLOPS: 0.8
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_800MF_V1.pt" "RegNet_X_800MF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_800MF_V1.pt" "RegNet_X_800MF_V1" "results/imagenet_results.csv"

# RegNet_X_800MF_V2
# Acc@1: 77.522,	Acc@5: 93.826,	Params: 7.3M,	GFLOPS: 0.8
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_800MF_V2.pt" "RegNet_X_800MF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_800MF_V2.pt" "RegNet_X_800MF_V2" "results/imagenet_results.csv"

# RegNet_X_8GF_V1
# Acc@1: 79.344,	Acc@5: 94.686,	Params: 39.6M,	GFLOPS: 8
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_8GF_V1.pt" "RegNet_X_8GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_8GF_V1.pt" "RegNet_X_8GF_V1" "results/imagenet_results.csv"

# RegNet_X_8GF_V2
# Acc@1: 81.682,	Acc@5: 95.678,	Params: 39.6M,	GFLOPS: 8
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_X_8GF_V2.pt" "RegNet_X_8GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_X_8GF_V2.pt" "RegNet_X_8GF_V2" "results/imagenet_results.csv"

# RegNet_Y_128GF_V1
# Acc@1: 88.228,	Acc@5: 98.682,	Params: 644.8M,	GFLOPS: 374.57
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_128GF_V1.pt" "RegNet_Y_128GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_128GF_V1.pt" "RegNet_Y_128GF_V1" "results/imagenet_results.csv"

# RegNet_Y_128GF_V2
# Acc@1: 86.068,	Acc@5: 97.844,	Params: 644.8M,	GFLOPS: 127.52
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_128GF_V2.pt" "RegNet_Y_128GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_128GF_V2.pt" "RegNet_Y_128GF_V2" "results/imagenet_results.csv"

# RegNet_Y_16GF_V1
# Acc@1: 80.424,	Acc@5: 95.24,	Params: 83.6M,	GFLOPS: 15.91
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_16GF_V1.pt" "RegNet_Y_16GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_16GF_V1.pt" "RegNet_Y_16GF_V1" "results/imagenet_results.csv"

# RegNet_Y_16GF_V2
# Acc@1: 82.886,	Acc@5: 96.328,	Params: 83.6M,	GFLOPS: 15.91
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_16GF_V2.pt" "RegNet_Y_16GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_16GF_V2.pt" "RegNet_Y_16GF_V2" "results/imagenet_results.csv"

# RegNet_Y_16GF_SWAG_E2E_V1
# Acc@1: 86.012,	Acc@5: 98.054,	Params: 83.6M,	GFLOPS: 46.73
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_16GF_SWAG_E2E_V1.pt" "RegNet_Y_16GF_SWAG_E2E_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_16GF_SWAG_E2E_V1.pt" "RegNet_Y_16GF_SWAG_E2E_V1" "results/imagenet_results.csv"

# RegNet_Y_16GF_SWAG_LINEAR_V1
# Acc@1: 83.976,	Acc@5: 97.244,	Params: 83.6M,	GFLOPS: 15.91
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_16GF_SWAG_LINEAR_V1.pt" "RegNet_Y_16GF_SWAG_LINEAR_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_16GF_SWAG_LINEAR_V1.pt" "RegNet_Y_16GF_SWAG_LINEAR_V1" "results/imagenet_results.csv"

# RegNet_Y_1_6GF_V1
# Acc@1: 77.95,	Acc@5: 93.966,	Params: 11.2M,	GFLOPS: 1.61
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_1_6GF_V1.pt" "RegNet_Y_1_6GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_1_6GF_V1.pt" "RegNet_Y_1_6GF_V1" "results/imagenet_results.csv"

# RegNet_Y_1_6GF_V2
# Acc@1: 80.876,	Acc@5: 95.444,	Params: 11.2M,	GFLOPS: 1.61
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_1_6GF_V2.pt" "RegNet_Y_1_6GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_1_6GF_V2.pt" "RegNet_Y_1_6GF_V2" "results/imagenet_results.csv"

# RegNet_Y_32GF_V1
# Acc@1: 80.878,	Acc@5: 95.34,	Params: 145.0M,	GFLOPS: 32.28
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_32GF_V1.pt" "RegNet_Y_32GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_32GF_V1.pt" "RegNet_Y_32GF_V1" "results/imagenet_results.csv"

# RegNet_Y_32GF_V2
# Acc@1: 83.368,	Acc@5: 96.498,	Params: 145.0M,	GFLOPS: 32.28
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_32GF_V2.pt" "RegNet_Y_32GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_32GF_V2.pt" "RegNet_Y_32GF_V2" "results/imagenet_results.csv"

# RegNet_Y_32GF_SWAG_E2E_V1
# Acc@1: 86.838,	Acc@5: 98.362,	Params: 145.0M,	GFLOPS: 94.83
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_32GF_SWAG_E2E_V1.pt" "RegNet_Y_32GF_SWAG_E2E_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_32GF_SWAG_E2E_V1.pt" "RegNet_Y_32GF_SWAG_E2E_V1" "results/imagenet_results.csv"

# RegNet_Y_32GF_SWAG_LINEAR_V1
# Acc@1: 84.622,	Acc@5: 97.48,	Params: 145.0M,	GFLOPS: 32.28
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_32GF_SWAG_LINEAR_V1.pt" "RegNet_Y_32GF_SWAG_LINEAR_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_32GF_SWAG_LINEAR_V1.pt" "RegNet_Y_32GF_SWAG_LINEAR_V1" "results/imagenet_results.csv"

# RegNet_Y_3_2GF_V1
# Acc@1: 78.948,	Acc@5: 94.576,	Params: 19.4M,	GFLOPS: 3.18
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_3_2GF_V1.pt" "RegNet_Y_3_2GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_3_2GF_V1.pt" "RegNet_Y_3_2GF_V1" "results/imagenet_results.csv"

# RegNet_Y_3_2GF_V2
# Acc@1: 81.982,	Acc@5: 95.972,	Params: 19.4M,	GFLOPS: 3.18
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_3_2GF_V2.pt" "RegNet_Y_3_2GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_3_2GF_V2.pt" "RegNet_Y_3_2GF_V2" "results/imagenet_results.csv"

# RegNet_Y_400MF_V1
# Acc@1: 74.046,	Acc@5: 91.716,	Params: 4.3M,	GFLOPS: 0.4
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_400MF_V1.pt" "RegNet_Y_400MF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_400MF_V1.pt" "RegNet_Y_400MF_V1" "results/imagenet_results.csv"

# RegNet_Y_400MF_V2
# Acc@1: 75.804,	Acc@5: 92.742,	Params: 4.3M,	GFLOPS: 0.4
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_400MF_V2.pt" "RegNet_Y_400MF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_400MF_V2.pt" "RegNet_Y_400MF_V2" "results/imagenet_results.csv"

# RegNet_Y_800MF_V1
# Acc@1: 76.42,	Acc@5: 93.136,	Params: 6.4M,	GFLOPS: 0.83
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_800MF_V1.pt" "RegNet_Y_800MF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_800MF_V1.pt" "RegNet_Y_800MF_V1" "results/imagenet_results.csv"

# RegNet_Y_800MF_V2
# Acc@1: 78.828,	Acc@5: 94.502,	Params: 6.4M,	GFLOPS: 0.83
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_800MF_V2.pt" "RegNet_Y_800MF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_800MF_V2.pt" "RegNet_Y_800MF_V2" "results/imagenet_results.csv"

# RegNet_Y_8GF_V1
# Acc@1: 80.032,	Acc@5: 95.048,	Params: 39.4M,	GFLOPS: 8.47
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_8GF_V1.pt" "RegNet_Y_8GF_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_8GF_V1.pt" "RegNet_Y_8GF_V1" "results/imagenet_results.csv"

# RegNet_Y_8GF_V2
# Acc@1: 82.828,	Acc@5: 96.33,	Params: 39.4M,	GFLOPS: 8.47
python train_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_8GF_V2.pt" "RegNet_Y_8GF_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_RegNet_Y_8GF_V2.pt" "RegNet_Y_8GF_V2" "results/imagenet_results.csv"

# ResNeXt101_32X8D_V1
# Acc@1: 79.312,	Acc@5: 94.526,	Params: 88.8M,	GFLOPS: 16.41
python train_feature_density_estimator.py "checkpoint/Flow_ResNeXt101_32X8D_V1.pt" "ResNeXt101_32X8D_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNeXt101_32X8D_V1.pt" "ResNeXt101_32X8D_V1" "results/imagenet_results.csv"

# ResNeXt101_32X8D_V2
# Acc@1: 82.834,	Acc@5: 96.228,	Params: 88.8M,	GFLOPS: 16.41
python train_feature_density_estimator.py "checkpoint/Flow_ResNeXt101_32X8D_V2.pt" "ResNeXt101_32X8D_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNeXt101_32X8D_V2.pt" "ResNeXt101_32X8D_V2" "results/imagenet_results.csv"

# ResNeXt101_64X4D
# Acc@1: 83.246,	Acc@5: 96.454,	Params: 83.5M,	GFLOPS: 15.46
python train_feature_density_estimator.py "checkpoint/Flow_ResNeXt101_64X4D.pt" "ResNeXt101_64X4D" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNeXt101_64X4D.pt" "ResNeXt101_64X4D" "results/imagenet_results.csv"

# ResNeXt50_32X4D_V1
# Acc@1: 77.618,	Acc@5: 93.698,	Params: 25.0M,	GFLOPS: 4.23
python train_feature_density_estimator.py "checkpoint/Flow_ResNeXt50_32X4D_V1.pt" "ResNeXt50_32X4D_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNeXt50_32X4D_V1.pt" "ResNeXt50_32X4D_V1" "results/imagenet_results.csv"

# ResNeXt50_32X4D_V2
# Acc@1: 81.198,	Acc@5: 95.34,	Params: 25.0M,	GFLOPS: 4.23
python train_feature_density_estimator.py "checkpoint/Flow_ResNeXt50_32X4D_V2.pt" "ResNeXt50_32X4D_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNeXt50_32X4D_V2.pt" "ResNeXt50_32X4D_V2" "results/imagenet_results.csv"

# ResNet101_V1
# Acc@1: 77.374,	Acc@5: 93.546,	Params: 44.5M,	GFLOPS: 7.8
python train_feature_density_estimator.py "checkpoint/Flow_ResNet101_V1.pt" "ResNet101_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNet101_V1.pt" "ResNet101_V1" "results/imagenet_results.csv"

# ResNet101_V2
# Acc@1: 81.886,	Acc@5: 95.78,	Params: 44.5M,	GFLOPS: 7.8
python train_feature_density_estimator.py "checkpoint/Flow_ResNet101_V2.pt" "ResNet101_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNet101_V2.pt" "ResNet101_V2" "results/imagenet_results.csv"

# ResNet152_V1
# Acc@1: 78.312,	Acc@5: 94.046,	Params: 60.2M,	GFLOPS: 11.51
python train_feature_density_estimator.py "checkpoint/Flow_ResNet152_V1.pt" "ResNet152_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNet152_V1.pt" "ResNet152_V1" "results/imagenet_results.csv"

# ResNet152_V2
# Acc@1: 82.284,	Acc@5: 96.002,	Params: 60.2M,	GFLOPS: 11.51
python train_feature_density_estimator.py "checkpoint/Flow_ResNet152_V2.pt" "ResNet152_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNet152_V2.pt" "ResNet152_V2" "results/imagenet_results.csv"

# ResNet18
# Acc@1: 69.758,	Acc@5: 89.078,	Params: 11.7M,	GFLOPS: 1.81
python train_feature_density_estimator.py "checkpoint/Flow_ResNet18.pt" "ResNet18" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNet18.pt" "ResNet18" "results/imagenet_results.csv"

# ResNet34
# Acc@1: 73.314,	Acc@5: 91.42,	Params: 21.8M,	GFLOPS: 3.66
python train_feature_density_estimator.py "checkpoint/Flow_ResNet34.pt" "ResNet34" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNet34.pt" "ResNet34" "results/imagenet_results.csv"

# ResNet50_V1
# Acc@1: 76.13,	Acc@5: 92.862,	Params: 25.6M,	GFLOPS: 4.09
python train_feature_density_estimator.py "checkpoint/Flow_ResNet50_V1.pt" "ResNet50_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNet50_V1.pt" "ResNet50_V1" "results/imagenet_results.csv"

# ResNet50_V2
# Acc@1: 80.858,	Acc@5: 95.434,	Params: 25.6M,	GFLOPS: 4.09
python train_feature_density_estimator.py "checkpoint/Flow_ResNet50_V2.pt" "ResNet50_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ResNet50_V2.pt" "ResNet50_V2" "results/imagenet_results.csv"

# ShuffleNet_V2_X0_5
# Acc@1: 60.552,	Acc@5: 81.746,	Params: 1.4M,	GFLOPS: 0.04
python train_feature_density_estimator.py "checkpoint/Flow_ShuffleNet_V2_X0_5.pt" "ShuffleNet_V2_X0_5" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ShuffleNet_V2_X0_5.pt" "ShuffleNet_V2_X0_5" "results/imagenet_results.csv"

# ShuffleNet_V2_X1_0
# Acc@1: 69.362,	Acc@5: 88.316,	Params: 2.3M,	GFLOPS: 0.14
python train_feature_density_estimator.py "checkpoint/Flow_ShuffleNet_V2_X1_0.pt" "ShuffleNet_V2_X1_0" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ShuffleNet_V2_X1_0.pt" "ShuffleNet_V2_X1_0" "results/imagenet_results.csv"

# ShuffleNet_V2_X1_5
# Acc@1: 72.996,	Acc@5: 91.086,	Params: 3.5M,	GFLOPS: 0.3
python train_feature_density_estimator.py "checkpoint/Flow_ShuffleNet_V2_X1_5.pt" "ShuffleNet_V2_X1_5" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ShuffleNet_V2_X1_5.pt" "ShuffleNet_V2_X1_5" "results/imagenet_results.csv"

# ShuffleNet_V2_X2_0
# Acc@1: 76.23,	Acc@5: 93.006,	Params: 7.4M,	GFLOPS: 0.58
python train_feature_density_estimator.py "checkpoint/Flow_ShuffleNet_V2_X2_0.pt" "ShuffleNet_V2_X2_0" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ShuffleNet_V2_X2_0.pt" "ShuffleNet_V2_X2_0" "results/imagenet_results.csv"

# SqueezeNet1_0
# Acc@1: 58.092,	Acc@5: 80.42,	Params: 1.2M,	GFLOPS: 0.82
python train_feature_density_estimator.py "checkpoint/Flow_SqueezeNet1_0.pt" "SqueezeNet1_0" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_SqueezeNet1_0.pt" "SqueezeNet1_0" "results/imagenet_results.csv"

# SqueezeNet1_1
# Acc@1: 58.178,	Acc@5: 80.624,	Params: 1.2M,	GFLOPS: 0.35
python train_feature_density_estimator.py "checkpoint/Flow_SqueezeNet1_1.pt" "SqueezeNet1_1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_SqueezeNet1_1.pt" "SqueezeNet1_1" "results/imagenet_results.csv"

# Swin_B
# Acc@1: 83.582,	Acc@5: 96.64,	Params: 87.8M,	GFLOPS: 15.43
python train_feature_density_estimator.py "checkpoint/Flow_Swin_B.pt" "Swin_B" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Swin_B.pt" "Swin_B" "results/imagenet_results.csv"

# Swin_S
# Acc@1: 83.196,	Acc@5: 96.36,	Params: 49.6M,	GFLOPS: 8.74
python train_feature_density_estimator.py "checkpoint/Flow_Swin_S.pt" "Swin_S" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Swin_S.pt" "Swin_S" "results/imagenet_results.csv"

# Swin_T
# Acc@1: 81.474,	Acc@5: 95.776,	Params: 28.3M,	GFLOPS: 4.49
python train_feature_density_estimator.py "checkpoint/Flow_Swin_T.pt" "Swin_T" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Swin_T.pt" "Swin_T" "results/imagenet_results.csv"

# Swin_V2_B
# Acc@1: 84.112,	Acc@5: 96.864,	Params: 87.9M,	GFLOPS: 20.32
python train_feature_density_estimator.py "checkpoint/Flow_Swin_V2_B.pt" "Swin_V2_B" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Swin_V2_B.pt" "Swin_V2_B" "results/imagenet_results.csv"

# Swin_V2_S
# Acc@1: 83.712,	Acc@5: 96.816,	Params: 49.7M,	GFLOPS: 11.55
python train_feature_density_estimator.py "checkpoint/Flow_Swin_V2_S.pt" "Swin_V2_S" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Swin_V2_S.pt" "Swin_V2_S" "results/imagenet_results.csv"

# Swin_V2_T
# Acc@1: 82.072,	Acc@5: 96.132,	Params: 28.4M,	GFLOPS: 5.94
python train_feature_density_estimator.py "checkpoint/Flow_Swin_V2_T.pt" "Swin_V2_T" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Swin_V2_T.pt" "Swin_V2_T" "results/imagenet_results.csv"

# VGG11_BN
# Acc@1: 70.37,	Acc@5: 89.81,	Params: 132.9M,	GFLOPS: 7.61
python train_feature_density_estimator.py "checkpoint/Flow_VGG11_BN.pt" "VGG11_BN" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_VGG11_BN.pt" "VGG11_BN" "results/imagenet_results.csv"

# VGG11
# Acc@1: 69.02,	Acc@5: 88.628,	Params: 132.9M,	GFLOPS: 7.61
python train_feature_density_estimator.py "checkpoint/Flow_VGG11.pt" "VGG11" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_VGG11.pt" "VGG11" "results/imagenet_results.csv"

# VGG13_BN
# Acc@1: 71.586,	Acc@5: 90.374,	Params: 133.1M,	GFLOPS: 11.31
python train_feature_density_estimator.py "checkpoint/Flow_VGG13_BN.pt" "VGG13_BN" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_VGG13_BN.pt" "VGG13_BN" "results/imagenet_results.csv"

# VGG13
# Acc@1: 69.928,	Acc@5: 89.246,	Params: 133.0M,	GFLOPS: 11.31
python train_feature_density_estimator.py "checkpoint/Flow_VGG13.pt" "VGG13" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_VGG13.pt" "VGG13" "results/imagenet_results.csv"

# VGG16_BN
# Acc@1: 73.36,	Acc@5: 91.516,	Params: 138.4M,	GFLOPS: 15.47
python train_feature_density_estimator.py "checkpoint/Flow_VGG16_BN.pt" "VGG16_BN" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_VGG16_BN.pt" "VGG16_BN" "results/imagenet_results.csv"

# VGG16_V1
# Acc@1: 71.592,	Acc@5: 90.382,	Params: 138.4M,	GFLOPS: 15.47
python train_feature_density_estimator.py "checkpoint/Flow_VGG16_V1.pt" "VGG16_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_VGG16_V1.pt" "VGG16_V1" "results/imagenet_results.csv"

# VGG16_FEATURES
# Acc@1: nan	nan,	Acc@5: 138.4M,	Params: 15.47,	GFLOPS: Acc@1: 74.218,	Acc@5: 91.842,	Params: 143.7M,	
python train_feature_density_estimator.py "checkpoint/Flow_VGG16_FEATURES.pt" "VGG16_FEATURES" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_VGG16_FEATURES.pt" "VGG16_FEATURES" "results/imagenet_results.csv"

# VGG19_BN
# GFLOPS: 19.63
python train_feature_density_estimator.py "checkpoint/Flow_VGG19_BN.pt" "VGG19_BN" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_VGG19_BN.pt" "VGG19_BN" "results/imagenet_results.csv"

# VGG19
# Acc@1: 72.376,	Acc@5: 90.876,	Params: 143.7M,	GFLOPS: 19.63
python train_feature_density_estimator.py "checkpoint/Flow_VGG19.pt" "VGG19" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_VGG19.pt" "VGG19" "results/imagenet_results.csv"

# ViT_B_16_V1
# Acc@1: 81.072,	Acc@5: 95.318,	Params: 86.6M,	GFLOPS: 17.56
python train_feature_density_estimator.py "checkpoint/Flow_ViT_B_16_V1.pt" "ViT_B_16_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_B_16_V1.pt" "ViT_B_16_V1" "results/imagenet_results.csv"

# ViT_B_16_SWAG_E2E_V1
# Acc@1: 85.304,	Acc@5: 97.65,	Params: 86.9M,	GFLOPS: 55.48
python train_feature_density_estimator.py "checkpoint/Flow_ViT_B_16_SWAG_E2E_V1.pt" "ViT_B_16_SWAG_E2E_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_B_16_SWAG_E2E_V1.pt" "ViT_B_16_SWAG_E2E_V1" "results/imagenet_results.csv"

# ViT_B_16_SWAG_LINEAR_V1
# Acc@1: 81.886,	Acc@5: 96.18,	Params: 86.6M,	GFLOPS: 17.56
python train_feature_density_estimator.py "checkpoint/Flow_ViT_B_16_SWAG_LINEAR_V1.pt" "ViT_B_16_SWAG_LINEAR_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_B_16_SWAG_LINEAR_V1.pt" "ViT_B_16_SWAG_LINEAR_V1" "results/imagenet_results.csv"

# ViT_B_32
# Acc@1: 75.912,	Acc@5: 92.466,	Params: 88.2M,	GFLOPS: 4.41
python train_feature_density_estimator.py "checkpoint/Flow_ViT_B_32.pt" "ViT_B_32" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_B_32.pt" "ViT_B_32" "results/imagenet_results.csv"

# ViT_H_14_SWAG_E2E_V1
# Acc@1: 88.552,	Acc@5: 98.694,	Params: 633.5M,	GFLOPS: 1016.72
python train_feature_density_estimator.py "checkpoint/Flow_ViT_H_14_SWAG_E2E_V1.pt" "ViT_H_14_SWAG_E2E_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_H_14_SWAG_E2E_V1.pt" "ViT_H_14_SWAG_E2E_V1" "results/imagenet_results.csv"

# ViT_H_14_SWAG_LINEAR_V1
# Acc@1: 85.708,	Acc@5: 97.73,	Params: 632.0M,	GFLOPS: 167.29
python train_feature_density_estimator.py "checkpoint/Flow_ViT_H_14_SWAG_LINEAR_V1.pt" "ViT_H_14_SWAG_LINEAR_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_H_14_SWAG_LINEAR_V1.pt" "ViT_H_14_SWAG_LINEAR_V1" "results/imagenet_results.csv"

# ViT_L_16_V1
# Acc@1: 79.662,	Acc@5: 94.638,	Params: 304.3M,	GFLOPS: 61.55
python train_feature_density_estimator.py "checkpoint/Flow_ViT_L_16_V1.pt" "ViT_L_16_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_L_16_V1.pt" "ViT_L_16_V1" "results/imagenet_results.csv"

# ViT_L_16_SWAG_E2E_V1
# Acc@1: 88.064,	Acc@5: 98.512,	Params: 305.2M,	GFLOPS: 361.99
python train_feature_density_estimator.py "checkpoint/Flow_ViT_L_16_SWAG_E2E_V1.pt" "ViT_L_16_SWAG_E2E_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_L_16_SWAG_E2E_V1.pt" "ViT_L_16_SWAG_E2E_V1" "results/imagenet_results.csv"

# ViT_L_16_SWAG_LINEAR_V1
# Acc@1: 85.146,	Acc@5: 97.422,	Params: 304.3M,	GFLOPS: 61.55
python train_feature_density_estimator.py "checkpoint/Flow_ViT_L_16_SWAG_LINEAR_V1.pt" "ViT_L_16_SWAG_LINEAR_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_L_16_SWAG_LINEAR_V1.pt" "ViT_L_16_SWAG_LINEAR_V1" "results/imagenet_results.csv"

# ViT_L_32
# Acc@1: 76.972,	Acc@5: 93.07,	Params: 306.5M,	GFLOPS: 15.38
python train_feature_density_estimator.py "checkpoint/Flow_ViT_L_32.pt" "ViT_L_32" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_ViT_L_32.pt" "ViT_L_32" "results/imagenet_results.csv"

# Wide_ResNet101_2_V1
# Acc@1: 78.848,	Acc@5: 94.284,	Params: 126.9M,	GFLOPS: 22.75
python train_feature_density_estimator.py "checkpoint/Flow_Wide_ResNet101_2_V1.pt" "Wide_ResNet101_2_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Wide_ResNet101_2_V1.pt" "Wide_ResNet101_2_V1" "results/imagenet_results.csv"

# Wide_ResNet101_2_V2
# Acc@1: 82.51,	Acc@5: 96.02,	Params: 126.9M,	GFLOPS: 22.75
python train_feature_density_estimator.py "checkpoint/Flow_Wide_ResNet101_2_V2.pt" "Wide_ResNet101_2_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Wide_ResNet101_2_V2.pt" "Wide_ResNet101_2_V2" "results/imagenet_results.csv"

# Wide_ResNet50_2_V1
# Acc@1: 78.468,	Acc@5: 94.086,	Params: 68.9M,	GFLOPS: 11.4
python train_feature_density_estimator.py "checkpoint/Flow_Wide_ResNet50_2_V1.pt" "Wide_ResNet50_2_V1" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Wide_ResNet50_2_V1.pt" "Wide_ResNet50_2_V1" "results/imagenet_results.csv"

# Wide_ResNet50_2_V2
# Acc@1: 81.602,	Acc@5: 95.758,	Params: 68.9M,	GFLOPS: 11.4
python train_feature_density_estimator.py "checkpoint/Flow_Wide_ResNet50_2_V2.pt" "Wide_ResNet50_2_V2" --seed=1
python test_feature_density_estimator.py "checkpoint/Flow_Wide_ResNet50_2_V2.pt" "Wide_ResNet50_2_V2" "results/imagenet_results.csv"

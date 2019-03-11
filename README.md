# PyTorch implementation of UNet++ (Nested U-Net)
This repository contains code for a image segmentation model based on [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) implemented in PyTorch.

## Requirements
- Python 3.6
- PyTorch 0.4
- scikit-learn 0.20
- scikit-image 0.14
- OpenCV 3

## Training on [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) dataset
1. Download dataset from [here](https://www.kaggle.com/c/data-science-bowl-2018/data) to input/
2. Preprocess.
```
python preprocess_dsb2018.py
```
3. Train the model.
```
python train.py --dataset dsb2018_96 --arch NestedUNet
```
4. Evaluate.
```
python test.py --name dsb2018_96_NestedUNet_wDS
```
### (Optional) Using LovaszHingeLoss
1. Clone LovaszSoftmax from [bermanmaxim/LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax).
```
git clone https://github.com/bermanmaxim/LovaszSoftmax.git
```
2. Train the model with LovaszHingeLoss.
```
python train.py --dataset dsb2018_96 --arch NestedUNet --loss LovaszHingeLoss
```

## Training on original dataset
make sure to put the files as the following structure:
```
<dataset name>
├── images
|   ├── 0a7e06.jpg
│   ├── 0aab0a.jpg
│   ├── 0b1761.jpg
│   ├── ...
|
└── masks
    ├── 0a7e06.png
    ├── 0aab0a.png
    ├── 0b1761.png
    ├── ...
```

1. Train the model.
```
python train.py --dataset <dataset name> --arch NestedUNet --image-ext jpg --mask-ext png
```
2. Evaluate.
```
python test.py --name <dataset name>_NestedUNet_wDS
```

## Results
### DSB2018 (96x96)

Here is the results on DSB2018 dataset (96x96) with LovaszHingeLoss.

| Model                           |   IoU   |  Loss   |
|:------------------------------- |:-------:|:-------:|
| U-Net                           |  0.839  |  0.365  |
| Nested U-Net                    |  0.842  |**0.354**|
| Nested U-Net w/ Deepsupervision |**0.843**|  0.362  |

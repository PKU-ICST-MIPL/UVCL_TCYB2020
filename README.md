# Introduction
This is the source code of our TCYB 2020 paper "Unsupervised Visual-textual Correlation Learning with Fine-grained Semantic Alignment". Please cite the following paper if you use our code.

Yuxin Peng, Zhaoda Ye, Jinwei Qi and Yunkan Zhuo, "Unsupervised Visual-textual Correlation Learning with Fine-grained Semantic Alignment", IEEE Transactions on Cybernetics (TCYB), DOI:10.1109/TCYB.2020.3015084, Sep. 2020.

# Dependency

The main code is implemented with pytorch.

# Data Preparation

## Obtain the image and text entity

We adopt the object detection model (https://github.com/peteanderson80/bottom-up-attention) and SceneGraphParser (https://github.com/vacancy/SceneGraphParser) to extracted the image and text entity. The entity files for flickr can be find in ./data folder.

## Unsupervised Caption Generation

1) Based on IOU: Uses the script in ./caption/IOU

2) Based on Generation model: We adopt OpenNMT (https://github.com/OpenNMT/OpenNMT) for Caption generation. The generation captions for flickr can be find in ./data folder.

# Usage

## Local representation learning

cd ./Cross-modal/local

Train the model: sh script.sh

Test and obtain the similarity score: python test.py

## Global representation learning

cd ./Cross-modal/global

Train the model: sh script.sh

Test and obtain the global representation: python test.py

## Merge and test

Use the script in ./Merge
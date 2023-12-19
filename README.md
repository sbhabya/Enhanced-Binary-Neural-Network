# Enhanced Binary Neural Network

Re-implementation of an improved BNN from the ReActNet paper ([Zechun Liu et al. (ECCV 2020)](https://arxiv.org/abs/2003.03488)).
ReActNet enhances plain BNNs by generalizing the activation functions by introducing learnable parameters. RPReLU and RSign activation functions are used in ReactNet.

This project implements knowledge distillation using a ResNet-18 model trained on CIFAR-10 as the model teacher, for training the ReactNet student model on the same dataset. 
ResNet-18 model is trained on CIFAR-10 from scratch, using transfer learning. All but the last 2 layers are frozen to perform this transfer learning. 

The baseline model for ReactNet and KDLoss code were taken from the original [ReActNet codebase](https://github.com/liuzechun/ReActNet).

## Instructions
1. Run the Jupyter notebook `resnet18.ipynb`. This will train ResNet18 on CIFAR-10 and save the model. 
2. Next, run `python train.py`. This will use the trained ResNet18 as model teacher and train ReActNet as student.

## Experimental results on CIFAR-10
* CIFAR-10 was used to validate the model after training for 10 epochs, and this gave ~58% top-1 and ~93% top-5 accuracy. 

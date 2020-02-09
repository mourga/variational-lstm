# Variational LSTM & MC dropout with Pytorch
This repository is based on the Salesforce code of [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm/). 

There is no official PyTorch code for the _Variational RNNs_ proposed by Gal and Ghahramani in the paper [A Theoretically Grounded Application of Dropout in
Recurrent Neural Networks](https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf). In this repository, we implement an RNN-based classifier with (optionally) a self-attention mechanism. We apply different variants of dropout to all layers, in order to implement a model equivalent to a Bayesian NN, using Monte Carlo dropout during inference (test time).

The difference bwtween naive and variational dropout is illustrated in the following figure.
![Alt text](https://user-images.githubusercontent.com/28900064/74105210-b2a8d800-4b53-11ea-9def-ddb79b9d5d45.png)
<!--
PyTorch implementation for Variational LSTM and Monte Carlo dropout.
-->


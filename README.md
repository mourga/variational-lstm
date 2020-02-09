# Variational LSTM & MC dropout with Pytorch
This repository is based on the Salesforce code of [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm/). 

There is no official PyTorch code for the _Variational RNNs_ proposed by Gal and Ghahramani in the paper [A Theoretically Grounded Application of Dropout in
Recurrent Neural Networks](https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf). In this repository, we implement an RNN-based classifier with (optionally) a self-attention mechanism. We apply different variants of dropout to all layers, in order to implement a model equivalent to a Bayesian NN, using Monte Carlo dropout during inference (test time).


![variational dropout](https://user-images.githubusercontent.com/28900064/74105210-b2a8d800-4b53-11ea-9def-ddb79b9d5d45.png)

Each *square* represents an RNN unit, *horizontal arrows* represent time dependence (recurrent connections), *vertical arrows* represent the input and output to each RNN unit, coloured connections represent dropped-out inputs, with *different colours corresponding to different dropout masks*. Dashed lines correspond to standard connections with no dropout. Current techniques (naive dropout, left) use different masks at differenttime steps, with no dropout on the recurrent layers. The proposed technique (Variational RNN, right) uses the **same dropout mask at each time step, including the recurrent layers**.

## Environment setup

First create a conda environment:

```
conda create -n var_env python=3.6
conda activate var_env
```
Then install the required torch package:

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
And finally the rest of the requirements:
```
pip install -r requirements.txt
```

## Test the model!
<!--
PyTorch implementation for Variational LSTM and Monte Carlo dropout.
-->


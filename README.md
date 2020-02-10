# Variational LSTM & MC dropout with PyTorch
This repository is based on the Salesforce code for [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm/). 

There is no official PyTorch code for the _Variational RNNs_ proposed by Gal and Ghahramani in the paper [A Theoretically Grounded Application of Dropout in
Recurrent Neural Networks](https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf). In this repository, we implement an RNN-based classifier with (optionally) a self-attention mechanism. We apply different variants of dropout to all layers, in order to implement a _model equivalent to a Bayesian NN_, using Monte Carlo dropout during inference (test time).


![variational dropout](https://user-images.githubusercontent.com/28900064/74105210-b2a8d800-4b53-11ea-9def-ddb79b9d5d45.png)

Each *square* represents an RNN unit, *horizontal arrows* represent time dependence (recurrent connections), *vertical arrows* represent the input and output to each RNN unit, coloured connections represent dropped-out inputs, with *different colours corresponding to different dropout masks*. Dashed lines correspond to standard connections with no dropout. Current techniques (naive dropout, left) use different masks at differenttime steps, with no dropout on the recurrent layers. The proposed technique (Variational RNN, right) uses the **same dropout mask at each time step, including the recurrent layers**. (Figure taken from the [paper](https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf)).

## Software Requirements
Python 3 and PyTorch=1.4.0 are required for the current codebase.

### Environment setup (optional)

First create a conda environment:

```
conda create -n var_env python=3
conda activate var_env
```
Then install the required [PyTorch](https://pytorch.org/) package:

```
conda install pytorch=1.4.0 python torchvision cudatoolkit=10.1 -c pytorch
```
And finally the rest of the requirements:
```
pip install -r requirements.txt
```

## Test the model!
<!--
PyTorch implementation for Variational LSTM and Monte Carlo dropout.
-->
Run `test_variational_rnn.py` to do a forward pass of the Variational RNN model.

## Dropout options
In the code you will see that there are various types of dropout that we can apply to different parts of our RNN-based model.
* `dropoute` is dropout to the _embedding_ layer 
* `dropouti` is dropout to the _inputs_ of the RNN
* `dropoutw` is dropout to the _recurrent_ layers of the RNN
* `dropouto` is dropout to the _outputs_ of the RNN

## Troubleshooting
* If you face the error `PackageNotFoundError: Package missing in current linux-64 channels: - cudatoolkit 10.1*`, first run `conda install -c anaconda cudatoolkit=10.1` and then `conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch`.

* If you face the error `PackageNotFoundError: Dependency missing in current linux-64 channels:` 
  `- pytorch 1.4.0* -> mkl >=2018`, try running `conda install -c anaconda mkl` and then again `conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch`.
  
* If you face the error `ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.21' not found` run `conda install libgcc`.

In order to make sure that all the required packages & dependecies are correct, I recommend running the following:
```
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())
print('device: {}'.format(device))
```
that should give:
```
torch: 1.4.0
Cuda: 10.1
CuDNN: 7603
device: cpu (or gpu)
```


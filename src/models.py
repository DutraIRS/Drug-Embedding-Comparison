import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational AutoEncoder

    Args:
        nn (_type_): _description_
    """
    ...

class GSA(nn.Module):
    """Graph Self-Attention

    Args:
        nn (_type_): _description_
    """
    ...

class GNN(nn.Module):
    """Graph Neural Network

    Args:
        nn (_type_): _description_
    """
    ...

class MPGNN(nn.Module):
    """Message Passing Graph Neural Network

    Args:
        nn (_type_): _description_
    """
    ...

class FP(nn.Module):
    """FingerPrint

    Args:
        nn (_type_): _description_
    """
    ...

class FCNN(nn.Module):
    """Fully Connected Neural Network

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_dim=10, hidden_dim=5, num_hidden=2, output_dim=1, activation=nn.Identity()):
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(activation)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
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
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 5, num_hidden: int = 2,
                output_dim: int = 1, activation: nn.Modeule = nn.Identity()):
        """Initialize the FCNN model

        Args:
            input_dim (int, optional): Size of the input tensor. Defaults to 10.
            hidden_dim (int, optional): Size of the hidden layer(s). Defaults to 5.
            num_hidden (int, optional): Number of hidden layers. Defaults to 2.
            output_dim (int, optional): Size of the output tensor. Defaults to 1.
            activation (nn.Module, optional): Activation function for the output layer. Defaults to nn.Identity().
        """
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FCNN model

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        
        return x
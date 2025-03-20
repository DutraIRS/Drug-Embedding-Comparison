import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layers import GCNConv, GATConv, MessagePassing

from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational AutoEncoder
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, latent_dim: int = 8):
        """Initialize the VAE model

        Args:
            input_dim (int, optional): Size of input tensor. Defaults to 128.
            hidden_dim (int, optional): Size of hidden layer. Defaults to 64.
            latent_dim (int, optional): Size of latent space. Defaults to 8.
        """
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Bottleneck
        self.fc4 = nn.Linear(hidden_dim, latent_dim*2)
        
        # Decoder
        self.fc5 = nn.Linear(latent_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, input_dim)
        self.fc8 = nn.Linear(input_dim, input_dim)
    
    def encode(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the input tensor

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            mu (torch.Tensor): Mean of latent space
            logvar (torch.Tensor): Log variance of latent space
        """
        h = F.relu(self.fc1(X))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        # Split the output into mean and log-variance
        mu, logvar = torch.chunk(self.fc4(h), 2, dim=1)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Create a sample from the latent space

        Args:
            mu (torch.Tensor): Mean of latent space
            logvar (torch.Tensor): Log variance of latent space

        Returns:
            torch.Tensor: Sample from the latent space
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn(std.size())
        z = mu + eps*std
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent space into the representation

        Args:
            z (torch.Tensor): Sample from the latent space

        Returns:
            torch.Tensor: Reconstructed input tensor
        """
        h = F.relu(self.fc5(z))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        
        return h
    
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            y (torch.Tensor): Reconstructed input tensor
            mu (torch.Tensor): Mean of latent space
            logvar (torch.Tensor): Log variance of latent space
        """
        mu, logvar = self.encode(X)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        
        return y, mu, logvar

class GNN(nn.Module):
    """Graph Neural Network
    """
    def __init__(self, layer: Literal["GCNConv", "GATConv", "MessagePassing"],
                num_layers: int, input_dim: int, output_dim: int, **kwargs):
        """Initialize a GNN model with specified type of layer

        Args:
            layer (Literal["GCNConv", "GATConv", "MessagePassing"]): Type of layer to be used
            num_layers (int): Number of layers to chain
            input_dim (int): Initial number of features in each node
            output_dim (int): Final number of features in each node
        
        Kwargs:
            hidden_dim (int): Size of the hidden layer(s) for each GNN layer. Used for GATConv and MessagePassing
            num_hidden (int): Number of hidden layers for each GNN layer. Used for MessagePassing
            activation (nn.Module): Activation function for the output layer of each GNN layer
        """
        super(GNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        if layer == "GCNConv":
            self.layers.append(
                    GCNConv(input_dim, output_dim)
                )
            
            for _ in range(num_layers - 1):
                self.layers.append(
                    GCNConv(output_dim, output_dim)
                )
        elif layer == "GATConv":
            if 'hidden_dim' not in kwargs.keys():
                raise TypeError("GATConv layer requires kwarg 'hidden_dim'")
            
            self.layers.append(
                    GATConv(input_dim, output_dim, kwargs['hidden_dim'])
                )
            
            for _ in range(num_layers - 1):
                self.layers.append(
                    GATConv(output_dim, output_dim, kwargs['hidden_dim'])
                )
        elif layer == "MessagePassing":
            if 'hidden_dim' not in kwargs.keys():
                raise TypeError("MessagePassing layer requires kwarg 'hidden_dim'")
            if 'num_hidden' not in kwargs.keys():
                raise TypeError("MessagePassing layer requires kwarg 'num_hidden'")
            if 'activation' not in kwargs.keys():
                raise TypeError("MessagePassing layer requires kwarg 'activation'")
            
            self.layers.append(
                    MessagePassing(input_dim, kwargs["hidden_dim"],
                        kwargs["num_hidden"], output_dim, kwargs["activation"])
                )
            
            for _ in range(num_layers - 1):
                self.layers.append(
                    MessagePassing(output_dim, kwargs["hidden_dim"],
                        kwargs["num_hidden"], output_dim, kwargs["activation"])
                )
        else:
            raise ValueError("Layer type not implemented")
    
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GNN model

        Args:
            X (torch.Tensor): Input graph of shape [num_nodes, input_dim]
            A (torch.Tensor): Adjacency matrix of shape [num_nodes, num_nodes]

        Returns:
            torch.Tensor: Output graph of shape [num_nodes, output_dim]
        """
        for layer in self.layers:
            X = layer(X, A)
        
        return X

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
                output_dim: int = 1, activation: nn.Module = nn.Identity()):
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
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FCNN model

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.layers:
            X = layer(X)
        
        return X
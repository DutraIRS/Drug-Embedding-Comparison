from typing import Tuple

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
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the input tensor

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            mu (torch.Tensor): Mean of latent space
            logvar (torch.Tensor): Log variance of latent space
        """
        h = F.relu(self.fc1(x))
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            y (torch.Tensor): Reconstructed input tensor
            mu (torch.Tensor): Mean of latent space
            logvar (torch.Tensor): Log variance of latent space
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        
        return y, mu, logvar

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
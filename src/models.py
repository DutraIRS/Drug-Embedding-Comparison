import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layers import GCNConv, GATConv, MessagePassing

from typing import Tuple, Literal

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational AutoEncoder
    """
    def __init__(self, input_dim: int = 100, hidden_dim: int = 64, latent_dim: int = 8):
        """Initialize the VAE model

        Args:
            input_dim (int, optional): Size of input tensor. Defaults to 100.
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
        
        self.predictor = FCNN(input_dim=latent_dim, hidden_dim=64, num_hidden=3, output_dim=994, activation=nn.ReLU())
    
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
        X_reconstructed = self.decode(z)
        
        y = self.predictor(z)

        return X_reconstructed, y, mu, logvar

class Transformer(nn.Module):
    """Transformer Encoder for molecular graphs
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3, dim_feedforward: int = 512, output_dim: int = 994):
        """Initialize the Transformer model

        Args:
            input_dim (int): Number of input features per atom
            d_model (int, optional): Dimension of embeddings. Defaults to 128.
            nhead (int, optional): Number of attention heads. Defaults to 8.
            num_layers (int, optional): Number of encoder layers. Defaults to 3.
            dim_feedforward (int, optional): Dimension of feedforward network. Defaults to 512.
            output_dim (int, optional): Number of outputs (side effects). Defaults to 994.
        """
        super(Transformer, self).__init__()
        
        # Project input features to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.max_seq_len = 200  # Maximum number of atoms in a molecule
        self.positional_encoding = nn.Parameter(torch.randn(1, self.max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head (after global pooling)
        self.prediction_head = FCNN(
            input_dim=d_model,
            hidden_dim=d_model,
            num_hidden=2,
            output_dim=output_dim,
            activation=nn.Identity()
        )
    
    def forward(self, X: torch.Tensor, A: torch.Tensor = None) -> torch.Tensor:
        """Forward pass

        Args:
            X (torch.Tensor): Input features of shape [num_atoms, input_dim]
            A (torch.Tensor, optional): Adjacency matrix (not used, for compatibility)

        Returns:
            torch.Tensor: Predictions of shape [output_dim]
        """
        # X shape: [num_atoms, input_dim]
        num_atoms = X.size(0)
        
        # Project to d_model
        X = self.input_projection(X)  # [num_atoms, d_model]
        
        # Add batch dimension and positional encoding
        X = X.unsqueeze(0)  # [1, num_atoms, d_model]
        X = X + self.positional_encoding[:, :num_atoms, :]  # Add positional encoding
        
        # Pass through transformer encoder
        X = self.transformer_encoder(X)  # [1, num_atoms, d_model]
        
        # Global mean pooling over atoms
        X = X.mean(dim=1)  # [1, d_model]
        X = X.squeeze(0)  # [d_model]
        
        # Prediction head
        y = self.prediction_head(X)  # [output_dim]
        
        return y

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
    """Fingerprint-based model using RDKit Morgan Fingerprints
    """
    def __init__(self, radius: int = 2, n_bits: int = 2048, output_dim: int = 994):
        """Initialize the Fingerprint model

        Args:
            radius (int, optional): Radius for Morgan fingerprint. Defaults to 2.
            n_bits (int, optional): Number of bits in fingerprint. Defaults to 2048.
            output_dim (int, optional): Number of outputs (side effects). Defaults to 994.
        """
        super(FP, self).__init__()
        
        self.radius = radius
        self.n_bits = n_bits
        
        # Single linear layer for prediction
        self.linear = nn.Linear(n_bits, output_dim)
    
    def forward(self, smiles: str, X: torch.Tensor = None, A: torch.Tensor = None) -> torch.Tensor:
        """Forward pass

        Args:
            smiles (str): SMILES string of the molecule
            X (torch.Tensor, optional): Not used, for compatibility
            A (torch.Tensor, optional): Not used, for compatibility

        Returns:
            torch.Tensor: Predictions of shape [output_dim]
        """
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # Generate Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        
        # Convert to tensor
        fp_array = torch.zeros(self.n_bits)
        fp_array[:] = torch.tensor(list(fp), dtype=torch.float32)
        
        # Predict
        y = self.linear(fp_array)
        
        return y

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
    
    def forward(self, X: torch.Tensor, A: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the FCNN model

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.layers:
            X = layer(X)
        
        return X
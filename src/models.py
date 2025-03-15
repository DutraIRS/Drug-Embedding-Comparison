import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational AutoEncoder

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_dim=128, hidden_dim=64, latent_dim=8):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, latent_dim*2)
        self.fc5 = nn.Linear(latent_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, input_dim)
        self.fc8 = nn.Linear(input_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        mu, logvar = torch.chunk(self.fc4(h), 2, dim=1)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn(std.size())
        z = mu + eps*std
        
        return z
    
    def decode(self, z):
        h = F.relu(self.fc5(z))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        
        return h
    
    def forward(self, x):
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

    Args:
        nn (_type_): _description_
    """
    ...
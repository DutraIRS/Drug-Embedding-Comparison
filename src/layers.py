import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, activation = nn.ReLU()):
        super(GCNConv, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize weights from a normal distribution with 0 mean and low variance
        self.W = nn.Parameter(torch.randn([input_dim, output_dim]) / 10)
    
    def forward(self, X, A):
        A += torch.eye(A.size()[0])
        
        D_tilde = torch.diag(A.sum(dim=0) ** (-1/2))
        A_tilde = D_tilde  @ A @ D_tilde
        X_tilde = A_tilde @ X @ self.W
        
        return self.activation(X_tilde)

class GATConv(nn.Module):
    def __init__(self):
        super(GATConv, self).__init__()
        
        ...
    
    def forward(self, X, A):
        ...
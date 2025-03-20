import torch
import torch.nn as nn

class GCNConv(nn.Module):
    """Graph Convolutional Network Convolution Layer
    """
    def __init__(self, input_dim: int, output_dim: int, activation: nn.Module = nn.ReLU()):
        """Initialize the GCNConv layer

        Args:
            input_dim (int): Number of features in input graph
            output_dim (int): Number of features in output graph
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
        """
        super(GCNConv, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize weights from a normal distribution with 0 mean and low variance
        self.W = nn.Parameter(torch.randn([input_dim, output_dim]) / 10)
    
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GCNConv layer

        Args:
            X (torch.Tensor): Input graph of shape [num_nodes, input_dim]
            A (torch.Tensor): Adjacency matrix of shape [num_nodes, num_nodes]

        Returns:
            torch.Tensor: Output graph of shape [num_nodes, output_dim]
        """
        A += torch.eye(A.size()[0])
        
        D_tilde = torch.diag(A.sum(dim=0) ** (-1/2))
        A_tilde = D_tilde  @ A @ D_tilde
        X_tilde = A_tilde @ X @ self.W
        
        return self.activation(X_tilde)

class GATConv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(GATConv, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights from a normal distribution with 0 mean and low variance
        self.Wq = nn.Parameter(torch.randn([input_dim, hidden_dim]) / 10)
        self.Wk = nn.Parameter(torch.randn([input_dim, hidden_dim]) / 10)
        self.Wv = nn.Parameter(torch.randn([input_dim, output_dim]) / 10)
    
    def masked_attention(self, Q, K, V, A):
        X = Q @ K.T
        X = X + torch.nan_to_num(- torch.inf * (1 - A), nan=0) # Masking before softmax
        X = torch.softmax(X, dim=1)
        X = X @ V
        
        return X
    
    def forward(self, X, A):
        A += torch.eye(A.size()[0])
        
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        
        return self.masked_attention(Q, K, V, A)
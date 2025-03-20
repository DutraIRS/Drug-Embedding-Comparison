import torch
import torch.nn as nn

from src import models

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
    """Graph Self-Attention Convolutional Layer
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        """Initialize the GATConv layer

        Args:
            input_dim (int): Number of features in input graph
            output_dim (int): Number of features in output graph
            hidden_dim (int): Dimension of Wq and Wk weight tensors
        """
        super(GATConv, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights from a normal distribution with 0 mean and low variance
        self.Wq = nn.Parameter(torch.randn([input_dim, hidden_dim]) / 10)
        self.Wk = nn.Parameter(torch.randn([input_dim, hidden_dim]) / 10)
        self.Wv = nn.Parameter(torch.randn([input_dim, output_dim]) / 10)
    
    def masked_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        A: torch.Tensor) -> torch.Tensor:
        """Perform masked attention
        
        No attention is paid to non-neighbor nodes.

        Args:
            Q (torch.Tensor): Queries matrix
            K (torch.Tensor): Keys matrix
            V (torch.Tensor): Values matrix
            A (torch.Tensor): Adjacency matrix of shape [num_nodes, num_nodes]

        Returns:
            torch.Tensor: Update tokens embeddings
        """
        X = Q @ K.T
        X = X + torch.nan_to_num(- torch.inf * (1 - A), nan=0) # Masking before softmax
        X = torch.softmax(X, dim=1) # -inf becomes 0 after softmax
        X = X @ V
        
        return X
    
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GATConv layer

        Args:
            X (torch.Tensor): Input graph of shape [num_nodes, input_dim]
            A (torch.Tensor): Adjacency matrix of shape [num_nodes, num_nodes]

        Returns:
            torch.Tensor: Output graph of shape [num_nodes, output_dim]
        """
        A += torch.eye(A.size()[0])
        
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        
        return self.masked_attention(Q, K, V, A)

class MessagePassing(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden, output_dim, activation):
        super(MessagePassing, self).__init__()
        
        self.MLP = models.FCNN(input_dim, hidden_dim, num_hidden, output_dim, activation)
    
    def forward(self, X, A):
        A += torch.eye(A.size()[0])
        
        return A @ self.MLP(X) # FCNN treats rows as independent datapoints in a batch
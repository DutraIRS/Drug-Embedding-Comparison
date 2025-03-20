import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import layers

import torch
import torch.nn as nn

class TestGCNConv:
    def test_init(self):
        """
        Test the initialization of the GCNConv layer
        """
        conv_layer = layers.GCNConv(2, 2)
        
        assert conv_layer is not None
        assert isinstance(conv_layer, nn.Module)
        assert isinstance(conv_layer, layers.GCNConv)
    
    def test_output_dim_case_shrink(self):
        """
        Test the dimension of the output in the case of a shrinkage
        """
        tensor_identity_two = torch.Tensor([[1, 0], [0, 1]])
        tensor_rectangular = torch.Tensor([[1.2, 1.5, -1.0, 9.1],
                                            [0, 5.121, -4.001, 1]])
        
        conv_layer = layers.GCNConv(input_dim = 4, output_dim = 2, activation = nn.ReLU())
        
        output = conv_layer(tensor_rectangular, tensor_identity_two)
        
        assert isinstance(output, torch.Tensor)
        assert output.size()[0] == 2
        assert output.size()[1] == 2
        assert torch.all(output >= 0)
    
    def test_output_dim_case_expand(self):
        """
        Test the dimension of the output in the case of an expansion
        """
        tensor_identity_two = torch.Tensor([[1, 0], [0, 1]])
        
        conv_layer = layers.GCNConv(input_dim = 2, output_dim = 4, activation = nn.ReLU())
        
        output = conv_layer(tensor_identity_two, tensor_identity_two)
        
        assert isinstance(output, torch.Tensor)
        assert output.size()[0] == 2
        assert output.size()[1] == 4
        assert torch.all(output >= 0)
    
    def test_disconnected_graph(self):
        """
        Test if using a graph without links breaks the code
        """
        tensor_rectangular = torch.Tensor([[1.2, 1.5, -1.0, 9.1],
                                            [0, 5.121, -4.001, 1]])
        input_adjacency_matrix = torch.zeros(2, 2)
        
        conv_layer = layers.GCNConv(input_dim = 4, output_dim = 1)
        
        output = conv_layer(tensor_rectangular, input_adjacency_matrix)
        
        assert isinstance(output, torch.Tensor)
        assert output.size()[0] == 2
        assert output.size()[1] == 1
    
    def test_number_of_parameters(self):
        """
        Test the number of trainable parameters in the layer
        """
        conv_layer = layers.GCNConv(input_dim = 2, output_dim = 2)
        
        n_params = sum(p.numel() for p in conv_layer.parameters())
        
        assert n_params == 4

class TestGATConv:
    def test_init(self):
        """
        Test the initialization of the GATConv layer
        """
        conv_layer = layers.GATConv(2, 2, 2)
        
        assert conv_layer is not None
        assert isinstance(conv_layer, nn.Module)
        assert isinstance(conv_layer, layers.GATConv)
    
    def test_number_of_parameters(self):
        """
        Test the number of trainable parameters in the layer
        """
        conv_layer = layers.GATConv(input_dim = 3, output_dim = 1, hidden_dim = 5)
        
        n_params = sum(p.numel() for p in conv_layer.parameters())
        
        assert n_params == 3*5 + 3*5 + 3*1
    
    def test_masked_attention(self):
        """
        Test the masked attention operation with mock tensors
        """
        eye = torch.eye(5)
        
        conv_layer = layers.GATConv(input_dim = 5, output_dim = 5, hidden_dim = 5)
        
        output = conv_layer.masked_attention(eye, eye, eye, eye)
        
        assert torch.all(output == torch.eye(5))
    
    def test_forward(self):
        """
        Test the forward pass of the layer
        """
        eye = torch.eye(3)
        zeros = torch.zeros(3, 3)
        
        conv_layer = layers.GATConv(input_dim = 3, output_dim = 3, hidden_dim = 3)
        
        conv_layer.Wq = nn.Parameter(eye)
        conv_layer.Wk = nn.Parameter(eye)
        conv_layer.Wv = nn.Parameter(eye)
        
        output = conv_layer(eye, zeros)
        
        assert torch.all(output == eye)
        
    def test_output_dim_case_shrink(self):
        """
        Test the dimension of the output in the case of a shrinkage
        """
        tensor_identity_two = torch.Tensor([[1, 0], [0, 1]])
        tensor_rectangular = torch.Tensor([[1.2, 1.5, -1.0, 9.1],
                                            [0, 5.121, -4.001, 1]])
        
        conv_layer = layers.GATConv(input_dim = 4, output_dim = 2, hidden_dim = 3)
        
        output = conv_layer(tensor_rectangular, tensor_identity_two)
        
        assert isinstance(output, torch.Tensor)
        assert output.size()[0] == 2
        assert output.size()[1] == 2
    
    def test_output_dim_case_expand(self):
        """
        Test the dimension of the output in the case of an expansion
        """
        tensor_identity_two = torch.Tensor([[1, 0], [0, 1]])
        
        conv_layer = layers.GATConv(input_dim = 2, output_dim = 4, hidden_dim = 3)
        
        output = conv_layer(tensor_identity_two, tensor_identity_two)
        
        assert isinstance(output, torch.Tensor)
        assert output.size()[0] == 2
        assert output.size()[1] == 4
    
    def test_disconnected_graph(self):
        """
        Test if using a graph without links breaks the code
        """
        tensor_rectangular = torch.Tensor([[1.2, 1.5, -1.0, 9.1],
                                            [0, 5.121, -4.001, 1]])
        input_adjacency_matrix = torch.zeros(2, 2)
        
        conv_layer = layers.GATConv(input_dim = 4, output_dim = 1, hidden_dim = 3)
        
        output = conv_layer(tensor_rectangular, input_adjacency_matrix)
        
        assert isinstance(output, torch.Tensor)
        assert output.size()[0] == 2
        assert output.size()[1] == 1

class TestMessagePassing:
    def test_init(self):
        """
        Test the initialization of the MessagePassing layer
        """
        conv_layer = layers.MessagePassing(2, 2, 2, 2, nn.ReLU())
        
        assert conv_layer is not None
        assert isinstance(conv_layer, nn.Module)
        assert isinstance(conv_layer, layers.MessagePassing)
    
    def test_forward(self):
        """
        Test the forward pass of the layer
        """
        eye = torch.eye(3)
        zeros = torch.zeros(3, 3)
        
        mp_layer = layers.MessagePassing(input_dim=10, hidden_dim=5, num_hidden=2,
                                            output_dim=1, activation=nn.ReLU())
        
        mp_layer.MLP = nn.Identity()
        
        output = mp_layer(eye, zeros)
        
        assert torch.all(output == eye)
    
    def test_number_of_parameters(self):
        """
        Test the number of parameters in the MessagePassing layer
        """
        mp_layer = layers.MessagePassing(input_dim=10, hidden_dim=5, num_hidden=2,
                                            output_dim=1, activation=nn.ReLU())
        n_params = sum(p.numel() for p in mp_layer.parameters() if p.requires_grad)
        
        assert n_params == 11*5 + 6*5 + 6*1
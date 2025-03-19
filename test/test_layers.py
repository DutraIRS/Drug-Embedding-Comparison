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
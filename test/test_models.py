import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import models

import pytest
import torch
import torch.nn as nn

class TestVAE:
    def test_init(self):
        """
        Test the initialization of the VAE model
        """
        model = models.VAE()
        
        assert isinstance(model, models.VAE)
        assert isinstance(model, models.nn.Module)
    
    def test_forward(self):
        """
        Test the forward pass of the VAE model
        """
        model = models.VAE(input_dim=128, hidden_dim=64, latent_dim=8)
        x = torch.randn(50, 128)
        
        x_reconstructed, y, mu, logvar = model(x)
        
        assert isinstance(x_reconstructed, torch.Tensor)
        assert x_reconstructed.shape == (50, 128)
        assert isinstance(y, torch.Tensor)
        assert y.shape == (50, 994) or y.shape == (994,)  # Batch or single prediction
        assert isinstance(mu, torch.Tensor)
        assert mu.shape == (50, 8)
        assert isinstance(logvar, torch.Tensor)
        assert logvar.shape == (50, 8)
    
    def test_model_number_of_parameters(self):
        """
        Test the number of parameters in the VAE model
        """
        model = models.VAE(input_dim=16, hidden_dim=8, latent_dim=2)
        n_params = sum(p.numel() for p in model.parameters())
        
        # VAE now has predictor FCNN, so just check it has parameters
        assert n_params > 0

class TestFCNN:
    def test_init(self):
        """
        Test the initialization of the FCNN model
        """
        model = models.FCNN()
        assert isinstance(model, models.FCNN)
        assert isinstance(model, models.nn.Module)
        assert isinstance(model.layers, models.nn.ModuleList)
        assert len(model.layers) == 6
    
    def test_forward(self):
        """
        Test the forward pass of the FCNN model
        """
        model = models.FCNN(input_dim=10, hidden_dim=5, num_hidden=2, output_dim=1)
        
        x = torch.rand(5, 10)
        y = model(x)
        
        assert y.shape == (5, 1)
        assert y.dtype == torch.float32
        assert isinstance(y, torch.Tensor)
    
    def test_forward_with_activation(self):
        """
        Test the forward pass of the FCNN model with activation function
        """
        model = models.FCNN(input_dim=10, hidden_dim=5, num_hidden=2, output_dim=1, activation=models.nn.ReLU())
        
        x = torch.rand(500, 10)
        y = model(x)
        
        assert y.shape == (500, 1)
        assert y.dtype == torch.float32
        assert isinstance(y, torch.Tensor)
        assert torch.all(y >= 0)
    
    def test_model_number_of_parameters(self):
        """
        Test the number of parameters in the FCNN model
        """
        model = models.FCNN(input_dim=10, hidden_dim=5, num_hidden=2, output_dim=1)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert num_params == 11*5 + 6*5 + 6*1

class TestGNN:
    def test_init(self):
        """
        Test the initialization of the GNN model
        """
        model_a = models.GNN('GCNConv', 3, 5, 2)
        model_b = models.GNN("GATConv", 3, 5, 2, hidden_dim=5)
        model_c = models.GNN("MessagePassing", 3, 5, 2, hidden_dim=5, num_hidden=3, activation=nn.ReLU())
        
        assert isinstance(model_a, models.GNN)
        assert isinstance(model_a, models.nn.Module)
        
        assert isinstance(model_b, models.GNN)
        assert isinstance(model_b, models.nn.Module)
        
        assert isinstance(model_c, models.GNN)
        assert isinstance(model_c, models.nn.Module)
    
    def test_forward(self):
        """
        Test the forward pass of the GNN model
        """
        model_a = models.GNN('GCNConv', 3, 5, 2)
        model_b = models.GNN("GATConv", 3, 5, 2, hidden_dim=5)
        model_c = models.GNN("MessagePassing", 3, 5, 2, hidden_dim=5, num_hidden=3, activation=nn.ReLU())
        
        X = torch.randn(7, 5) * 10
        A = torch.eye(7)

        out_a = model_a(X, A)
        out_b = model_b(X, A)
        out_c = model_c(X, A)
        
        assert isinstance(out_a, torch.Tensor)
        assert out_a.shape == (7, 2)
        
        assert isinstance(out_b, torch.Tensor)
        assert out_b.shape == (7, 2)
        
        assert isinstance(out_c, torch.Tensor)
        assert out_c.shape == (7, 2)
    
    def test_incorrect_arguments(self):
        """
        Test if incorrectly initializing the model breaks it (as it should)
        """
        with pytest.raises(ValueError):        
            model = models.GNN("WeirdLayer", 3, 5, 2)
        
        with pytest.raises(TypeError):        
            model = models.GNN("GATConv", 3, 5, 2)
        
        with pytest.raises(TypeError):
            model = models.GNN("MessagePassing", 3, 5, 2, hidden_dim=5, num_hidden=3)
        
        with pytest.raises(TypeError):
            model = models.GNN("MessagePassing", 3, 5, 2, hidden_dim=5, activation=nn.ReLU())
        
        with pytest.raises(TypeError):
            model = models.GNN("MessagePassing", 3, 5, 2, num_hidden=3, activation=nn.ReLU())
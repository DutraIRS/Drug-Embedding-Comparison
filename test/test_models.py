import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import models

import torch

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
        
        y, mu, logvar = model(x)
        
        assert isinstance(y, torch.Tensor)
        assert y.shape == (50, 128)
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
        
        assert n_params == 17*16 + 17*8 + 9*8 + 9*2*2 + 3*8 + 9*8 + 9*16 + 17*16
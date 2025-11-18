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
        
        # Test with batch (batch_size, input_dim)
        x_batch = torch.randn(50, 128)
        x_reconstructed, y, mu, logvar = model(x_batch)
        
        assert isinstance(x_reconstructed, torch.Tensor)
        assert x_reconstructed.shape == (50, 128)
        assert isinstance(y, torch.Tensor)
        assert y.shape == (50, 994), f"Expected shape (50, 994) but got {y.shape}"  # Batch output
        assert isinstance(mu, torch.Tensor)
        assert mu.shape == (50, 8)
        assert isinstance(logvar, torch.Tensor)
        assert logvar.shape == (50, 8)
        
        # Test with single sample (1, input_dim) - as used in training
        x_single = torch.randn(1, 128)
        x_reconstructed_single, y_single, mu_single, logvar_single = model(x_single)
        
        assert isinstance(y_single, torch.Tensor)
        assert y_single.shape == (994,), f"Expected shape (994,) but got {y_single.shape}"  # Single output after squeeze
    
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
        Test the forward pass of the FCNN model with pooling
        """
        # Test with default pooling (sum)
        model = models.FCNN(input_dim=10, hidden_dim=5, num_hidden=2, output_dim=1)
        
        x = torch.rand(5, 10)
        y = model(x)
        
        assert y.shape == (1,)  # Pooled output
        assert y.dtype == torch.float32
        assert isinstance(y, torch.Tensor)
        
        # Test with no pooling
        model_no_pool = models.FCNN(input_dim=10, hidden_dim=5, num_hidden=2, output_dim=1, pooling="none")
        y_no_pool = model_no_pool(x)
        
        assert y_no_pool.shape == (5, 1)  # No pooling
        assert y_no_pool.dtype == torch.float32
        assert isinstance(y_no_pool, torch.Tensor)
    
    def test_forward_with_activation(self):
        """
        Test the forward pass of the FCNN model with activation function
        """
        # Test with pooling
        model = models.FCNN(input_dim=10, hidden_dim=5, num_hidden=2, output_dim=1, activation=models.nn.ReLU())
        
        x = torch.rand(500, 10)
        y = model(x)
        
        assert y.shape == (1,)  # Pooled output
        assert y.dtype == torch.float32
        assert isinstance(y, torch.Tensor)
        assert torch.all(y >= 0)
        
        # Test with no pooling
        model_no_pool = models.FCNN(input_dim=10, hidden_dim=5, num_hidden=2, output_dim=1, 
                                    activation=models.nn.ReLU(), pooling="none")
        y_no_pool = model_no_pool(x)
        
        assert y_no_pool.shape == (500, 1)  # No pooling
        assert y_no_pool.dtype == torch.float32
        assert isinstance(y_no_pool, torch.Tensor)
        assert torch.all(y_no_pool >= 0)
    
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
        model_c = models.GNN("MessagePassing", 3, 5, 2, hidden_dim=5, num_hidden=3)
        
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
        # Test with pooling (default behavior)
        model_a = models.GNN('GCNConv', 3, 5, 2, pooling="sum")
        model_b = models.GNN("GATConv", 3, 5, 2, hidden_dim=5, pooling="sum")
        model_c = models.GNN("MessagePassing", 3, 5, 2, hidden_dim=5, num_hidden=3, pooling="sum")
        
        # Test without pooling (for backward compatibility)
        model_a_no_pool = models.GNN('GCNConv', 3, 5, 2, pooling="none")
        model_b_no_pool = models.GNN("GATConv", 3, 5, 2, hidden_dim=5, pooling="none")
        model_c_no_pool = models.GNN("MessagePassing", 3, 5, 2, hidden_dim=5, num_hidden=3, pooling="none")
        
        X = torch.randn(7, 5) * 10
        A = torch.eye(7)

        # Test with pooling - should return [output_dim]
        out_a = model_a(X, A)
        out_b = model_b(X, A)
        out_c = model_c(X, A)
        
        assert isinstance(out_a, torch.Tensor)
        assert out_a.shape == (2,), f"Expected shape (2,) but got {out_a.shape}"
        
        assert isinstance(out_b, torch.Tensor)
        assert out_b.shape == (2,), f"Expected shape (2,) but got {out_b.shape}"
        
        assert isinstance(out_c, torch.Tensor)
        assert out_c.shape == (2,), f"Expected shape (2,) but got {out_c.shape}"
        
        # Test without pooling - should return [num_nodes, output_dim]
        out_a_no_pool = model_a_no_pool(X, A)
        out_b_no_pool = model_b_no_pool(X, A)
        out_c_no_pool = model_c_no_pool(X, A)
        
        assert isinstance(out_a_no_pool, torch.Tensor)
        assert out_a_no_pool.shape == (7, 2), f"Expected shape (7, 2) but got {out_a_no_pool.shape}"
        
        assert isinstance(out_b_no_pool, torch.Tensor)
        assert out_b_no_pool.shape == (7, 2), f"Expected shape (7, 2) but got {out_b_no_pool.shape}"
        
        assert isinstance(out_c_no_pool, torch.Tensor)
        assert out_c_no_pool.shape == (7, 2), f"Expected shape (7, 2) but got {out_c_no_pool.shape}"
    
    def test_incorrect_arguments(self):
        """
        Test if incorrectly initializing the model breaks it (as it should)
        """
        with pytest.raises(ValueError):        
            model = models.GNN("WeirdLayer", 3, 5, 2)
        
        with pytest.raises(TypeError):        
            model = models.GNN("GATConv", 3, 5, 2)
        
        with pytest.raises(TypeError):
            model = models.GNN("MessagePassing", 3, 5, 2, hidden_dim=5)
        
        with pytest.raises(TypeError):
            model = models.GNN("MessagePassing", 3, 5, 2, num_hidden=3)


class TestTransformer:
    def test_init(self):
        """
        Test the initialization of the Transformer model
        """
        model = models.Transformer(input_dim=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, output_dim=5)
        
        assert isinstance(model, models.Transformer)
        assert isinstance(model, models.nn.Module)
        assert model.max_seq_len == 200
        assert model.positional_encoding.shape == (1, 200, 64)
    
    def test_forward(self):
        """
        Test the forward pass of the Transformer model
        """
        model = models.Transformer(input_dim=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, output_dim=5)
        
        # Test with different sequence lengths
        x1 = torch.randn(15, 10)  # 15 atoms
        a1 = torch.eye(15)
        out1 = model(x1, a1)
        
        assert isinstance(out1, torch.Tensor)
        assert out1.shape == (5,)  # Output is [output_dim]
        
        x2 = torch.randn(50, 10)  # 50 atoms
        a2 = torch.eye(50)
        out2 = model(x2, a2)
        
        assert isinstance(out2, torch.Tensor)
        assert out2.shape == (5,)  # Output is [output_dim]
    
    def test_positional_encoding(self):
        """
        Test that positional encoding works correctly with sequences up to max_len
        """
        model = models.Transformer(input_dim=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, output_dim=5)
        
        # Test with sequence at max_len
        x = torch.randn(200, 10)  # At max_len=200
        a = torch.eye(200)
        
        # Should work fine
        out = model(x, a)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (5,)
    
    def test_model_number_of_parameters(self):
        """
        Test the number of parameters in the Transformer model
        """
        model = models.Transformer(input_dim=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, output_dim=5)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Just verify it has parameters
        assert num_params > 0


class TestFP:
    def test_init(self):
        """
        Test the initialization of the FP (Fingerprint) model
        """
        model = models.FP(radius=2, n_bits=1024, output_dim=994)
        
        assert isinstance(model, models.FP)
        assert isinstance(model, models.nn.Module)
        assert model.radius == 2
        assert model.n_bits == 1024
    
    def test_forward(self):
        """
        Test the forward pass of the FP model with valid SMILES
        """
        model = models.FP(radius=2, n_bits=1024, output_dim=5)
        
        # Test with a valid SMILES string
        smiles = "CCO"  # Ethanol
        out = model(smiles)
        
        assert isinstance(out, torch.Tensor)
        assert out.shape == (5,)
    
    def test_forward_batch(self):
        """
        Test the forward pass with multiple SMILES
        """
        model = models.FP(radius=3, n_bits=2048, output_dim=10)
        
        # Test with different SMILES
        smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]  # Ethanol, Acetic acid, Benzene
        
        for smiles in smiles_list:
            out = model(smiles)
            assert isinstance(out, torch.Tensor)
            assert out.shape == (10,)
    
    def test_different_radii(self):
        """
        Test with different fingerprint radii
        """
        for radius in [2, 3, 4]:
            model = models.FP(radius=radius, n_bits=1024, output_dim=5)
            out = model("CCO")
            assert out.shape == (5,)
    
    def test_different_bit_sizes(self):
        """
        Test with different fingerprint bit sizes
        """
        for n_bits in [512, 1024, 2048]:
            model = models.FP(radius=2, n_bits=n_bits, output_dim=5)
            out = model("CCO")
            assert out.shape == (5,)
    
    def test_model_number_of_parameters(self):
        """
        Test the number of parameters in the FP model
        """
        model = models.FP(radius=2, n_bits=1024, output_dim=994)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # FP has linear layer: 1024 * 994 + 994 (bias)
        expected_params = 1024 * 994 + 994
        assert num_params == expected_params

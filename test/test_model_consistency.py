"""
Consistency tests to verify that all models return shape [output_dim]
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import GNN, Transformer, FP, FCNN, VAE


class TestModelOutputConsistency:
    """Tests consistency of outputs across all models"""
    
    @pytest.fixture
    def test_data(self):
        """Fixture with test data"""
        return {
            'input_dim': 10,
            'output_dim': 994,
            'num_atoms': 30,
            'x': torch.randn(30, 10),
            'a': torch.eye(30),
            'smiles': 'CCO'
        }
    
    def test_gcn_output_shape(self, test_data):
        """Tests that GCN returns shape [output_dim] with pooling"""
        model = GNN(
            layer="GCNConv",
            num_layers=3,
            input_dim=test_data['input_dim'],
            output_dim=test_data['output_dim'],
            pooling="sum"
        )
        model.eval()
        
        with torch.no_grad():
            output = model(test_data['x'], test_data['a'])
        
        assert output.shape == torch.Size([test_data['output_dim']]), \
            f"Expected shape ({test_data['output_dim']},) but got {output.shape}"
    
    def test_gat_output_shape(self, test_data):
        """Tests that GAT returns shape [output_dim] with pooling"""
        model = GNN(
            layer="GATConv",
            num_layers=3,
            input_dim=test_data['input_dim'],
            output_dim=test_data['output_dim'],
            hidden_dim=64,
            pooling="sum"
        )
        model.eval()
        
        with torch.no_grad():
            output = model(test_data['x'], test_data['a'])
        
        assert output.shape == torch.Size([test_data['output_dim']]), \
            f"Expected shape ({test_data['output_dim']},) but got {output.shape}"
    
    def test_mpnn_output_shape(self, test_data):
        """Tests that MPNN returns shape [output_dim] with pooling"""
        model = GNN(
            layer="MessagePassing",
            num_layers=3,
            input_dim=test_data['input_dim'],
            output_dim=test_data['output_dim'],
            hidden_dim=64,
            num_hidden=2,
            activation=nn.ReLU(),
            pooling="sum"
        )
        model.eval()
        
        with torch.no_grad():
            output = model(test_data['x'], test_data['a'])
        
        assert output.shape == torch.Size([test_data['output_dim']]), \
            f"Expected shape ({test_data['output_dim']},) but got {output.shape}"
    
    def test_transformer_output_shape(self, test_data):
        """Tests that Transformer returns shape [output_dim]"""
        model = Transformer(
            input_dim=test_data['input_dim'],
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            output_dim=test_data['output_dim']
        )
        model.eval()
        
        with torch.no_grad():
            output = model(test_data['x'], test_data['a'])
        
        assert output.shape == torch.Size([test_data['output_dim']]), \
            f"Expected shape ({test_data['output_dim']},) but got {output.shape}"
    
    def test_fp_output_shape(self, test_data):
        """Tests that FP returns shape [output_dim]"""
        model = FP(
            radius=2,
            n_bits=1024,
            output_dim=test_data['output_dim']
        )
        model.eval()
        
        with torch.no_grad():
            output = model(test_data['smiles'])
        
        assert output.shape == torch.Size([test_data['output_dim']]), \
            f"Expected shape ({test_data['output_dim']},) but got {output.shape}"
    
    def test_vae_output_shape_single(self, test_data):
        """Tests that VAE returns shape [output_dim] for single sample"""
        model = VAE(input_dim=100, hidden_dim=64, latent_dim=8)
        model.eval()
        
        # Prepare input as in training code
        x = torch.argmax(test_data['x'], dim=1).float()
        if len(x) < 100:
            x = F.pad(x, (0, 100 - len(x)), "constant", 0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            x_recon, output, mu, logvar = model(x)
        
        assert output.shape == torch.Size([test_data['output_dim']]), \
            f"Expected shape ({test_data['output_dim']},) but got {output.shape}"
    
    def test_gnn_pooling_methods(self, test_data):
        """Tests different GNN pooling methods"""
        for pooling in ["sum", "mean", "max"]:
            model = GNN(
                layer="GCNConv",
                num_layers=2,
                input_dim=test_data['input_dim'],
                output_dim=test_data['output_dim'],
                pooling=pooling
            )
            model.eval()
            
            with torch.no_grad():
                output = model(test_data['x'], test_data['a'])
            
            assert output.shape == torch.Size([test_data['output_dim']]), \
                f"Pooling={pooling}: Expected shape ({test_data['output_dim']},) but got {output.shape}"
    
    def test_gnn_no_pooling(self, test_data):
        """Tests GNN without pooling (backward compatibility)"""
        model = GNN(
            layer="GCNConv",
            num_layers=2,
            input_dim=test_data['input_dim'],
            output_dim=test_data['output_dim'],
            pooling="none"
        )
        model.eval()
        
        with torch.no_grad():
            output = model(test_data['x'], test_data['a'])
        
        assert output.shape == torch.Size([test_data['num_atoms'], test_data['output_dim']]), \
            f"Expected shape ({test_data['num_atoms']}, {test_data['output_dim']}) but got {output.shape}"
    
    def test_all_models_consistent(self, test_data):
        """Tests that all main models return same shape"""
        models = {
            'GCN': GNN('GCNConv', 2, test_data['input_dim'], test_data['output_dim'], pooling="sum"),
            'GAT': GNN('GATConv', 2, test_data['input_dim'], test_data['output_dim'], hidden_dim=32, pooling="sum"),
            'MPNN': GNN('MessagePassing', 2, test_data['input_dim'], test_data['output_dim'], 
                       hidden_dim=32, num_hidden=2, activation=nn.ReLU(), pooling="sum"),
            'Transformer': Transformer(test_data['input_dim'], 32, 4, 2, 64, test_data['output_dim']),
        }
        
        expected_shape = torch.Size([test_data['output_dim']])
        
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                output = model(test_data['x'], test_data['a'])
            
            assert output.shape == expected_shape, \
                f"{name}: Expected {expected_shape} but got {output.shape}"


class TestGNNPoolingEdgeCases:
    """Tests for GNN pooling edge cases and error handling"""
    
    @pytest.fixture
    def test_data(self):
        """Fixture with test data"""
        return {
            'input_dim': 10,
            'output_dim': 20,
            'num_atoms': 15,
            'x': torch.randn(15, 10),
            'a': torch.eye(15),
        }
    
    def test_gnn_invalid_pooling(self, test_data):
        """Test that GNN raises error for invalid pooling method"""
        model = GNN(
            layer="GCNConv",
            num_layers=2,
            input_dim=test_data['input_dim'],
            output_dim=test_data['output_dim'],
            pooling="invalid"
        )
        
        with pytest.raises(ValueError, match="Unknown pooling method"):
            model(test_data['x'], test_data['a'])
    
    def test_gnn_invalid_layer_type(self, test_data):
        """Test that GNN raises error for invalid layer type"""
        with pytest.raises(ValueError, match="Layer type not implemented"):
            model = GNN(
                layer="InvalidLayer",
                num_layers=2,
                input_dim=test_data['input_dim'],
                output_dim=test_data['output_dim']
            )
    
    def test_gnn_missing_hidden_dim_gat(self, test_data):
        """Test that GATConv requires hidden_dim"""
        with pytest.raises(TypeError, match="GATConv layer requires kwarg 'hidden_dim'"):
            model = GNN(
                layer="GATConv",
                num_layers=2,
                input_dim=test_data['input_dim'],
                output_dim=test_data['output_dim']
            )
    
    def test_gnn_missing_hidden_dim_mpnn(self, test_data):
        """Test that MessagePassing requires hidden_dim"""
        with pytest.raises(TypeError, match="MessagePassing layer requires kwarg 'hidden_dim'"):
            model = GNN(
                layer="MessagePassing",
                num_layers=2,
                input_dim=test_data['input_dim'],
                output_dim=test_data['output_dim'],
                num_hidden=2,
                activation=nn.ReLU()
            )
    
    def test_gnn_missing_num_hidden_mpnn(self, test_data):
        """Test that MessagePassing requires num_hidden"""
        with pytest.raises(TypeError, match="MessagePassing layer requires kwarg 'num_hidden'"):
            model = GNN(
                layer="MessagePassing",
                num_layers=2,
                input_dim=test_data['input_dim'],
                output_dim=test_data['output_dim'],
                hidden_dim=32,
                activation=nn.ReLU()
            )
    
    def test_gnn_missing_activation_mpnn(self, test_data):
        """Test that MessagePassing requires activation"""
        with pytest.raises(TypeError, match="MessagePassing layer requires kwarg 'activation'"):
            model = GNN(
                layer="MessagePassing",
                num_layers=2,
                input_dim=test_data['input_dim'],
                output_dim=test_data['output_dim'],
                hidden_dim=32,
                num_hidden=2
            )


class TestVAEComponents:
    """Tests for VAE internal components"""
    
    def test_vae_encode(self):
        """Test VAE encode method"""
        model = VAE(input_dim=64, hidden_dim=32, latent_dim=8)
        x = torch.randn(1, 64)
        
        mu, logvar = model.encode(x)
        
        assert mu.shape == (1, 8)
        assert logvar.shape == (1, 8)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(logvar, torch.Tensor)
    
    def test_vae_reparameterize(self):
        """Test VAE reparameterize method"""
        model = VAE(input_dim=64, hidden_dim=32, latent_dim=8)
        mu = torch.randn(1, 8)
        logvar = torch.randn(1, 8)
        
        z = model.reparameterize(mu, logvar)
        
        assert z.shape == (1, 8)
        assert isinstance(z, torch.Tensor)
    
    def test_vae_decode(self):
        """Test VAE decode method"""
        model = VAE(input_dim=64, hidden_dim=32, latent_dim=8)
        z = torch.randn(1, 8)
        
        x_recon = model.decode(z)
        
        assert x_recon.shape == (1, 64)
        assert isinstance(x_recon, torch.Tensor)
    
    def test_vae_batch_output(self):
        """Test VAE with batch size > 1"""
        model = VAE(input_dim=64, hidden_dim=32, latent_dim=8)
        x = torch.randn(10, 64)  # Batch of 10
        
        x_recon, y, mu, logvar = model(x)
        
        assert x_recon.shape == (10, 64)
        assert y.shape == (10, 994)  # Batch output
        assert mu.shape == (10, 8)
        assert logvar.shape == (10, 8)


class TestTransformerComponents:
    """Tests for Transformer components"""
    
    def test_transformer_positional_encoding(self):
        """Test that positional encoding is properly initialized"""
        model = Transformer(input_dim=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        
        assert hasattr(model, 'positional_encoding')
        assert model.positional_encoding.shape == (1, 200, 64)
        assert model.max_seq_len == 200
    
    def test_transformer_with_max_sequence_length(self):
        """Test Transformer with sequence at max length"""
        model = Transformer(input_dim=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, output_dim=100)
        x = torch.randn(200, 10)  # Max sequence length
        a = torch.eye(200)
        
        model.eval()
        with torch.no_grad():
            output = model(x, a)
        
        assert output.shape == (100,)
    
    def test_transformer_different_sequence_lengths(self):
        """Test Transformer with various sequence lengths"""
        model = Transformer(input_dim=10, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, output_dim=50)
        model.eval()
        
        for seq_len in [5, 20, 50, 100]:
            x = torch.randn(seq_len, 10)
            a = torch.eye(seq_len)
            
            with torch.no_grad():
                output = model(x, a)
            
            assert output.shape == (50,), f"Failed for sequence length {seq_len}"


class TestFPEdgeCases:
    """Tests for FP (Morgan Fingerprint) edge cases"""
    
    def test_fp_different_radii(self):
        """Test FP with different radii"""
        for radius in [1, 2, 3, 4]:
            model = FP(radius=radius, n_bits=512, output_dim=100)
            output = model("CCO")
            assert output.shape == (100,)
    
    def test_fp_different_bit_sizes(self):
        """Test FP with different bit sizes"""
        for n_bits in [256, 512, 1024, 2048]:
            model = FP(radius=2, n_bits=n_bits, output_dim=100)
            output = model("CCO")
            assert output.shape == (100,)
    
    def test_fp_complex_molecules(self):
        """Test FP with more complex molecules"""
        model = FP(radius=2, n_bits=1024, output_dim=50)
        
        smiles_list = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC(C)C",  # Isobutane
            "C1CCCCC1",  # Cyclohexane
        ]
        
        for smiles in smiles_list:
            output = model(smiles)
            assert output.shape == (50,), f"Failed for SMILES: {smiles}"


class TestFCNNVariations:
    """Tests for FCNN variations"""
    
    def test_fcnn_different_architectures(self):
        """Test FCNN with different architectures and pooling"""
        configs = [
            {'input_dim': 10, 'hidden_dim': 20, 'num_hidden': 1, 'output_dim': 5},
            {'input_dim': 50, 'hidden_dim': 100, 'num_hidden': 3, 'output_dim': 10},
            {'input_dim': 5, 'hidden_dim': 10, 'num_hidden': 5, 'output_dim': 2},
        ]
        
        for config in configs:
            # Test with default pooling (sum)
            model = FCNN(**config, activation=nn.ReLU())
            x = torch.randn(8, config['input_dim'])
            output = model(x)
            
            assert output.shape == (config['output_dim'],), f"Expected shape {(config['output_dim'],)}, got {output.shape}"
            assert torch.all(output >= 0)  # ReLU activation
            
            # Test with no pooling
            model_no_pool = FCNN(**config, activation=nn.ReLU(), pooling="none")
            output_no_pool = model_no_pool(x)
            
            assert output_no_pool.shape == (8, config['output_dim']), f"Expected shape {(8, config['output_dim'])}, got {output_no_pool.shape}"
    
    def test_fcnn_with_different_activations(self):
        """Test FCNN with different activation functions"""
        x = torch.randn(5, 10)
        
        model_relu = FCNN(input_dim=10, hidden_dim=20, num_hidden=2, output_dim=5, activation=nn.ReLU())
        model_sigmoid = FCNN(input_dim=10, hidden_dim=20, num_hidden=2, output_dim=5, activation=nn.Sigmoid())
        model_tanh = FCNN(input_dim=10, hidden_dim=20, num_hidden=2, output_dim=5, activation=nn.Tanh())
        
        output_relu = model_relu(x)
        output_sigmoid = model_sigmoid(x)
        output_tanh = model_tanh(x)
        
        assert output_relu.shape == (5,)
        assert output_sigmoid.shape == (5,)
        assert output_tanh.shape == (5,)
        # Note: After pooling, activation bounds don't hold (e.g., sum of sigmoids can be > 1)
        assert torch.all(output_relu >= 0)  # ReLU is non-negative
    
    def test_fcnn_unused_adjacency_parameter(self):
        """Test that FCNN accepts but ignores adjacency matrix"""
        model = FCNN(input_dim=10, hidden_dim=20, num_hidden=2, output_dim=5)
        x = torch.randn(7, 10)
        a = torch.eye(7)
        
        output_with_a = model(x, a)
        output_without_a = model(x)
        
        # Both should produce same result since A is unused
        assert output_with_a.shape == (5,)
        assert output_without_a.shape == (5,)
    
    def test_fcnn_pooling_methods(self):
        """Test all pooling methods for FCNN"""
        x = torch.randn(10, 15)
        
        model_sum = FCNN(input_dim=15, hidden_dim=20, num_hidden=2, output_dim=5, pooling="sum")
        model_mean = FCNN(input_dim=15, hidden_dim=20, num_hidden=2, output_dim=5, pooling="mean")
        model_max = FCNN(input_dim=15, hidden_dim=20, num_hidden=2, output_dim=5, pooling="max")
        model_none = FCNN(input_dim=15, hidden_dim=20, num_hidden=2, output_dim=5, pooling="none")
        
        output_sum = model_sum(x)
        output_mean = model_mean(x)
        output_max = model_max(x)
        output_none = model_none(x)
        
        assert output_sum.shape == (5,)
        assert output_mean.shape == (5,)
        assert output_max.shape == (5,)
        assert output_none.shape == (10, 5)
    
    def test_fcnn_invalid_pooling(self):
        """Test that FCNN raises error for invalid pooling method"""
        model = FCNN(input_dim=10, hidden_dim=20, num_hidden=2, output_dim=5, pooling="invalid")
        x = torch.randn(5, 10)
        
        with pytest.raises(ValueError, match="Invalid pooling method: invalid"):
            model(x)

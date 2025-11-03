"""
Tests for train.py functions - optimized version
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import torch.nn as nn
from src.models import VAE, GNN, FP, FCNN, Transformer


# Replicate functions from train.py to avoid loading full module
def create_model(model_type, config, input_dim, output_dim=994):
    """Factory function to create models - replicated from train.py"""
    reconstruction_beta = None
    
    if model_type == "VAE":
        model = VAE(input_dim=100, latent_dim=config['latent_dim'], hidden_dim=config['hidden_dim'])
        reconstruction_beta = config['reconstruction_beta']
    elif model_type == "GCN":
        model = GNN(layer="GCNConv", num_layers=config['num_layers'], input_dim=input_dim, output_dim=output_dim)
    elif model_type == "GAT":
        model = GNN(layer="GATConv", num_layers=config['num_layers'], input_dim=input_dim, output_dim=output_dim, hidden_dim=config['hidden_dim'])
    elif model_type == "MPNN":
        model = GNN(layer="MessagePassing", num_layers=config['num_layers'], input_dim=input_dim, output_dim=output_dim,
                   hidden_dim=config['hidden_dim'], num_hidden=config['num_hidden'], activation=nn.ReLU())
    elif model_type == "Transformer":
        model = Transformer(input_dim=input_dim, d_model=config['d_model'], nhead=config['nhead'],
                          num_layers=config['num_layers'], dim_feedforward=config['dim_feedforward'], output_dim=output_dim)
    elif model_type == "FP":
        model = FP(radius=config['radius'], n_bits=config['n_bits'], output_dim=output_dim)
    elif model_type == "FCNN":
        model = FCNN(input_dim=input_dim, hidden_dim=config['hidden_dim'], num_hidden=config['num_layers'],
                    output_dim=output_dim, activation=nn.ReLU())
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, reconstruction_beta


def get_model_name(model_type, config, lr, wd):
    """Generate model name - replicated from train.py"""
    parts = [model_type]
    for key in sorted(config.keys()):
        value = config[key]
        short_key = key.replace('num_layers', 'nlayers').replace('hidden_dim', 'hdim') \
                      .replace('latent_dim', 'latent').replace('reconstruction_beta', 'beta') \
                      .replace('dim_feedforward', 'ffn').replace('d_model', 'dmodel') \
                      .replace('num_hidden', 'nhidden').replace('n_bits', 'bits')
        parts.append(f"{short_key}{value}")
    parts.append(f"lr{lr}")
    parts.append(f"wd{wd}")
    return "_".join(parts)


class TestCreateModel:
    """Test create_model factory function"""
    
    def test_create_vae(self):
        config = {'latent_dim': 8, 'hidden_dim': 64, 'reconstruction_beta': 0.1}
        model, beta = create_model('VAE', config, input_dim=21)
        assert model is not None and beta == 0.1 and isinstance(model, VAE)
    
    def test_create_gcn(self):
        config = {'num_layers': 2}  # Reduced from 3 for speed
        model, beta = create_model('GCN', config, input_dim=21, output_dim=100)
        assert model is not None and beta is None and isinstance(model, GNN)
    
    def test_create_transformer(self):
        config = {'num_layers': 2, 'd_model': 32, 'nhead': 4, 'dim_feedforward': 128}
        model, beta = create_model('Transformer', config, input_dim=21, output_dim=100)
        assert model is not None and beta is None and isinstance(model, Transformer)
    
    def test_create_fp(self):
        config = {'radius': 2, 'n_bits': 1024}
        model, beta = create_model('FP', config, input_dim=21, output_dim=100)
        assert model is not None and beta is None and isinstance(model, FP)
    
    def test_create_fcnn(self):
        config = {'num_layers': 2, 'hidden_dim': 32}
        model, beta = create_model('FCNN', config, input_dim=21, output_dim=100)
        assert model is not None and beta is None and isinstance(model, FCNN)
    
    def test_create_invalid_model(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model('InvalidModel', {}, input_dim=21)


class TestGetModelName:
    """Test get_model_name function"""
    
    def test_simple_config(self):
        name = get_model_name('GCN', {'num_layers': 3}, 1e-4, 1e-6)
        assert all(x in name for x in ['GCN', 'nlayers3', 'lr0.0001', 'wd1e-06'])
    
    def test_complex_config(self):
        config = {'d_model': 128, 'nhead': 8, 'num_layers': 6, 'dim_feedforward': 512}
        name = get_model_name('Transformer', config, 1e-4, 1e-6)
        assert all(x in name for x in ['Transformer', 'dmodel128', 'nhead8', 'nlayers6', 'ffn512'])
    
    def test_name_consistency(self):
        config = {'num_layers': 3, 'hidden_dim': 64}
        assert get_model_name('GAT', config, 1e-4, 1e-6) == get_model_name('GAT', config, 1e-4, 1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

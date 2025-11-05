import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import numpy as np

from itertools import product
from torchinfo import summary

from src.utils import *
from src.models import *

### PARSE COMMAND LINE ARGUMENTS ###
parser = argparse.ArgumentParser(description='Train drug side effect prediction models')
parser.add_argument('--task', type=str, default='regression', 
                    choices=['regression', 'classification'],
                    help='Task type: regression (predict 0-5 scores) or classification (predict presence/absence)')
args = parser.parse_args()

### HYPERPARAMETERS ###
TASK = args.task  # "regression" or "classification"
FILE_PATH = './data/R_100.csv'
VAL_RATIO = 0.2
TEST_RATIO = 0.2
BATCH_SIZE = 64
EPOCHS = 00
N_RUNS = 1

### SETUP ###
device = (
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(device)
print(f"Using device: {device}")
print(f"Task: {TASK}")

train_loader, val_loader, test_loader = get_loaders(FILE_PATH, VAL_RATIO, TEST_RATIO, BATCH_SIZE, task=TASK)

# Get input dimension from dataset
sample_X = train_loader.dataset.dataset.data[0][0]
input_dim = sample_X.size(1)

### GRID SEARCH CONFIGURATION ###

# General hyperparameters (all models)
general_configs = {
    'learning_rates': [1e-4, 1e-3],
    'weight_decays': [1e-6, 1e-4, 1e-2],
}

# Model-specific hyperparameters
model_specific_configs = {
    'VAE': {
        'latent_dim': [8, 16],
        'hidden_dim': [64],
        'reconstruction_beta': [0.001, 0.01, 0.1]
    },
    'GCN': {
        'num_layers': [3, 5],
    },
    'GAT': {
        'num_layers': [3, 5],
        'hidden_dim': [64],
    },
    'MPNN': {
        'num_layers': [3, 5],
        'hidden_dim': [64],
        'num_hidden': [2, 3]
    },
    'Transformer': {
        'num_layers': [3, 5],
        'd_model': [64],
        'nhead': [4, 8],
        'dim_feedforward': [64]
    },
    'FP': {
        'radius': [2, 3],
        'n_bits': [1024, 2048]
    },
    'FCNN': {
        'num_layers': [3, 5],
        'hidden_dim': [64, 128]
    }
}

def create_model(model_type: str, config: dict, input_dim: int, output_dim: int = 994) -> tuple[nn.Module, float | None]:
    """Factory function to create models with specific configurations
    
    Args:
        model_type: Type of model to create ("VAE", "GCN", "GAT", etc.)
        config: Model-specific configuration parameters
        input_dim: Input dimension (number of atom features)
        output_dim: Output dimension (number of side effects)
    
    Returns:
        Tuple of (model, reconstruction_beta) where reconstruction_beta is None for non-VAE models
        
    Raises:
        ValueError: If model_type is not recognized
    """
    reconstruction_beta = None
    
    if model_type == "VAE":
        model = VAE(
            input_dim=100,
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim']
        )
        reconstruction_beta = config['reconstruction_beta']
    
    elif model_type == "GCN":
        model = GNN(
            layer="GCNConv",
            num_layers=config['num_layers'],
            input_dim=input_dim,
            output_dim=output_dim,
            activation=nn.Identity()
        )
    
    elif model_type == "GAT":
        model = GNN(
            layer="GATConv",
            num_layers=config['num_layers'],
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config['hidden_dim']
        )
    
    elif model_type == "MPNN":
        model = GNN(
            layer="MessagePassing",
            num_layers=config['num_layers'],
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config['hidden_dim'],
            num_hidden=config['num_hidden'],
            activation=nn.Identity()
        )
    
    elif model_type == "Transformer":
        model = Transformer(
            input_dim=input_dim,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            output_dim=output_dim
        )
    
    elif model_type == "FP":
        model = FP(
            radius=config['radius'],
            n_bits=config['n_bits'],
            output_dim=output_dim
        )
    
    elif model_type == "FCNN":
        model = FCNN(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_hidden=config['num_layers'],
            output_dim=output_dim,
            activation=nn.Identity()
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, reconstruction_beta

def get_model_name(model_type: str, config: dict, lr: float, wd: float) -> str:
    """Generate descriptive model name with all hyperparameters
    
    Args:
        model_type: Type of model ("VAE", "GCN", etc.)
        config: Model-specific configuration parameters
        lr: Learning rate
        wd: Weight decay
    
    Returns:
        Descriptive model name string with hyperparameters
    """
    parts = [model_type]
    
    # Add model-specific configs (sorted for consistency)
    for key in sorted(config.keys()):
        value = config[key]
        # Shorten key names for readability
        short_key = key.replace('num_layers', 'nlayers') \
                        .replace('hidden_dim', 'hdim') \
                        .replace('latent_dim', 'latent') \
                        .replace('reconstruction_beta', 'beta') \
                        .replace('dim_feedforward', 'ffn') \
                        .replace('d_model', 'dmodel') \
                        .replace('num_hidden', 'nhidden') \
                        .replace('n_bits', 'bits')
        
        parts.append(f"{short_key}{value}")
    
    # Add general configs
    parts.append(f"lr{lr}")
    parts.append(f"wd{wd}")
    
    return "_".join(parts)

### GRID SEARCH ###
model_types = ["GCN", "GAT", "MPNN", "Transformer", "FP", "FCNN", "VAE"]

def _safe_predict(model_type, model, x, a, w, smile, loss_fn):
    """Helper function to safely call model and loss function based on model type"""
    if model_type == "VAE":
        x = torch.argmax(x, dim=1).float()
        
        if len(x) < 100:
            x = F.pad(x, (0, 100 - len(x)), "constant", 0)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x_reconstructed, y_pred, mu, logvar = model(x)
        l = loss_fn(y_pred, w, x, x_reconstructed)
    
    elif model_type == "FP":
        y_pred = model(smile)
        l = loss_fn(y_pred, w)
    
    else:
        y_pred = model(x, a)
        l = loss_fn(y_pred, w)
    
    return l

for model_type in model_types:
    print(f"\n{'='*50}\nTraining {model_type} models\n{'='*50}")
    
    # Get model-specific configs
    specific_configs = model_specific_configs[model_type]
    
    # Get all combinations of model-specific hyperparameters
    config_keys = list(specific_configs.keys())
    config_values = list(specific_configs.values())
    
    for config_combo in product(*config_values):
        # Create config dict
        config = dict(zip(config_keys, config_combo))
        
        for learning_rate in general_configs['learning_rates']:
            for weight_decay in general_configs['weight_decays']:
                
                # Run each configuration N_RUNS times
                run_results = []
                base_model_name = get_model_name(model_type, config, learning_rate, weight_decay)
                
                print(f"\n>>> Training: {base_model_name} ({N_RUNS} runs)")
                
                for run_idx in range(N_RUNS):
                    print(f"\n  Run {run_idx + 1}/{N_RUNS}")
                    
                    # Create model for this run
                    model, reconstruction_beta = create_model(model_type, config, input_dim)
                    model_name = f"{base_model_name}_run{run_idx + 1}"
                    
                    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay
                    )
                    
                    # Select loss function based on task
                    if TASK == "classification":
                        loss_fn = WeightedBCELoss(reconstruction_beta=reconstruction_beta)
                    else:
                        loss_fn = DataDrivenLoss(reconstruction_beta=reconstruction_beta)

                    train_losses = []
                    val_losses = []
                    
                    for epoch in range(EPOCHS):
                        epoch_train_loss = 0
                        epoch_val_loss = 0
                        
                        # Train loop
                        model.train()
                        for X, A, y, smiles in train_loader:
                            optimizer.zero_grad()
                            batch_loss = 0
                            
                            for i in range(len(X)):
                                x = X[i]
                                a = A[i]
                                w = y[i]
                                smile = smiles[i]
                                
                                l = _safe_predict(model_type, model, x, a, w, smile, loss_fn)
                                
                                epoch_train_loss += l.item()
                                batch_loss += l
                            
                            # Average loss over batch before backward
                            batch_loss = batch_loss / len(X)
                            batch_loss.backward()
                            optimizer.step()
                        
                        # Validation loop
                        model.eval()
                        with torch.no_grad():
                            for X, A, y, smiles in val_loader:
                                for i in range(len(X)):
                                    x = X[i]
                                    a = A[i]
                                    w = y[i]
                                    smile = smiles[i]
                                    
                                    l = _safe_predict(model_type, model, x, a, w, smile, loss_fn)
                                    
                                    epoch_val_loss += l.item()
                        
                        # Average over drugs in loader
                        epoch_train_loss /= len(train_loader.dataset)
                        epoch_val_loss /= len(val_loader.dataset)
                        
                        # Average over side effects
                        epoch_train_loss /= len(w)
                        epoch_val_loss /= len(w)
                        
                        train_losses.append(epoch_train_loss)
                        val_losses.append(epoch_val_loss)
                        
                        if (epoch + 1) % 100 == 0 or epoch == 0:
                            print(f"  Epoch {epoch+1}/{EPOCHS} - "
                                f"Train Loss: {epoch_train_loss:.8f} - "
                                f"Val Loss: {epoch_val_loss:.8f}")
                    
                    # Save final model
                    save_model(model_name, model, task=TASK)
                    
                    # Save and plot losses
                    save_losses(model_name, train_losses, val_losses, task=TASK)
                    
                    # Save model architecture
                    with torch.no_grad():
                        if model_type == "VAE":
                            # Create a sample input for VAE (expects [batch, seq_len] with seq_len=100)
                            sample_input = torch.randint(0, input_dim, (1, 100)).float()
                            model_architecture = str(summary(model, input_data=sample_input, verbose=0))
                        elif model_type == "FP":
                            # torchinfo doesn't support string inputs, so use str(model) instead
                            model_architecture = str(model)
                        else:
                            model_architecture = str(summary(model, input_data=(x, a), verbose=0))
                    
                    # Save model specs for this run
                    last_val_loss = val_losses[-1]
                    
                    specs = {
                        'model_name': model_name,
                        'model_type': model_type,
                        'n_parameters': n_params,
                        'last_val_loss': last_val_loss,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'model_architecture': model_architecture,
                        'run': run_idx + 1
                    }
                    
                    # Add model-specific hyperparameters to specs
                    specs.update(config)
                    
                    save_specs(model_name, specs, task=TASK)
                    
                    # Save KDE predictions for train, val, and test sets
                    save_preds_kde(model, model_name, train_loader, model_type=model_type, task=TASK, split="train")
                    save_preds_kde(model, model_name, val_loader, model_type=model_type, task=TASK, split="val")
                    
                    # Store result for this run
                    run_results.append({
                        'run': run_idx + 1,
                        'last_val_loss': last_val_loss,
                        'model_name': model_name
                    })
                    
                    print(f'  Run {run_idx + 1} complete! Last val loss: {last_val_loss:.4f}')
                
                # Calculate statistics across runs
                val_losses_all_runs = [r['last_val_loss'] for r in run_results]
            
            mean_val_loss = np.mean(val_losses_all_runs)
            std_val_loss = np.std(val_losses_all_runs)
            
            print(f'\n  Summary for {base_model_name}:')
            print(f'    Mean val loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}')
            print(f'    Individual runs: {[f"{v:.4f}" for v in val_losses_all_runs]}')
            
            # Save summary specs for the configuration (average across runs)
            summary_model_name = base_model_name
            summary_specs = {
                'model_name': summary_model_name,
                'model_type': model_type,
                'n_parameters': n_params,
                'mean_val_loss': mean_val_loss,
                'std_val_loss': std_val_loss,
                'best_val_loss': min(val_losses_all_runs),
                'worst_val_loss': max(val_losses_all_runs),
                'n_runs': N_RUNS,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
            }
            
            # Add model-specific hyperparameters
            summary_specs.update(config)
            
            # Save summary specs
            save_specs(summary_model_name, summary_specs, task=TASK)
            
            print(f'\n{"="*5} Configuration {base_model_name} complete ({N_RUNS} runs)! {"="*5}\n')
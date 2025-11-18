import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from itertools import product

import torch

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
EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

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
model_specific_configs = {
    'VAE': {
        'latent_dim': [8, 16],
        'hidden_dim': [64],
        'reconstruction_beta': [0.001, 0.1]
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
        'nhead': [8, 16],
        'dim_feedforward': [64]
    },
    'FP': {
        'radius': [2, 3],
        'n_bits': [1024, 2048]
    },
    'FCNN': {
        'num_layers': [3, 5],
        'hidden_dim': [64]
    }
}

### GRID SEARCH ###
model_types = ["GCN", "GAT", "MPNN", "Transformer", "FP", "FCNN", "VAE"]

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
        
        model_name = get_model_name(model_type, config)
        
        print(f"\n>>> Training: {model_name}")
        
        # Create model for this run
        model, reconstruction_beta = create_model(model_type, config, input_dim)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
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
                    
                    y_pred, l = safe_predict(model_type, model, x, a, w, smile, loss_fn)
                    
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
                        
                        y_pred, l = safe_predict(model_type, model, x, a, w, smile, loss_fn)
                        
                        epoch_val_loss += l.item()
            
            # Average over drugs in loader
            epoch_train_loss /= len(train_loader.dataset)
            epoch_val_loss /= len(val_loader.dataset)
            
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} - "
                    f"Train Loss: {epoch_train_loss:.8f} - "
                    f"Val Loss: {epoch_val_loss:.8f}")
        
        # Save and plot losses
        save_losses(model_name, train_losses, val_losses, task=TASK)
        
        # Save model weights
        save_model(model_name, model, task=TASK)
        
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
        
        specs = {
            'model_name': model_name,
            'model_type': model_type,
            'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'val_loss': val_losses[-1],
            'val_metric': eval_metric(model, model_type, val_loader, TASK, loss_fn),
            'model_architecture': model_architecture,
        }
        
        # Add model-specific hyperparameters to specs
        specs.update(config)
        
        save_specs(model_name, specs, task=TASK)
        
        # Save KDE predictions for train, val, and test sets
        save_preds_kde(model, model_name, train_loader, model_type=model_type, task=TASK, split="train")
        save_preds_kde(model, model_name, val_loader, model_type=model_type, task=TASK, split="val")

        print(f'\n{"="*5} Configuration {model_name} complete! {"="*5}\n')
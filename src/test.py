import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from src.utils import *
from src.models import *

### PARSE COMMAND LINE ARGUMENTS ###
parser = argparse.ArgumentParser(description='Test best drug side effect prediction models')
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
EPOCHS = 3_000
N_TEST_RUNS = 3
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

# Load best configurations
best_configs_path = f'./diagnostics/{TASK}/best_configs.csv'
if not os.path.exists(best_configs_path):
    raise FileNotFoundError(
        f"Best configs file not found at {best_configs_path}. "
        f"Please run analyze_results.py --task {TASK} first to generate best_configs.csv"
    )

best_configs = pd.read_csv(best_configs_path)
print(f"\nLoaded {len(best_configs)} best configurations:")
print(best_configs[['model_type', 'val_metric']])

# Get data loaders
train_loader, val_loader, test_loader = get_loaders(FILE_PATH, VAL_RATIO, TEST_RATIO, BATCH_SIZE, task=TASK)

# Get input dimension from dataset
sample_X = train_loader.dataset.dataset.data[0][0]
input_dim = sample_X.size(1)

# Combine train and val datasets for final training
from torch.utils.data import ConcatDataset
train_val_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])

generator = torch.Generator(device=device)
full_dataset = train_loader.dataset.dataset  # Get the original DrugSideEffectsDataset
train_val_loader = DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                               generator=generator, collate_fn=full_dataset.collate_fn)

def create_model_from_config(row: pd.Series, input_dim: int, output_dim: int = 994) -> tuple[nn.Module, float | None]:
    """Create model from best config row
    
    Args:
        row: DataFrame row with model configuration
        input_dim: Input dimension (number of atom features)
        output_dim: Output dimension (number of side effects)
    
    Returns:
        Tuple of (model, reconstruction_beta) where reconstruction_beta is None for non-VAE models
        
    Raises:
        ValueError: If model_type is not recognized
    """
    model_type = row['model_type']
    reconstruction_beta = None
    
    if model_type == "VAE":
        model = VAE(
            input_dim=100,
            latent_dim=int(row['latent_dim']),
            hidden_dim=int(row['hidden_dim'])
        )
        reconstruction_beta = float(row['reconstruction_beta'])
    
    elif model_type == "GCN":
        model = GNN(
            layer="GCNConv",
            num_layers=int(row['num_layers']),
            input_dim=input_dim,
            output_dim=output_dim
        )
    
    elif model_type == "GAT":
        model = GNN(
            layer="GATConv",
            num_layers=int(row['num_layers']),
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(row['hidden_dim'])
        )
    
    elif model_type == "MPNN":
        model = GNN(
            layer="MessagePassing",
            num_layers=int(row['num_layers']),
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(row['hidden_dim']),
            num_hidden=int(row['num_hidden'])
        )
    
    elif model_type == "Transformer":
        model = Transformer(
            input_dim=input_dim,
            d_model=int(row['d_model']),
            nhead=int(row['nhead']),
            num_layers=int(row['num_layers']),
            dim_feedforward=int(row['dim_feedforward']),
            output_dim=output_dim
        )
    
    elif model_type == "FP":
        model = FP(
            radius=int(row['radius']),
            n_bits=int(row['n_bits']),
            output_dim=output_dim
        )
    
    elif model_type == "FCNN":
        model = FCNN(
            input_dim=input_dim,
            hidden_dim=int(row['hidden_dim']),
            num_hidden=int(row['num_layers']),
            output_dim=output_dim
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, reconstruction_beta


print(f"\nDataset sizes:")
print(f"  Train+Val: {len(train_val_dataset)} samples")
print(f"  Test: {len(test_loader.dataset)} samples")

# Store results for all models
test_results = []

### TRAIN AND TEST EACH BEST MODEL ###
for idx, row in best_configs.iterrows():
    model_type = row['model_type']
    model_name = f"{model_type}_best_final"
    
    print(f"\n{'='*60}\nTraining and Testing: {model_type}\n{'='*60}")
    
    print(f"\nRunning {N_TEST_RUNS} test runs for {model_type}...")
    
    test_run_results = []
    
    for test_run_idx in range(N_TEST_RUNS):
        print(f"\n  Test run {test_run_idx + 1}/{N_TEST_RUNS}")
        
        model, reconstruction_beta = create_model_from_config(row, input_dim)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
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
        model_name = f"{model_type}_best_final_run{test_run_idx + 1}"
        
        # Train on combined train+val dataset
        if test_run_idx == 0:
            print(f"  Training on train+val dataset...")
        
        # Start timing training
        train_start_time = time.time()
            
        for epoch in range(EPOCHS):
            epoch_train_loss = 0
            
            # Train loop
            model.train()
            for X, A, y, smiles in train_val_loader:
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
                
                batch_loss /= len(X)
                batch_loss.backward()
                optimizer.step()
            
            # Average over drugs in loader
            epoch_train_loss /= len(train_val_loader.dataset)
            
            train_losses.append(epoch_train_loss)
            
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f}")
        
        # End timing training
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        
        save_model(model_name, model, task=TASK)
        last_train_loss = train_losses[-1]
        print(f"\nTraining complete! Last train loss: {last_train_loss:.4f}, Time: {train_time:.2f}s")
        
        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        
        test_metric = eval_metric(model, model_type, test_loader, TASK, loss_fn)
        
        metric_name = "AUROC" if TASK == "classification" else "RMSE"
        
        print(f"  Run {test_run_idx + 1}: {metric_name}: {test_metric:.4f}, Train Time: {train_time:.2f}s")
        
        # Store results for this run
        test_run_results.append({
            'run': test_run_idx + 1,
            metric_name.lower(): test_metric,
            'last_train_loss': last_train_loss,
            'train_time_seconds': train_time
        })
        
        # Save this run's model and results
        save_losses(model_name, train_losses, task=TASK)
        
        specs = {
            'model_name': model_name,
            'model_type': model_type,
            'n_parameters': n_params,
            'last_train_loss': last_train_loss,
            metric_name.lower(): test_metric,
            'metric_name': metric_name,
            'train_time_seconds': train_time,
            'run': test_run_idx + 1
        }
        
        save_specs(model_name, specs, task=TASK)
        
        save_preds_kde(model, model_name, train_val_loader, model_type=model_type, task=TASK, split="train_val")
        save_preds_kde(model, model_name, test_loader, model_type=model_type, task=TASK, split="test")
    
    # Calculate statistics across test runs=
    metrics_all_runs = [r[metric_name.lower()] for r in test_run_results]
    train_times_all_runs = [r['train_time_seconds'] for r in test_run_results]
    
    mean_metric = np.mean(metrics_all_runs)
    std_metric = np.std(metrics_all_runs)
    mean_train_time = np.mean(train_times_all_runs)
    std_train_time = np.std(train_times_all_runs)
    
    print(f"\n  Summary for {model_type}:")
    print(f"    {metric_name}: {mean_metric:.4f} ± {std_metric:.4f}")
    print(f"    Train Time: {mean_train_time:.2f}s ± {std_train_time:.2f}s")
    print(f"    Individual {metric_name}s: {[f'{v:.4f}' for v in metrics_all_runs]}")
    print(f"    Individual train times: {[f'{v:.2f}s' for v in train_times_all_runs]}")
    
    # Store aggregate results
    test_results.append({
        'model_type': model_type,
        'n_parameters': n_params,
        f'mean_{metric_name.lower()}': mean_metric,
        f'std_{metric_name.lower()}': std_metric,
        'mean_train_time_seconds': mean_train_time,
        'std_train_time_seconds': std_train_time,
        'n_runs': N_TEST_RUNS
    })
    
    print(f'\n{"="*5} Model {model_type} testing complete ({N_TEST_RUNS} runs)! {"="*5}\n')

### SAVE AND VISUALIZE FINAL RESULTS ###
results_df = pd.DataFrame(test_results)
results_df = results_df.sort_values(f'mean_{metric_name.lower()}')

print("\n" + "="*60)
print("FINAL TEST RESULTS (sorted by mean test loss)")
print("="*60)
print(results_df.to_string(index=False))

# Save results to CSV
os.makedirs(f'./diagnostics/{TASK}', exist_ok=True)
results_path = f'./diagnostics/{TASK}/final_test_results.csv'
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")

# Create visualizations
vis_dir = f'./diagnostics/{TASK}/final_test_plots'
os.makedirs(vis_dir, exist_ok=True)

# 1. Test loss comparison with error bars
plt.figure(figsize=(12, 6))
plt.bar(results_df['model_type'], results_df[f'mean_{metric_name.lower()}'], 
        yerr=results_df[f'std_{metric_name.lower()}'], capsize=5, color='steelblue', alpha=0.7)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel(f'Mean Test {metric_name}', fontsize=12)
plt.title(f'Test {metric_name} Comparison Across Models ({N_TEST_RUNS} runs each)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, f'test_{metric_name.lower()}_comparison.png'), dpi=300)
plt.close()

# 2. Parameters vs Performance (using mean test loss)
plt.figure(figsize=(10, 6))
plt.errorbar(results_df['n_parameters'], results_df[f'mean_{metric_name.lower()}'],
             yerr=results_df[f'std_{metric_name.lower()}'], fmt='o', markersize=10, 
             alpha=0.6, capsize=5, elinewidth=2)
for idx, row in results_df.iterrows():
    plt.annotate(row['model_type'], 
                (row['n_parameters'], row[f'mean_{metric_name.lower()}']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
plt.xlabel('Number of Parameters', fontsize=12)
plt.ylabel(f'Mean Test {metric_name}', fontsize=12)
plt.title(f'Model Complexity vs Test Performance ({N_TEST_RUNS} runs)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'parameters_vs_performance.png'), dpi=300)
plt.close()

# 3. Training time comparison
plt.figure(figsize=(12, 6))
plt.bar(results_df['model_type'], results_df['mean_train_time_seconds'], 
        yerr=results_df['std_train_time_seconds'], capsize=5, color='green', alpha=0.7)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('Mean Training Time (seconds)', fontsize=12)
plt.title(f'Training Time Comparison ({N_TEST_RUNS} runs, {EPOCHS} epochs)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'training_time_comparison.png'), dpi=300)
plt.close()

print(f"\nVisualization plots saved to: {vis_dir}")
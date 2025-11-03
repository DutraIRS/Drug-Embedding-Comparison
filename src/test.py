import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time

from torchinfo import summary
from sklearn.metrics import roc_auc_score

from src.utils import *
from src.models import VAE, GNN, FP, FCNN, Transformer

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
EPOCHS = 1_000
N_TEST_RUNS = 5

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
print(best_configs[['model_type', 'best_val_loss']])

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

print(f"\nDataset sizes:")
print(f"  Train+Val: {len(train_val_dataset)} samples")
print(f"  Test: {len(test_loader.dataset)} samples")

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
            num_hidden=int(row['num_hidden']),
            activation=nn.ReLU()
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
            output_dim=output_dim,
            activation=nn.ReLU()
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, reconstruction_beta

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
        
        # Create model from best config
        model, reconstruction_beta = create_model_from_config(row, input_dim)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Setup optimizer with best hyperparameters
        learning_rate = float(row['learning_rate'])
        weight_decay = float(row['weight_decay'])
        
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
                
                for i in range(len(X)):
                    x = X[i]
                    a = A[i]
                    w = y[i]
                    smile = smiles[i]
                    
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
                        y_pred = y_pred.sum(dim=0)
                        l = loss_fn(y_pred, w)
                    
                    epoch_train_loss += l.item()
                    l.backward()
                
                optimizer.step()
            
            # Average over drugs in loader
            epoch_train_loss /= len(train_val_loader.dataset)
            
            # Average over side effects
            epoch_train_loss /= len(w)
            
            train_losses.append(epoch_train_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f}")
        
        # End timing training
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        
        save_model(model_name, model, task=TASK)
        last_train_loss = train_losses[-1]
        print(f"\nTraining complete! Last train loss: {last_train_loss:.4f}, Time: {train_time:.2f}s")
        
        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X, A, y, smiles in test_loader:
                for i in range(len(X)):
                    x = X[i]
                    a = A[i]
                    w = y[i]
                    smile = smiles[i]
                    
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
                        y_pred = y_pred.sum(dim=0)
                        l = loss_fn(y_pred, w)
                    
                    test_loss += l.item()
                    all_preds.append(y_pred.cpu().numpy())
                    all_targets.append(w.cpu().numpy())
            
            # Average test loss
            test_loss /= len(test_loader.dataset)
            test_loss /= len(w)
            
            # Calculate metric (AUROC for classification, RMSE for regression)
            all_preds = np.vstack(all_preds)  # Stack to shape (n_samples, 994)
            all_targets = np.vstack(all_targets)  # Stack to shape (n_samples, 994)
            
            if TASK == "classification":
                # Calculate AUROC for each side effect
                aurocs = []
                for se_idx in range(all_targets.shape[1]):
                    try:
                        # Only calculate if we have both classes
                        if len(np.unique(all_targets[:, se_idx])) > 1:
                            auroc = roc_auc_score(all_targets[:, se_idx], all_preds[:, se_idx])
                            aurocs.append(auroc)
                    except:
                        pass  # Skip if error (e.g., all one class)
                
                metric_value = np.mean(aurocs) if aurocs else 0.0
                metric_name = "AUROC"
            else:
                # Root Mean Squared Error for regression
                metric_value = np.sqrt(np.mean((all_preds - all_targets) ** 2))
                metric_name = "RMSE"
            
            print(f"  Run {test_run_idx + 1}: Test Loss: {test_loss:.4f}, {metric_name}: {metric_value:.4f}, Train Time: {train_time:.2f}s")
            
            # Store results for this run
            test_run_results.append({
                'run': test_run_idx + 1,
                'test_loss': test_loss,
                metric_name.lower(): metric_value,
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
                'test_loss': test_loss,
                metric_name.lower(): metric_value,
                'metric_name': metric_name,
                'train_time_seconds': train_time,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'run': test_run_idx + 1
            }
            
            # Add model-specific hyperparameters
            for col in best_configs.columns:
                if col not in ['model_name', 'model_type', 'n_parameters', 'best_val_loss', 'mean_val_loss', 'std_val_loss',
                               'learning_rate', 'weight_decay', 'model_architecture']:
                    if col in row and pd.notna(row[col]):
                        specs[col] = row[col]
            
            save_specs(model_name, specs, task=TASK)
            
            # Save KDE plot for first run only
            if test_run_idx == 0:
                save_preds_kde(model, model_name, test_loader, model_type=model_type, task=TASK)
    
    # Calculate statistics across test runs
    test_losses_all_runs = [r['test_loss'] for r in test_run_results]
    metrics_all_runs = [r[metric_name.lower()] for r in test_run_results]
    train_times_all_runs = [r['train_time_seconds'] for r in test_run_results]
    
    mean_test_loss = np.mean(test_losses_all_runs)
    std_test_loss = np.std(test_losses_all_runs)
    mean_metric = np.mean(metrics_all_runs)
    std_metric = np.std(metrics_all_runs)
    mean_train_time = np.mean(train_times_all_runs)
    std_train_time = np.std(train_times_all_runs)
    
    print(f"\n  Summary for {model_type}:")
    print(f"    Test Loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}")
    print(f"    {metric_name}: {mean_metric:.4f} ± {std_metric:.4f}")
    print(f"    Train Time: {mean_train_time:.2f}s ± {std_train_time:.2f}s")
    print(f"    Individual test losses: {[f'{v:.4f}' for v in test_losses_all_runs]}")
    print(f"    Individual {metric_name}s: {[f'{v:.4f}' for v in metrics_all_runs]}")
    print(f"    Individual train times: {[f'{v:.2f}s' for v in train_times_all_runs]}")
    
    # Store aggregate results
    test_results.append({
        'model_type': model_type,
        'n_parameters': n_params,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'mean_test_loss': mean_test_loss,
        'std_test_loss': std_test_loss,
        f'mean_{metric_name.lower()}': mean_metric,
        f'std_{metric_name.lower()}': std_metric,
        'mean_train_time_seconds': mean_train_time,
        'std_train_time_seconds': std_train_time,
        'n_runs': N_TEST_RUNS
    })
    
    print(f'\n{"="*5} Model {model_type} testing complete ({N_TEST_RUNS} runs)! {"="*5}\n')

### SAVE AND VISUALIZE FINAL RESULTS ###
results_df = pd.DataFrame(test_results)
results_df = results_df.sort_values('mean_test_loss')

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
plt.bar(results_df['model_type'], results_df['mean_test_loss'], 
        yerr=results_df['std_test_loss'], capsize=5, color='steelblue', alpha=0.7)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('Mean Test Loss', fontsize=12)
plt.title(f'Test Loss Comparison Across Models ({N_TEST_RUNS} runs each)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'test_loss_comparison.png'), dpi=300)
plt.close()

# 2. Multiple metrics comparison with error bars
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(results_df['model_type'], results_df['mean_test_loss'], 
            yerr=results_df['std_test_loss'], capsize=5, color='steelblue', alpha=0.7)
axes[0].set_title(f'Mean Test Loss ({N_TEST_RUNS} runs)')
axes[0].set_ylabel('Loss')
axes[0].tick_params(axis='x', rotation=45)

metric_col = f'mean_{metric_name.lower()}'
std_col = f'std_{metric_name.lower()}'
axes[1].bar(results_df['model_type'], results_df[metric_col], 
            yerr=results_df[std_col], capsize=5, color='coral', alpha=0.7)
axes[1].set_title(f'Mean {metric_name} ({N_TEST_RUNS} runs)')
axes[1].set_ylabel(metric_name)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'all_metrics_comparison.png'), dpi=300)
plt.close()

# 3. Parameters vs Performance (using mean test loss)
plt.figure(figsize=(10, 6))
plt.errorbar(results_df['n_parameters'], results_df['mean_test_loss'], 
             yerr=results_df['std_test_loss'], fmt='o', markersize=10, 
             alpha=0.6, capsize=5, elinewidth=2)
for idx, row in results_df.iterrows():
    plt.annotate(row['model_type'], 
                (row['n_parameters'], row['mean_test_loss']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
plt.xlabel('Number of Parameters', fontsize=12)
plt.ylabel('Mean Test Loss', fontsize=12)
plt.title(f'Model Complexity vs Test Performance ({N_TEST_RUNS} runs)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'parameters_vs_performance.png'), dpi=300)
plt.close()

# 4. Training time comparison
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

# Print best model
best_model = results_df.iloc[0]
print("\n" + "="*60)
print("BEST MODEL")
print("="*60)
print(f"Model Type: {best_model['model_type']}")
print(f"Mean Test Loss: {best_model['mean_test_loss']:.4f} ± {best_model['std_test_loss']:.4f}")
print(f"Mean {metric_name}: {best_model[metric_col]:.4f} ± {best_model[std_col]:.4f}")
print(f"Mean Train Time: {best_model['mean_train_time_seconds']:.2f}s ± {best_model['std_train_time_seconds']:.2f}s")
print(f"Parameters: {best_model['n_parameters']:,}")
print(f"Number of runs: {best_model['n_runs']}")
print("="*60)

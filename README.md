# Drug Embedding Comparison
Final Project for Bachelor's Degree in Data Science and AI

## Overview
This project compares different molecular embedding approaches for predicting drug side effects. I implemented and evaluated 7 different model architectures on a dataset of molecules and their associated 994 side effects.

The pipeline supports **two distinct tasks**:
- **Regression**: Predict frequency scores (0-5 scale)
- **Classification**: Predict presence/absence of side effects (binary)

The pipeline includes comprehensive grid search with multiple runs for statistical significance, automated analysis, and final testing on held-out test sets.

## Models Implemented
- **VAE** (Variational Autoencoder): Encodes molecules into latent space with reconstruction loss
- **GCN** (Graph Convolutional Network): Uses graph convolutions on molecular structure
- **GAT** (Graph Attention Network): Applies attention mechanism to molecular graphs
- **MPNN** (Message Passing Neural Network): Custom message passing on molecular graphs
- **Transformer**: Self-attention over atomic sequences with positional encoding
- **FP** (Fingerprint): Morgan fingerprint (ECFP) + linear layer for prediction
- **FCNN** (Fully Connected Neural Network): Baseline MLP on atomic features

## Project Structure
```
src/
├── models.py          # Model architectures (VAE, GNN, Transformer, FP, FCNN)
├── layers.py          # Custom GNN layers (GCNConv, GATConv, MessagePassing)
├── utils.py           # Data loading, loss functions, saving utilities
├── train.py           # Grid search training per configuration
├── analyze_results.py # Consolidate results and find best configurations
├── test.py            # Final evaluation on test set with 3 runs per model
└── trim_data.ipynb    # Notebook for data preprocessing and subset creation

test/
├── test_layers.py     # Unit tests for GNN layers
└── test_models.py     # Unit tests for model architectures

data/
├── R.csv              # Full dataset (molecules × side effects)
├── R_100.csv          # Subset for quick experiments (samples with less than 100 atoms)
├── smiles.csv         # SMILES strings for full dataset
└── smiles_100.csv     # SMILES strings for subset

saved_models/          # Trained model checkpoints (organized by task)
├── regression/        # Regression task models
│   └── [model_name]/
│       ├── model_weights.pt      # Model state dict
│       ├── specs.csv             # Hyperparameters and architecture
│       ├── losses.csv            # Training history
│       ├── loss_plot.png         # Training curves
│       └── kde_plot.png          # Prediction distribution
└── classification/    # Classification task models
    └── [model_name]/
        ├── model_weights.pt      # Model state dict
        ├── specs.csv             # Hyperparameters and architecture
        ├── losses.csv            # Training history
        ├── loss_plot.png         # Training curves
        └── kde_plot.png          # Prediction distribution

diagnostics/           # Analysis results and visualizations (organized by task)
├── regression/        # Regression task diagnostics
│   ├── all_results.csv           # All training configurations and results
│   ├── best_configs.csv          # Best configuration per model type
│   ├── final_test_results.csv    # Test set evaluation results
│   ├── analysis_plots/           # Training analysis visualizations
│   │   ├── val_loss_by_model_type.png
│   │   ├── best_configs_comparison.png
│   │   ├── params_vs_loss.png
│   │   └── [model]_lr_wd_heatmap.png
│   └── final_test_plots/         # Test set evaluation visualizations
│       ├── test_loss_comparison.png
│       ├── test_metric_comparison.png
│       ├── params_vs_performance.png
│       └── training_time_comparison.png
└── classification/    # Classification task diagnostics
    ├── all_results.csv           # All training configurations and results
    ├── best_configs.csv          # Best configuration per model type
    ├── final_test_results.csv    # Test set evaluation results
    ├── analysis_plots/           # Training analysis visualizations
    │   ├── val_loss_by_model_type.png
    │   ├── best_configs_comparison.png
    │   ├── params_vs_loss.png
    │   └── [model]_lr_wd_heatmap.png
    └── final_test_plots/         # Test set evaluation visualizations
        ├── test_loss_comparison.png
        ├── test_metric_comparison.png
        ├── params_vs_performance.png
        └── training_time_comparison.png

run_pipeline_regression.ps1      # PowerShell script for regression task
run_pipeline_classification.ps1  # PowerShell script for classification task
```

## Task Configuration

### Regression Task
Predicts frequency scores on a 0-5 scale where:
- 0 = not observed
- 1-5 = increasing frequency (very rare to very frequent)

**Loss Function:** Weighted RMSE (Root Mean Squared Error)
$$\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{DS}\sum_{i,j} w_{ij}(y_{ij}-\hat{y}_{ij})^2}$$

Where $w_{ij} = \alpha$ if $y_{ij}=0$, else $1$ (default $\alpha=0.03$)

**Evaluation Metric:** RMSE

### Classification Task
Binary prediction of side effect presence/absence:
- 1 = association exists (original score > 0)
- 0 = no association (original score = 0)

**Loss Function:** Weighted Binary Cross-Entropy
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{DS}\sum_{i,j}[y_{ij}\log(\hat{p}_{ij}) + \alpha(1-y_{ij})\log(1-\hat{p}_{ij})]$$

Where $\alpha=0.03$ down-weights negative samples

**Evaluation Metric:** AUROC (Area Under ROC Curve)

## Quick Start

### 1. Install Dependencies
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Pipeline

**For Regression Task:**
```powershell
.\run_pipeline_regression.ps1
```

**For Classification Task:**
```powershell
.\run_pipeline_classification.ps1
```

**Run Both Tasks in Parallel:**
You can run both pipelines simultaneously in separate terminals without conflicts:
- Terminal 1: `.\run_pipeline_regression.ps1`
- Terminal 2: `.\run_pipeline_classification.ps1`

Each pipeline saves results to task-specific directories:
- Regression: `saved_models/regression/` and `diagnostics/regression/`
- Classification: `saved_models/classification/` and `diagnostics/classification/`

Each pipeline will sequentially:
1. Train all models with grid search
2. Analyze results and select best configurations
3. Test best models on test set (3 runs per model)


**Features:**
- Grid search over model-specific hyperparameters
- **3 independent runs** per configuration for statistical robustness
- Automatic model checkpointing
- Tracks mean ± std validation loss across runs
- **Task-specific metrics**: RMSE for regression, AUROC for classification
- Generates loss curves and KDE plots

**Hyperparameter Grids:**
- **General**: learning_rate, weight_decay
- **VAE**: latent_dim, hidden_dim, reconstruction_beta
- **GCN**: num_layers
- **GAT**: num_layers, hidden_dim
- **MPNN**: num_layers, hidden_dim, num_hidden
- **Transformer**: num_layers, d_model, nhead, dim_feedforward
- **FP**: radius, n_bits
- **FCNN**: num_layers, hidden_dim

**Training Configuration:**
- Dataset: `data/R_100.csv` (samples with less than 100 atoms)
- Split: 60% train / 20% val / 20% test
- Epochs: 300
- Device: CUDA if available, else CPU
- **Task-specific loss**:
  - Regression: `DataDrivenLoss` (Weighted RMSE with α=0.03)
  - Classification: `WeightedBCELoss` (Weighted BCE with α=0.03)

**Generates:**
- `diagnostics/{task}/all_results.csv` - Complete grid search results
- `diagnostics/{task}/best_configs.csv` - Best configuration per model type (selected by mean validation loss)
- `diagnostics/{task}/analysis_plots/` - Comparative visualizations:
  - Validation loss distribution by model type (boxplot)
  - Best configurations comparison (bar chart)
  - Model complexity vs performance (scatter plot)
  - Hyperparameter impact heatmaps (learning rate × weight decay)

### 3. Final Testing
**Features:**
- Loads best configurations from `analyze_results.py`
- Trains each model on combined train+val set for 3000 epochs
- **3 independent test runs** per model for robust evaluation
- Evaluates on held-out test set
- Reports mean ± std for test loss and **task-specific metric** (RMSE or AUROC)
- Tracks and reports training time for each model
- Generates comparative visualizations with error bars

**Outputs:**
- `diagnostics/{task}/final_test_results.csv` - Test metrics for all models
- `diagnostics/{task}/final_test_plots/` - Comparative test visualizations

### 4. Run Tests
```powershell
pytest
```

**Test Configuration:**
- Configured in `pytest.ini` to focus on core modules
- Coverage targets: `src.layers` and `src.models` only
- Automatic HTML report generation in `htmlcov/`

**Test Coverage:**
- **layers.py**: 100% coverage (43 statements)
- **models.py**: 100% coverage (151 statements)  
- **utils.py**: 70% coverage (291 statements)
- **87 tests total** covering:
  - Loss functions (DataDrivenLoss, WeightedBCELoss)
  - Dataset loading and preprocessing
  - Custom GNN layers and pooling operations
  - All model architectures (VAE, FCNN, GNN variants, Transformer, FP)
  - Save/load utilities
  - Edge cases and error handling
- HTML coverage report: `htmlcov/index.html`

## Model Architecture Details

### Transformer
- Processes atomic sequences with multi-head self-attention
- Learnable positional encodings (max 200 atoms)
- Global mean pooling → FCNN prediction head
- No explicit adjacency matrix needed
- Handles variable-length molecular graphs

### Fingerprint (FP)
- Uses RDKit Morgan fingerprints (ECFP)
- Configurable radius and bit size
- Direct linear mapping to side effects
- Fastest inference time
- No graph structure required (works from SMILES)

### VAE (Variational Autoencoder)
- Encodes linearized atom sequence to latent space
- Reparameterization trick for sampling
- Reconstruction loss + prediction loss
- Bottleneck regularization with KL divergence
- Can generate new molecular representations

### Graph Neural Networks (GCN, GAT, MPNN)
- Operate on molecular graphs (atoms as nodes, bonds as edges)
- Aggregate information from neighboring atoms
- **GCN**: Symmetric normalized aggregation
- **GAT**: Attention-weighted aggregation
- **MPNN**: Custom message passing with configurable hidden layers

## Data Format
- **Input**: Molecular graphs
  - SMILES strings → RDKit molecular objects
  - Atomic features: one-hot encoded atom types
  - Adjacency matrix: binary connectivity
- **Output**: 994-dimensional vector
  - **Regression**: Side effect frequencies (0-5 scale)
    - 0 = not observed, 5 = very frequent
  - **Classification**: Binary labels (0 or 1)
    - 1 = association exists (original score > 0)
    - 0 = no association (original score = 0)
- **Loss Functions**:
  - **Regression**: Weighted RMSE - penalizes errors on non-zero targets more than zero targets
  - **Classification**: Weighted BCE - penalizes false positives (predicting 1 when true is 0) less than false negatives
  - Both use α=0.03 to handle 95% sparsity (only ~5% of pairs have associations)
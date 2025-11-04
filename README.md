# Drug Embedding Comparison
Final Project for Bachelor's Degree in Data Science and AI

## Overview
This project compares different molecular embedding approaches for predicting drug side effects. We implement and evaluate 7 different model architectures on a dataset of molecules and their associated 994 side effects.

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
├── train.py           # Grid search training with 3 runs per configuration
├── analyze_results.py # Consolidate results and find best configurations
├── test.py            # Final evaluation on test set with 5 runs per model
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

### Option 1: Run Full Pipeline (Automated)

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
1. Train all models with grid search (3 runs per configuration)
2. Analyze results and select best configurations
3. Test best models on test set (5 runs per model)

### Option 2: Manual Execution

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch 2.6.0+ (with CUDA support for GPU training)
- RDKit (for molecular fingerprints)
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn (for AUROC metric)
- pytest, pytest-cov (for testing)

#### 2. Train Models
```bash
# Regression task (default)
python src/train.py --task regression

# Classification task
python src/train.py --task classification
```

**Command-line Arguments:**
- `--task`: Task type - `regression` (default) or `classification`

**Features:**
- Grid search over model-specific hyperparameters
- **3 independent runs** per configuration for statistical robustness
- Automatic model checkpointing (saves final model, not best)
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
- Epochs: 1_000 (configurable via `EPOCHS` variable)
- Runs per config: 3 (configurable via `N_RUNS`)
- Device: CUDA if available, else CPU
- **Task-specific loss**:
  - Regression: `DataDrivenLoss` (Weighted RMSE with α=0.03)
  - Classification: `WeightedBCELoss` (Weighted BCE with α=0.03)

#### 3. Analyze Results
```bash
# Regression task (default)
python src/analyze_results.py --task regression

# Classification task
python src/analyze_results.py --task classification
```

**Command-line Arguments:**
- `--task`: Task type - `regression` (default) or `classification`

**Generates:**
- `diagnostics/{task}/all_results.csv` - Complete grid search results
- `diagnostics/{task}/best_configs.csv` - Best configuration per model type (selected by mean validation metric)
- `diagnostics/{task}/analysis_plots/` - Comparative visualizations:
  - Validation loss distribution by model type (boxplot)
  - Best configurations comparison (bar chart)
  - Model complexity vs performance (scatter plot)
  - Hyperparameter impact heatmaps (learning rate × weight decay)

#### 4. Final Testing
```bash
# Regression task (default)
python src/test.py --task regression

# Classification task
python src/test.py --task classification
```

**Command-line Arguments:**
- `--task`: Task type - `regression` (default) or `classification`

**Features:**
- Loads best configurations from `analyze_results.py`
- Trains each model on combined train+val set
- **5 independent test runs** per model for robust evaluation
- Evaluates on held-out test set
- Reports mean ± std for test loss and **task-specific metric** (RMSE or AUROC)
- Tracks and reports training time for each model
- Generates comparative visualizations with error bars

**Outputs:**
- `diagnostics/{task}/final_test_results.csv` - Test metrics for all models
- `diagnostics/{task}/final_test_plots/` - Comparative test visualizations

#### 5. Run Tests
```bash
# Run all tests with coverage

# View coverage report
# After running pytest, open htmlcov/index.html in a browser
```

**Test Configuration:**
- Configured in `pytest.ini` to focus on core modules
- Coverage targets: `src.layers` and `src.models` only
- Automatic HTML report generation in `htmlcov/`

**Test Coverage:**
- **layers.py**: 100% coverage - All GNN layer types tested
- **models.py**: 100% coverage - All model architectures tested
- **35 tests total** covering:
  - Custom GNN layers: GCNConv, GATConv, MessagePassing (15 tests)
  - Model architectures: VAE, FCNN, GNN, Transformer, FP (20 tests)
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

## Experimental Design

### Multiple Runs for Statistical Robustness
- **Training**: 3 runs per hyperparameter configuration
  - Reports: mean ± std validation loss
  - Selects best config based on **mean** validation loss
- **Testing**: 5 runs per best model
  - Reports: mean ± std test loss and task-specific metric (RMSE or AUROC)
  - Provides confidence intervals for model comparison

### No Early Stopping
- Models are trained for fixed number of epochs
- **Final model** (last epoch) is saved and used for evaluation
- This ensures consistent training across all runs
- Simpler and more reproducible than early stopping

### Reproducibility
- Fixed random seeds (42) throughout pipeline
- All hyperparameters logged in `specs.json`
- Model architectures saved as text summaries
- Complete experiment tracking for reproducibility

## Results Interpretation

### Key Metrics
- **Validation Loss**: Used for hyperparameter selection (lower is better for both tasks)
- **Test Loss**: Final model performance on held-out data
- **Regression Metric**: RMSE (Root Mean Squared Error) - lower is better
- **Classification Metric**: AUROC (Area Under ROC Curve) - higher is better (range: 0-1)
- **Statistical Significance**: Error bars show std across multiple runs

### Expected Outputs
After running the full pipeline, you can compare:
1. Model ranking by test performance (task-specific metric)
2. Trade-offs between model complexity and accuracy
3. Impact of hyperparameters on each model type
4. Prediction distributions (KDE plots)
5. Regression vs Classification performance differences

## Hardware Requirements
- **GPU**: Recommended for faster training (CUDA-compatible NVIDIA GPU)
- **CPU**: Works but significantly slower
- **RAM**: 8GB+ recommended
- **Storage**: ~500MB for models and results (depends on grid search size)
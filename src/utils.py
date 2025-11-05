import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from rdkit import Chem

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### Random number generator seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

### SETUP ###
device = (
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(device)

MODEL_FOLDER = "./saved_models/"

class DataDrivenLoss(nn.Module):
    """
    Weighted RMSE loss for regression task
    Applies lower weight (alpha) to zero-value predictions
    Computes Root Mean Squared Error as per methodology
    """
    def __init__(self, alpha=0.03, reconstruction_beta=None):
        super(DataDrivenLoss, self).__init__()
        self.alpha = alpha
        self.reconstruction_beta = reconstruction_beta

    def forward(self, y_pred, y_true, x=None, x_reconstructed=None):
        # Compute squared error
        squared_error = (y_pred - y_true) ** 2

        # Apply scaling factor for targets that are zero
        scaled_error = torch.where(y_true == 0, squared_error * self.alpha, squared_error)
        
        # Compute RMSE (Root Mean Squared Error) as per methodology
        mse = scaled_error.mean()
        rmse = torch.sqrt(mse)
        
        if self.reconstruction_beta is not None:
            # Avoid in-place operation - create new tensor
            reconstruction_loss = ((x - x_reconstructed) ** 2).mean() * self.reconstruction_beta
            rmse = rmse + reconstruction_loss

        return rmse

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy loss for classification task
    Applies lower weight (alpha) to negative samples (y_true=0)
    Uses mean instead of sum for better numerical stability
    """
    def __init__(self, alpha=0.03, reconstruction_beta=None):
        super(WeightedBCELoss, self).__init__()
        self.alpha = alpha
        self.reconstruction_beta = reconstruction_beta

    def forward(self, y_pred, y_true, x=None, x_reconstructed=None):
        # Ensure predictions are probabilities (0-1 range)
        y_pred = torch.sigmoid(y_pred)
        
        # Clamp to avoid log(0)
        y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
        
        # Compute BCE for each element
        bce = -(y_true * torch.log(y_pred) + self.alpha * (1 - y_true) * torch.log(1 - y_pred))
        bce = bce.mean()
        
        if self.reconstruction_beta is not None:
            # Avoid in-place operation - create new tensor
            reconstruction_loss = ((x - x_reconstructed) ** 2).mean() * self.reconstruction_beta
            bce = bce + reconstruction_loss

        return bce

class DrugSideEffectsDataset(Dataset):
    """Dataset for drug-side effect prediction from molecular graphs
    
    Converts SMILES strings to molecular graphs with node features and adjacency matrices.
    Supports both regression (predicting 0-5 scores) and classification (binary prediction) tasks.
    """
    
    def __init__(self, smiles: list[str], side_effects_data: torch.Tensor | np.ndarray, task: str = "regression"):
        """Initialize the dataset
        
        Args:
            smiles: List of SMILES strings representing molecules
            side_effects_data: Target values (scores or binary labels)
            task: Task type - "regression" or "classification"
        """
        self.smiles = smiles
        self.task = task
        
        # Binarize for classification task
        if task == "classification":
            # Handle both numpy arrays and torch tensors
            if isinstance(side_effects_data, torch.Tensor):
                self.side_effects_data = (side_effects_data > 0).float()
            else:
                self.side_effects_data = (side_effects_data > 0).astype(float)
        else:
            self.side_effects_data = side_effects_data
        
        self.atomic_num_order = []
        for smiles in self.smiles:
            mol = Chem.MolFromSmiles(smiles)
    
            for atom in mol.GetAtoms():
                num = atom.GetAtomicNum()
                
                if num not in self.atomic_num_order:
                    self.atomic_num_order.append(num)
        
        self.atomic_num_order.sort()
        self.num_different_atoms = len(self.atomic_num_order)
        
        self.data = [self.build_graph(smiles) for smiles in self.smiles]
    
    def __len__(self) -> int:
        """Return the number of molecules in the dataset"""
        return len(self.smiles)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Get a molecule and its side effect data
        
        Args:
            idx: Index of the molecule
            
        Returns:
            Tuple of (node_features, adjacency_matrix, side_effects, smiles_string)
        """
        X, A = self.data[idx]
        y = self.side_effects_data[idx, :]
        smiles = self.smiles[idx]
        
        return X, A, y, smiles
    
    def build_graph(self, smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert SMILES string to graph representation
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            Tuple of (node_features, adjacency_matrix)
        """
        mol = Chem.MolFromSmiles(smiles)
        
        X = self.get_atom_features(mol)
        A = self.get_adjacency_matrix(mol)
        
        return X, A
    
    def get_adjacency_matrix(self, mol: Chem.Mol) -> torch.Tensor:
        """Build adjacency matrix from molecular bonds
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Adjacency matrix with self-loops (identity + bonds)
        """
        num_atoms = mol.GetNumAtoms()

        A = torch.eye(num_atoms)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            A[i, j] = 1
            A[j, i] = 1

        return A

    def get_atom_features(self, mol: Chem.Mol) -> torch.Tensor:
        """Extract one-hot encoded atom features
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            One-hot encoded atomic features [num_atoms, num_atom_types]
        """
        num_atoms = mol.GetNumAtoms()

        X = torch.zeros(num_atoms, self.num_different_atoms)

        for i, atom in enumerate(mol.GetAtoms()):
            X[i, self.atomic_num_order.index(atom.GetAtomicNum())] = 1

        return X
    
    def collate_fn(self, batch: list) -> tuple:
        """Custom collate function for batching variable-size graphs
        
        Args:
            batch: List of (X, A, y, smiles) tuples
            
        Returns:
            Tuple of lists for (X, A, y, smiles)
        """
        X, A, y, smiles = zip(*batch)
        
        return X, A, y, smiles

def train_val_test_split(len_dataset: int, val_ratio: float, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset indices into train/validation/test sets
    
    Args:
        len_dataset: Total number of samples in dataset
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    indices = np.arange(len_dataset)
    np.random.shuffle(indices)
    
    val_size = int(len_dataset * val_ratio)
    test_size = int(len_dataset * test_ratio)
    
    val_indices = indices[:val_size]
    test_indices = indices[val_size:val_size + test_size]
    train_indices = indices[val_size + test_size:]
    
    return train_indices, val_indices, test_indices

def get_loaders(path: str, val_ratio: float, test_ratio: float, batch_size: int, 
                task: str = "regression") -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing
    
    Args:
        path: Path to CSV file with SMILES and side effect data
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        batch_size: Batch size for data loaders
        task: Task type - "regression" or "classification"
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(path, header=None, sep=';')
    
    smiles = list(df.iloc[:, 0].values)
    side_effects_data = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float)
    
    full_dataset = DrugSideEffectsDataset(smiles, side_effects_data, task=task)
    
    total_size = len(full_dataset)
    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size - test_size

    generator = torch.Generator(device=device).manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                generator=torch.Generator(device=device).manual_seed(42),
                                collate_fn=full_dataset.collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                generator=torch.Generator(device=device).manual_seed(42),
                                collate_fn=full_dataset.collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                generator=torch.Generator(device=device).manual_seed(42),
                                collate_fn=full_dataset.collate_fn)

    return train_loader, val_loader, test_loader


def save_model(name: str, model: nn.Module, task: str = "regression") -> None:
    """Save model weights to disk
    
    Args:
        name: Model name (used as subdirectory name)
        model: PyTorch model to save
        task: Task type for directory organization - "regression" or "classification"
    """
    folder_path = os.path.join(MODEL_FOLDER, task, name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save model weights only (not the full model object to save memory)
    model_path = os.path.join(folder_path, 'model_weights.pt')
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {folder_path}")

def save_losses(name: str, train_losses: list[float], val_losses: list[float] = None, task: str = "regression"):
    """
    Save losses to CSV and generate training curves plot
    
    Args:
        name: Model name
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        task: Task type ('regression' or 'classification')
    """
    folder_path = os.path.join(MODEL_FOLDER, task, name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save losses to CSV
    losses_path = os.path.join(folder_path, 'losses.csv')
    data_dict = {'train_losses': train_losses}
    
    if val_losses is not None and len(val_losses) > 0:
        data_dict['val_losses'] = val_losses
    
    losses_df = pd.DataFrame(data_dict)
    losses_df.to_csv(losses_path, index=False)

    # Plot only train and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2, color='tab:blue')
    if val_losses and len(val_losses) > 0:
        plt.plot(val_losses, label='Validation Loss', linewidth=2, color='tab:orange')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(folder_path, 'loss_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Losses saved to {folder_path}")

def save_specs(name: str, specs: dict, task: str = "regression") -> None:
    """Save model specifications to CSV file
    
    Args:
        name: Model name (used as subdirectory name)
        specs: Dictionary of model specifications and hyperparameters
        task: Task type for directory organization - "regression" or "classification"
    """
    folder_path = os.path.join(MODEL_FOLDER, task, name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    df = pd.DataFrame.from_dict(specs, orient='index', columns=['value'])
    df.index.name = 'specification'
    df.to_csv(os.path.join(folder_path, 'specs.csv'))

def save_preds_kde(model: nn.Module, model_name: str, loader: DataLoader, 
                   model_type: str = None, task: str = "regression", split: str = "val") -> None:
    """Generate KDE plots of predicted side effect frequencies
    
    Args:
        model: Trained PyTorch model
        model_name: Name of the model (for loading weights and saving plot)
        loader: DataLoader for generating predictions
        model_type: Type of model ("VAE", "FP", etc.) - determines preprocessing
        task: Task type for directory organization - "regression" or "classification"
        split: Dataset split name ("train", "val", or "test") for plot labeling
    """
    model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, task, model_name, 'model_weights.pt')))
    
    model.eval() # freeze learning
    
    if task == "classification":
        # For classification: collect predictions for each true class (0 or 1)
        preds = {0: [], 1: []}  # Keys are true labels (0=absent, 1=present)
    else:
        # For regression: collect predictions for each true score (0-5)
        preds = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    
    with torch.no_grad(): # stop tracking gradients
        for X, A, y, smiles in loader:
            for i in range(len(X)):
                x = X[i]
                a = A[i]
                w = y[i]
                smile = smiles[i]
                
                # Handle different model types
                if model_type == "VAE":
                    x_processed = torch.argmax(x, dim=1).float()
                    if len(x_processed) < 100:
                        x_processed = F.pad(x_processed, (0, 100 - len(x_processed)), "constant", 0)
                    if x_processed.dim() == 1:
                        x_processed = x_processed.unsqueeze(0)
                    _, y_pred, _, _ = model(x_processed)
                    if y_pred.dim() > 1:
                        y_pred = y_pred.squeeze(0)
                elif model_type == "FP":
                    y_pred = model(smile)
                else:
                    y_pred = model(x, a)
                
                # Ensure y_pred is 1D with shape (994,)
                if y_pred.dim() == 0:
                    raise ValueError(f"y_pred is a scalar! Shape: {y_pred.shape}, model_type: {model_type}")
                elif y_pred.dim() > 1:
                    y_pred = y_pred.squeeze()

                # Apply sigmoid for classification to get probabilities
                if task == "classification":
                    y_pred = torch.sigmoid(y_pred)
                
                # Group predictions by true label
                for true_label in preds.keys():
                    mask = (w == true_label)
                    preds_for_label = y_pred[mask].cpu().numpy().tolist()
                    preds[true_label].extend(preds_for_label)
        
        # KDE per side effect frequency/class
        plt.figure(figsize=(10, 6))
        
        if task == "classification":
            # For classification: plot probability distributions for each true class
            for true_label in [0, 1]:
                if len(preds[true_label]) > 0:
                    sns.kdeplot(
                        preds[true_label],
                        label=f'True Class {true_label} ({"Absent" if true_label == 0 else "Present"})',
                        fill=True,
                        alpha=0.5
                    )
            plt.title(f'KDE of Predicted Probabilities ({split} set)')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.xlim(-0.1, 1.1)
        else:
            # For regression: plot score distributions for each true score
            for score in range(6):
                if len(preds[score]) > 0:
                    sns.kdeplot(
                        preds[score],
                        label=f'True Score {score}',
                        fill=True,
                        alpha=0.5
                    )
            plt.title(f'KDE of Predicted Scores ({split} set)')
            plt.xlabel('Predicted Score')
            plt.ylabel('Density')
            plt.xlim(-1, 6)
        
        plt.legend()
        plt.savefig(os.path.join(MODEL_FOLDER, task, model_name, f'kde_plot_{split}.png'))
        plt.close()

if __name__ == "__main__":
    a, b, c = get_loaders('./data/R.csv', 0.1, 0.1, 10)
    
    for X, A, y, smiles in a:
        x = X[0]
        a = A[0]
        y = y[0]
        smile = smiles[0]
        
        print(x.size(), a.size(), y.size(), smile)
        break

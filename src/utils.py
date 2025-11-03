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
    def __init__(self, smiles, side_effects_data, task="regression"):        
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
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        X, A = self.data[idx]
        y = self.side_effects_data[idx, :]
        smiles = self.smiles[idx]
        
        return X, A, y, smiles
    
    def build_graph(self, smiles: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(smiles)
        
        X = self.get_atom_features(mol)
        A = self.get_adjacency_matrix(mol)
        
        return X, A
    
    def get_adjacency_matrix(self, mol: Chem.Mol) -> torch.Tensor:
        num_atoms = mol.GetNumAtoms()

        A = torch.eye(num_atoms)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            A[i, j] = 1
            A[j, i] = 1

        return A

    def get_atom_features(self, mol: Chem.Mol) -> torch.Tensor:
        num_atoms = mol.GetNumAtoms()

        X = torch.zeros(num_atoms, self.num_different_atoms)

        for i, atom in enumerate(mol.GetAtoms()):
            X[i, self.atomic_num_order.index(atom.GetAtomicNum())] = 1

        return X
    
    def collate_fn(self, batch):
        X, A, y, smiles = zip(*batch)
        
        return X, A, y, smiles

def train_val_test_split(len_dataset: int, val_ratio: float, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len_dataset)
    np.random.shuffle(indices)
    
    val_size = int(len_dataset * val_ratio)
    test_size = int(len_dataset * test_ratio)
    
    val_indices = indices[:val_size]
    test_indices = indices[val_size:val_size + test_size]
    train_indices = indices[val_size + test_size:]
    
    return train_indices, val_indices, test_indices

def get_loaders(path, val_ratio, test_ratio, batch_size, task="regression"):
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


def save_model(name: str, model: torch.nn.Module, task: str = "regression"):
    folder_path = os.path.join(MODEL_FOLDER, task, name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save model weights only (not the full model object to save memory)
    model_path = os.path.join(folder_path, 'model_weights.pt')
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {folder_path}")

def save_losses(name: str, train_losses: list[float], val_losses: list[float] = None, task: str = "regression"):
    folder_path = os.path.join(MODEL_FOLDER, task, name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save losses
    losses_path = os.path.join(folder_path, 'losses.csv')
    if val_losses is None or len(val_losses) == 0:
        losses_df = pd.DataFrame({'train_losses': train_losses})
    else:
        losses_df = pd.DataFrame({'train_losses': train_losses, 'val_losses': val_losses})
    losses_df.to_csv(losses_path, index=False)

    # Plot losses and save fig
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    if val_losses and len(val_losses) > 0:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(folder_path, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Losses saved to {folder_path}")

def save_specs(name: str, specs: dict, task: str = "regression"):
    folder_path = os.path.join(MODEL_FOLDER, task, name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    df = pd.DataFrame.from_dict(specs, orient='index', columns=['value'])
    df.index.name = 'specification'
    df.to_csv(os.path.join(folder_path, 'specs.csv'))

def save_preds_kde(model, model_name, loader, model_type=None, task="regression"):
    model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, task, model_name, 'model_weights.pt')))
    
    model.eval() # freeze learning
    
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
                    if y_pred.dim() == 2:  # (num_atoms, 994)
                        y_pred = y_pred.sum(dim=0)  # (994,)
                
                # Ensure y_pred is 1D with shape (994,)
                if y_pred.dim() == 0:
                    raise ValueError(f"y_pred is a scalar! Shape: {y_pred.shape}, model_type: {model_type}")
                elif y_pred.dim() > 1:
                    y_pred = y_pred.squeeze()

                for j in range(6):
                    mask = (w == j)
                    
                    preds_for_i = y_pred[mask].cpu().numpy().tolist()
                    
                    preds[j].extend(preds_for_i)
        
        # KDE per side effect frequency        
        plt.figure(figsize=(10, 6))
        for i in range(6):
            sns.kdeplot(
                preds[i],
                label=f'Side Effect {i}',
                fill=True,
                alpha=0.5
            )
        plt.title('KDE of Predicted Side Effects')
        plt.xlabel('Predicted Frequency')
        plt.ylabel('Density')
        plt.xlim(-5, 10)
        plt.legend()
        plt.savefig(os.path.join(MODEL_FOLDER, task, model_name, 'kde_plot.png'))
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

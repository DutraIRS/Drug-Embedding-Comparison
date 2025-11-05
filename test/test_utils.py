import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure matplotlib to use non-interactive backend for tests
import matplotlib
matplotlib.use('Agg')

from src import utils
import pytest
import torch
import numpy as np

class TestDataDrivenLoss:
    def test_init_default(self):
        loss = utils.DataDrivenLoss()
        assert loss.alpha == 0.03
        assert loss.reconstruction_beta is None

    def test_init_custom(self):
        loss = utils.DataDrivenLoss(alpha=0.05, reconstruction_beta=0.1)
        assert loss.alpha == 0.05
        assert loss.reconstruction_beta == 0.1

    def test_forward_without_reconstruction(self):
        loss_fn = utils.DataDrivenLoss(alpha=0.03)
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.5, 0.0, 3.0])
        loss_value = loss_fn(y_pred, y_true)
        assert loss_value > 0
        assert isinstance(loss_value, torch.Tensor)

    def test_forward_with_reconstruction(self):
        loss_fn = utils.DataDrivenLoss(alpha=0.03, reconstruction_beta=0.1)
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.5, 0.0, 3.0])
        x = torch.tensor([1.0, 2.0, 3.0])
        x_reconstructed = torch.tensor([1.1, 2.1, 2.9])
        loss_value = loss_fn(y_pred, y_true, x, x_reconstructed)
        assert loss_value > 0

    def test_forward_zero_scaling(self):
        """Test that zeros get alpha scaling"""
        loss_fn = utils.DataDrivenLoss(alpha=0.01)
        y_pred = torch.tensor([1.0, 1.0])
        y_true = torch.tensor([0.0, 1.0])
        loss_value = loss_fn(y_pred, y_true)
        assert loss_value > 0

class TestWeightedBCELoss:
    def test_init_default(self):
        loss = utils.WeightedBCELoss()
        assert loss.alpha == 0.03
        assert loss.reconstruction_beta is None

    def test_init_custom(self):
        loss = utils.WeightedBCELoss(alpha=0.05, reconstruction_beta=0.2)
        assert loss.alpha == 0.05
        assert loss.reconstruction_beta == 0.2

    def test_forward_without_reconstruction(self):
        loss_fn = utils.WeightedBCELoss(alpha=0.03)
        y_pred = torch.tensor([0.5, -0.5, 2.0])
        y_true = torch.tensor([1.0, 0.0, 1.0])
        loss_value = loss_fn(y_pred, y_true)
        assert loss_value > 0
        assert isinstance(loss_value, torch.Tensor)

    def test_forward_with_reconstruction(self):
        loss_fn = utils.WeightedBCELoss(alpha=0.03, reconstruction_beta=0.1)
        y_pred = torch.tensor([0.5, -0.5])
        y_true = torch.tensor([1.0, 0.0])
        x = torch.tensor([1.0, 2.0])
        x_reconstructed = torch.tensor([1.1, 2.1])
        loss_value = loss_fn(y_pred, y_true, x, x_reconstructed)
        assert loss_value > 0

    def test_sigmoid_clamping(self):
        """Test that sigmoid is applied and values are clamped"""
        loss_fn = utils.WeightedBCELoss()
        y_pred = torch.tensor([100.0, -100.0])  # Extreme values
        y_true = torch.tensor([1.0, 0.0])
        loss_value = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss_value)

class TestDrugSideEffectsDataset:
    def test_init_regression(self):
        smiles = ["CCO", "CC"]
        side_effects = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float)
        dataset = utils.DrugSideEffectsDataset(smiles, side_effects, task="regression")
        assert len(dataset) == 2
        assert dataset.task == "regression"

    def test_init_classification(self):
        smiles = ["CCO", "CC"]
        side_effects = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
        dataset = utils.DrugSideEffectsDataset(smiles, side_effects, task="classification")
        assert len(dataset) == 2
        assert dataset.task == "classification"

    def test_getitem(self):
        smiles = ["CCO", "CC"]
        side_effects = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float)
        dataset = utils.DrugSideEffectsDataset(smiles, side_effects)
        x, a, y, smile = dataset[0]
        assert x.shape[1] == dataset.num_different_atoms  # One-hot encoding size depends on unique atoms
        assert a.shape[0] == a.shape[1]  # Square adjacency matrix
        assert y.shape[0] == 3  # 3 side effects
        assert smile == "CCO"

    def test_collate_fn(self):
        smiles = ["CCO", "CC", "C"]
        side_effects = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
        dataset = utils.DrugSideEffectsDataset(smiles, side_effects)
        batch = [dataset[0], dataset[1], dataset[2]]
        X, A, Y, smiles_batch = dataset.collate_fn(batch)
        assert len(X) == 3
        assert len(A) == 3
        assert len(Y) == 3
        assert len(smiles_batch) == 3

class TestDataLoaders:
    def test_get_loaders_regression(self):
        loaders = utils.get_loaders('./data/R_100.csv', 0.2, 0.2, 32, task="regression")
        train_loader, val_loader, test_loader = loaders
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_get_loaders_classification(self):
        loaders = utils.get_loaders('./data/R_100.csv', 0.2, 0.2, 32, task="classification")
        train_loader, val_loader, test_loader = loaders
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

class TestSaveFunctions:
    def test_save_specs(self):
        model_name = "test_model"
        specs = {"test_key": "test_value", "learning_rate": 0.001}
        utils.save_specs(model_name, specs, task="regression")
        
        # Check if file was created
        expected_path = f"./saved_models/regression/{model_name}/specs.csv"
        assert os.path.exists(expected_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(f"./saved_models/regression/{model_name}")

    def test_save_losses(self):
        model_name = "test_model_losses"
        train_losses = [0.5, 0.4, 0.3]
        val_losses = [0.6, 0.5, 0.4]
        utils.save_losses(model_name, train_losses, val_losses, task="regression")
        
        # Check if file was created
        expected_path = f"./saved_models/regression/{model_name}/losses.csv"
        assert os.path.exists(expected_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(f"./saved_models/regression/{model_name}")

    def test_save_model(self):
        model_name = "test_model_weights"
        # Create a simple model
        model = torch.nn.Linear(10, 5)
        utils.save_model(model_name, model, task="regression")
        
        # Check if file was created
        expected_path = f"./saved_models/regression/{model_name}/model_weights.pt"
        assert os.path.exists(expected_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(f"./saved_models/regression/{model_name}")

    def test_save_preds_kde_regression(self):
        """Test save_preds_kde for regression task"""
        from src.models import FCNN
        
        # Create a simple model and save it
        model_name = "test_kde_regression"
        model = FCNN(input_dim=2, hidden_dim=16, num_hidden=1, output_dim=3)
        utils.save_model(model_name, model, task="regression")
        
        # Create simple dataset
        smiles = ["CCO", "CC", "C"]
        side_effects = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [2.0, 0.0, 4.0]], dtype=torch.float)
        dataset = utils.DrugSideEffectsDataset(smiles, side_effects, task="regression")
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
        
        # Test save_preds_kde
        utils.save_preds_kde(model, model_name, loader, model_type="FCNN", task="regression", split="test")
        
        # Check if file was created
        expected_path = f"./saved_models/regression/{model_name}/kde_plot_test.png"
        assert os.path.exists(expected_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(f"./saved_models/regression/{model_name}")

class TestUtilityFunctions:
    def test_train_val_test_split(self):
        """Test the train_val_test_split function"""
        from src.utils import train_val_test_split
        
        len_dataset = 100
        val_ratio = 0.2
        test_ratio = 0.2
        
        train_idx, val_idx, test_idx = train_val_test_split(len_dataset, val_ratio, test_ratio)
        
        # Check sizes
        assert len(val_idx) == 20
        assert len(test_idx) == 20
        assert len(train_idx) == 60
        
        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0
        
        # Check all indices covered
        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_indices) == len_dataset
        assert set(all_indices) == set(range(len_dataset))

    def test_build_graph(self):
        """Test the build_graph method of DrugSideEffectsDataset"""
        smiles = ["CCO", "CC"]
        side_effects = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
        dataset = utils.DrugSideEffectsDataset(smiles, side_effects)
        
        # Test build_graph directly
        X, A = dataset.build_graph("CCO")
        
        # CCO has 3 atoms (C, C, O)
        assert X.shape[0] == 3
        assert A.shape == (3, 3)
        
        # Adjacency matrix should be symmetric
        assert torch.allclose(A, A.T)
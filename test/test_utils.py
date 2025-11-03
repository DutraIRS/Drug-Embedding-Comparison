import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils
import pytest
import torch

class TestDataDrivenLoss:
    def test_init_default(self):
        loss = utils.DataDrivenLoss()
        assert loss.alpha == 0.03

    def test_forward_without_reconstruction(self):
        loss_fn = utils.DataDrivenLoss(alpha=0.03)
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.5, 0.0, 3.0])
        loss_value = loss_fn(y_pred, y_true)
        assert loss_value > 0

class TestDrugSideEffectsDataset:
    def test_init(self):
        smiles = ["CCO", "CC"]
        side_effects = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
        dataset = utils.DrugSideEffectsDataset(smiles, side_effects)
        assert len(dataset) == 2

class TestDataLoaders:
    def test_get_loaders(self):
        loaders = utils.get_loaders('./data/R_100.csv', 0.2, 0.2, 32)
        train_loader, val_loader, test_loader = loaders
        assert train_loader is not None

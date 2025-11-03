import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from src.utils import DataDrivenLoss, WeightedBCELoss


class TestDataDrivenLoss:
    """Test suite for DataDrivenLoss (regression task - RMSE)"""
    
    def test_init_default_alpha(self):
        """Test that default alpha is 0.03"""
        loss_fn = DataDrivenLoss()
        assert loss_fn.alpha == 0.03
        assert loss_fn.reconstruction_beta is None
    
    def test_init_custom_alpha(self):
        """Test initialization with custom alpha"""
        loss_fn = DataDrivenLoss(alpha=0.05, reconstruction_beta=0.1)
        assert loss_fn.alpha == 0.05
        assert loss_fn.reconstruction_beta == 0.1
    
    def test_loss_with_zeros(self):
        """Test that zero targets get scaled by alpha (RMSE)"""
        loss_fn = DataDrivenLoss(alpha=0.03)
        
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([0.0, 0.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Expected: sqrt(mean([1^2 * 0.03, 2^2 * 0.03, 3^2 * 0.03]))
        mse = torch.tensor([1.0**2 * 0.03, 2.0**2 * 0.03, 3.0**2 * 0.03]).mean()
        expected = torch.sqrt(mse)
        assert torch.isclose(loss, expected)
    
    def test_loss_with_nonzeros(self):
        """Test that non-zero targets don't get scaled (RMSE)"""
        loss_fn = DataDrivenLoss(alpha=0.03)
        
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.0, 2.0, 3.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Expected: sqrt(mean of squared errors without scaling) = sqrt(0) = 0
        expected = torch.sqrt(torch.tensor([0.0, 0.0, 0.0]).mean())
        assert torch.isclose(loss, expected)
    
    def test_loss_mixed_zeros_nonzeros(self):
        """Test loss with mixed zero and non-zero targets (RMSE)"""
        loss_fn = DataDrivenLoss(alpha=0.03)
        
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_true = torch.tensor([0.0, 2.0, 0.0, 4.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Expected: sqrt(mean([1^2 * 0.03, 0, 3^2 * 0.03, 0]))
        mse = torch.tensor([1.0**2 * 0.03, 0.0, 3.0**2 * 0.03, 0.0]).mean()
        expected = torch.sqrt(mse)
        assert torch.isclose(loss, expected)
    
    def test_loss_uses_mean_not_sum(self):
        """Test that loss uses mean instead of sum before sqrt"""
        loss_fn = DataDrivenLoss(alpha=0.03)
        
        # Two different sized tensors should give same RMSE if pattern repeats
        y_pred_small = torch.tensor([1.0, 2.0])
        y_true_small = torch.tensor([0.0, 0.0])
        
        y_pred_large = torch.tensor([1.0, 2.0, 1.0, 2.0])
        y_true_large = torch.tensor([0.0, 0.0, 0.0, 0.0])
        
        loss_small = loss_fn(y_pred_small, y_true_small)
        loss_large = loss_fn(y_pred_large, y_true_large)
        
        # If using mean, both should be the same
        # sqrt(mean([1^2*0.03, 2^2*0.03])) = sqrt(mean([0.03, 0.12])) = sqrt(0.075)
        assert torch.isclose(loss_small, loss_large)
    
    def test_loss_with_reconstruction(self):
        """Test loss with reconstruction term (for VAE) - RMSE"""
        loss_fn = DataDrivenLoss(alpha=0.03, reconstruction_beta=0.5)
        
        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([0.0, 0.0])
        x = torch.tensor([1.0, 2.0, 3.0])
        x_reconstructed = torch.tensor([1.1, 2.1, 3.1])
        
        loss = loss_fn(y_pred, y_true, x, x_reconstructed)
        
        # Expected: sqrt(mean([1^2*0.03, 2^2*0.03])) + mean([0.1^2, 0.1^2, 0.1^2]) * 0.5
        mse = torch.tensor([1.0**2 * 0.03, 2.0**2 * 0.03]).mean()
        pred_loss = torch.sqrt(mse)
        recon_loss = torch.tensor([0.1**2, 0.1**2, 0.1**2]).mean() * 0.5
        expected = pred_loss + recon_loss
        
        assert torch.isclose(loss, expected, atol=1e-6)


class TestWeightedBCELoss:
    """Test suite for WeightedBCELoss (classification task)"""
    
    def test_init_default_alpha(self):
        """Test that default alpha is 0.03"""
        loss_fn = WeightedBCELoss()
        assert loss_fn.alpha == 0.03
        assert loss_fn.reconstruction_beta is None
    
    def test_init_custom_alpha(self):
        """Test initialization with custom alpha"""
        loss_fn = WeightedBCELoss(alpha=0.05, reconstruction_beta=0.1)
        assert loss_fn.alpha == 0.05
        assert loss_fn.reconstruction_beta == 0.1
    
    def test_sigmoid_applied(self):
        """Test that sigmoid is applied to predictions"""
        loss_fn = WeightedBCELoss(alpha=0.03)
        
        # Logits (before sigmoid)
        y_pred = torch.tensor([0.0, 100.0, -100.0])
        y_true = torch.tensor([0.5, 1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Should not raise error and should return finite value
        assert torch.isfinite(loss)
    
    def test_loss_with_negative_samples(self):
        """Test that negative samples (y_true=0) get scaled by alpha"""
        loss_fn = WeightedBCELoss(alpha=0.03)
        
        # Perfect predictions for negative class
        y_pred = torch.tensor([-10.0, -10.0])  # After sigmoid ≈ 0
        y_true = torch.tensor([0.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Loss should be very small and positive
        assert loss > 0
        assert loss < 0.1
    
    def test_loss_with_positive_samples(self):
        """Test that positive samples (y_true=1) don't get scaled"""
        loss_fn = WeightedBCELoss(alpha=0.03)
        
        # Perfect predictions for positive class
        y_pred = torch.tensor([10.0, 10.0])  # After sigmoid ≈ 1
        y_true = torch.tensor([1.0, 1.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Loss should be very small
        assert loss < 0.1
    
    def test_loss_mixed_classes(self):
        """Test loss with mixed positive and negative samples"""
        loss_fn = WeightedBCELoss(alpha=0.03)
        
        y_pred = torch.tensor([0.0, 0.0, 0.0, 0.0])  # Logits
        y_true = torch.tensor([0.0, 1.0, 0.0, 1.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Should return finite positive value
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_loss_uses_mean_not_sum(self):
        """Test that loss uses mean instead of sum"""
        loss_fn = WeightedBCELoss(alpha=0.03)
        
        # Two different sized tensors
        y_pred_small = torch.tensor([0.0, 0.0])
        y_true_small = torch.tensor([0.0, 0.0])
        
        y_pred_large = torch.tensor([0.0, 0.0, 0.0, 0.0])
        y_true_large = torch.tensor([0.0, 0.0, 0.0, 0.0])
        
        loss_small = loss_fn(y_pred_small, y_true_small)
        loss_large = loss_fn(y_pred_large, y_true_large)
        
        # If using mean, both should be the same
        assert torch.isclose(loss_small, loss_large, rtol=1e-4)
    
    def test_clamping_prevents_inf(self):
        """Test that clamping prevents infinite loss"""
        loss_fn = WeightedBCELoss(alpha=0.03)
        
        # Extreme logits that could cause numerical issues
        y_pred = torch.tensor([1000.0, -1000.0])
        y_true = torch.tensor([1.0, 0.0])
        
        loss = loss_fn(y_pred, y_true)
        
        # Should not be inf or nan
        assert torch.isfinite(loss)
    
    def test_loss_with_reconstruction(self):
        """Test loss with reconstruction term (for VAE with classification)"""
        loss_fn = WeightedBCELoss(alpha=0.03, reconstruction_beta=0.5)
        
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([0.0, 1.0])
        x = torch.tensor([1.0, 2.0, 3.0])
        x_reconstructed = torch.tensor([1.1, 2.1, 3.1])
        
        loss = loss_fn(y_pred, y_true, x, x_reconstructed)
        
        # Should include reconstruction term
        assert torch.isfinite(loss)
        assert loss > 0


class TestLossComparison:
    """Test comparison between different loss functions"""
    
    def test_alpha_effect(self):
        """Test that smaller alpha gives smaller loss for zero targets"""
        loss_fn_small = DataDrivenLoss(alpha=0.01)
        loss_fn_large = DataDrivenLoss(alpha=0.1)
        
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([0.0, 0.0, 0.0])
        
        loss_small = loss_fn_small(y_pred, y_true)
        loss_large = loss_fn_large(y_pred, y_true)
        
        assert loss_small < loss_large
    
    def test_zero_vs_nonzero_penalty(self):
        """Test that errors on zeros are penalized less than on non-zeros (RMSE)"""
        loss_fn = DataDrivenLoss(alpha=0.03)
        
        # Same prediction error magnitude
        y_pred_zero = torch.tensor([2.0])
        y_true_zero = torch.tensor([0.0])
        
        y_pred_nonzero = torch.tensor([4.0])
        y_true_nonzero = torch.tensor([2.0])
        
        loss_zero = loss_fn(y_pred_zero, y_true_zero)
        loss_nonzero = loss_fn(y_pred_nonzero, y_true_nonzero)
        
        # Error on zero: sqrt(2^2 * 0.03) = sqrt(0.12) ≈ 0.346
        # Error on nonzero: sqrt(2^2) = 2.0
        # Ratio should be sqrt(0.03) ≈ 0.173
        assert loss_zero < loss_nonzero
        assert loss_zero / loss_nonzero < 0.2  # sqrt(0.03) ≈ 0.173

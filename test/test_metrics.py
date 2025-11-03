import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


class TestAUROCCalculation:
    """Tests for AUROC metric calculation in classification task."""
    
    def test_perfect_predictions(self):
        """AUROC should be 1.0 for perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        
        auroc = roc_auc_score(y_true, y_pred)
        assert auroc == 1.0, "Perfect predictions should yield AUROC of 1.0"
    
    def test_random_predictions(self):
        """AUROC should be around 0.5 for random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_pred = np.random.rand(100)
        
        auroc = roc_auc_score(y_true, y_pred)
        # Random should be around 0.5, but we allow some variance
        assert 0.3 < auroc < 0.7, f"Random predictions should yield AUROC around 0.5, got {auroc}"
    
    def test_worst_predictions(self):
        """AUROC should be 0.0 for completely wrong predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        
        auroc = roc_auc_score(y_true, y_pred)
        assert auroc == 0.0, "Completely wrong predictions should yield AUROC of 0.0"
    
    def test_multi_side_effect_auroc(self):
        """Test AUROC calculation across multiple side effects."""
        # Simulate predictions for 3 side effects, 10 drugs
        n_drugs = 10
        n_side_effects = 3
        
        y_true = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ])
        
        # Good predictions for side effect 0
        # Perfect for side effect 1
        # Random for side effect 2
        y_pred = np.array([
            [0.1, 0.9, 0.3],
            [0.9, 0.1, 0.7],
            [0.2, 0.8, 0.6],
            [0.8, 0.0, 0.2],
            [0.1, 0.9, 0.8],
            [0.7, 1.0, 0.4],
            [0.3, 0.1, 0.9],
            [0.9, 0.8, 0.3],
            [0.2, 0.0, 0.7],
            [0.8, 0.9, 0.9]
        ])
        
        aurocs = []
        for se_idx in range(n_side_effects):
            if len(np.unique(y_true[:, se_idx])) > 1:
                auroc = roc_auc_score(y_true[:, se_idx], y_pred[:, se_idx])
                aurocs.append(auroc)
        
        assert len(aurocs) == 3, "Should calculate AUROC for all 3 side effects"
        assert aurocs[0] > 0.7, "Good predictions should have high AUROC"
        assert aurocs[1] == 1.0, "Perfect predictions should have AUROC of 1.0"
        # Side effect 2 is more random, so we don't check it strictly
        
        mean_auroc = np.mean(aurocs)
        assert 0.5 < mean_auroc <= 1.0, f"Mean AUROC should be reasonable, got {mean_auroc}"
    
    def test_single_class_handling(self):
        """Test that single-class side effects are handled properly."""
        # All zeros for one side effect
        y_true_single = np.zeros(10)
        y_pred_single = np.random.rand(10)
        
        # Should raise an error or be skipped
        with pytest.raises(ValueError):
            roc_auc_score(y_true_single, y_pred_single)
    
    def test_auroc_range(self):
        """AUROC should always be between 0 and 1."""
        np.random.seed(123)
        for _ in range(10):
            y_true = np.random.randint(0, 2, size=50)
            y_pred = np.random.rand(50)
            
            # Only calculate if we have both classes
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_pred)
                assert 0 <= auroc <= 1, f"AUROC should be in [0, 1], got {auroc}"


class TestRMSECalculation:
    """Tests for RMSE metric calculation in regression task."""
    
    def test_perfect_predictions(self):
        """RMSE should be 0.0 for perfect predictions."""
        y_true = np.array([1.5, 2.3, 3.7, 0.5])
        y_pred = np.array([1.5, 2.3, 3.7, 0.5])
        
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        assert rmse == 0.0, "Perfect predictions should yield RMSE of 0.0"
    
    def test_constant_error(self):
        """RMSE should equal the constant error."""
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        assert rmse == 1.0, "Constant error of 1.0 should yield RMSE of 1.0"
    
    def test_rmse_vs_mse(self):
        """RMSE should be the square root of MSE."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5])
        
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        calculated_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        assert np.isclose(rmse, calculated_rmse), "RMSE should equal sqrt(MSE)"
    
    def test_multi_side_effect_rmse(self):
        """Test RMSE calculation across multiple side effects."""
        # Simulate predictions for 3 side effects, 5 drugs
        y_true = np.array([
            [0.0, 2.5, 1.0],
            [3.0, 0.0, 2.0],
            [1.5, 1.0, 0.0],
            [0.0, 3.5, 2.5],
            [2.0, 0.5, 1.5]
        ])
        
        y_pred = np.array([
            [0.5, 2.0, 1.2],
            [2.8, 0.3, 2.1],
            [1.7, 0.8, 0.2],
            [0.2, 3.3, 2.3],
            [1.9, 0.6, 1.4]
        ])
        
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        assert 0 < rmse < 1, f"RMSE should be reasonable for small errors, got {rmse}"
    
    def test_rmse_non_negative(self):
        """RMSE should always be non-negative."""
        np.random.seed(456)
        for _ in range(10):
            y_true = np.random.rand(20) * 5
            y_pred = np.random.rand(20) * 5
            
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            assert rmse >= 0, f"RMSE should be non-negative, got {rmse}"

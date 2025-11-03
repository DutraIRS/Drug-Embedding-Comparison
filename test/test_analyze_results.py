"""
Tests for analyze_results.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import tempfile
from pathlib import Path

# Import analyze_results module
import importlib.util


class TestCollectResults:
    """Test collect_all_results function"""
    
    def setup_method(self):
        """Setup test environment"""
        spec = importlib.util.spec_from_file_location(
            "analyze_results", "src/analyze_results.py"
        )
        self.analyze_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.analyze_module)
        
        # Create temporary directory for test models
        self.temp_dir = tempfile.mkdtemp()
        self.original_model_folder = self.analyze_module.MODEL_FOLDER
        self.analyze_module.MODEL_FOLDER = self.temp_dir + "/"
    
    def teardown_method(self):
        """Cleanup"""
        self.analyze_module.MODEL_FOLDER = self.original_model_folder
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_dummy_model(self, name, model_type, val_loss, n_params=1000):
        """Helper to create dummy model specs"""
        model_dir = Path(self.temp_dir) / name
        model_dir.mkdir(exist_ok=True)
        
        specs_data = {
            'specification': ['model_name', 'model_type', 'n_parameters', 
                            'best_val_loss', 'learning_rate', 'weight_decay'],
            'value': [name, model_type, n_params, val_loss, 0.0001, 1e-6]
        }
        specs_df = pd.DataFrame(specs_data)
        specs_df.to_csv(model_dir / 'specs.csv', index=False)
    
    def test_collect_empty_folder(self):
        """Test collecting from empty folder"""
        df = self.analyze_module.collect_all_results()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_collect_single_model(self):
        """Test collecting single model"""
        self.create_dummy_model('GCN_test', 'GCN', 0.5)
        
        df = self.analyze_module.collect_all_results()
        
        assert len(df) == 1
        assert df.iloc[0]['model_name'] == 'GCN_test'
        assert df.iloc[0]['model_type'] == 'GCN'
        assert df.iloc[0]['best_val_loss'] == 0.5
    
    def test_collect_multiple_models(self):
        """Test collecting multiple models"""
        self.create_dummy_model('GCN_1', 'GCN', 0.5)
        self.create_dummy_model('GAT_1', 'GAT', 0.4)
        self.create_dummy_model('Transformer_1', 'Transformer', 0.3)
        
        df = self.analyze_module.collect_all_results()
        
        assert len(df) == 3
        assert set(df['model_type']) == {'GCN', 'GAT', 'Transformer'}
    
    def test_collect_with_missing_specs(self):
        """Test collecting when some models don't have specs.csv"""
        self.create_dummy_model('GCN_1', 'GCN', 0.5)
        
        # Create directory without specs
        bad_dir = Path(self.temp_dir) / 'bad_model'
        bad_dir.mkdir(exist_ok=True)
        
        df = self.analyze_module.collect_all_results()
        
        # Should only collect the good one
        assert len(df) == 1


class TestFindBestConfigs:
    """Test find_best_configs function"""
    
    def setup_method(self):
        """Setup test environment"""
        spec = importlib.util.spec_from_file_location(
            "analyze_results", "src/analyze_results.py"
        )
        self.analyze_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.analyze_module)
    
    def test_find_best_single_type(self):
        """Test finding best config with single model type"""
        data = {
            'model_name': ['GCN_1', 'GCN_2', 'GCN_3'],
            'model_type': ['GCN', 'GCN', 'GCN'],
            'best_val_loss': [0.5, 0.3, 0.7],
            'n_parameters': [1000, 1000, 1000]
        }
        df = pd.DataFrame(data)
        
        best = self.analyze_module.find_best_configs(df)
        
        assert len(best) == 1
        assert best.iloc[0]['model_name'] == 'GCN_2'
        assert best.iloc[0]['best_val_loss'] == 0.3
    
    def test_find_best_multiple_types(self):
        """Test finding best configs for multiple model types"""
        data = {
            'model_name': ['GCN_1', 'GCN_2', 'GAT_1', 'GAT_2', 'FCNN_1'],
            'model_type': ['GCN', 'GCN', 'GAT', 'GAT', 'FCNN'],
            'best_val_loss': [0.5, 0.3, 0.4, 0.2, 0.6],
            'n_parameters': [1000, 1000, 1000, 1000, 1000]
        }
        df = pd.DataFrame(data)
        
        best = self.analyze_module.find_best_configs(df)
        
        assert len(best) == 3
        assert set(best['model_type']) == {'GCN', 'GAT', 'FCNN'}
        assert best[best['model_type'] == 'GCN'].iloc[0]['model_name'] == 'GCN_2'
        assert best[best['model_type'] == 'GAT'].iloc[0]['model_name'] == 'GAT_2'
    
    def test_best_configs_sorted(self):
        """Test that best configs are sorted by val_loss"""
        data = {
            'model_name': ['GCN_1', 'GAT_1', 'FCNN_1'],
            'model_type': ['GCN', 'GAT', 'FCNN'],
            'best_val_loss': [0.5, 0.2, 0.7],
            'n_parameters': [1000, 1000, 1000]
        }
        df = pd.DataFrame(data)
        
        best = self.analyze_module.find_best_configs(df)
        
        # Should be sorted by val_loss
        assert best.iloc[0]['model_type'] == 'GAT'  # Lowest loss
        assert best.iloc[1]['model_type'] == 'GCN'
        assert best.iloc[2]['model_type'] == 'FCNN'  # Highest loss


class TestIntegration:
    """Integration tests for analyze_results"""
    
    def setup_method(self):
        """Setup test environment"""
        spec = importlib.util.spec_from_file_location(
            "analyze_results", "src/analyze_results.py"
        )
        self.analyze_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.analyze_module)
        
        self.temp_dir = tempfile.mkdtemp()
        self.original_model_folder = self.analyze_module.MODEL_FOLDER
        self.original_output_file = self.analyze_module.OUTPUT_FILE
        
        self.analyze_module.MODEL_FOLDER = self.temp_dir + "/"
        self.analyze_module.OUTPUT_FILE = self.temp_dir + "/best_configs.csv"
    
    def teardown_method(self):
        """Cleanup"""
        self.analyze_module.MODEL_FOLDER = self.original_model_folder
        self.analyze_module.OUTPUT_FILE = self.original_output_file
        
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_dummy_model(self, name, model_type, val_loss, n_params=1000):
        """Helper to create dummy model specs"""
        model_dir = Path(self.temp_dir) / name
        model_dir.mkdir(exist_ok=True)
        
        specs_data = {
            'specification': ['model_name', 'model_type', 'n_parameters', 
                            'best_val_loss', 'learning_rate', 'weight_decay'],
            'value': [name, model_type, n_params, val_loss, 0.0001, 1e-6]
        }
        specs_df = pd.DataFrame(specs_data)
        specs_df.to_csv(model_dir / 'specs.csv', index=False)
    
    def test_full_pipeline(self):
        """Test complete analysis pipeline"""
        # Create dummy models
        self.create_dummy_model('GCN_1', 'GCN', 0.5, 1000)
        self.create_dummy_model('GCN_2', 'GCN', 0.3, 1200)
        self.create_dummy_model('GAT_1', 'GAT', 0.4, 1500)
        self.create_dummy_model('Transformer_1', 'Transformer', 0.2, 2000)
        
        # Collect results
        df = self.analyze_module.collect_all_results()
        assert len(df) == 4
        
        # Find best configs
        best = self.analyze_module.find_best_configs(df)
        assert len(best) == 3  # 3 different model types
        
        # Check best config for each type
        assert best[best['model_type'] == 'GCN'].iloc[0]['model_name'] == 'GCN_2'
        assert best[best['model_type'] == 'GAT'].iloc[0]['model_name'] == 'GAT_1'
        assert best[best['model_type'] == 'Transformer'].iloc[0]['model_name'] == 'Transformer_1'


class TestHelperFunctions:
    """Test helper functions"""
    
    def setup_method(self):
        """Setup test environment"""
        spec = importlib.util.spec_from_file_location(
            "analyze_results", "src/analyze_results.py"
        )
        self.analyze_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.analyze_module)
    
    def test_print_summary_runs(self):
        """Test print_summary doesn't crash"""
        data = {
            'model_name': ['GCN_1', 'GAT_1'],
            'model_type': ['GCN', 'GAT'],
            'best_val_loss': [0.5, 0.3],
            'n_parameters': [1000, 1500],
            'learning_rate': [0.0001, 0.0001],
            'weight_decay': [1e-6, 1e-6]
        }
        df = pd.DataFrame(data)
        best = df.copy()
        
        # Should not raise exception
        try:
            self.analyze_module.print_summary(df, best)
        except Exception as e:
            pytest.fail(f"print_summary raised exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

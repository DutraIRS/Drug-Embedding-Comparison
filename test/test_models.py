import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import models

class TestVAE:
    def test_init(self):
        """
        Test the initialization of the VAE model
        """
        model = models.VAE()
        assert model is not None
        assert isinstance(model, models.VAE)
        assert isinstance(model, models.nn.Module)
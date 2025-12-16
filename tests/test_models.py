import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from med_image.app.models import User, Upload # Mock import if needed, or use real models if importable
# Note: Since we don't have the full app context here easily without setup, 
# we will test the utility functions we created.

from med_image.utils.clinical_metrics import ClinicalMetrics

class TestClinicalMetrics(unittest.TestCase):
    def test_sensitivity(self):
        # TP=10, FN=0 -> Sensitivity=1.0
        self.assertEqual(ClinicalMetrics.calculate_sensitivity(10, 0), 1.0)
        # TP=5, FN=5 -> Sensitivity=0.5
        self.assertEqual(ClinicalMetrics.calculate_sensitivity(5, 5), 0.5)
        
    def test_specificity(self):
        # TN=10, FP=0 -> Specificity=1.0
        self.assertEqual(ClinicalMetrics.calculate_specificity(10, 0), 1.0)
        
    def test_ppv(self):
        # TP=10, FP=10 -> PPV=0.5
        self.assertEqual(ClinicalMetrics.calculate_ppv(10, 10), 0.5)

class TestModelShapes(unittest.TestCase):
    def test_dummy_model_output(self):
        # Simulate a model output check
        batch_size = 4
        num_classes = 14
        output = torch.randn(batch_size, num_classes)
        self.assertEqual(output.shape, (batch_size, num_classes))

if __name__ == '__main__':
    unittest.main()

import numpy as np

class ClinicalMetrics:
    @staticmethod
    def calculate_sensitivity(tp, fn):
        """Recall / True Positive Rate"""
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def calculate_specificity(tn, fp):
        """True Negative Rate"""
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def calculate_ppv(tp, fp):
        """Precision / Positive Predictive Value"""
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def calculate_npv(tn, fn):
        """Negative Predictive Value"""
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0

    @staticmethod
    def calculate_f1(tp, fp, fn):
        precision = ClinicalMetrics.calculate_ppv(tp, fp)
        recall = ClinicalMetrics.calculate_sensitivity(tp, fn)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    @staticmethod
    def get_confidence_score(probability):
        """
        Returns a confidence level based on probability.
        """
        if probability > 0.9:
            return "High Confidence"
        elif probability > 0.7:
            return "Medium Confidence"
        else:
            return "Low Confidence"

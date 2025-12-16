# safe_range.py

# Medical-Grade Thresholds (Calibrated for CheXNet/DenseNet121)
# These values are lower because medical models trained on real-world data
# often output low absolute probabilities even for positive cases.
# A score > 0.06 (6%) for Hernia or > 0.08 (8%) for Edema is often clinically significant.

safe_ranges = {
    # High Sensitivity Group (Hard to detect, so threshold is LOW)
    'Hernia': {'val': 0.05, 'direction': 'below'},      # Very rare, ANY signal is suspicious
    'Pneumonia': {'val': 0.08, 'direction': 'below'},   # Often confused with others
    'Fibrosis': {'val': 0.08, 'direction': 'below'},
    'Edema': {'val': 0.08, 'direction': 'below'},       # FLUID: Critical to catch early
    'Emphysema': {'val': 0.08, 'direction': 'below'},
    
    # Medium Sensitivity Group
    'Cardiomegaly': {'val': 0.10, 'direction': 'below'}, # Heart size
    'Pleural_Thickening': {'val': 0.10, 'direction': 'below'},
    'Consolidation': {'val': 0.10, 'direction': 'below'},
    'Pneumothorax': {'val': 0.10, 'direction': 'below'}, # Collapsed lung
    'Mass': {'val': 0.12, 'direction': 'below'},
    'Nodule': {'val': 0.12, 'direction': 'below'},

    # Lower Sensitivity Group (Model is usually confident here)
    'Atelectasis': {'val': 0.15, 'direction': 'below'},
    'Effusion': {'val': 0.15, 'direction': 'below'},
    'Infiltration': {'val': 0.15, 'direction': 'below'}
}
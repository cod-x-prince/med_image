"""
Application configuration with proper model path handling.
Place in: src/med_image/config.py
"""

import os
from pathlib import Path

class Config:
    """Application configuration."""
    
    # Base directory
    BASE_DIR = Path(__file__).parent.parent.parent
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', '1') == '1'
    
    # Upload settings
    # Correctly point to src/med_image/app/static/uploads
    UPLOAD_FOLDER = BASE_DIR / 'src' / 'med_image' / 'app' / 'static' / 'uploads'
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS = {'.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    
    # Model paths (multiple fallbacks)
    MODEL_PATHS = {
        'primary': BASE_DIR / 'src' / 'med_image' / 'models' / 'chexnet_model.pth',
        'balanced': BASE_DIR / 'src' / 'med_image' / 'models' / 'balanced_robust_chexnet.pth',
        'highres': BASE_DIR / 'src' / 'med_image' / 'models' / 'high_res_chexnet.pth',
        'backup': BASE_DIR / 'src' / 'med_image' / 'models' / 'chexnet_real_finetuned.pth'
    }
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///medimage.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Celery
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'memory://')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'db+sqlite:///results.db')
    CELERY_TASK_ALWAYS_EAGER = os.environ.get('CELERY_TASK_ALWAYS_EAGER', 'true') == 'true'
    CELERY_TASK_STORE_EAGER_RESULT = True # Ensure this is set for eager mode result retrieval
    
    # PACS
    PACS_SERVER = os.environ.get('PACS_SERVER', 'localhost')
    PACS_PORT = int(os.environ.get('PACS_PORT', 11112))
    
    # Security
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'false') == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    @classmethod
    def get_model_path(cls):
        """Get the first available model path."""
        for name, path in cls.MODEL_PATHS.items():
            if path.exists():
                return path
        return None
    
    @classmethod
    def init_app(cls, app):
        """Initialize application with config."""
        # Create necessary directories
        directories = [
            cls.UPLOAD_FOLDER,
            cls.UPLOAD_FOLDER.parent / 'heatmaps',
            cls.UPLOAD_FOLDER.parent / 'reports',
            Path(cls.SQLALCHEMY_DATABASE_URI.replace('sqlite:///', '')).parent if 'sqlite' in cls.SQLALCHEMY_DATABASE_URI else Path('.')
        ]
        
        for directory in directories:
            if isinstance(directory, Path):
                directory.mkdir(parents=True, exist_ok=True)
        
        # Set upload folder
        app.config['UPLOAD_FOLDER'] = str(cls.UPLOAD_FOLDER)
        app.config['HEATMAP_FOLDER'] = str(cls.UPLOAD_FOLDER.parent / 'heatmaps')
        app.config['REPORT_FOLDER'] = str(cls.UPLOAD_FOLDER.parent / 'reports')



    # Smart Dual-Mode Deployment Config
    MODEL_CONFIGS = {
        'standard': {
            'path': BASE_DIR / 'src' / 'med_image' / 'models' / 'chexnet_model.pth',
            'resolution': 224,
            'batch_size': 32,
            'name': 'Standard Scan (224px)',
            'description': 'Fast analysis for routine screening'
        },
        'deep': {
            'path': BASE_DIR / 'src' / 'med_image' / 'models' / 'CheXNet_Stage2_Final.pth',
            'resolution': 1024,
            'batch_size': 8,
            'name': 'Deep Scan (1024px)',
            'description': 'High-resolution analysis for subtle findings'
        },
        'ensemble': {
            'paths': [
                BASE_DIR / 'src' / 'med_image' / 'models' / 'chexnet_model.pth',
                BASE_DIR / 'src' / 'med_image' / 'models' / 'balanced_robust_chexnet.pth'
            ],
            'weights': [0.6, 0.4],
            'resolutions': [224, 224],
            'name': 'Ensemble Scan',
            'description': 'Combined model analysis for maximum accuracy'
        }
    }
    
    PERFORMANCE_METRICS = {
        'standard': {
            'auc': 0.8956,
            'ap': 0.3844,
            'speed': 'Fast',  
            'use_cases': ['Routine Screening', 'Obvious Findings']
        },
        'deep': {
            'auc': 0.8690,
            'ap': 0.3013,
            'speed': 'Moderate', 
            'use_cases': ['Subtle Nodules', 'High Precision']
        },
        'ensemble': {
            'auc': 0.8961,
            'ap': 0.3858,
            'speed': 'Fast',
            'use_cases': ['Maximum Accuracy', 'Difficult Cases']
        }
    }

config = Config()

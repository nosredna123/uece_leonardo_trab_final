"""
Model Export Module for Transit Coverage Classification

This module exports trained models to ONNX format for efficient inference
and validates the exported models.
"""

import pickle
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Exports trained models to ONNX format.
    
    Attributes:
        models_dir: Directory containing trained models
        model: Loaded best model
        metadata: Model metadata dictionary
        feature_names: List of feature names
    """
    
    def __init__(self, models_dir: str = "models/transit_coverage"):
        """
        Initialize ModelExporter.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.metadata = {}
        self.feature_names = []
        
        logger.info(f"ModelExporter initialized")
        logger.info(f"Models directory: {self.models_dir}")
    
    def load_best_model(self) -> Any:
        """
        Load the best trained model.
        
        Returns:
            Loaded model object
        """
        logger.info("Loading best model...")
        
        model_path = self.models_dir / 'best_model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Best model not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Loaded model from {model_path}")
        
        # Load metadata if available
        metadata_path = self.models_dir / 'best_model_metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded metadata: {self.metadata.get('model_name', 'Unknown')}")
        
        return self.model
    
    def get_feature_names(self, test_path: str = "data/processed/features/test.parquet") -> list:
        """
        Extract feature names from test dataset.
        
        Args:
            test_path: Path to test parquet file
            
        Returns:
            List of feature names
        """
        logger.info("Extracting feature names from test dataset...")
        
        test_df = pd.read_parquet(test_path)
        
        # Get feature columns (same logic as training/evaluation)
        # IMPORTANT: Exclude normalized features to match training
        feature_cols = [col for col in test_df.columns 
                       if col not in ['cell_id', 'label', 'composite_score', 'threshold_used', 'weights']
                       and not col.endswith('_norm')  # Exclude normalized features
                       and not col.endswith('_normalized')]
        
        self.feature_names = feature_cols
        logger.info(f"Found {len(feature_cols)} features: {feature_cols}")
        
        return feature_cols
    
    def convert_to_onnx(self, output_path: str = None) -> str:
        """
        Convert model to ONNX format.
        
        Args:
            output_path: Output path for ONNX model (default: models/transit_coverage/best_model.onnx)
            
        Returns:
            Path to saved ONNX model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_best_model() first.")
        
        if not self.feature_names:
            self.get_feature_names()
        
        logger.info("Converting model to ONNX format...")
        
        # Define input type (number of features)
        n_features = len(self.feature_names)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        logger.info(f"Input shape: [None, {n_features}]")
        
        # Convert to ONNX
        try:
            onnx_model = convert_sklearn(
                self.model,
                initial_types=initial_type,
                target_opset=12,  # Use opset 12 for compatibility
                options={'zipmap': False}  # Disable zipmap for cleaner output
            )
            
            logger.info("Model successfully converted to ONNX")
        except Exception as e:
            logger.error(f"Error converting model to ONNX: {e}")
            raise
        
        # Save ONNX model
        if output_path is None:
            output_path = self.models_dir / 'best_model.onnx'
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        file_size_kb = output_path.stat().st_size / 1024
        file_size_mb = file_size_kb / 1024
        
        logger.info(f"ONNX model saved to {output_path}")
        logger.info(f"File size: {file_size_kb:.2f} KB ({file_size_mb:.2f} MB)")
        
        return str(output_path)
    
    def validate_onnx_model(self, 
                           onnx_path: str = None,
                           test_path: str = "data/processed/features/test.parquet",
                           n_samples: int = 100) -> bool:
        """
        Validate ONNX model predictions match original model.
        
        Args:
            onnx_path: Path to ONNX model
            test_path: Path to test data
            n_samples: Number of samples to test
            
        Returns:
            True if predictions match, False otherwise
        """
        logger.info("Validating ONNX model predictions...")
        
        if onnx_path is None:
            onnx_path = self.models_dir / 'best_model.onnx'
        else:
            onnx_path = Path(onnx_path)
        
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
        # Load test data
        test_df = pd.read_parquet(test_path)
        X_test = test_df[self.feature_names].values[:n_samples].astype(np.float32)
        
        logger.info(f"Testing with {len(X_test)} samples")
        
        # Get predictions from original model
        original_predictions = self.model.predict(X_test)
        original_probabilities = self.model.predict_proba(X_test)
        
        # Get predictions from ONNX model
        sess = rt.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        output_names = [output.name for output in sess.get_outputs()]
        
        onnx_outputs = sess.run(output_names, {input_name: X_test})
        onnx_predictions = onnx_outputs[0]
        onnx_probabilities = onnx_outputs[1]
        
        # Compare predictions
        predictions_match = np.array_equal(original_predictions, onnx_predictions)
        
        # Compare probabilities (allowing small numerical differences)
        prob_diff = np.abs(original_probabilities - onnx_probabilities).max()
        probabilities_match = prob_diff < 1e-5
        
        logger.info(f"Predictions match: {predictions_match}")
        logger.info(f"Probabilities match (max diff: {prob_diff:.2e}): {probabilities_match}")
        
        if predictions_match and probabilities_match:
            logger.info("✓ PASS: ONNX model validation successful")
            return True
        else:
            logger.warning("✗ FAIL: ONNX model predictions do not match original")
            return False
    
    def verify_file_size(self, onnx_path: str = None, max_size_mb: float = 100.0) -> bool:
        """
        Verify ONNX model file size is within constraint.
        
        Args:
            onnx_path: Path to ONNX model
            max_size_mb: Maximum allowed file size in MB
            
        Returns:
            True if size is within constraint, False otherwise
        """
        logger.info(f"Verifying file size constraint (max: {max_size_mb} MB)...")
        
        if onnx_path is None:
            onnx_path = self.models_dir / 'best_model.onnx'
        else:
            onnx_path = Path(onnx_path)
        
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
        file_size_bytes = onnx_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        logger.info(f"ONNX model size: {file_size_mb:.2f} MB")
        logger.info(f"Maximum allowed: {max_size_mb} MB")
        
        if file_size_mb <= max_size_mb:
            logger.info(f"✓ PASS: File size within constraint")
            return True
        else:
            logger.warning(f"✗ FAIL: File size exceeds constraint")
            return False
    
    def save_model_metadata(self, onnx_path: str = None):
        """
        Save comprehensive model metadata to JSON file.
        
        Args:
            onnx_path: Path to ONNX model (for size calculation)
        """
        logger.info("Saving model metadata...")
        
        if onnx_path is None:
            onnx_path = self.models_dir / 'best_model.onnx'
        else:
            onnx_path = Path(onnx_path)
        
        # Compile metadata
        metadata = {
            'model_name': self.metadata.get('model_name', 'Transit Coverage Classifier'),
            'model_type': self.metadata.get('model_type', 'unknown'),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'model_version': '1.0.0',
            'training_date': self.metadata.get('training_date', datetime.now().isoformat()),
            'export_date': datetime.now().isoformat(),
            'best_params': self.metadata.get('best_params', {}),
            'val_f1_score': self.metadata.get('val_f1_score', None),
            'cv_f1_score': self.metadata.get('cv_f1_score', None),
            'training_time_seconds': self.metadata.get('training_time_seconds', None),
            'onnx_model_path': str(onnx_path.name),
            'onnx_file_size_mb': onnx_path.stat().st_size / (1024 * 1024) if onnx_path.exists() else None,
            'target_classes': ['underserved', 'well_served'],
            'class_labels': [0, 1]
        }
        
        # Save to JSON
        output_file = self.models_dir / 'model_metadata.json'
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {output_file}")
        
        # Display metadata
        logger.info("\nModel Metadata:")
        for key, value in metadata.items():
            if key not in ['best_params', 'feature_names']:
                logger.info(f"  {key}: {value}")
    
    def export_pipeline(self):
        """Run complete export pipeline."""
        logger.info("="*60)
        logger.info("STARTING MODEL EXPORT PIPELINE")
        logger.info("="*60)
        
        # Load best model
        self.load_best_model()
        
        # Get feature names
        self.get_feature_names()
        
        # Convert to ONNX
        onnx_path = self.convert_to_onnx()
        
        # Validate ONNX model
        validation_passed = self.validate_onnx_model(onnx_path)
        
        # Verify file size
        size_ok = self.verify_file_size(onnx_path)
        
        # Save metadata
        self.save_model_metadata(onnx_path)
        
        logger.info("="*60)
        logger.info("MODEL EXPORT COMPLETE")
        logger.info("="*60)
        
        return {
            'onnx_path': onnx_path,
            'validation_passed': validation_passed,
            'size_ok': size_ok
        }


def main():
    """Main function to export model."""
    exporter = ModelExporter()
    results = exporter.export_pipeline()
    
    # Display summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"ONNX Model: {results['onnx_path']}")
    print(f"Validation: {'✓ PASSED' if results['validation_passed'] else '✗ FAILED'}")
    print(f"Size Check: {'✓ PASSED' if results['size_ok'] else '✗ FAILED'}")
    print("="*60)


if __name__ == "__main__":
    main()

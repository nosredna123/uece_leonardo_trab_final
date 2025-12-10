"""
Prediction Service for Transit Coverage Classification

This module provides prediction capabilities using the ONNX model.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

import onnxruntime as rt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for making predictions using ONNX model.
    
    Attributes:
        model_path: Path to ONNX model
        metadata: Model metadata dictionary
        session: ONNX Runtime inference session
        feature_names: List of expected feature names
        input_name: Name of model input
        output_names: Names of model outputs
    """
    
    def __init__(self, 
                 model_path: str = "models/transit_coverage/best_model.onnx",
                 metadata_path: str = "models/transit_coverage/model_metadata.json"):
        """
        Initialize PredictionService.
        
        Args:
            model_path: Path to ONNX model file
            metadata_path: Path to model metadata JSON file
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        
        self.session = None
        self.metadata = {}
        self.feature_names = []
        self.input_name = None
        self.output_names = []
        
        logger.info("PredictionService initializing...")
        self._load_model()
        self._load_metadata()
        logger.info("PredictionService ready")
    
    def _load_model(self):
        """Load ONNX model into inference session."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        logger.info(f"Loading ONNX model from {self.model_path}")
        
        # Create inference session
        self.session = rt.InferenceSession(str(self.model_path))
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Input name: {self.input_name}")
        logger.info(f"Output names: {self.output_names}")
    
    def _load_metadata(self):
        """Load model metadata from JSON file."""
        if not self.metadata_path.exists():
            logger.warning(f"Metadata not found at {self.metadata_path}")
            return
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata.get('feature_names', [])
        
        logger.info(f"Metadata loaded: {self.metadata.get('model_name', 'Unknown')}")
        logger.info(f"Expected features ({len(self.feature_names)}): {self.feature_names}")
    
    def validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and order features according to model expectations.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Ordered dictionary of features
            
        Raises:
            ValueError: If required features are missing or invalid
        """
        # Check for missing features
        missing_features = set(self.feature_names) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check for extra features (warning only)
        extra_features = set(features.keys()) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Ignoring extra features: {extra_features}")
        
        # Order features according to model expectations
        ordered_features = {name: features[name] for name in self.feature_names}
        
        # Validate feature types
        for name, value in ordered_features.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature '{name}' must be numeric, got {type(value)}")
        
        return ordered_features
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Validate and order features
        ordered_features = self.validate_features(features)
        
        # Convert to numpy array
        X = np.array([[ordered_features[name] for name in self.feature_names]], dtype=np.float32)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: X})
        
        prediction = int(outputs[0][0])
        probabilities = outputs[1][0].tolist()
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Map prediction to class label
        class_labels = self.metadata.get('target_classes', ['underserved', 'well_served'])
        predicted_class = class_labels[prediction]
        
        result = {
            'prediction': prediction,
            'predicted_class': predicted_class,
            'probabilities': {
                class_labels[0]: probabilities[0],
                class_labels[1]: probabilities[1]
            },
            'confidence': max(probabilities),
            'latency_ms': round(latency_ms, 2)
        }
        
        logger.debug(f"Prediction: {predicted_class} (latency: {latency_ms:.2f}ms)")
        
        return result
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        start_time = time.time()
        
        # Validate and convert all features
        X_list = []
        for features in features_list:
            ordered_features = self.validate_features(features)
            X_list.append([ordered_features[name] for name in self.feature_names])
        
        X = np.array(X_list, dtype=np.float32)
        
        # Run batch inference
        outputs = self.session.run(self.output_names, {self.input_name: X})
        
        predictions = outputs[0].tolist()
        probabilities_array = outputs[1]
        
        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000
        avg_latency_ms = total_latency_ms / len(features_list)
        
        # Map predictions to class labels
        class_labels = self.metadata.get('target_classes', ['underserved', 'well_served'])
        
        results = []
        for i, pred in enumerate(predictions):
            probs = probabilities_array[i].tolist()
            results.append({
                'prediction': int(pred),
                'predicted_class': class_labels[pred],
                'probabilities': {
                    class_labels[0]: probs[0],
                    class_labels[1]: probs[1]
                },
                'confidence': max(probs)
            })
        
        logger.info(f"Batch prediction: {len(features_list)} samples "
                   f"(total: {total_latency_ms:.2f}ms, avg: {avg_latency_ms:.2f}ms)")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.metadata.get('model_name', 'Unknown'),
            'model_type': self.metadata.get('model_type', 'Unknown'),
            'model_version': self.metadata.get('model_version', 'Unknown'),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'target_classes': self.metadata.get('target_classes', []),
            'val_f1_score': self.metadata.get('val_f1_score'),
            'export_date': self.metadata.get('export_date')
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Dictionary with health status
        """
        is_healthy = self.session is not None and len(self.feature_names) > 0
        
        return {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'model_loaded': self.session is not None,
            'metadata_loaded': len(self.feature_names) > 0,
            'model_version': self.metadata.get('model_version', 'Unknown')
        }


# Global service instance
_prediction_service = None


def get_prediction_service() -> PredictionService:
    """
    Get or create global prediction service instance.
    
    Returns:
        PredictionService instance
    """
    global _prediction_service
    
    if _prediction_service is None:
        _prediction_service = PredictionService()
    
    return _prediction_service

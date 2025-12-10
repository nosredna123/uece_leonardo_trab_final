"""
Label Generator for Transit Coverage Classification

This module generates binary labels (well-served vs underserved) based on
composite transit coverage scores.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Generates binary classification labels for grid cells.
    
    Labels are generated using a composite score that combines normalized
    transit features with configurable weights.
    
    Attributes:
        config: Configuration dictionary with label parameters
        weights: Feature weights for composite score
        threshold_quantile: Quantile threshold for classification
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize LabelGenerator with configuration.
        
        Args:
            config_path: Path to model configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # Extract label configuration
        self.weights = {
            'stops': self.config['features']['stop_count_weight'],
            'routes': self.config['features']['route_count_weight'],
            'trips': self.config['features']['daily_trips_weight']
        }
        self.threshold_quantile = self.config['labels']['threshold_quantile']
        self.min_minority_pct = self.config['labels']['min_minority_class_pct']
        
        logger.info(f"LabelGenerator initialized")
        logger.info(f"Feature weights: {self.weights}")
        logger.info(f"Threshold: {self.threshold_quantile*100:.0f}th percentile "
                   f"(top {(1-self.threshold_quantile)*100:.0f}% = well-served)")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_features(self, features_path: str = "data/processed/features/grid_features.parquet") -> pd.DataFrame:
        """
        Load normalized features from parquet file.
        
        Args:
            features_path: Path to features parquet file
            
        Returns:
            DataFrame with normalized features
        """
        logger.info(f"Loading features from {features_path}...")
        features = pd.read_parquet(features_path)
        logger.info(f"Loaded features for {len(features)} cells")
        
        # Verify required columns exist
        required_cols = ['cell_id', 'stop_count_norm', 'route_count_norm', 'daily_trips_norm']
        missing_cols = [col for col in required_cols if col not in features.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return features
    
    def calculate_composite_score(self, features: pd.DataFrame) -> pd.Series:
        """
        Calculate composite coverage score using weighted average.
        
        Formula: score = w1*stop_count_norm + w2*route_count_norm + w3*daily_trips_norm
        
        Args:
            features: DataFrame with normalized features
            
        Returns:
            Series with composite scores
        """
        logger.info("Calculating composite coverage scores...")
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            logger.warning(f"Weights sum to {weight_sum:.3f}, not 1.0. Normalizing...")
            for key in self.weights:
                self.weights[key] /= weight_sum
        
        # Calculate weighted composite score
        composite_score = (
            self.weights['stops'] * features['stop_count_norm'] +
            self.weights['routes'] * features['route_count_norm'] +
            self.weights['trips'] * features['daily_trips_norm']
        )
        
        logger.info(f"Composite score - Mean: {composite_score.mean():.3f}, "
                   f"Std: {composite_score.std():.3f}, "
                   f"Min: {composite_score.min():.3f}, "
                   f"Max: {composite_score.max():.3f}")
        
        return composite_score
    
    def calculate_threshold(self, composite_score: pd.Series) -> float:
        """
        Calculate threshold using quantile method.
        
        Args:
            composite_score: Series with composite scores
            
        Returns:
            Threshold value
        """
        threshold = composite_score.quantile(self.threshold_quantile)
        logger.info(f"Calculated threshold: {threshold:.3f} "
                   f"({self.threshold_quantile*100:.0f}th percentile)")
        
        return threshold
    
    def assign_labels(self, composite_score: pd.Series, threshold: float) -> pd.Series:
        """
        Assign binary labels based on threshold.
        
        Args:
            composite_score: Series with composite scores
            threshold: Classification threshold
            
        Returns:
            Series with binary labels (0=underserved, 1=well-served)
        """
        labels = (composite_score >= threshold).astype(int)
        
        # Calculate distribution
        well_served_count = labels.sum()
        underserved_count = len(labels) - well_served_count
        well_served_pct = well_served_count / len(labels) * 100
        underserved_pct = underserved_count / len(labels) * 100
        
        logger.info(f"Label distribution:")
        logger.info(f"  Well-served (1): {well_served_count} ({well_served_pct:.1f}%)")
        logger.info(f"  Underserved (0): {underserved_count} ({underserved_pct:.1f}%)")
        
        return labels
    
    def validate_labels(self, labels: pd.Series) -> bool:
        """
        Validate label distribution meets requirements.
        
        Args:
            labels: Series with binary labels
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating label distribution...")
        
        # Calculate minority class percentage
        label_counts = labels.value_counts()
        minority_pct = label_counts.min() / len(labels)
        
        # Check minimum minority class percentage
        min_ok = minority_pct >= self.min_minority_pct
        logger.info(f"Minority class: {minority_pct*100:.1f}% "
                   f"(min required: {self.min_minority_pct*100:.0f}%) - "
                   f"{'✓ PASS' if min_ok else '✗ FAIL'}")
        
        # Check expected well-served percentage (target ~30%)
        target_well_served = self.config['labels'].get('target_well_served_pct', 0.30)
        actual_well_served = labels.sum() / len(labels)
        well_served_ok = abs(actual_well_served - target_well_served) <= 0.05  # Allow 5% deviation
        logger.info(f"Well-served percentage: {actual_well_served*100:.1f}% "
                   f"(target: {target_well_served*100:.0f}%) - "
                   f"{'✓ PASS' if well_served_ok else '⚠ WARNING'}")
        
        validation_passed = min_ok
        
        if validation_passed:
            logger.info("✓ Label validation PASSED")
        else:
            logger.warning("✗ Label validation FAILED")
        
        return validation_passed
    
    def generate_labels(self) -> pd.DataFrame:
        """
        Generate labels for all grid cells.
        
        Returns:
            DataFrame with cell_id, composite_score, label, threshold_used, weights
        """
        logger.info("="*60)
        logger.info("Starting label generation...")
        logger.info("="*60)
        
        # Load features
        features = self.load_features()
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(features)
        
        # Calculate threshold
        threshold = self.calculate_threshold(composite_score)
        
        # Assign labels
        labels = self.assign_labels(composite_score, threshold)
        
        # Validate labels
        is_valid = self.validate_labels(labels)
        
        if not is_valid:
            logger.warning("Label validation failed, but continuing...")
        
        # Create labels DataFrame
        labels_df = pd.DataFrame({
            'cell_id': features['cell_id'],
            'composite_score': composite_score.values,
            'label': labels.values,
            'threshold_used': threshold,
            'weights': str(self.weights)  # Convert dict to string for storage
        })
        
        logger.info(f"Generated labels for {len(labels_df)} cells")
        
        return labels_df
    
    def save_labels(self,
                   labels_df: pd.DataFrame,
                   output_path: str = "data/processed/labels/grid_labels.parquet"):
        """
        Save labels to parquet file.
        
        Args:
            labels_df: Labels DataFrame
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        labels_df.to_parquet(output_path, index=False)
        
        logger.info(f"Labels saved to {output_path}")
        logger.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
    
    def load_labels(self, input_path: str = "data/processed/labels/grid_labels.parquet") -> pd.DataFrame:
        """
        Load labels from parquet file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Labels DataFrame
        """
        labels = pd.read_parquet(input_path)
        logger.info(f"Loaded labels for {len(labels)} cells from {input_path}")
        return labels


def main():
    """Main function to generate and save labels."""
    # Initialize generator
    generator = LabelGenerator()
    
    # Generate labels
    labels = generator.generate_labels()
    
    # Save labels
    generator.save_labels(labels)
    
    # Display summary statistics
    print("\n" + "="*60)
    print("Label Generation Summary")
    print("="*60)
    print(f"Total cells: {len(labels)}")
    print(f"\nLabel Distribution:")
    print(labels['label'].value_counts().sort_index())
    print(f"\nComposite Score Statistics:")
    print(labels['composite_score'].describe())
    print(f"\nThreshold: {labels['threshold_used'].iloc[0]:.3f}")
    print(f"Weights: {labels['weights'].iloc[0]}")
    print("\nFirst 10 cells:")
    print(labels[['cell_id', 'composite_score', 'label']].head(10))
    print("="*60)


if __name__ == "__main__":
    main()

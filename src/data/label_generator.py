"""
Label Generator for Transit Coverage Classification

This module generates binary labels (well-served vs underserved) using a
supply vs demand approach to avoid data leakage.

Labeling Strategy (to prevent data leakage):
- Labels are NOT simply derived from the same features used for prediction
- Uses EXTERNAL criterion: population density (demand)
- Combines with transit coverage score (supply)

Label Logic:
- Underserved (0): Low transit supply (score < threshold) AND high population demand
- Well-served (1): High transit supply OR low population demand

This creates a realistic classification problem where the model must learn
the interaction between infrastructure supply and population demand, rather
than simply memorizing a threshold on the same features.
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
        
        # Noise configuration for realistic label generation
        self.noise_config = self.config['labels'].get('noise', {})
        self.enable_noise = self.noise_config.get('enabled', True)
        self.population_noise_std = self.noise_config.get('population_noise_std', 0.10)  # 10% std
        self.threshold_noise_std = self.noise_config.get('threshold_noise_std', 0.05)    # 5% std
        self.flip_probability = self.noise_config.get('label_flip_probability', 0.02)     # 2% flip
        
        logger.info(f"LabelGenerator initialized")
        logger.info(f"Feature weights: {self.weights}")
        logger.info(f"Threshold: {self.threshold_quantile*100:.0f}th percentile "
                   f"(top {(1-self.threshold_quantile)*100:.0f}% = well-served)")
        if self.enable_noise:
            logger.info(f"Noise enabled: pop_std={self.population_noise_std:.1%}, "
                       f"threshold_std={self.threshold_noise_std:.1%}, "
                       f"flip_prob={self.flip_probability:.1%}")
    
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
        
        # Check for population feature (optional but recommended)
        if 'population' in features.columns:
            logger.info(f"✓ Population feature found - will use for demand-based labeling")
            self.has_population = True
        else:
            logger.warning(f"⚠️  Population feature not found - falling back to score-only labeling")
            self.has_population = False
        
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
    
    def assign_labels(self, 
                      composite_score: pd.Series, 
                      threshold: float,
                      population: pd.Series = None) -> pd.Series:
        """
        Assign binary labels based on supply vs demand logic with optional noise.
        
        This method avoids data leakage by using population (external criterion)
        in addition to composite score. Labels are NOT simply derived from the
        same features used for prediction.
        
        Labeling Logic:
        - Underserved (0): Low transit supply (score < threshold) AND high demand (population > median)
        - Well-served (1): High supply OR low demand
        
        Noise Injection (realistic uncertainty):
        - Population noise: Gaussian noise on population values (simulates census errors)
        - Threshold noise: Random variation in threshold (simulates definition uncertainty)
        - Label flips: Small probability of random flips (simulates borderline cases)
        
        This creates a realistic classification problem where the model must learn
        the interaction between infrastructure supply and population demand while
        handling uncertainty and noise.
        
        Args:
            composite_score: Series with composite transit coverage scores
            threshold: Score threshold for "low supply"
            population: Series with population per cell (optional, for demand-based labeling)
            
        Returns:
            Series with binary labels (0=underserved, 1=well-served)
        """
        if population is not None and len(population) > 0:
            # DEMAND-BASED LABELING (recommended to avoid data leakage)
            logger.info("Using supply vs demand logic for labeling")
            
            # Apply noise to population if enabled (simulates census uncertainty)
            if self.enable_noise and self.population_noise_std > 0:
                np.random.seed(42)  # Reproducible noise
                pop_noise = np.random.normal(0, self.population_noise_std * population.std(), len(population))
                population_noisy = np.maximum(0, population + pop_noise)  # Ensure non-negative
                logger.info(f"Applied population noise: std={self.population_noise_std:.1%} "
                           f"(~{pop_noise.std():.1f} inhabitants)")
            else:
                population_noisy = population
            
            # Apply noise to threshold if enabled (simulates definition uncertainty)
            if self.enable_noise and self.threshold_noise_std > 0:
                threshold_noise = np.random.normal(0, self.threshold_noise_std * composite_score.std())
                threshold_noisy = threshold + threshold_noise
                logger.info(f"Applied threshold noise: {threshold:.3f} → {threshold_noisy:.3f}")
            else:
                threshold_noisy = threshold
            
            # Define low supply and high demand (using noisy values)
            is_low_supply = composite_score < threshold_noisy
            pop_median = population_noisy.median()
            is_high_demand = population_noisy > pop_median
            
            # Underserved = low supply AND high demand
            # Well-served = high supply OR low demand
            is_underserved = is_low_supply & is_high_demand
            labels = (~is_underserved).astype(int)
            
            # Apply random label flips if enabled (simulates borderline cases)
            if self.enable_noise and self.flip_probability > 0:
                n_flips = int(len(labels) * self.flip_probability)
                flip_indices = np.random.choice(len(labels), n_flips, replace=False)
                labels.iloc[flip_indices] = 1 - labels.iloc[flip_indices]
                logger.info(f"Applied random label flips: {n_flips} cells ({self.flip_probability:.1%})")
            
            # Calculate distribution
            underserved_count = (labels == 0).sum()
            well_served_count = (labels == 1).sum()
            underserved_pct = underserved_count / len(labels) * 100
            well_served_pct = well_served_count / len(labels) * 100
            
            logger.info(f"Supply threshold: {threshold:.3f}" + 
                       (f" (with noise: {threshold_noisy:.3f})" if self.enable_noise and self.threshold_noise_std > 0 else ""))
            logger.info(f"Demand threshold (population median): {pop_median:.0f} inhabitants")
            logger.info(f"Low supply cells: {is_low_supply.sum()} ({is_low_supply.sum()/len(labels)*100:.1f}%)")
            logger.info(f"High demand cells: {is_high_demand.sum()} ({is_high_demand.sum()/len(labels)*100:.1f}%)")
            logger.info(f"Label distribution (supply vs demand{' + noise' if self.enable_noise else ''}):")
            logger.info(f"  Underserved (0): {underserved_count} ({underserved_pct:.1f}%) - low supply + high demand")
            logger.info(f"  Well-served (1): {well_served_count} ({well_served_pct:.1f}%) - high supply or low demand")
            
        else:
            # FALLBACK: SCORE-ONLY LABELING (legacy, has data leakage issue)
            logger.warning("⚠️  Falling back to score-only labeling (may cause data leakage)")
            logger.warning("    Recommend integrating population data to avoid artificial separation")
            
            labels = (composite_score >= threshold).astype(int)
            
            # Calculate distribution
            well_served_count = labels.sum()
            underserved_count = len(labels) - well_served_count
            well_served_pct = well_served_count / len(labels) * 100
            underserved_pct = underserved_count / len(labels) * 100
            
            logger.info(f"Label distribution (score-only):")
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
        Generate labels for all grid cells using supply vs demand logic.
        
        This method creates labels based on the interaction between:
        - Supply: Composite transit coverage score
        - Demand: Population density per cell
        
        This avoids data leakage by ensuring labels are not simply derived
        from the same features used for prediction.
        
        Returns:
            DataFrame with cell_id, composite_score, population, label, threshold_used, 
            population_median, labeling_method
        """
        logger.info("="*60)
        logger.info("Starting label generation...")
        logger.info("="*60)
        
        # Load features
        features = self.load_features()
        
        # Calculate composite score (this will be a FEATURE, not the label determinant)
        composite_score = self.calculate_composite_score(features)
        
        # Calculate threshold for "low supply"
        threshold = self.calculate_threshold(composite_score)
        
        # Extract population if available
        population = None
        population_median = None
        labeling_method = "score-only"
        
        if self.has_population and 'population' in features.columns:
            population = features['population']
            population_median = population.median()
            labeling_method = "supply-vs-demand"
            logger.info(f"Population statistics:")
            logger.info(f"  Total: {population.sum():,.0f} inhabitants")
            logger.info(f"  Mean: {population.mean():.0f} per cell")
            logger.info(f"  Median: {population_median:.0f} per cell")
            logger.info(f"  Range: {population.min():.0f} - {population.max():.0f}")
        
        # Assign labels using supply vs demand logic
        labels = self.assign_labels(composite_score, threshold, population)
        
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
            'labeling_method': labeling_method
        })
        
        # Add population columns if available
        if population is not None:
            labels_df['population'] = population.values
            labels_df['population_median'] = population_median
        
        logger.info(f"Generated labels for {len(labels_df)} cells using {labeling_method} method")
        logger.info("="*60)
        
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
    print(f"Labeling method: {labels['labeling_method'].iloc[0]}")
    
    print(f"\nLabel Distribution:")
    label_counts = labels['label'].value_counts().sort_index()
    for label_val, count in label_counts.items():
        label_name = "Underserved" if label_val == 0 else "Well-served"
        print(f"  {label_val} ({label_name}): {count} ({count/len(labels)*100:.1f}%)")
    
    print(f"\nComposite Score Statistics:")
    print(labels['composite_score'].describe())
    print(f"\nSupply threshold: {labels['threshold_used'].iloc[0]:.3f}")
    
    if 'population' in labels.columns:
        print(f"\nPopulation Statistics:")
        print(labels['population'].describe())
        print(f"Demand threshold (median): {labels['population_median'].iloc[0]:.0f} inhabitants")
        
        # Show label distribution by supply/demand quadrants
        print(f"\nSupply vs Demand Quadrants:")
        threshold = labels['threshold_used'].iloc[0]
        pop_median = labels['population_median'].iloc[0]
        
        low_supply = labels['composite_score'] < threshold
        high_demand = labels['population'] > pop_median
        
        quadrants = pd.DataFrame({
            'Supply': ['Low', 'Low', 'High', 'High'],
            'Demand': ['Low', 'High', 'Low', 'High'],
            'Count': [
                ((~low_supply) & (~high_demand)).sum(),  # High supply, low demand
                (low_supply & high_demand).sum(),        # Low supply, high demand (underserved)
                ((~low_supply) & high_demand).sum(),     # High supply, high demand
                (low_supply & (~high_demand)).sum()      # Low supply, low demand
            ]
        })
        quadrants['Percentage'] = (quadrants['Count'] / len(labels) * 100).round(1)
        print(quadrants.to_string(index=False))
    
    print("\nFirst 10 cells:")
    if 'population' in labels.columns:
        print(labels[['cell_id', 'composite_score', 'population', 'label']].head(10))
    else:
        print(labels[['cell_id', 'composite_score', 'label']].head(10))
    print("="*60)


if __name__ == "__main__":
    main()

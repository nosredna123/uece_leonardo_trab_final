"""
Data preprocessing utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """
    Class for data preprocessing operations
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            filepath: Path to data file
            
        Returns:
            DataFrame with loaded data
        """
        filepath = str(filepath)
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if strategy == 'drop':
            df = df.dropna()
        else:
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['int64', 'float64']:
                        if strategy == 'mean':
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif strategy == 'median':
                            df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale numerical features
        
        Args:
            X: Input features
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Scaled features
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42):
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


class DatasetPreparator:
    """
    Prepares dataset for model training by merging features and labels,
    then performing stratified train/validation/test splits.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize DatasetPreparator with configuration.
        
        Args:
            config_path: Path to model configuration YAML file
        """
        import yaml
        import logging
        
        self.logger = logging.getLogger(__name__)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.random_seed = self.config['training']['random_seed']
        self.test_size = self.config['training']['test_size']
        self.val_size = self.config['training']['val_size']
        
        self.logger.info(f"DatasetPreparator initialized")
        self.logger.info(f"Random seed: {self.random_seed}")
        self.logger.info(f"Split ratios - Test: {self.test_size}, Val: {self.val_size}, Train: {1-self.test_size-self.val_size}")
    
    def load_features(self, features_path: str = "data/processed/features/grid_features.parquet") -> pd.DataFrame:
        """Load features from parquet file."""
        self.logger.info(f"Loading features from {features_path}...")
        features = pd.read_parquet(features_path)
        self.logger.info(f"Loaded features for {len(features)} cells")
        return features
    
    def load_labels(self, labels_path: str = "data/processed/labels/grid_labels.parquet") -> pd.DataFrame:
        """Load labels from parquet file."""
        self.logger.info(f"Loading labels from {labels_path}...")
        labels = pd.read_parquet(labels_path)
        self.logger.info(f"Loaded labels for {len(labels)} cells")
        return labels
    
    def merge_features_labels(self, features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        """
        Merge features and labels on cell_id.
        
        Args:
            features: Features DataFrame
            labels: Labels DataFrame
            
        Returns:
            Merged DataFrame
        """
        self.logger.info("Merging features and labels...")
        
        # Merge on cell_id
        merged = features.merge(labels[['cell_id', 'label', 'composite_score']], on='cell_id', how='inner')
        
        self.logger.info(f"Merged dataset size: {len(merged)} cells")
        
        # Verify no data loss
        if len(merged) != len(features):
            self.logger.warning(f"Data loss during merge: {len(features)} -> {len(merged)}")
        
        return merged
    
    def prepare_dataset(self) -> tuple:
        """
        Prepare complete dataset by loading, merging, and splitting data.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info("="*60)
        self.logger.info("Starting dataset preparation...")
        self.logger.info("="*60)
        
        # Load data
        features = self.load_features()
        labels = self.load_labels()
        
        # Merge
        dataset = self.merge_features_labels(features, labels)
        
        # Split
        train_df, val_df, test_df = self.stratified_split(dataset)
        
        return train_df, val_df, test_df
    
    def stratified_split(self, dataset: pd.DataFrame) -> tuple:
        """
        Perform stratified train/validation/test split.
        
        Args:
            dataset: Merged dataset with features and labels
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info("Performing stratified split...")
        
        # First split: separate test set
        train_val, test = train_test_split(
            dataset,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=dataset['label']
        )
        
        self.logger.info(f"Test set split: {len(test)} cells ({self.test_size*100:.0f}%)")
        
        # Second split: separate validation from training
        # Calculate validation size as proportion of remaining data
        val_size_adjusted = self.val_size / (1 - self.test_size)
        
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            stratify=train_val['label']
        )
        
        self.logger.info(f"Validation set split: {len(val)} cells ({self.val_size*100:.0f}% of original)")
        self.logger.info(f"Training set split: {len(train)} cells ({(1-self.test_size-self.val_size)*100:.0f}% of original)")
        
        # Validate splits
        self.validate_splits(train, val, test, dataset)
        
        return train, val, test
    
    def validate_splits(self, train: pd.DataFrame, val: pd.DataFrame, 
                       test: pd.DataFrame, original: pd.DataFrame) -> bool:
        """
        Validate split sizes and class balance.
        
        Args:
            train: Training set
            val: Validation set
            test: Test set
            original: Original dataset
            
        Returns:
            True if validation passes
        """
        self.logger.info("Validating splits...")
        
        # Check sizes
        total_split = len(train) + len(val) + len(test)
        size_ok = total_split == len(original)
        self.logger.info(f"Total cells preserved: {total_split} == {len(original)} - "
                        f"{'✓ PASS' if size_ok else '✗ FAIL'}")
        
        # Check class balance in each split
        for name, split in [('Train', train), ('Val', val), ('Test', test)]:
            label_dist = split['label'].value_counts(normalize=True)
            well_served_pct = label_dist.get(1, 0) * 100
            self.logger.info(f"{name} set - Well-served: {well_served_pct:.1f}%, "
                           f"Underserved: {(100-well_served_pct):.1f}%")
        
        # Check that balance is similar across splits (within 5%)
        train_balance = train['label'].mean()
        val_balance = val['label'].mean()
        test_balance = test['label'].mean()
        
        balance_ok = (abs(train_balance - val_balance) < 0.05 and 
                     abs(train_balance - test_balance) < 0.05)
        self.logger.info(f"Class balance similar across splits: {'✓ PASS' if balance_ok else '⚠ WARNING'}")
        
        validation_passed = size_ok
        
        if validation_passed:
            self.logger.info("✓ Split validation PASSED")
        else:
            self.logger.warning("✗ Split validation FAILED")
        
        return validation_passed
    
    def save_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   output_dir: str = "data/processed/features"):
        """
        Save train/val/test splits to parquet files.
        
        Args:
            train: Training set
            val: Validation set
            test: Test set
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        train_path = output_path / "train.parquet"
        val_path = output_path / "val.parquet"
        test_path = output_path / "test.parquet"
        
        train.to_parquet(train_path, index=False)
        val.to_parquet(val_path, index=False)
        test.to_parquet(test_path, index=False)
        
        self.logger.info(f"Splits saved to {output_dir}/")
        self.logger.info(f"  Train: {train_path} ({train_path.stat().st_size / 1024:.2f} KB)")
        self.logger.info(f"  Val: {val_path} ({val_path.stat().st_size / 1024:.2f} KB)")
        self.logger.info(f"  Test: {test_path} ({test_path.stat().st_size / 1024:.2f} KB)")


def main():
    """Main function to prepare and save dataset splits."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize preparator
    preparator = DatasetPreparator()
    
    # Prepare dataset
    train, val, test = preparator.prepare_dataset()
    
    # Save splits
    preparator.save_splits(train, val, test)
    
    # Display summary
    print("\n" + "="*60)
    print("Dataset Preparation Summary")
    print("="*60)
    print(f"Total cells: {len(train) + len(val) + len(test)}")
    print(f"\nSplit sizes:")
    print(f"  Training: {len(train)} cells ({len(train)/(len(train)+len(val)+len(test))*100:.1f}%)")
    print(f"  Validation: {len(val)} cells ({len(val)/(len(train)+len(val)+len(test))*100:.1f}%)")
    print(f"  Test: {len(test)} cells ({len(test)/(len(train)+len(val)+len(test))*100:.1f}%)")
    
    print(f"\nClass distribution:")
    for name, split in [('Train', train), ('Val', val), ('Test', test)]:
        dist = split['label'].value_counts()
        print(f"  {name}: Well-served={dist.get(1, 0)}, Underserved={dist.get(0, 0)}")
    
    print(f"\nFeature columns: {len([col for col in train.columns if col.endswith('_norm')])} normalized features")
    print("="*60)


if __name__ == "__main__":
    main()

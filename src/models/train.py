"""
Model Trainer for Transit Coverage Classification

This module trains multiple classification models with hyperparameter optimization.
Supports Logistic Regression, Random Forest, and Gradient Boosting classifiers.
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains classification models with hyperparameter optimization.
    
    Attributes:
        config: Configuration dictionary with model parameters
        random_seed: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 = all cores)
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config_path: Path to model configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.random_seed = self.config['training']['random_seed']
        self.cv_folds = self.config['training']['cv_folds']
        self.n_jobs = self.config['training']['n_jobs']
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        self.trained_models = {}
        self.search_results = {}
        
        logger.info(f"ModelTrainer initialized with random_seed={self.random_seed}, "
                   f"cv_folds={self.cv_folds}, n_jobs={self.n_jobs}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_datasets(self, 
                     train_path: str = "data/processed/features/train.parquet",
                     val_path: str = "data/processed/features/val.parquet"):
        """
        Load training and validation datasets.
        
        Args:
            train_path: Path to training parquet file
            val_path: Path to validation parquet file
        """
        logger.info("Loading datasets...")
        
        # Load train and validation sets
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        
        logger.info(f"Loaded training set: {len(train_df)} samples")
        logger.info(f"Loaded validation set: {len(val_df)} samples")
        
        # Identify feature columns (exclude cell_id, label, and any metadata)
        # IMPORTANT: Exclude normalized features to avoid circular dependency
        # (labels were generated FROM normalized features, so we can't train ON them)
        feature_cols = [col for col in train_df.columns 
                       if col not in ['cell_id', 'label', 'composite_score', 'threshold_used', 'weights']
                       and not col.endswith('_norm')  # Exclude normalized features
                       and not col.endswith('_normalized')]
        
        logger.info(f"Using raw features: {feature_cols}")
        
        # Split features and labels
        self.X_train = train_df[feature_cols].values
        self.y_train = train_df['label'].values
        self.X_val = val_df[feature_cols].values
        self.y_val = val_df['label'].values
        
        logger.info(f"Feature shape: {self.X_train.shape}")
        logger.info(f"Training labels: {np.sum(self.y_train == 1)} well-served, "
                   f"{np.sum(self.y_train == 0)} underserved")
        logger.info(f"Validation labels: {np.sum(self.y_val == 1)} well-served, "
                   f"{np.sum(self.y_val == 0)} underserved")
    
    def train_logistic_regression(self) -> Tuple[Any, Dict]:
        """
        Train Logistic Regression with GridSearchCV.
        
        Returns:
            Tuple of (best_model, search_results_dict)
        """
        logger.info("="*60)
        logger.info("Training Logistic Regression with GridSearchCV...")
        
        start_time = time.time()
        
        # Get hyperparameter grid from config
        model_config = self.config['models']['logistic_regression']
        param_grid = model_config['param_grid']
        
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        # Create base model
        base_model = LogisticRegression(random_state=self.random_seed)
        
        # Create GridSearchCV with F1 score
        f1_scorer = make_scorer(f1_score, pos_label=1)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=f1_scorer,
            n_jobs=self.n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)
        
        elapsed_time = time.time() - start_time
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation set
        val_f1 = f1_score(self.y_val, best_model.predict(self.X_val), pos_label=1)
        
        # Compile results
        results = {
            'model_name': 'Logistic Regression',
            'search_method': 'GridSearchCV',
            'best_params': grid_search.best_params_,
            'best_cv_f1': grid_search.best_score_,
            'val_f1': val_f1,
            'training_time_seconds': elapsed_time,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV F1 score: {results['best_cv_f1']:.4f}")
        logger.info(f"Validation F1 score: {results['val_f1']:.4f}")
        logger.info(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        return best_model, results
    
    def train_random_forest(self) -> Tuple[Any, Dict]:
        """
        Train Random Forest with RandomizedSearchCV.
        
        Returns:
            Tuple of (best_model, search_results_dict)
        """
        logger.info("="*60)
        logger.info("Training Random Forest with RandomizedSearchCV...")
        
        start_time = time.time()
        
        # Get hyperparameter distributions from config
        model_config = self.config['models']['random_forest']
        param_distributions = model_config['param_distributions']
        n_iter = model_config['n_iter']
        
        logger.info(f"Parameter distributions: {param_distributions}")
        logger.info(f"Random iterations: {n_iter}")
        
        # Create base model
        base_model = RandomForestClassifier(random_state=self.random_seed)
        
        # Create RandomizedSearchCV with F1 score
        f1_scorer = make_scorer(f1_score, pos_label=1)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=self.cv_folds,
            scoring=f1_scorer,
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=self.random_seed,
            return_train_score=True
        )
        
        # Fit random search
        random_search.fit(self.X_train, self.y_train)
        
        elapsed_time = time.time() - start_time
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Evaluate on validation set
        val_f1 = f1_score(self.y_val, best_model.predict(self.X_val), pos_label=1)
        
        # Compile results
        results = {
            'model_name': 'Random Forest',
            'search_method': 'RandomizedSearchCV',
            'best_params': random_search.best_params_,
            'best_cv_f1': random_search.best_score_,
            'val_f1': val_f1,
            'training_time_seconds': elapsed_time,
            'cv_results': random_search.cv_results_
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV F1 score: {results['best_cv_f1']:.4f}")
        logger.info(f"Validation F1 score: {results['val_f1']:.4f}")
        logger.info(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        return best_model, results
    
    def train_gradient_boosting(self) -> Tuple[Any, Dict]:
        """
        Train Gradient Boosting with RandomizedSearchCV.
        
        Returns:
            Tuple of (best_model, search_results_dict)
        """
        logger.info("="*60)
        logger.info("Training Gradient Boosting with RandomizedSearchCV...")
        
        start_time = time.time()
        
        # Get hyperparameter distributions from config
        model_config = self.config['models']['gradient_boosting']
        param_distributions = model_config['param_distributions']
        n_iter = model_config['n_iter']
        
        logger.info(f"Parameter distributions: {param_distributions}")
        logger.info(f"Random iterations: {n_iter}")
        
        # Create base model
        base_model = GradientBoostingClassifier(random_state=self.random_seed)
        
        # Create RandomizedSearchCV with F1 score
        f1_scorer = make_scorer(f1_score, pos_label=1)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=self.cv_folds,
            scoring=f1_scorer,
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=self.random_seed,
            return_train_score=True
        )
        
        # Fit random search
        random_search.fit(self.X_train, self.y_train)
        
        elapsed_time = time.time() - start_time
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Evaluate on validation set
        val_f1 = f1_score(self.y_val, best_model.predict(self.X_val), pos_label=1)
        
        # Compile results
        results = {
            'model_name': 'Gradient Boosting',
            'search_method': 'RandomizedSearchCV',
            'best_params': random_search.best_params_,
            'best_cv_f1': random_search.best_score_,
            'val_f1': val_f1,
            'training_time_seconds': elapsed_time,
            'cv_results': random_search.cv_results_
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV F1 score: {results['best_cv_f1']:.4f}")
        logger.info(f"Validation F1 score: {results['val_f1']:.4f}")
        logger.info(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        return best_model, results
    
    def train_all_models(self) -> Dict[str, Tuple[Any, Dict]]:
        """
        Train all models sequentially.
        
        Returns:
            Dictionary mapping model names to (model, results) tuples
        """
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        overall_start_time = time.time()
        
        # Train Logistic Regression
        lr_model, lr_results = self.train_logistic_regression()
        self.trained_models['logistic_regression'] = lr_model
        self.search_results['logistic_regression'] = lr_results
        
        # Train Random Forest
        rf_model, rf_results = self.train_random_forest()
        self.trained_models['random_forest'] = rf_model
        self.search_results['random_forest'] = rf_results
        
        # Train Gradient Boosting
        gb_model, gb_results = self.train_gradient_boosting()
        self.trained_models['gradient_boosting'] = gb_model
        self.search_results['gradient_boosting'] = gb_results
        
        overall_elapsed_time = time.time() - overall_start_time
        
        logger.info("="*60)
        logger.info("ALL MODELS TRAINED SUCCESSFULLY")
        logger.info(f"Total training time: {overall_elapsed_time:.2f} seconds "
                   f"({overall_elapsed_time/60:.2f} minutes)")
        logger.info("="*60)
        
        return {
            'logistic_regression': (lr_model, lr_results),
            'random_forest': (rf_model, rf_results),
            'gradient_boosting': (gb_model, gb_results)
        }
    
    def save_models(self, output_dir: str = "models/transit_coverage"):
        """
        Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        logger.info("Saving trained models...")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_file = output_path / f"{model_name}.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            file_size_kb = model_file.stat().st_size / 1024
            logger.info(f"Saved {model_name} to {model_file} ({file_size_kb:.2f} KB)")
    
    def save_search_results(self, output_dir: str = "models/transit_coverage"):
        """
        Save hyperparameter search results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        logger.info("Saving search results...")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "search_results.pkl"
        
        with open(results_file, 'wb') as f:
            pickle.dump(self.search_results, f)
        
        file_size_kb = results_file.stat().st_size / 1024
        logger.info(f"Saved search results to {results_file} ({file_size_kb:.2f} KB)")
        
        # Also save a human-readable summary
        summary_file = output_path / "training_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for model_name, results in self.search_results.items():
                f.write(f"Model: {results['model_name']}\n")
                f.write(f"Search Method: {results['search_method']}\n")
                f.write(f"Best Parameters: {results['best_params']}\n")
                f.write(f"Best CV F1 Score: {results['best_cv_f1']:.4f}\n")
                f.write(f"Validation F1 Score: {results['val_f1']:.4f}\n")
                f.write(f"Training Time: {results['training_time_seconds']:.2f} seconds "
                       f"({results['training_time_seconds']/60:.2f} minutes)\n")
                f.write("\n" + "-"*60 + "\n\n")
        
        logger.info(f"Saved training summary to {summary_file}")
    
    def select_best_model(self) -> Tuple[str, Any, Dict]:
        """
        Select the best model based on validation F1 score.
        
        Returns:
            Tuple of (model_name, best_model, results)
        """
        logger.info("Selecting best model based on validation F1 score...")
        
        best_model_name = None
        best_val_f1 = -1
        
        for model_name, results in self.search_results.items():
            val_f1 = results['val_f1']
            logger.info(f"{results['model_name']}: Validation F1 = {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_name = model_name
        
        best_model = self.trained_models[best_model_name]
        best_results = self.search_results[best_model_name]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST MODEL: {best_results['model_name']}")
        logger.info(f"Validation F1 Score: {best_val_f1:.4f}")
        logger.info(f"Best Parameters: {best_results['best_params']}")
        logger.info(f"{'='*60}")
        
        return best_model_name, best_model, best_results
    
    def save_best_model(self, output_dir: str = "models/transit_coverage"):
        """
        Save the best model as best_model.pkl.
        
        Args:
            output_dir: Directory to save best model
        """
        best_model_name, best_model, best_results = self.select_best_model()
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        best_model_file = output_path / "best_model.pkl"
        
        with open(best_model_file, 'wb') as f:
            pickle.dump(best_model, f)
        
        file_size_kb = best_model_file.stat().st_size / 1024
        logger.info(f"Saved best model to {best_model_file} ({file_size_kb:.2f} KB)")
        
        # Save metadata about best model
        metadata = {
            'model_name': best_results['model_name'],
            'model_type': best_model_name,
            'best_params': best_results['best_params'],
            'val_f1_score': best_results['val_f1'],
            'cv_f1_score': best_results['best_cv_f1'],
            'training_time_seconds': best_results['training_time_seconds']
        }
        
        metadata_file = output_path / "best_model_metadata.pkl"
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved best model metadata to {metadata_file}")
    
    def validate_training_time(self, max_time_minutes: float = 40.0):
        """
        Validate that total training time is within constraints.
        
        Args:
            max_time_minutes: Maximum allowed training time in minutes
        """
        logger.info("Validating training time constraint...")
        
        total_time = sum(results['training_time_seconds'] 
                        for results in self.search_results.values())
        total_time_minutes = total_time / 60
        
        logger.info(f"Total training time: {total_time:.2f} seconds ({total_time_minutes:.2f} minutes)")
        logger.info(f"Maximum allowed time: {max_time_minutes:.2f} minutes")
        
        if total_time_minutes <= max_time_minutes:
            logger.info(f"✓ PASS: Training time within constraint")
        else:
            logger.warning(f"✗ WARNING: Training time exceeded constraint by "
                          f"{total_time_minutes - max_time_minutes:.2f} minutes")
            logger.warning("Consider reducing cv_folds or n_iter in config")


def main():
    """Main function to train all models."""
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load datasets
    trainer.load_datasets()
    
    # Train all models
    trainer.train_all_models()
    
    # Save all models and results
    trainer.save_models()
    trainer.save_search_results()
    
    # Select and save best model
    trainer.save_best_model()
    
    # Validate training time
    trainer.validate_training_time()
    
    # Display final summary
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    
    for model_name, results in trainer.search_results.items():
        print(f"\n{results['model_name']}:")
        print(f"  CV F1 Score: {results['best_cv_f1']:.4f}")
        print(f"  Validation F1 Score: {results['val_f1']:.4f}")
        print(f"  Training Time: {results['training_time_seconds']/60:.2f} minutes")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

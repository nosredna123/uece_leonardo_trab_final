"""
Model Evaluator for Transit Coverage Classification

This module evaluates trained models and generates comprehensive performance metrics,
confusion matrices, ROC curves, and feature importance visualizations.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Evaluates classification models and generates performance reports.
    
    Attributes:
        models_dir: Directory containing trained models
        output_dir: Directory for reports and visualizations
        models: Dictionary of loaded models
        X_test: Test features
        y_test: Test labels
        metrics: Dictionary of evaluation metrics
    """
    
    def __init__(self, 
                 models_dir: str = "models/transit_coverage",
                 output_dir: str = "reports"):
        """
        Initialize ModelEvaluator.
        
        Args:
            models_dir: Directory containing trained models
            output_dir: Directory for output reports and figures
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.metrics = {}
        
        logger.info(f"ModelEvaluator initialized")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_test_dataset(self, test_path: str = "data/processed/features/test.parquet"):
        """
        Load test dataset.
        
        Args:
            test_path: Path to test parquet file
        """
        logger.info("Loading test dataset...")
        
        test_df = pd.read_parquet(test_path)
        logger.info(f"Loaded test set: {len(test_df)} samples")
        
        # Identify feature columns (exclude cell_id, label, and metadata)
        # IMPORTANT: Exclude normalized features to match training
        # (models were trained WITHOUT normalized features to avoid circular dependency)
        feature_cols = [col for col in test_df.columns 
                       if col not in ['cell_id', 'label', 'composite_score', 'threshold_used', 'weights']
                       and not col.endswith('_norm')  # Exclude normalized features
                       and not col.endswith('_normalized')]
        
        logger.info(f"Using features: {feature_cols}")
        
        self.feature_names = feature_cols
        self.X_test = test_df[feature_cols].values
        self.y_test = test_df['label'].values
        
        logger.info(f"Test features shape: {self.X_test.shape}")
        logger.info(f"Test labels: {np.sum(self.y_test == 1)} well-served, "
                   f"{np.sum(self.y_test == 0)} underserved")
    
    def load_models(self):
        """Load all trained models from models directory."""
        logger.info("Loading trained models...")
        
        model_files = {
            'logistic_regression': self.models_dir / 'logistic_regression.pkl',
            'random_forest': self.models_dir / 'random_forest.pkl',
            'gradient_boosting': self.models_dir / 'gradient_boosting.pkl'
        }
        
        for model_name, model_path in model_files.items():
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"Loaded {model_name}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def calculate_metrics(self, model_name: str, model: Any) -> Dict:
        """
        Calculate evaluation metrics for a model.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Calculating metrics for {model_name}...")
        
        # Get predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name.replace('_', ' ').title(),
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, pos_label=1),
            'recall': recall_score(self.y_test, y_pred, pos_label=1),
            'f1_score': f1_score(self.y_test, y_pred, pos_label=1),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}, "
                   f"AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """
        Evaluate all loaded models and compile metrics.
        
        Returns:
            DataFrame with all model metrics
        """
        logger.info("="*60)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*60)
        
        for model_name, model in self.models.items():
            self.metrics[model_name] = self.calculate_metrics(model_name, model)
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'algorithm'})
        
        logger.info("\nMetrics Summary:")
        logger.info(f"\n{metrics_df.to_string(index=False)}")
        
        return metrics_df
    
    def generate_confusion_matrix(self, model_name: str, model: Any):
        """
        Generate and save confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            model: Trained model object
        """
        logger.info(f"Generating confusion matrix for {model_name}...")
        
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Underserved', 'Well-served'],
                   yticklabels=['Underserved', 'Well-served'])
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save figure
        output_file = self.figures_dir / f'confusion_matrix_{model_name}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_file}")
    
    def generate_roc_curves_comparison(self):
        """Generate ROC curves for all models on the same plot."""
        logger.info("Generating ROC curves comparison...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{model_name.replace("_", " ").title()} (AUC = {auc:.4f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison - All Models')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_file = self.figures_dir / 'roc_curves_comparison.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ROC curves to {output_file}")
    
    def extract_feature_importance(self, model_name: str, model: Any) -> np.ndarray:
        """
        Extract feature importance from a model.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            
        Returns:
            Array of feature importances (normalized to [0, 1])
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (RF, GB)
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models (Logistic Regression)
            importances = np.abs(model.coef_[0])
        else:
            logger.warning(f"Cannot extract feature importance from {model_name}")
            return np.zeros(len(self.feature_names))
        
        # Normalize to [0, 1]
        if importances.max() > 0:
            importances = importances / importances.max()
        
        return importances
    
    def generate_feature_importance_comparison(self):
        """Generate feature importance comparison plot for all models."""
        logger.info("Generating feature importance comparison...")
        
        # Extract importances for all models
        importance_data = {}
        for model_name, model in self.models.items():
            importance_data[model_name] = self.extract_feature_importance(model_name, model)
        
        # Create DataFrame
        importance_df = pd.DataFrame(
            importance_data,
            index=self.feature_names
        )
        importance_df.columns = [col.replace('_', ' ').title() for col in importance_df.columns]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (col, ax) in enumerate(zip(importance_df.columns, axes)):
            importance_df[col].sort_values(ascending=True).plot(
                kind='barh', ax=ax, color=sns.color_palette("husl", 3)[idx]
            )
            ax.set_title(f'{col}\nFeature Importance')
            ax.set_xlabel('Normalized Importance')
            ax.set_ylabel('Feature')
            ax.grid(True, alpha=0.3)
        
        # Save figure
        output_file = self.figures_dir / 'feature_importance_comparison.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance comparison to {output_file}")
        
        # Also save as CSV
        csv_file = self.tables_dir / 'feature_importance.csv'
        importance_df.to_csv(csv_file)
        logger.info(f"Saved feature importance data to {csv_file}")
    
    def save_metrics_table(self, metrics_df: pd.DataFrame):
        """
        Save metrics comparison table to CSV.
        
        Args:
            metrics_df: DataFrame with model metrics
        """
        output_file = self.tables_dir / 'model_comparison.csv'
        metrics_df.to_csv(output_file, index=False)
        logger.info(f"Saved metrics table to {output_file}")
    
    def generate_classification_report(self):
        """Generate classification report for best model."""
        logger.info("Generating classification report for best model...")
        
        # Load best model
        best_model_path = self.models_dir / 'best_model.pkl'
        with open(best_model_path, 'rb') as f:
            best_model = pickle.load(f)
        
        # Load metadata to get model name
        metadata_path = self.models_dir / 'best_model_metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            best_model_name = metadata.get('model_name', 'Best Model')
        else:
            best_model_name = 'Best Model'
        
        # Generate predictions
        y_pred = best_model.predict(self.X_test)
        
        # Generate classification report
        report = classification_report(
            self.y_test, y_pred,
            target_names=['Underserved', 'Well-served'],
            digits=4
        )
        
        # Save to file
        output_file = self.tables_dir / 'classification_report.txt'
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"CLASSIFICATION REPORT - {best_model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(report)
            f.write("\n" + "="*60 + "\n")
        
        logger.info(f"Saved classification report to {output_file}")
        logger.info(f"\n{report}")
    
    def validate_f1_threshold(self, threshold: float = 0.70):
        """
        Validate that best model meets F1 score threshold.
        
        Args:
            threshold: Minimum required F1 score
        """
        logger.info(f"Validating F1 score threshold ({threshold:.2f})...")
        
        # Load best model
        best_model_path = self.models_dir / 'best_model.pkl'
        with open(best_model_path, 'rb') as f:
            best_model = pickle.load(f)
        
        # Calculate F1 score
        y_pred = best_model.predict(self.X_test)
        test_f1 = f1_score(self.y_test, y_pred, pos_label=1)
        
        logger.info(f"Best model test F1 score: {test_f1:.4f}")
        logger.info(f"Required threshold: {threshold:.2f}")
        
        if test_f1 >= threshold:
            logger.info(f"✓ PASS: F1 score meets requirement ({test_f1:.4f} >= {threshold:.2f})")
        else:
            logger.warning(f"✗ FAIL: F1 score below requirement ({test_f1:.4f} < {threshold:.2f})")
        
        return test_f1 >= threshold
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        logger.info("="*60)
        logger.info("STARTING FULL EVALUATION PIPELINE")
        logger.info("="*60)
        
        # Load data and models
        self.load_test_dataset()
        self.load_models()
        
        # Evaluate all models
        metrics_df = self.evaluate_all_models()
        
        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        
        # Confusion matrices for each model
        for model_name, model in self.models.items():
            self.generate_confusion_matrix(model_name, model)
        
        # ROC curves comparison
        self.generate_roc_curves_comparison()
        
        # Feature importance comparison
        self.generate_feature_importance_comparison()
        
        # Save metrics table
        self.save_metrics_table(metrics_df)
        
        # Generate classification report
        self.generate_classification_report()
        
        # Validate F1 threshold
        self.validate_f1_threshold()
        
        logger.info("="*60)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*60)


def main():
    """Main function to run full evaluation."""
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()
    
    # Display summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print("\nGenerated Files:")
    print("\nFigures:")
    for fig_file in sorted(evaluator.figures_dir.glob("*.png")):
        print(f"  - {fig_file.name}")
    print("\nTables:")
    for table_file in sorted(evaluator.tables_dir.glob("*")):
        print(f"  - {table_file.name}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

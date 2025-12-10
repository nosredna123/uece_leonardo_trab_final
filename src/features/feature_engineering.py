"""
Feature engineering utilities
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class FeatureEngineer:
    """
    Class for feature engineering operations
    """
    
    def __init__(self):
        self.feature_selector = None
        self.selected_features = None
    
    def create_polynomial_features(self, df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features for
            degree: Degree of polynomial
            
        Returns:
            DataFrame with polynomial features
        """
        df = df.copy()
        
        for col in columns:
            for d in range(2, degree + 1):
                df[f'{col}_pow_{d}'] = df[col] ** d
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, col_pairs: list) -> pd.DataFrame:
        """
        Create interaction features between column pairs
        
        Args:
            df: Input DataFrame
            col_pairs: List of column pairs [(col1, col2), ...]
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        for col1, col2 in col_pairs:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return df
    
    def create_binned_features(self, df: pd.DataFrame, column: str, bins: int = 5) -> pd.DataFrame:
        """
        Create binned categorical features from numerical columns
        
        Args:
            df: Input DataFrame
            column: Column to bin
            bins: Number of bins
            
        Returns:
            DataFrame with binned feature
        """
        df = df.copy()
        df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=False)
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 10, method: str = 'f_classif'):
        """
        Select top k features using statistical tests
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            method: Selection method ('f_classif' or 'mutual_info')
            
        Returns:
            X with selected features
        """
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Store selected feature names
        self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            DataFrame with feature importance scores
        """
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        })
        
        return scores.sort_values('score', ascending=False)


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    print("FeatureEngineer initialized successfully")
